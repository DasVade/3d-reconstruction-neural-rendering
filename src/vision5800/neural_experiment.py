import csv
import time

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from .artifacts import save_image, save_metrics
from .cli import ROOT
from .image_field import model_2d, normalize_coord
from .nerf import nerf_model, get_rays, stratified_sampling, positional_encoding, volumetric_rendering


def render_rays(model, origins, directions, samples, xf, df):
    # The course depth convention and deterministic sampling are preserved.
    points, depths = stratified_sampling(origins[:, None, :], directions[:, None, :], 0.667, 2.0, samples)
    x = positional_encoding(points.reshape(-1, 3), xf)
    d = F.normalize(directions, dim=-1)[:, None, :].expand(-1, samples, -1)
    d = positional_encoding(d.reshape(-1, 3), df)
    rgb, sigma = model(x, d)
    return volumetric_rendering(rgb.reshape(-1, 1, samples, 3), sigma.reshape(-1, 1, samples), depths,
                               directions[:, None, :]).reshape(-1, 3)


@torch.no_grad()
def render_view(model, h, w, intrinsics, pose, samples, xf, df, chunk):
    origins, directions = get_rays(h, w, intrinsics, pose[:3, :3], pose[:3, 3])
    origins, directions = origins.reshape(-1, 3), directions.reshape(-1, 3)
    return torch.cat([render_rays(model, origins[i:i+chunk], directions[i:i+chunk], samples, xf, df)
                      for i in range(0, h*w, chunk)]).reshape(h, w, 3)


def psnr(prediction, target):
    mse = F.mse_loss(prediction, target).item()
    return mse, -10 * np.log10(max(mse, 1e-12))


def load_state(path, device):
    if not path.is_file():
        raise FileNotFoundError(f'Missing weights: {path}. Run scripts/prepare_data.py with the supplied ZIP.')
    return torch.load(path, map_location=device, weights_only=True)


def run_neural(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu or install a compatible CUDA PyTorch build.')
    args.output.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    records = []
    metadata = {'experiment': args.task, 'mode': args.mode, 'device': str(device), 'seed': args.seed,
                'scale': args.scale, 'iterations': args.iterations if args.mode == 'train' else 0}
    if args.task == 'image':
        source = np.asarray(imageio.imread(args.data / 'aurora.jpg'))[..., :3].astype(np.float32) / 255.0
        if args.scale != 1:
            source = cv2.resize(source, (max(1, round(source.shape[1]*args.scale)), max(1, round(source.shape[0]*args.scale))), interpolation=cv2.INTER_AREA)
        target = torch.as_tensor(source, device=device)
        h, w = source.shape[:2]
        coords = normalize_coord(h, w, args.freq).to(device)
        model = model_2d(num_frequencies=args.freq).to(device)
        checkpoint = args.checkpoint or ROOT / f'checkpoints/model_2d_{args.freq}freq.pt'
        if args.mode == 'eval':
            model.load_state_dict(load_state(checkpoint, device))
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            for step in range(args.iterations):
                idx = torch.randperm(h*w, device=device)[:args.rays_per_step]
                optimizer.zero_grad()
                loss = F.mse_loss(model(coords[idx]), target.reshape(-1, 3)[idx])
                loss.backward()
                optimizer.step()
                if step == 0 or (step+1) % max(1, args.iterations//10) == 0:
                    row = {'step': step+1, 'train_mse': loss.item()}
                    records.append(row)
                    print(row, flush=True)
            checkpoint = args.output / f'model_2d_{args.freq}freq.pt'
            torch.save(model.state_dict(), checkpoint)
        model.eval()
        with torch.no_grad():
            prediction = torch.cat([model(coords[i:i+8192]) for i in range(0, h*w, 8192)]).reshape(h, w, 3)
        mse, quality = psnr(prediction, target)
        metadata.update(frequencies=args.freq, image_shape=[h, w], reconstruction_mse=mse,
                        reconstruction_psnr_db=quality, note='Image fitting quality on the input image; not a generalization metric.')
    else:
        with np.load(args.data / 'lego_data.npz', allow_pickle=False) as data:
            source = data['images'][..., :3].astype(np.float32)
            poses = torch.as_tensor(data['poses'].astype(np.float32), device=device)
            intrinsics = data['intrinsics'].astype(np.float32).copy()
        h0, w0 = source.shape[1:3]
        h, w = max(1, round(h0*args.scale)), max(1, round(w0*args.scale))
        if (h, w) != (h0, w0):
            source = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA) for x in source])
            intrinsics = np.diag([w/w0, h/h0, 1]).astype(np.float32) @ intrinsics
        images = torch.as_tensor(source, device=device)
        intrinsics = torch.as_tensor(intrinsics, device=device)
        checkpoint = args.checkpoint or ROOT / 'checkpoints/model_nerf.pt'
        xf, df, width = 10, 4, 256
        if args.mode == 'eval':
            state = load_state(checkpoint, device)
            width, x_dim = state['layers.layer_1.weight'].shape
            xf = (x_dim//3-1)//2
            df = ((state['layers.layer_11.weight'].shape[1]-width)//3-1)//2
        model = nerf_model(width, xf, df).to(device)
        if args.mode == 'eval':
            model.load_state_dict(state)
        else:
            # Fresh split: training 0:90, validation 90:100, test 100:.
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            for step in range(args.iterations):
                index = torch.randint(90, (1,), device=device).item()
                origins, directions = get_rays(h, w, intrinsics, poses[index, :3, :3], poses[index, :3, 3])
                indices = torch.randperm(h*w, device=device)[:args.rays_per_step]
                optimizer.zero_grad()
                pred = render_rays(model, origins.reshape(-1, 3)[indices], directions.reshape(-1, 3)[indices], args.samples, xf, df)
                loss = F.mse_loss(pred, images[index].reshape(-1, 3)[indices])
                loss.backward()
                optimizer.step()
                if step == 0 or (step+1) % max(1, args.iterations//10) == 0:
                    row = {'step': step+1, 'train_mse': loss.item(), 'train_view': index}
                    records.append(row)
                    print(row, flush=True)
            checkpoint = args.output / 'model_nerf.pt'
            torch.save(model.state_dict(), checkpoint)
        model.eval()
        # Index 101 was used for selection by the original training script; label it explicitly.
        eval_index = 101 if args.mode == 'eval' else 100
        target = images[eval_index]
        prediction = render_view(model, h, w, intrinsics, poses[eval_index], args.samples, xf, df, args.ray_chunk)
        mse, quality = psnr(prediction, target)
        metadata.update(image_shape=[h, w], samples=args.samples, position_frequencies=xf, direction_frequencies=df,
                        eval_view=eval_index, eval_mse=mse, eval_psnr_db=quality,
                        note='Supplied checkpoint: view 101 was used for model selection, so this is validation reconstruction quality.'
                        if args.mode == 'eval' else 'Fresh training uses views 0:90. Views 90:100 are reserved for validation; this run evaluates test view 100 after fixed-step training.')
        if args.render_orbit:
            frames = []
            indices = np.linspace(0, len(poses)-1, min(args.render_orbit, len(poses)), dtype=int)
            for index in indices:
                print(f'Rendering supplied camera pose {index}', flush=True)
                frame = render_view(model, h, w, intrinsics, poses[index], args.samples, xf, df, args.ray_chunk)
                frames.append(np.round(frame.cpu().numpy()*255).astype(np.uint8))
            imageio.mimsave(args.output / 'camera_sweep.gif', frames, duration=250, loop=0)
            metadata['sweep_pose_indices'] = indices.tolist()
            metadata['sweep_note'] = 'Uses supplied camera poses, including training poses; not an unseen-view benchmark.'
    pred_np, target_np = prediction.detach().cpu().numpy(), target.detach().cpu().numpy()
    save_image(args.output / 'prediction.png', pred_np)
    save_image(args.output / 'target.png', target_np)
    save_image(args.output / 'comparison.png', np.concatenate([target_np, pred_np], axis=1))
    if records:
        with open(args.output / 'training.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    metadata.update(checkpoint=str(checkpoint), elapsed_seconds=time.perf_counter()-t)
    save_metrics(args.output, metadata)
