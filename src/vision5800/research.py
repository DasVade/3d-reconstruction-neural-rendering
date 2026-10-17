"""Audited experiments: explicit splits, validation selection, finite-volume rendering."""
import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

from .artifacts import save_image, save_map, save_ply, save_cloud_preview
from .cli import ROOT
from .image_field import model_2d, normalize_coord
from .nerf import get_rays, nerf_model, positional_encoding, stratified_sampling, volumetric_rendering


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def hardware():
    info = {'platform': platform.platform(), 'python': platform.python_version(), 'torch': torch.__version__,
            'logical_cpu_count': os.cpu_count(), 'cuda_available': torch.cuda.is_available(), 'cuda_runtime': torch.version.cuda}
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            info['cpu'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(x, ctypes.c_ulonglong) for x in
                      ['total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'extended']]
        status = Status();status.length = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        info['ram_bytes'] = status.total_phys
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_bytes=p.total_memory)
    try:
        info['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return info


def quality(prediction, target):
    a = np.asarray(prediction, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    mse = np.mean((a-b)**2)
    # Wang-style SSIM: 11x11 Gaussian, sigma=1.5, population covariance,
    # RGB channel mean, data_range=1, valid center crop (5px border).
    kwargs = dict(sigma=(1.5, 1.5, 0), truncate=3.5, mode='reflect')
    ma, mb = gaussian_filter(a, **kwargs), gaussian_filter(b, **kwargs)
    va = gaussian_filter(a*a, **kwargs)-ma*ma
    vb = gaussian_filter(b*b, **kwargs)-mb*mb
    cov = gaussian_filter(a*b, **kwargs)-ma*mb
    ssim = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return {'psnr_db': float(-10*np.log10(max(mse, 1e-12))), 'ssim': float(ssim[5:-5, 5:-5].mean()), 'mse': float(mse)}


def rays(model, origins, directions, samples, xf, df, near=.667, far=2.):
    points, depths = stratified_sampling(origins[:, None], directions[:, None], near, far, samples)
    x = positional_encoding(points.reshape(-1, 3), xf)
    d = F.normalize(directions, dim=-1)[:, None].expand(-1, samples, -1)
    rgb, raw = model(x, positional_encoding(d.reshape(-1, 3), df))
    return volumetric_rendering(rgb.reshape(-1, 1, samples, 3), raw.reshape(-1, 1, samples), depths,
                               directions[:, None], (far-near)/samples).reshape(-1, 3)


@torch.no_grad()
def view(model, origin, direction, config, samples=None):
    return torch.cat([rays(model, origin[i:i+512], direction[i:i+512], samples or config['samples'],
                           config['xf'], config['df']) for i in range(0, len(origin), 512)])


def dataset(device):
    path = ROOT / 'data/nerf/lego_data.npz'
    with np.load(path) as z:
        images = torch.tensor(z['images'], device=device)
        poses = torch.tensor(z['poses'], device=device)
        k = torch.tensor(z['intrinsics'], device=device)
    h, w = images.shape[1:3]
    origins, directions = zip(*[get_rays(h, w, k, p[:3, :3], p[:3, 3]) for p in poses])
    return images, torch.stack(origins).reshape(len(images), -1, 3), torch.stack(directions).reshape(len(images), -1, 3), hashlib.sha256(path.read_bytes()).hexdigest()


def initialize(config, device):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    model = nerf_model(config['width'], config['xf'], config['df']).to(device)
    # Finite softplus density starts low but nonzero, preserving gradients.
    with torch.no_grad():
        model.layers['layer_9'].bias.fill_(-1.)
    return model


def sync(device):
    if device.type == 'cuda':torch.cuda.synchronize()


def train(config, output, benchmark=False):
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    images, origins, directions, digest = dataset(device)
    h, w = images.shape[1:3]
    train_ids = list(range(90));val_ids = [90, 92, 94, 96, 98];test_ids = list(range(100, 106))
    # 91,93,95,97,99 remain unused; never silently merged into training.
    model = initialize(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    generator = torch.Generator(device=device).manual_seed(config['seed']+1)
    log = [];best = -float('inf');best_step = None
    sync(device);start = time.perf_counter()
    if device.type == 'cuda':torch.cuda.reset_peak_memory_stats()
    measured = None
    for step in range(1, config['steps']+1):
        if step == 11:
            sync(device);measured = time.perf_counter()
        ids = torch.randint(len(train_ids), (config['batch'],), generator=generator, device=device)
        pixels = torch.randint(h*w, (config['batch'],), generator=generator, device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = rays(model, origins[ids, pixels], directions[ids, pixels], config['samples'], config['xf'], config['df'])
        loss = F.mse_loss(prediction, images.reshape(len(images), -1, 3)[ids, pixels])
        loss.backward();optimizer.step()
        for group in optimizer.param_groups:
            group['lr'] = config['lr'] * (.1**(step/10000))
        if step == 1 or step % config['log_every'] == 0:
            row = {'step': step, 'train_mse': loss.item(), 'train_psnr_db': -10*np.log10(max(loss.item(),1e-12)),
                   'elapsed_seconds': time.perf_counter()-start}
            if not benchmark and (step % config['validate_every'] == 0 or step == config['steps']):
                model.eval()
                qs = [quality(view(model, origins[i], directions[i], config).reshape(h,w,3).cpu(), images[i].cpu()) for i in val_ids]
                row['val_psnr_db'] = float(np.mean([q['psnr_db'] for q in qs]))
                row['val_ssim'] = float(np.mean([q['ssim'] for q in qs]))
                if row['val_psnr_db'] > best:
                    best = row['val_psnr_db'];best_step = step
                    torch.save({'state_dict': model.state_dict(), 'config':config, 'step':step, 'val_psnr_db':best}, output/'best.pt')
                model.train()
            log.append(row);print(json.dumps(row), flush=True)
            write_json(output/'curve.json', log)
    sync(device);loop_end=time.perf_counter();elapsed = loop_end-start
    result = {'config':config, 'hardware':hardware(), 'dataset_sha256':digest, 'train_views':train_ids,
              'validation_views':val_ids, 'test_views':test_ids, 'elapsed_seconds':elapsed,
              'best_step':best_step, 'best_validation_psnr_db':best if not benchmark else None,
              'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
              'benchmark':benchmark, 'precision':'float32; no mixed precision; TF32 default disabled',
              'seconds_per_training_step_after_warmup':(loop_end-measured)/(config['steps']-10) if benchmark and measured else None}
    write_json(output/'run.json',result)
    if not benchmark:
        with open(output/'curve.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['step','train_mse','train_psnr_db','elapsed_seconds','val_psnr_db','val_ssim'])
            writer.writeheader();writer.writerows(log)
    return result


def evaluate(run, output, validation_sweep=False):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    checkpoint=torch.load(run/'best.pt',map_location=device,weights_only=True)
    config=checkpoint['config'];model=initialize(config,device);model.load_state_dict(checkpoint['state_dict']);model.eval()
    images,origins,directions,digest=dataset(device);h,w=images.shape[1:3]
    ids=[90,92,94,96,98] if validation_sweep else list(range(100,106))
    counts=[16,32,64,128] if validation_sweep else [config['samples']]
    rows=[];output.mkdir(parents=True,exist_ok=True)
    for samples in counts:
        # Warm up each inference configuration before timing.
        view(model,origins[ids[0]][:512],directions[ids[0]][:512],config,samples)
        for i in ids:
            sync(device);start=time.perf_counter()
            pred=view(model,origins[i],directions[i],config,samples).reshape(h,w,3)
            sync(device);elapsed=time.perf_counter()-start
            a=pred.cpu().numpy();b=images[i].cpu().numpy();q=quality(a,b)
            row={'view':i,'samples':samples,'render_seconds':elapsed,**q};rows.append(row)
            folder=output/f's{samples}_view{i}';folder.mkdir(exist_ok=True)
            save_image(folder/'target.png',b);save_image(folder/'prediction.png',a)
            # Fixed scale absolute RGB error, not per-image normalized color maps.
            save_image(folder/'error.png',np.abs(a-b))
            save_image(folder/'comparison.png',np.concatenate([b,a,np.abs(a-b)],axis=1))
        print('evaluated samples',samples,flush=True)
    result={'checkpoint_step':checkpoint['step'],'checkpoint_sha256':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
            'dataset_sha256':digest,'split':'validation' if validation_sweep else 'test',
            'experiment':'render-only sample count; weights frozen; training samples fixed at '+str(config['samples']) if validation_sweep else 'final fixed-configuration independent test views',
            'rows':rows,'summary':[{ 'samples':s,**{k:float(np.mean([r[k] for r in rows if r['samples']==s])) for k in ['psnr_db','ssim','render_seconds']}} for s in counts]}
    write_json(output/'metrics.json',result);return result


def image_ablation(output, steps=2000):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    target=torch.tensor(cv2.cvtColor(cv2.imread(str(ROOT/'data/nerf/aurora.jpg')),cv2.COLOR_BGR2RGB)/255.,dtype=torch.float32,device=device)
    h,w=target.shape[:2];rows=[]
    for seed in [5670,5671,5672]:
        for freq in [0,2,6]:
            torch.manual_seed(seed);model=model_2d(num_frequencies=freq).to(device)
            coords=normalize_coord(h,w,freq).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=5e-4)
            generator=torch.Generator(device=device).manual_seed(seed+100)
            curve=[];sync(device);start=time.perf_counter()
            for step in range(1,steps+1):
                ids=torch.randint(h*w,(1024,),generator=generator,device=device)
                optimizer.zero_grad(set_to_none=True)
                loss=F.mse_loss(model(coords[ids]),target.reshape(-1,3)[ids]);loss.backward();optimizer.step()
                if step%100==0:curve.append({'step':step,'mse':loss.item()})
            sync(device);elapsed=time.perf_counter()-start
            with torch.no_grad():a=model(coords).reshape(h,w,3).cpu().numpy()
            folder=output/f'L{freq}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
            q=quality(a,target.cpu().numpy());rows.append({'seed':seed,'frequency':freq,'seconds':elapsed,**q})
            save_image(folder/'prediction.png',a);write_json(folder/'curve.json',curve)
            torch.save(model.state_dict(),folder/'model.pt')
            print(rows[-1],flush=True)
    result={'hardware':hardware(),'settings':{'steps':steps,'batch':1024,'lr':5e-4,'image_shape':[h,w],'hidden_width':128,
                        'optimizer':'Adam','seeds':[5670,5671,5672],'frequencies':[0,2,6],
                        'control':'Same target, steps, optimizer, minibatch RNG stream per seed. Encoding changes first-layer parameter count.',
                        'metric_scope':'Input-image reconstruction, not held-out generalization.'},'rows':rows}
    write_json(output/'metrics.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['benchmark','train','test','sample-sweep','image'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/nerf_formal.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/formal_nerf')
    args=parser.parse_args()
    if args.task in ['benchmark','train']:
        config=json.loads(args.config.read_text())
        if args.task=='benchmark':config.update(steps=100,log_every=20)
        train(config,args.output,args.task=='benchmark')
    elif args.task=='image':image_ablation(args.output)
    else:evaluate(args.run,args.output,args.task=='sample-sweep')


if __name__=='__main__':main()
