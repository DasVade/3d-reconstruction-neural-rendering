import time

import cv2
import numpy as np

from .artifacts import save_cloud_preview, save_image, save_map, save_metrics, save_ply
from .data import load_middlebury_data
from .stereo import two_view, ssd_kernel, sad_kernel, zncc_kernel


def resize_view(view, scale):
    out = dict(view)
    h, w = view['rgb'].shape[:2]
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    out['rgb'] = cv2.resize(view['rgb'], (nw, nh), interpolation=cv2.INTER_AREA)
    out['K'] = np.diag([nw / w, nh / h, 1.0]) @ view['K']
    return out


def run_stereo(args):
    if not args.data.is_dir():
        raise FileNotFoundError(f'Missing stereo data: {args.data}. Run scripts/prepare_data.py with the supplied ZIP.')
    data = [resize_view(v, args.scale) for v in load_middlebury_data(str(args.data))]
    pairs = [(0, 2), (2, 4), (5, 7), (8, 10), (13, 15), (16, 18), (19, 21), (22, 24), (25, 27)] if args.aggregate else [tuple(args.views)]
    if any(i < 0 or j < 0 or i >= len(data) or j >= len(data) or i == j for i, j in pairs):
        raise ValueError('View indices must be distinct and within the dataset.')
    args.output.mkdir(parents=True, exist_ok=True)
    kernel = {'ssd': ssd_kernel, 'sad': sad_kernel, 'zncc': zncc_kernel}[args.kernel]
    points, colors, records = [], [], []
    start = time.perf_counter()
    for i, j in pairs:
        t = time.perf_counter()
        p, c, disparity, depth = two_view(data[i], data[j], args.patch_size, kernel)
        prefix = f'{i:02d}_{j:02d}'
        np.savez_compressed(args.output / f'{prefix}_maps.npz', disparity=disparity, depth=depth)
        save_image(args.output / f'{prefix}_input.png', np.concatenate([data[i]['rgb'], data[j]['rgb']], axis=1))
        save_map(args.output / f'{prefix}_disparity.png', disparity)
        save_map(args.output / f'{prefix}_depth.png', depth)
        points.append(p)
        colors.append(c)
        records.append({'views': [i, j], 'points': len(p), 'elapsed_seconds': time.perf_counter() - t})
    points, colors = np.concatenate(points), np.concatenate(colors)
    save_ply(args.output / 'reconstruction.ply', points, colors)
    save_cloud_preview(args.output / 'pointcloud.png', points, colors)
    save_metrics(args.output, {'experiment': 'calibrated_stereo', 'scale': args.scale, 'kernel': args.kernel,
        'patch_size': args.patch_size, 'total_points': len(points), 'pairs': records,
        'elapsed_seconds': time.perf_counter() - start,
        'note': 'Point count and runtime describe this run, not reconstruction accuracy. Supplied camera calibration is used.'})
    if not len(points):
        raise RuntimeError('Reconstruction produced no points; inspect the saved depth maps and course depth filters.')
