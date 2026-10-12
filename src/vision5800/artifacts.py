import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def save_image(path, image):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def save_map(path, values):
    configure_plotting(path)
    from matplotlib import colormaps
    values = np.asarray(values)
    valid = np.isfinite(values) & (values > 0)
    scaled = np.zeros_like(values, dtype=float)
    if valid.any():
        low, high = np.percentile(values[valid], [2, 98])
        scaled[valid] = np.clip((values[valid] - low) / max(high - low, 1e-8), 0, 1)
    rgb = colormaps['viridis'](scaled)[..., :3]
    rgb[~valid] = 0
    save_image(path, rgb)


def save_ply(path, points, colors):
    points, colors = np.asarray(points), np.asarray(colors)
    if colors.dtype != np.uint8:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    header = ('ply\nformat ascii 1.0\nelement vertex %d\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n') % len(points)
    with open(path, 'w', encoding='ascii') as f:
        f.write(header)
        np.savetxt(f, np.column_stack([points, colors]), fmt=['%.7f'] * 3 + ['%d'] * 3)


def save_metrics(output, metrics):
    path = Path(output) / 'metrics.json'
    path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


def save_cloud_preview(path, points, colors):
    configure_plotting(path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    stride = max(1, len(points) // 20000)
    p = points[::stride]
    if len(p):
        ax.scatter(*p.T, c=colors[::stride] / 255.0, s=0.5)
        extent = np.ptp(p, axis=0)
        ax.set_box_aspect(np.maximum(extent, 1e-6))
    ax.set(xlabel='World X', ylabel='World Y', zlabel='World Z', title=f'Stereo reconstruction ({len(points):,} points)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def configure_plotting(path):
    cache = Path(path).resolve().parent / '.mplconfig'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
