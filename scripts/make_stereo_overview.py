"""Compose a README figure from saved stereo artifacts; no reconstruction is run."""
from pathlib import Path
import json
import os

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = ROOT / 'outputs/stereo_controlled/0_3_zncc'
metrics = json.loads((ROOT / 'docs/results/research/stereo.json').read_text())
row = next(r for r in metrics['rows'] if r['pair'] == [0, 3] and r['kernel'] == 'zncc')
inputs = np.asarray(Image.open(folder / 'rectified.png'))
with np.load(folder / 'maps.npz') as saved:
    depth = saved['depth']
    keep = saved['final_mask'] > 0
with (folder / 'reconstruction.ply').open() as stream:
    for line in stream:
        if line.strip() == 'end_header':
            break
    cloud = np.loadtxt(stream)
assert len(cloud) == row['point_count'] == int(keep.sum())
assert list(depth.shape) == row['shape']

fig = plt.figure(figsize=(10.5, 3.4), facecolor='white')
grid = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.5], wspace=.28)
left = grid[0].subgridspec(2, 1, hspace=.42)
for slot, index in zip(left, [0, 3]):
    ax = fig.add_subplot(slot)
    image = inputs[:, :inputs.shape[1]//2] if index == 0 else inputs[:, inputs.shape[1]//2:]
    ax.imshow(image, interpolation='nearest')
    ax.set_title(f'Rectified input: view {index}', fontsize=10)
    ax.axis('off')
ax = fig.add_subplot(grid[1])
colored = plt.get_cmap('viridis').copy()
colored.set_bad('black')
shown = ax.imshow(np.ma.array(depth, mask=~keep), vmin=.5, vmax=.6,
                  cmap=colored, interpolation='nearest')
ax.set_title('Filtered depth', fontsize=11)
ax.axis('off')
bar = fig.colorbar(shown, ax=ax, orientation='horizontal', fraction=.06, pad=.07, ticks=[.5,.55,.6])
bar.ax.tick_params(labelsize=8)
bar.set_label('Depth (dataset units)', fontsize=8)
ax = fig.add_subplot(grid[2], projection='3d')
ax.scatter(*cloud[:, :3].T, c=cloud[:, 3:6]/255, s=2, depthshade=False)
ax.set_box_aspect(np.maximum(np.ptp(cloud[:, :3], axis=0), 1e-6))
ax.set_title(f'Colored cloud: {len(cloud):,} points', fontsize=11)
ax.tick_params(labelsize=6, pad=0)
ax.set(xlabel='X', ylabel='Y', zlabel='Z')
fig.subplots_adjust(top=.86, bottom=.12, left=.03, right=.98)
fig.suptitle('TempleRing | views 0/3 | ZNCC | scale 0.25 | patch 5×5', fontsize=12, y=.99)
target = ROOT / 'docs/assets/research/stereo_overview.png'
fig.savefig(target, dpi=160)
plt.close(fig)
print(target)
