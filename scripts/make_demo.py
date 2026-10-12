"""Build a labelled comparison panel from previously generated experiment outputs."""
from pathlib import Path
import json
import os

root = Path(__file__).resolve().parents[1]
assets = root / 'docs/assets'
assets.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(root / 'outputs/.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

runs = [('Target', 'image_eval', 'target.png'), ('L=0', 'image_L0', 'prediction.png'),
        ('L=2', 'image_L2', 'prediction.png'), ('L=6', 'image_eval', 'prediction.png')]
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, (label, run, filename) in zip(axes, runs):
    folder = root / 'outputs' / run
    metrics = json.loads((folder / 'metrics.json').read_text())
    ax.imshow(Image.open(folder / filename))
    title = label if label == 'Target' else f"{label}: {metrics['reconstruction_psnr_db']:.2f} dB"
    ax.set_title(title)
    ax.axis('off')
fig.suptitle('Coordinate MLP: supplied checkpoints, input-image reconstruction')
fig.tight_layout()
fig.savefig(assets / 'image_frequency_comparison.png', dpi=160)
plt.close(fig)
print(assets / 'image_frequency_comparison.png')
