import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description='CIS 5800 geometric and neural vision experiments')
    sub = parser.add_subparsers(dest='command', required=True)
    stereo = sub.add_parser('stereo', help='Reconstruct a calibrated pair or aggregate several pairs')
    stereo.add_argument('--data', type=Path, default=ROOT / 'data/stereo/templeRing')
    stereo.add_argument('--views', nargs=2, type=int, default=[0, 3])
    stereo.add_argument('--aggregate', action='store_true', help='Use the nine pairs specified in the original notebook')
    stereo.add_argument('--scale', type=float, default=0.25)
    stereo.add_argument('--kernel', choices=['ssd', 'sad', 'zncc'], default='zncc')
    stereo.add_argument('--patch-size', type=int, default=5)
    stereo.add_argument('--output', type=Path, default=ROOT / 'outputs/stereo')
    neural = sub.add_parser('neural', help='Evaluate saved weights or train coordinate models')
    neural.add_argument('--task', choices=['image', 'nerf'], default='nerf')
    neural.add_argument('--mode', choices=['eval', 'train'], default='eval')
    neural.add_argument('--data', type=Path, default=ROOT / 'data/nerf')
    neural.add_argument('--checkpoint', type=Path)
    neural.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    neural.add_argument('--freq', type=int, choices=[0, 2, 6], default=6)
    neural.add_argument('--iterations', type=int, default=100)
    neural.add_argument('--scale', type=float, default=1.0)
    neural.add_argument('--samples', type=int, default=64)
    neural.add_argument('--rays-per-step', type=int, default=1024)
    neural.add_argument('--ray-chunk', type=int, default=256)
    neural.add_argument('--seed', type=int, default=5670)
    neural.add_argument('--threads', type=int, default=4)
    neural.add_argument('--render-orbit', type=int, default=0, help='Render this many supplied camera poses as a GIF')
    neural.add_argument('--output', type=Path, default=ROOT / 'outputs/neural')
    args = parser.parse_args()
    if not 0 < args.scale <= 1:
        parser.error('--scale must be in (0, 1]')
    if args.command == 'stereo':
        if args.patch_size < 1 or args.patch_size % 2 == 0:
            parser.error('--patch-size must be positive and odd')
        from .experiments import run_stereo
        run_stereo(args)
    else:
        if min(args.iterations, args.samples, args.rays_per_step, args.ray_chunk, args.threads) < 1 or args.render_orbit < 0:
            parser.error('iterations, samples, batch/chunk sizes and threads must be positive; orbit must be nonnegative')
        from .neural_experiment import run_neural
        run_neural(args)
