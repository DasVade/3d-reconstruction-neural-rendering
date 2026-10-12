"""Extract only the required assets from a local course ZIP. No downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--zip', type=Path, required=True)
args = parser.parse_args()
manifest = []
with zipfile.ZipFile(args.zip) as archive:
    for info in archive.infolist():
        name = info.filename
        if info.is_dir():
            continue
        target = None
        if name.startswith('Project A/Part_A_student 2/data/templeRing/'):
            target = root / 'data/stereo/templeRing' / Path(name).name
        elif name.startswith('Project B/Project B/data/'):
            target = root / 'data/nerf' / Path(name).name
        elif name.startswith('Project B/Project B/code/') and name.endswith('.pt'):
            target = root / 'checkpoints' / Path(name).name
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            content = archive.read(info)
            target.write_bytes(content)
            manifest.append({'path': target.relative_to(root).as_posix(), 'source': name,
                             'sha256': hashlib.sha256(content).hexdigest()})
(root / 'docs/input_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(f'Prepared {len(manifest)} assets from {args.zip.name}')
