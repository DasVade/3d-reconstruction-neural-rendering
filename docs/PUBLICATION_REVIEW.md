# Release and data notes

The owner confirmed on October 24, 2026 that course rules impose no restrictions on publication and authorized making this existing repository public. This supersedes the October 17 private/local-only publication notice. Coursework and dataset attribution remain intact. No experiments, algorithms or numerical results changed for this release.

[Live project page](https://yixiaoliu.me/projects/3d-reconstruction-neural-rendering/) · [Experiment report](EXPERIMENT_REPORT.md) · [Source provenance](PROVENANCE.md)

## Included and excluded

Source, tests, configuration, selected display figures/animations and small numerical summaries are included. The owner previously confirmed permission for the selected website visuals. Raw TempleRing images/calibration packages, `lego_data.npz`, original and trained checkpoints, course archives/PDF packages, environments and full run outputs remain local and ignored. Git history contains no tracked raw dataset or model-weight files in the scanned reachable objects.

## Acquiring inputs

The CLI tests run without the private data. For full reproduction, use your authorized local course package with `scripts/prepare_data.py`; the script extracts inputs locally and does not download or redistribute them. It cannot run data-dependent experiments from this repository alone.

TempleRing originates from the [Middlebury MVS benchmark](https://vision.middlebury.edu/mview/data/); use its official download and original creator attribution. The local `lego_data.npz` is a course repack; do not infer input-image or checkpoint licensing from a software license. The [bmild NeRF software license](https://github.com/bmild/nerf/blob/master/LICENSE) applies to that software, not automatically to this course repack. Existing source references, including the rectification reference in `stereo.py`, are retained. No blanket license is assigned to third-party data or course source.

The public page distinguishes the fresh six-view 100×100 NeRF test benchmark from the historical camera sweep, which contains training poses. No depth ground-truth accuracy is reported for stereo.
