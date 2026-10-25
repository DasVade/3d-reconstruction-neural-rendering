# Data preparation and source references

Tests run without datasets. Full reproduction requires TempleRing images and
calibration plus the course-format `lego_data.npz`. Prepare inputs from a local
course package using `scripts/prepare_data.py`, as shown in the [quick start](../README.md#installation-and-quick-start).

TempleRing originates from the [Middlebury MVS benchmark](https://vision.middlebury.edu/mview/data/).
Use the official download and retain the dataset's creator attribution.
The Lego NPZ is a course repack. The [NeRF software license](https://github.com/bmild/nerf/blob/master/LICENSE)
applies to that software; dataset terms are separate. Source references,
including the rectification reference in `stereo.py`, remain with the code.

Raw inputs, calibration packages, checkpoints, source archives, environments
and full generated runs are excluded from Git. Selected display assets and
small numerical summaries accompany the [experiment report](EXPERIMENT_REPORT.md).
Local input hashes and archive-entry names are recorded in `input_manifest.json`.
