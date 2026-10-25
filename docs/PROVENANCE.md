# Implementation and source references

Developed from Penn CIS 5800 Project A/B coursework using the supplied tasks,
calibration, network architecture and frameworks.

| Repository module | Origin | Implementation changes |
|---|---|---|
| `stereo.py` | `Project A/Part_A_student 2/two_view_stereo.py` | Package imports; replace Open3D statistical filtering with a SciPy nearest-neighbor implementation. Uses nearest-neighbor statistics. |
| `data.py` | Project A `dataloader.py` | Package placement; course loader retained. |
| `image_field.py` | Project B `part1_code.py` | Preserve positional encoding and model; legacy notebook training function replaced with CLI runner. |
| `baselines/nerf.py` | Project B `part2_code.py`, as consolidated at `5c29239` | Frozen historical baseline; original supplied weights use this path. |
| `nerf.py` | Derived from the frozen baseline | October 17 audit: image-down y correction, midpoint sampling, finite softplus-density integration; see AUDIT.md. |
| CLI, experiment runners, exports, tests | Project extensions | Reproducible outputs, chunked inference, mini-batch training, explicit metric labels and fresh training split. |


The original NeRF implementation is preserved at `5c29239` and in
`src/vision5800/baselines/`. The research workflow adds camera/projection tests,
finite rendering checks, validation-selected training, independent test-view
evaluation, matched stereo comparisons and controlled encoding/sampling studies.

See the [experiment report](EXPERIMENT_REPORT.md) for the formal runs,
[historical reproduction record](REPRODUCTION.md) for the earlier supplied-weight
checks, and [data preparation](DATA.md) for input sources and excluded files.
