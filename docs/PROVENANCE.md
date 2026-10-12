# Source and contribution boundaries

Source archive: `HWs&Project.zip`, supplied by the user. This is a local reproducibility consolidation of CIS 5800 coursework, not a new research method.

| Repository module | Origin | Consolidation changes |
|---|---|---|
| `stereo.py` | `Project A/Part_A_student 2/two_view_stereo.py` | Package imports; replace Open3D statistical filtering with a SciPy nearest-neighbor implementation. Not guaranteed bit-identical. |
| `data.py` | Project A `dataloader.py` | Package placement; course loader retained. |
| `image_field.py` | Project B `part1_code.py` | Preserve positional encoding and model; legacy notebook training function replaced with CLI runner. |
| `nerf.py` | Project B `part2_code.py` | Core model/ray/rendering functions preserved. |
| CLI, experiment runners, exports, tests | Added during consolidation | Reproducible outputs, chunked inference, mini-batch training, explicit metric labels and fresh training split. |

Both source projects use course-provided frameworks with student implementation regions. File existence does not establish authorship of every line. Retain attribution and describe personal contributions according to actual work completed.

The course PDFs and template notebooks are intentionally not copied into the publishable source tree. Homework 1–5 are not included as separate polished projects in this first repository: the requested initial reproduction focuses on Project A/B. HW3 and HW4 contain conflicting template/completed versions and need a separate reconciliation before integration.

Input data and weights are present locally, ignored by Git, and described in `input_manifest.json` with source ZIP entry names and SHA-256 hashes. Do not infer quality from checkpoint presence. The original training logs are unavailable.

No open-source license is asserted for third-party coursework or data. The local data extraction process does not establish redistribution rights. Public release should preserve source attribution and use an appropriate data acquisition path.
