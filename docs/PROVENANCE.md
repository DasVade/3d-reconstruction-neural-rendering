# Source and contribution boundaries

Source archive: `HWs&Project.zip`, supplied by the user. This is a local reproducibility consolidation of CIS 5800 coursework, not a new research method.

| Repository module | Origin | Consolidation changes |
|---|---|---|
| `stereo.py` | `Project A/Part_A_student 2/two_view_stereo.py` | Package imports; replace Open3D statistical filtering with a SciPy nearest-neighbor implementation. Not guaranteed bit-identical. |
| `data.py` | Project A `dataloader.py` | Package placement; course loader retained. |
| `image_field.py` | Project B `part1_code.py` | Preserve positional encoding and model; legacy notebook training function replaced with CLI runner. |
| `baselines/nerf.py` | Project B `part2_code.py`, as consolidated at `5c29239` | Frozen historical baseline; original supplied weights use this path. |
| `nerf.py` | Derived from the frozen baseline | October 17 audit: image-down y correction, midpoint sampling, finite softplus-density integration; see AUDIT.md. |
| CLI, experiment runners, exports, tests | Added during consolidation | Reproducible outputs, chunked inference, mini-batch training, explicit metric labels and fresh training split. |

Both source projects use course-provided frameworks with student implementation regions. File existence does not establish authorship of every line. Retain attribution and describe personal contributions according to actual work completed.

The course PDFs and template notebooks are intentionally not copied into the publishable source tree. Homework 1–5 are not included as separate polished projects in this first repository: the requested initial reproduction focuses on Project A/B. HW3 and HW4 contain conflicting template/completed versions and need a separate reconciliation before integration.

Input data and weights are present locally, ignored by Git, and described in `input_manifest.json` with source ZIP entry names and SHA-256 hashes. Do not infer quality from checkpoint presence. The original training logs are unavailable.

No open-source license is asserted for third-party coursework or data. The local data extraction process does not establish redistribution rights. Public release should preserve source attribution and use an appropriate data acquisition path.

## October 17 experimental extension

`research.py`, audit and experiment scripts, new analytic/projection tests, configuration, results, English report and local page were added with coding-assistant support. They implement validation-based checkpoint selection, explicit train/validation/test sets, independent-view PSNR/SSIM, matched stereo runs and controlled studies. These are engineering/experimental extensions to the course architecture, not original NeRF/stereo methods. Applicant-specific claims of personal authorship require familiarity with and truthful attribution of the work.

Verified new runs are documented in EXPERIMENT_REPORT.md; the October 12 REPRODUCTION.md remains a historical record. PUBLICATION_REVIEW.md records checked sources and unresolved release conditions. No public upload was performed.
