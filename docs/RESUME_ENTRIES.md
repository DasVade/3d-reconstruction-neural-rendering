# English resume entries — use only contributions you can explain

**Calibrated Stereo Reconstruction and Neural Rendering | University of Pennsylvania, CIS 5800**

## Computer vision / robotics internship version

- Reproduced and audited a PyTorch neural-rendering pipeline, correcting camera-ray conventions and adding validation-based checkpoint selection, reproducible configurations, and GPU training on an RTX 4070 Ti.
- Achieved 24.19 dB PSNR and 0.831 SSIM averaged over six held-out views of a 100×100 synthetic Lego scene, with target/prediction/error visualizations and recorded training curves.
- Benchmarked SSD, SAD, and ZNCC stereo matching under matched settings across three calibrated image pairs, measuring runtime, depth coverage, left–right consistency, and exported colored point clouds.

## RA / PhD version

- Audited geometric and neural reconstruction implementations in a CIS 5800 course project; tested camera-coordinate consistency against known-pose correspondences and retained a traceable original baseline.
- Designed controlled studies of positional-encoding frequency across three seeds and inference-only ray-sampling density, distinguishing image-fit quality, validation selection, and independent-view evaluation.
- Trained a static-scene NeRF from scratch with disjoint training/validation/test views, obtaining mean 24.19 dB PSNR / 0.831 SSIM over six test views; documented limitations, failure cases, and reproducible experiment settings.

These describe reproduction/experimental work, not an original algorithm, SLAM system or dynamic world model. The framework, data and architecture were course-provided; the applicant should identify their own original course contributions separately. This iteration used coding-assistant support. Date the project according to actual work completed. Public Code/Demo links require publication clearance.
