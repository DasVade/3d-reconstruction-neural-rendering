# NeRF camera and rendering conventions

The research workflow uses corrected camera rays, finite volume rendering,
validation-selected training and explicit test-view evaluation. The earlier
supplied-weight checks are recorded in [REPRODUCTION.md](REPRODUCTION.md).

## Camera convention

The supplied 106 poses are camera-to-world matrices. Rotation determinants are approximately +1. The +Z camera axis points toward the origin (mean forward/radial cosine -1), so a blanket Blender -Z change would be wrong. Camera +Y points down for this converted dataset.

`scripts/audit_coordinates.py` compares known-pose epipolar consistency using 412 Lowe-ratio-filtered SIFT matches across 15 nearby pairs. Median Sampson distance is 0.145 px with camera direction `(x,+y,+1)`, versus 1.074 px with the previous `(x,-y,+1)`. Matches use the same Lowe-ratio filter for both conventions, without geometric inlier filtering. Sampson distance measures epipolar consistency. The new ray generator uses the positive image-down axis. A synthetic rotated-pose projection test checks the same convention independently.

## Encoding and rendering

Sin/cos encoding uses `2**i*pi*x`, concatenates input coordinates and is internally consistent. Position frequencies and view-direction frequencies are separate. Directions are normalized for direction encoding, while their world norm scales optical path lengths.

New experiments use deterministic midpoint quadrature over [.667,2.0]. The finite integration interval replaces the infinite terminal interval; unabsorbed light sees the black dataset background. Softplus replaces ReLU raw-density activation to keep gradients at negative initial density. These rendering/training choices define the new model parameterization. They change the parameterization, so old weights are not reused in new experiments. Constant-medium tests compare opacity with an analytic finite-path integral.

## Splits and checkpoint selection

New training starts from scratch. Views 0–89 train; validation is 90/92/94/96/98; test is 100–105. Other views are unused. The checkpoint maximizes mean validation PSNR. All test views are evaluated after selection using the predeclared training sample count. Test scores do not select checkpoints or the render sample count. Legacy runs previously inspected some of these images; independence here means disjoint from this new model's training/selection, not an entirely untouched external benchmark.

The sample-count study is **render-only on validation views**, with a single fixed checkpoint and 16/32/64/128 samples. It cannot establish the effect of changing training sample count. The 2D frequency study is **fresh training**, equal steps/optimizer/minibatches per seed and three seeds. Frequency changes input dimensionality and parameter count, so it is not an equal-parameter comparison.

## Traceable baseline

Git commit `5c29239` preserves the complete earlier repository. `src/vision5800/baselines/nerf.py` is byte-preserved from that state; its archived runner changes relative imports only. The historical `vision5800 neural` CLI retains baseline rendering for supplied weights. Use `python -m vision5800.research` for audited experiments. Original checkpoints and old results are not overwritten. The baseline and research renderer differ in ray sign, quadrature, interval and density activation; their score difference combines these changes.
