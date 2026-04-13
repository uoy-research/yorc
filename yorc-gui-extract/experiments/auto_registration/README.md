# Automatic Registration Experiments

This directory is intentionally separate from the production GUI pipeline.

It is for testing ideas for a more automatic coregistration workflow, especially
the harder outside-LIDAR to MRI step.

## Current hypothesis

Whole-head surface registration is often too symmetric to trust on its own.
The face and anterior scalp provide stronger asymmetry, so a useful direction is:

- coarse whole-head alignment
- candidate rigid hypotheses
- ICP refinement with extra emphasis on the anterior or face region
- confidence scoring that can reject weak solutions

## Included script

- `explore_face_weighted_registration.py`

This script compares several outside-to-MRI registration strategies:

- `full`: whole-cloud ICP only
- `anterior_crop`: ICP on the anterior subset only
- `face_weighted`: ICP on the full cloud with anterior points duplicated to
  approximate weighting
- initialization search includes global registration, PCA sign-flip variants,
  canonical MRI-to-head starts, plus ordered one-step and two-step `±90`
  `x`/`y`/`z` rotation pairs to catch common start-pose mismatches such as
  `y90` then `z-90`
- the JSON report keeps alias labels for equivalent ordered quarter-turn pairs,
  so you can still see which pair recipes map onto the same rigid start pose

It writes a JSON report and can optionally save transformed point clouds for
inspection.
When `--write-clouds` is used it also writes starting-pose contact sheets for
the canonical initial rotations as `starting_pose_variants_xy.png`,
`starting_pose_variants_xz.png`, and `starting_pose_variants_yz.png`.
It also writes `final_registration_selected.png`, and the MRI face region is
selected in the aligned target frame so the colored face overlay matches the
outside scan convention instead of raw MRI axes.
If you already know the start pose you want, `--init-label` restricts the run to
one named candidate or alias such as `ras_to_head_rot_x_-90`.
`--canonical-init-compose translate_then_rotate` changes the canonical
`ras_to_head` starts so centroid translation is applied before rotation, instead
of keeping the rotated source centroid locked to the target centroid.
`--translation-first` adds a staged optimizer: translation-only ICP first, then
the normal rigid ICP pass. Use `--translation-first-threshold` to widen the
coarse translation stage if the initial overlap is poor.
`--translation-last` adds a final translation-only polish after rigid ICP when
the orientation looks right but the residual translation still appears off.
`--face-polish` adds one more low-threshold rigid ICP pass on the aligned face
region to finesse an already-good result without re-running the full search.

## Assumptions

This is an exploration harness, not a production pipeline.

- It assumes the two surfaces are at least roughly head-oriented.
- By default, the anterior direction is `+x`, matching the YORC head-standard
  convention used elsewhere in the package.
- The candidate search now explicitly tests ordered one-step and two-step
  quarter-turn starts around the canonical MRI-to-head orientation, including
  `y90` and `z-90` combinations.
- If your data is not already in a roughly consistent orientation, the results
  will be informative but not necessarily correct.

## Example

```bash
cd yorc-gui-extract

.venv/bin/python experiments/auto_registration/explore_face_weighted_registration.py \
  --outside sampleData/Rxxxx_01_Outside.ply \
  --mri sampleData/FS/surf/mriscalp.stl \
  --init-label ras_to_head_rot_x_-90 \
  --translation-first \
  --translation-last \
  --translation-first-threshold 30 \
  --visualize \
  --write-clouds experiments/auto_registration/output
```

## Next ideas to test

- automatic face-region detection instead of fixed anterior crops
- ear and nose weighting separately from the broader anterior scalp
- confidence margins between top candidate transforms
- outside-to-inside automation with the same scoring framework
- hybrid fallback rules that request only 3 MRI anatomy points when confidence is low
