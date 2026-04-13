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

It writes a JSON report and can optionally save transformed point clouds for
inspection.

## Assumptions

This is an exploration harness, not a production pipeline.

- It assumes the two surfaces are at least roughly head-oriented.
- By default, the anterior direction is `+x`, matching the YORC head-standard
  convention used elsewhere in the package.
- If your data is not already in a roughly consistent orientation, the results
  will be informative but not necessarily correct.

## Example

```bash
cd yorc-gui-extract

.venv/bin/python experiments/auto_registration/explore_face_weighted_registration.py \
  --outside sampleData/Rxxxx_01_Outside.ply \
  --mri sampleData/FS/surf/mriscalp.stl \
  --visualize \
  --write-clouds experiments/auto_registration/output
```

## Next ideas to test

- automatic face-region detection instead of fixed anterior crops
- ear and nose weighting separately from the broader anterior scalp
- confidence margins between top candidate transforms
- outside-to-inside automation with the same scoring framework
- hybrid fallback rules that request only 3 MRI anatomy points when confidence is low