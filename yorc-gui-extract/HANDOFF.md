# HANDOFF

## Current Goal

Refine and validate the experimental outside-LIDAR to MRI registration workflow in `experiments/auto_registration`.

## Best Known Sample Settings

Command pattern:

```bash
.venv/bin/python experiments/auto_registration/explore_face_weighted_registration.py \
  --outside sampleData/Rxxxx_01_Outside.ply \
  --mri sampleData/FS/surf/mriscalp.stl \
  --max-points 20000 \
  --input-points 20000 \
  --translation-first \
  --translation-last \
  --translation-last-threshold 50 \
  --neck-filter-radius 100 \
  --face-polish
```

Observed sample outcome:

- selected method: `anterior_crop`
- winning initializer: `ras_to_head_rot_x_-90`
- face RMSE: about `3.44 mm`
- full RMSE: about `6.06 mm`

## Recent Changes

- Added iterative sphere-based neck filtering so the center estimate is less biased by neck points.
- Added automatic timestamped output directories under `yorc-gui-extract/runs/`.
- Improved face alignment by allowing all initializers to compete and enabling final face-only polish.

## Outstanding Questions

- Whether neck filtering and face polish should become CLI defaults.
- Whether the production GUI should adopt any of this experiment logic later.
- Whether generated `runs/` and experiment output folders should be ignored in git.

## Relevant Paths

- `experiments/auto_registration/explore_face_weighted_registration.py`
- `experiments/auto_registration/README.md`
- `runs/`

## Next Actions

- If continuing experiments, compare a few neighboring neck radii around `100`.
- If stabilizing the tool, consider promoting the current best recipe to defaults.
- Keep generated outputs out of commits unless explicitly requested.
