# AGENTS

This subproject uses GitHub Copilot workspace agents for exploratory registration work and targeted implementation.

## Available Agent Roles

- **Default Assistant**
  - Primary point of contact for code changes, experiment iteration, debugging, and repo guidance.
  - Uses the VS Code workspace tools to read, edit, validate, and run the experiment harness.

- **Explore**
  - Fast, read-only codebase exploration.
  - Use when you need quick answers about structure, symbol definitions, file-level context, or existing experiment outputs.

## Scope

This file applies to the `yorc-gui-extract` package subtree.

- Keep legacy root scripts `YORC.py` and `YORC_BIDS.py` unchanged.
- Keep production GUI logic separate from experiments under `experiments/auto_registration`.
- Store active session context in `HANDOFF.md` within this directory.

## Memory and Handoff

- Use repo memory for stable facts about this subproject.
- Update `HANDOFF.md` when a run changes the preferred defaults, best transform recipe, or output locations.
- Prefer concise notes: current goal, best-known settings, next actions, and relevant file paths.

## Current Work Context

- Main focus is the experimental outside-LIDAR to MRI alignment harness.
- Current best recipe on sample data uses:
  - neck filter radius `100`
  - `--translation-first`
  - `--translation-last --translation-last-threshold 50`
  - `--face-polish`
- Current best sample result selected `anterior_crop` with `ras_to_head_rot_x_-90` and face RMSE around `3.44 mm`.
