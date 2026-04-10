# YORC User Guide (Step-by-Step)

This guide takes you from a fresh checkout to a working tri-panel registration run.

## 1. Install Prerequisites

You need:
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/) (recommended) or `pip`

Check Python:

```bash
python3 --version
```

## 2. Clone and Enter the Repo

```bash
git clone https://github.com/wadelab/yorc-gui.git
cd yorc-gui
```

## 3. Create Environment and Install Dependencies

### Using uv (recommended)

```bash
uv sync
```

This resolves dependencies from `pyproject.toml` and `uv.lock` and installs everything into a local `.venv`. The lockfile ensures you get exactly the tested dependency versions.

For dev/test extras:

```bash
uv sync --extra dev
```

### Using pip

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -e .
```

## 4. Verify Setup

```bash
uv run yorc-tripanel-gui --help      # if using uv
yorc-tripanel-gui --help             # if using pip with venv activated
```

If this prints help text, your setup is ready.

> **Note:** The rest of this guide uses `uv run` to prefix commands. If you installed with pip and have your venv activated, just drop the `uv run` prefix — e.g., `uv run yorc-tripanel-gui` becomes `yorc-tripanel-gui`.

## 5. Try with Sample Data

Sample data is hosted on OSF (161 MB total: 2 LIDAR scans, MRI scalp, MEG recording).

Download:

```bash
./download_sample_data.sh
```

Launch with sample files pre-loaded:

```bash
./launch_sample.sh
```

This is the quickest way to verify the GUI is working end-to-end before using your own data.

## 6. Prepare Your Own Input Files

You need 4 files:
- `inside` LIDAR mesh/point cloud (`.ply/.stl/.obj/.pcd`) — scan with participant in helmet
- `outside` LIDAR mesh/point cloud (`.ply/.stl/.obj/.pcd`) — scan without helmet
- `mri scalp` surface (`.stl` or `.fif`)
- `meg` raw FIF (`.fif`)

## 7. Generate MRI Scalp Surface

You can use either option below.

### Option A: MNE watershed BEM (creates `.fif`)

Use this if you already have a FreeSurfer subject (`recon-all` done).

```bash
export SUBJECTS_DIR=/path/to/subjects_dir
subject=subj01

uv run mne watershed_bem --subject ${subject} --overwrite
```

Then use:
- `${SUBJECTS_DIR}/${subject}/bem/${subject}-head.fif`

### Option B: FreeSurfer scalp surface to STL

If you already ran `recon-all`:

```bash
export SUBJECTS_DIR=/path/to/subjects_dir
subject=subj01

mkheadsurf -subjid ${subject}

mris_convert ${SUBJECTS_DIR}/${subject}/surf/lh.seghead \
             ${SUBJECTS_DIR}/${subject}/surf/mriscalp.stl
```

Then use:
- `${SUBJECTS_DIR}/${subject}/surf/mriscalp.stl`

## 8. Run the Tri-Panel GUI

```bash
uv run yorc-tripanel-gui \
  -im /path/to/inside_mesh.ply \
  -om /path/to/outside_mesh.ply \
  -s /path/to/mri_scalp.stl \
  -m /path/to/meg_raw.fif \
  --auto-load
```

Available flags:

- `-im, --inside-mesh` — inside helmet scan (`.ply/.stl/.obj/.pcd`)
- `-om, --outside-mesh` — outside head scan (`.ply/.stl/.obj/.pcd`)
- `-s, --mri-scalp` — MRI scalp surface (`.fif/.ply/.stl/.obj`)
- `-m, --megdata` — MEG FIF file
- `--auto-load` — trigger Load Data automatically on startup
- `--renderer {auto, pyvista, native-vtk}` — 3D backend (default: auto)

Notes:
- If `inside` and `outside` are accidentally swapped, the GUI auto-detects common filename patterns and corrects.
- On macOS, `native-vtk` is typically most stable.

## 9. In-GUI Workflow

1. `Load Data` (or auto-load from CLI)
2. Pick points in order:
   1. Inside fiducials (7)
   2. Outside anatomy (3)
   3. Inside face (3)
   4. Outside face (3)
   5. MRI face (3)
3. `Compute Transforms`
4. `Preview Sensors`
5. `Apply to FIF` (writes transforms)

Picking controls:
- `Shift + Left Click`: add point
- `Shift + Right Click`: remove last point

Useful controls:
- `Fast Mode`: faster ICP
- `Stabilize ICP`: extra ICP rounds
- `Save Picks` / `Load Picks`: reuse landmarks
- `Show X2 Overlay`, `Show X3 Overlay`, `Restore Views`

## 10. Sensor Display and Distances

Preview supports:
- `Display contact points (pad at scalp)`
- `Display detector centers (+6 mm)`

Coloring rule:
- Inside/outside colors are based on detector-center position relative to scalp mesh.

Preview logs include both:
- median contact->scalp distance
- median detector->scalp distance

## 11. Output Files

When you click `Apply to FIF`:
- input MEG FIF is updated with dev->head transform
- a separate `_trans.fif` (MRI->head) is written beside the MEG file

## 12. Common Issues

If `yorc-tripanel-gui` fails to start:
- uv: ensure `.venv` exists (`uv sync` to recreate), try `uv sync --reinstall`
- pip: ensure your venv is activated, try `pip install -e .` again

If MRI `.fif` loading fails with SciPy errors:
- SciPy 1.11.4 is pinned in `pyproject.toml` for MNE compatibility
- uv: `uv sync` should handle this automatically
- pip: `pip install scipy==1.11.4`

If alignment is poor:
- re-pick landmarks carefully (especially face points)
- check MRI scalp quality
- try disabling `Fast Mode` for final pass
