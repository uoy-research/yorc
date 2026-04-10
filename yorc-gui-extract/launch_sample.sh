#!/usr/bin/env bash
# Launch the tripanel registration GUI with sample data.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE="$SCRIPT_DIR/sampleData"

exec "$SCRIPT_DIR/.venv/bin/python" -m scripts.tripanel_registration_gui \
    --inside-mesh  "$SAMPLE/Rxxxx_InHelmet.ply" \
    --outside-mesh "$SAMPLE/Rxxxx_01_Outside.ply" \
    --mri-scalp    "$SAMPLE/FS/surf/mriscalp.stl" \
    --megdata      "$SAMPLE/VEP_DS-raw.fif"
