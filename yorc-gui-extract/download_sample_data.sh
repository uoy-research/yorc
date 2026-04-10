#!/usr/bin/env bash
# Download YORC sample data from OSF into sampleData/.
# Usage: ./download_sample_data.sh

set -euo pipefail

# ── OSF download URLs ────────────────────────────────────────────────────────
declare -A FILES=(
    ["sampleData/Rxxxx_InHelmet.ply"]="https://osf.io/emvf7/download"
    ["sampleData/Rxxxx_01_Outside.ply"]="https://osf.io/vp9e7/download"
    ["sampleData/FS/surf/mriscalp.stl"]="https://osf.io/d4exr/download"
    ["sampleData/VEP_DS-raw.fif"]="https://osf.io/ztkyf/download"
)
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for DEST in "${!FILES[@]}"; do
    URL="${FILES[$DEST]}"
    FULL_PATH="$SCRIPT_DIR/$DEST"

    if [[ -f "$FULL_PATH" ]]; then
        echo "Already exists, skipping: $DEST"
        continue
    fi

    echo "Downloading: $DEST"
    mkdir -p "$(dirname "$FULL_PATH")"
    curl -fL --progress-bar -o "$FULL_PATH" "$URL"
done

echo "Done. Sample data is in sampleData/"
