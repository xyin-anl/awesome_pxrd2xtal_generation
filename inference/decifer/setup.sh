#!/usr/bin/env bash
# One-time setup for deCIFer inference: creates the conda environment, clones upstream at the
# commit verified in manifest.yaml, and downloads the published checkpoint (~665 MB).
# Works with conda, mamba, or micromamba.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/FrederikLizakJohansen/deCIFer.git"
COMMIT="5b22c0151b1129fad9b30dc3016a5d7977ca3232"
CKPT_URL="https://www.erda.dk/archives/b7342461e7c932bd99e8273c6a49e97b/Nanostructure_Data/Data_for_MachineLearning/deCIFer/deCIFer-frozen/decifer_v1_ckpt.pt"
CKPT_SHA256="6d97e352f99dc6dbc5ced102ed10ee0693651a81f429f20ea50c2bc1fe18b78b"
ENV_NAME="${ENV_NAME:-decifer}"

CONDA="$(command -v micromamba || command -v mamba || command -v conda || true)"
[ -n "$CONDA" ] || { echo "Install conda, mamba, or micromamba first." >&2; exit 1; }

"$CONDA" env create -y -n "$ENV_NAME" -f "$HERE/environment.yml"

if [ ! -d "$HERE/.upstream/.git" ]; then
  git clone "$REPO_URL" "$HERE/.upstream"
fi
git -C "$HERE/.upstream" fetch -q origin "$COMMIT"
git -C "$HERE/.upstream" checkout -q "$COMMIT"

mkdir -p "$HERE/.weights"
CKPT="$HERE/.weights/decifer_v1_ckpt.pt"
if [ ! -f "$CKPT" ]; then
  curl -L --fail -o "$CKPT.part" "$CKPT_URL" && mv "$CKPT.part" "$CKPT"
fi
python3 - "$CKPT" "$CKPT_SHA256" <<'PY'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
if h.hexdigest() != sys.argv[2]:
    sys.exit(f"Checksum mismatch for {sys.argv[1]}; delete it and rerun setup.sh")
print("Checkpoint checksum OK")
PY

echo "Done. Example:"
echo "  $CONDA run -n $ENV_NAME python $HERE/run.py --pattern my_scan.xy --wavelength CuKa --composition TiO2 --z 2 --out results/"
