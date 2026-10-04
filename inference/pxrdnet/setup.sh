#!/usr/bin/env bash
# One-time setup for PXRDnet inference: creates the conda environment, clones upstream at the
# commit verified in manifest.yaml, and downloads both checkpoints (~120 MB) at a pinned Hugging
# Face revision. Works with conda, mamba, or micromamba.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/gabeguo/cdvae_xrd.git"
COMMIT="1ce3845f5b30c7dab703793122298423dcec79bc"
HF_REVISION="299e965ef2a8b2216ef542e031e1f550f65f9158"
ENV_NAME="${ENV_NAME:-pxrdnet}"

CONDA="$(command -v micromamba || command -v mamba || command -v conda || true)"
[ -n "$CONDA" ] || { echo "Install conda, mamba, or micromamba first." >&2; exit 1; }

"$CONDA" env create -y -n "$ENV_NAME" -f "$HERE/environment.yml"

if [ ! -d "$HERE/.upstream/.git" ]; then
  git clone "$REPO_URL" "$HERE/.upstream"
fi
git -C "$HERE/.upstream" fetch -q origin "$COMMIT"
git -C "$HERE/.upstream" checkout -q "$COMMIT"

"$CONDA" run -n "$ENV_NAME" python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('therealgabeguo/cdvae_xrd_sinc10', revision='$HF_REVISION', local_dir='$HERE/.weights'))"

echo "Done. Example:"
echo "  $CONDA run -n $ENV_NAME python $HERE/run.py --pattern my_scan.xy --wavelength CuKa --composition KCaCO3F --z 1 --out results/"
