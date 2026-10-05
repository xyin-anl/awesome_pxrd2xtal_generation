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

if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment $ENV_NAME already exists; reusing it (remove it to rebuild from environment.yml)."
else
  "$CONDA" env create -y -n "$ENV_NAME" -f "$HERE/environment.yml"
fi

if [ ! -d "$HERE/.upstream/.git" ]; then
  git clone "$REPO_URL" "$HERE/.upstream"
fi
git -C "$HERE/.upstream" fetch -q origin "$COMMIT"
git -C "$HERE/.upstream" checkout -q "$COMMIT"

"$CONDA" run -n "$ENV_NAME" python -c "
import hashlib
from huggingface_hub import snapshot_download
path = snapshot_download('therealgabeguo/cdvae_xrd_sinc10', revision='$HF_REVISION', local_dir='$HERE/.weights')
# SHA-256 of both checkpoints at that revision (Hugging Face LFS object ids).
for rel, sha in [('mp_20_sinc100/epoch=954-step=304645.ckpt', 'ddff14ab8e4c05ea8442ebdc25a7908979155c0a8742565e4c26def0f004d631'),
                 ('mp_20_sinc10/epoch=939-step=299860.ckpt', 'eedb29796f5f223f517516a09362df2969f766d8ba4a395eb91f95a67f96c9a4')]:
    if hashlib.sha256(open(f'{path}/{rel}', 'rb').read()).hexdigest() != sha:
        raise SystemExit(f'Checksum mismatch for {rel}')
print(path, '(checksums OK)')"

echo "Done. Example:"
echo "  $CONDA run -n $ENV_NAME python $HERE/run.py --pattern my_scan.xy --wavelength CuKa --composition KCaCO3F --z 1 --out results/"
