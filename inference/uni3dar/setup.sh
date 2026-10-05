#!/usr/bin/env bash
# One-time setup for Uni-3DAR PXRD inference: creates the conda environment, installs Uni-Core
# (pure-PyTorch build, no CUDA compiler needed), clones upstream at the commit verified in
# manifest.yaml, and downloads mp20_pxrd.pt (~1.3 GB) at a pinned Hugging Face revision.
# Works with conda, mamba, or micromamba. Requires an NVIDIA GPU with compute capability >= 8.0.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/dptech-corp/Uni-3DAR.git"
COMMIT="bba82f5ffdcd85963e1dff68a52415cbf046e6c7"
UNICORE="git+https://github.com/dptech-corp/Uni-Core.git@ace6fae1c8479a9751f2bb1e1d6e4047427bc134"
HF_REVISION="f51306a4b12f7a2f8df45a124b1238f72e37805d"
CKPT_SHA256="0cf91fb568a04e5699df1dd6c80cfb00df50b8676d1b58f977f2596baf1db6ab"
ENV_NAME="${ENV_NAME:-uni3dar}"

CONDA="$(command -v micromamba || command -v mamba || command -v conda || true)"
[ -n "$CONDA" ] || { echo "Install conda, mamba, or micromamba first." >&2; exit 1; }

if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment $ENV_NAME already exists; reusing it (remove it to rebuild from environment.yml)."
else
  "$CONDA" env create -y -n "$ENV_NAME" -f "$HERE/environment.yml"
fi
# Uni-Core's setup.py imports torch, so it must build against the environment's torch.
"$CONDA" run -n "$ENV_NAME" pip install --no-build-isolation "$UNICORE"

if [ ! -d "$HERE/.upstream/.git" ]; then
  git clone "$REPO_URL" "$HERE/.upstream"
fi
git -C "$HERE/.upstream" fetch -q origin "$COMMIT"
git -C "$HERE/.upstream" checkout -q "$COMMIT"

mkdir -p "$HERE/.weights"
CKPT="$HERE/.weights/mp20_pxrd.pt"
if [ ! -f "$CKPT" ]; then
  curl -L --fail -o "$CKPT.part" "https://huggingface.co/dptech/Uni-3DAR/resolve/$HF_REVISION/mp20_pxrd.pt" && mv "$CKPT.part" "$CKPT"
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
