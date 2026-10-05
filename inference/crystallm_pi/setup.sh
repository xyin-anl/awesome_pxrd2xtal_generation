#!/usr/bin/env bash
# One-time setup for CrystaLLM-pi inference: creates the conda environment and clones
# upstream at the commit verified in manifest.yaml. Works with conda, mamba, or micromamba.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/C-Bone-UCL/CrystaLLM-pi.git"
COMMIT="12f1d728ca49fd1797e9ff2bfd40c5cbb0a457b3"
ENV_NAME="${ENV_NAME:-crystallm_pi}"

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
"$CONDA" run -n "$ENV_NAME" pip install -e "$HERE/.upstream"
# Download both XRD checkpoints at the Hugging Face revisions verified in manifest.yaml.
"$CONDA" run -n "$ENV_NAME" python - "$HERE/.weights" <<'PY'
import sys
from huggingface_hub import snapshot_download
pins = {
    "c-bone/CrystaLLM-pi_Mattergen-XRD": "0051c70854553f1e95682770a83cad82657e190a",
    "c-bone/CrystaLLM-pi_Chili100K-XRD": "f5a4e7848693889f5591702009fe6ee198002a5c",
}
# SHA-256 of model.safetensors at those revisions (Hugging Face LFS object ids).
sha = {
    "c-bone/CrystaLLM-pi_Mattergen-XRD": "67aee98d637b33db829f0408374865bdf2600235bcd6975bea0abaf9678c434f",
    "c-bone/CrystaLLM-pi_Chili100K-XRD": "aaf7f836dc5e78caf4a195304eb901b2c484fd6b5ab3dc39ae33292e6811a836",
}
import hashlib
for repo, rev in pins.items():
    path = snapshot_download(repo, revision=rev, local_dir=f"{sys.argv[1]}/{repo.split('/')[1]}")
    digest = hashlib.sha256(open(f"{path}/model.safetensors", "rb").read()).hexdigest()
    if digest != sha[repo]:
        raise SystemExit(f"Checksum mismatch for {repo} model.safetensors: {digest}")
    print(f"{repo}@{rev[:7]} -> {path} (checksum OK)")
PY

echo "Done. Example:"
echo "  $CONDA run -n $ENV_NAME python $HERE/run.py --pattern my_scan.xy --wavelength CuKa --composition TiO2 --out results/"
