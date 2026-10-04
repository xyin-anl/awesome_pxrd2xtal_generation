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

"$CONDA" env create -y -n "$ENV_NAME" -f "$HERE/environment.yml"

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
for repo, rev in pins.items():
    path = snapshot_download(repo, revision=rev, local_dir=f"{sys.argv[1]}/{repo.split('/')[1]}")
    print(f"{repo}@{rev[:7]} -> {path}")
PY

echo "Done. Example:"
echo "  $CONDA run -n $ENV_NAME python $HERE/run.py --pattern my_scan.xy --wavelength CuKa --composition TiO2 --out results/"
