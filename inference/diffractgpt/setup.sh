#!/usr/bin/env bash
# One-time setup for DiffractGPT inference: creates the conda environment and downloads the LoRA
# adapter and its 4-bit Mistral-7B base (~4 GB) at the Hugging Face revisions verified in
# manifest.yaml. Works with conda, mamba, or micromamba.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-diffractgpt}"

CONDA="$(command -v micromamba || command -v mamba || command -v conda || true)"
[ -n "$CONDA" ] || { echo "Install conda, mamba, or micromamba first." >&2; exit 1; }

"$CONDA" env create -y -n "$ENV_NAME" -f "$HERE/environment.yml"

"$CONDA" run -n "$ENV_NAME" python - "$HERE/.weights" <<'PY'
import sys
from huggingface_hub import snapshot_download
pins = {
    "knc6/diffractgpt_mistral_chemical_formula": "a4b14720aa2b5ef124f794e581dfd5ce4dc2ab49",
    "unsloth/mistral-7b-bnb-4bit": "3c47be0aa392c058e26ede776ecd2d5416fa5d28",
}
for repo, rev in pins.items():
    path = snapshot_download(repo, revision=rev, local_dir=f"{sys.argv[1]}/{repo.split('/')[1]}",
                             allow_patterns=["*.json", "*.safetensors", "*.model"])
    print(f"{repo}@{rev[:7]} -> {path}")
PY

echo "Done. Example:"
echo "  $CONDA run -n $ENV_NAME python $HERE/run.py --pattern my_scan.xy --wavelength CuKa --composition TiO2 --out results/"
