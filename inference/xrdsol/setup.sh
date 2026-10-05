#!/usr/bin/env bash
# One-time setup for XRDSol inference: creates the conda environment, clones upstream at the
# commit verified in manifest.yaml, and downloads the checkpoint (148 MB, stored with git LFS)
# with checksums. Works with conda, mamba, or micromamba; git-lfs is not required.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/ai4mat-zhu/XRDSol.git"
COMMIT="b2144df1968e4b03b188080c26089646f1293e88"
ENV_NAME="${ENV_NAME:-xrdsol}"

CONDA="$(command -v micromamba || command -v mamba || command -v conda || true)"
[ -n "$CONDA" ] || { echo "Install conda, mamba, or micromamba first." >&2; exit 1; }

if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment $ENV_NAME already exists; reusing it (remove it to rebuild from environment.yml)."
else
  "$CONDA" env create -y -n "$ENV_NAME" -f "$HERE/environment.yml"
fi

if [ ! -d "$HERE/.upstream/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone "$REPO_URL" "$HERE/.upstream"
fi
git -C "$HERE/.upstream" fetch -q origin "$COMMIT"
git -C "$HERE/.upstream" checkout -q "$COMMIT"

MODEL="$HERE/.weights/mp20"
mkdir -p "$MODEL"
cp "$HERE/.upstream/xrdsol/prop_models/mp20/"{hparams.yaml,lattice_scaler.pt,prop_scaler.pt} "$MODEL/"
CKPT="$MODEL/epoch=189-step=5129.ckpt"
if [ ! -f "$CKPT" ]; then
  curl -L --fail -o "$CKPT.part" \
    "https://media.githubusercontent.com/media/ai4mat-zhu/XRDSol/$COMMIT/xrdsol/prop_models/mp20/epoch=189-step=5129.ckpt"
  mv "$CKPT.part" "$CKPT"
fi
python3 - "$MODEL" <<'PY'
import hashlib, os, sys
expected = {
    "epoch=189-step=5129.ckpt": "7e3c48dbe4c055841e069f26d31551cc23d3dd3ea55921665328e827ec88e54e",
    "hparams.yaml": "bd7e61a7bb15cf9ef2e048e1a49dadc49cc7ed41f6a337e9d787909ef63a1ece",
    "lattice_scaler.pt": "2a961ab2b2d33ee8eca8a0306d29fcdd958ef22701c9c97787700c6e8b00580e",
    "prop_scaler.pt": "e8cdf089d3b8d2a2b53545598cfb49bc3833c2b5be3f8b1093fe6d4974783377",
}
for name, sha in expected.items():
    h = hashlib.sha256(open(os.path.join(sys.argv[1], name), "rb").read()).hexdigest()
    if h != sha:
        sys.exit(f"Checksum mismatch for {name}; delete .weights and rerun setup.sh")
print("Checkpoint files OK")
PY

echo "Done. Example:"
echo "  $CONDA run -n $ENV_NAME python $HERE/run.py --pattern scan.xy --wavelength CuKa --composition LuOF --z 2 --spacegroup P4/nmm --cell 3.85,3.85,5.31,90,90,90 --out results/"
