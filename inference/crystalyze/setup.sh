#!/usr/bin/env bash
# One-time setup for Crystalyze inference: creates the conda environment, clones upstream at the
# commit verified in manifest.yaml, and downloads model_folder (~415 MB) from the authors' Google
# Drive folder, verifying checksums. Works with conda, mamba, or micromamba.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/ML-PXRD/Crystalyze.git"
COMMIT="d265e8b4687fda9586a73f3eb881bb881171b05b"
ENV_NAME="${ENV_NAME:-crystalyze}"

CONDA="$(command -v micromamba || command -v mamba || command -v conda || true)"
[ -n "$CONDA" ] || { echo "Install conda, mamba, or micromamba first." >&2; exit 1; }

"$CONDA" env create -y -n "$ENV_NAME" -f "$HERE/environment.yml"

if [ ! -d "$HERE/.upstream/.git" ]; then
  git clone "$REPO_URL" "$HERE/.upstream"
fi
git -C "$HERE/.upstream" fetch -q origin "$COMMIT"
git -C "$HERE/.upstream" checkout -q "$COMMIT"

# Google Drive file ids from https://drive.google.com/drive/folders/1iANYLKp4pscNSA1VirSSSrPnt-2BNfzx
"$CONDA" run -n "$ENV_NAME" python - "$HERE/.weights/model_folder" <<'PY'
import hashlib, os, sys
import gdown
dest = sys.argv[1]
os.makedirs(dest, exist_ok=True)
files = {
    "epoch=804-step=57154.ckpt": ("1T2AB3Qv3ktZRwMtE8IYbSKCbfBlUIyIJ", "ee95964d8ca104cdc9fbb55ce54f9e52d891154d35ee1a47031fe01b0d523ee0"),
    "hparams.yaml": ("1q1QCrmdSdcfmtTVetm_LTWbpY7L2dLO9", "d1053e8bbaa5a0d1dbfe528c28d2750dc29b193f75046028ae06a3ec3051af3f"),
    "lattice_scaler.pt": ("1n1rNWCPFF0bVlTWONfrF7pbAOxHJxq13", "8d099eecd61b35f49eb77ef8444312a6def60e7b830ac1fb97b44723424a1e2b"),
    "prop_scaler.pt": ("1li2VNiqaCFlCMtZ8QDiz4rvs5-9unHsG", "84c119ca9b32ac7278373c0c3591315064940bef5af192f9faccca54bc08a6af"),
}
for name, (file_id, sha) in files.items():
    path = os.path.join(dest, name)
    if not os.path.exists(path):
        gdown.download(id=file_id, output=path, quiet=True)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != sha:
        sys.exit(f"Checksum mismatch for {path}; delete it and rerun setup.sh")
print("Checkpoint files OK")
PY

echo "Done. Example:"
echo "  $CONDA run -n $ENV_NAME python $HERE/run.py --pattern my_scan.xy --wavelength CuKa1 --composition LuOF --z 2 --out results/"
