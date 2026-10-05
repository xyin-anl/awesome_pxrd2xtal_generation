#!/usr/bin/env bash
# One-time setup for Ab-PXRD-Solver: creates the conda environment (GSAS-II is compiled from source
# with conda-forge's Fortran toolchain; this takes several minutes) and clones upstream at the
# commit verified in manifest.yaml. The solver's own models ship with upstream; MACE downloads its
# foundation model on first use. Works with conda, mamba, or micromamba.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/MaterSim/Ab-PXRD-Solver.git"
COMMIT="0d67964dfcbd2e2fa09e79bbbac2f5d020a121e9"
ENV_NAME="${ENV_NAME:-ab_pxrd_solver}"

CONDA="$(command -v micromamba || command -v mamba || command -v conda || true)"
[ -n "$CONDA" ] || { echo "Install conda, mamba, or micromamba first." >&2; exit 1; }

if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment $ENV_NAME already exists; reusing it (remove it to rebuild from environment.yml)."
else
  "$CONDA" env create -y -n "$ENV_NAME" -f "$HERE/environment.yml"
fi
"$CONDA" run -n "$ENV_NAME" python -c "from GSASII import GSASIIscriptable; print('GSAS-II OK')"

if [ ! -d "$HERE/.upstream/.git" ]; then
  git clone "$REPO_URL" "$HERE/.upstream"
fi
git -C "$HERE/.upstream" fetch -q origin "$COMMIT"
git -C "$HERE/.upstream" checkout -q "$COMMIT"

# Upstream calls mace_mp(model="small"), which downloads this foundation model on first use; fetch
# it now and check it is the model the wrapper was verified with.
"$CONDA" run -n "$ENV_NAME" python - <<'PY'
import hashlib, os
from mace.calculators import mace_mp
mace_mp(model="small", device="cpu")
path = os.path.expanduser("~/.cache/mace/20231210mace128L0_energy_epoch249model")
sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
if sha != "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736":
    raise SystemExit(f"Unexpected MACE model at {path} (sha256 {sha})")
print("MACE small model OK")
PY

echo "Done. Example:"
echo "  $CONDA run -n $ENV_NAME python $HERE/run.py --pattern scan.xy --wavelength CuKa --composition PrYMg2 --spacegroup P4/mmm --out results/"
