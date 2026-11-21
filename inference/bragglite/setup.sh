#!/usr/bin/env bash
# One-time setup for Bragglite inference: creates the conda environment, clones upstream PXRD-Bench
# at the commit verified in manifest.yaml, downloads the official checkpoint archive (1.58 GB,
# Zenodo record 23878332), checks its deposited md5 and its sha256, and extracts the four
# single-phase checkpoints (the two-phase files are not used). Works with conda, mamba, or
# micromamba. BRAGGLITE_UPSTREAM and BRAGGLITE_WEIGHTS move the checkout and the checkpoints
# elsewhere (run.py reads the same variables); KEEP_ARCHIVE=1 keeps the downloaded archive.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/dongyeop3813/PXRD-Bench.git"
COMMIT="2a791ca33d2adc10ec1f247817cddd8adb9f6fc1"
ARCHIVE_URL="https://zenodo.org/records/23878332/files/bragglite-checkpoints-v1.tar.gz?download=1"
ARCHIVE_MD5="6f1f8e2d60e438b79349e8f84f8eb292"  # deposited on the Zenodo record
ARCHIVE_SHA256="ccc981c46af394b61e4193d488a1cdb5b876c7bfab2659e8926637252c31a39c"
ENV_NAME="${ENV_NAME:-bragglite}"
UPSTREAM="${BRAGGLITE_UPSTREAM:-$HERE/.upstream}"
WEIGHTS="${BRAGGLITE_WEIGHTS:-$HERE/.weights}"

CONDA="$(command -v micromamba || command -v mamba || command -v conda || true)"
[ -n "$CONDA" ] || { echo "Install conda, mamba, or micromamba first." >&2; exit 1; }

if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment $ENV_NAME already exists; reusing it (remove it to rebuild from environment.yml)."
else
  "$CONDA" env create -y -n "$ENV_NAME" -f "$HERE/environment.yml"
fi

if [ ! -d "$UPSTREAM/.git" ]; then
  git clone "$REPO_URL" "$UPSTREAM"
fi
git -C "$UPSTREAM" fetch -q origin "$COMMIT"
git -C "$UPSTREAM" checkout -q "$COMMIT"

mkdir -p "$WEIGHTS"
"$CONDA" run -n "$ENV_NAME" python - "$WEIGHTS" "$ARCHIVE_URL" "$ARCHIVE_MD5" "$ARCHIVE_SHA256" "${KEEP_ARCHIVE:-0}" <<'PY'
import hashlib, os, subprocess, sys, tarfile
dest, url, md5, sha, keep = sys.argv[1:6]
prefix = "bragglite-checkpoints-v1/checkpoints/single_phase/"
files = {  # sha256 of the extracted single-phase checkpoints
    "mp20_element_set.pt": "253a47efbf3c278c69ab0d21675b805c5b3e99c75b4a9f2617c5e563df3ff883",
    "mp20_stoichiometry.pt": "241bde118d1ad00ac09bb848b2159d9d9d98e4057fd7e47ed6a16a1abcc39296",
    "pxrd_bench_element_set.pt": "fd274604696492fa2394f824acfbd2e36d01ea6431d89295336171b552440805",
    "pxrd_bench_stoichiometry.pt": "2b6fd3ff608322aa0b48fd03349e37d0751026923a24bedcfa6a0208632a1400",
}

def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def extracted_ok():
    return all(os.path.exists(os.path.join(dest, prefix, n)) and digest(os.path.join(dest, prefix, n), "sha256") == s
               for n, s in files.items())

if extracted_ok():
    print("Single-phase checkpoints already present and verified")
    sys.exit(0)
archive = os.path.join(dest, "bragglite-checkpoints-v1.tar.gz")
if not os.path.exists(archive):
    subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-o", archive + ".part", url], check=True)
    os.replace(archive + ".part", archive)
if digest(archive, "md5") != md5 or digest(archive, "sha256") != sha:
    sys.exit(f"Checksum mismatch for {archive}; delete it and rerun setup.sh")
with tarfile.open(archive) as tar:
    members = [m for m in tar.getmembers() if m.name.startswith(prefix) and m.isfile()]
    tar.extractall(dest, members=members, filter="data")
if not extracted_ok():
    sys.exit("Extracted checkpoints do not match the recorded sha256 values")
if keep != "1":
    os.remove(archive)
print("Checkpoint archive OK; single-phase checkpoints extracted to", os.path.join(dest, prefix))
PY

echo "Done. Example:"
echo "  $CONDA run -n $ENV_NAME python $HERE/run.py --pattern my_scan.xy --wavelength CuKa1 --composition BaTiO3 --z 1 --out results/"
