# Shared PXRD input handling for the inference scripts in this repository.
# Loads a measured pattern, converts it between wavelengths, and picks peaks for models
# that condition on peak lists instead of full profiles.
# Dependencies: numpy, scipy (pymatgen only for experimental CIF input).
# Curated by: Xiangyu Yin (xiangyu-yin.com)

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks, peak_widths, savgol_filter

CU_KA1 = 1.54056
CU_KA2 = 1.54439
CU_KA = 1.54184  # weighted average, the pymatgen "CuKa" default
# Q-space input is stored as 2theta at this short virtual wavelength so every Q up to ~125 1/A
# maps to a real angle; converting back to Q (or to any other wavelength) is exact.
VIRTUAL_Q_WAVELENGTH = 0.1

NAMED_WAVELENGTHS = {
    "CuKa": CU_KA,
    "CuKa1": CU_KA1,
    "MoKa": 0.71073,
    "MoKa1": 0.70930,
    "CoKa": 1.79026,
    "CoKa1": 1.78897,
    "FeKa": 1.93735,
    "CrKa": 2.29100,
    "AgKa": 0.560885,
}


@dataclass
class Pattern:
    two_theta: np.ndarray  # degrees
    intensity: np.ndarray
    wavelength: float  # Angstrom
    source: str = ""


def parse_wavelength(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    lam = NAMED_WAVELENGTHS[value] if value in NAMED_WAVELENGTHS else float(value)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError(f"Wavelength must be a positive number of Angstrom, got {value!r}")
    return lam


def _clean(x: np.ndarray, y: np.ndarray, path: str) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite rows, sort, and average intensities recorded at the same x."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ux, inverse = np.unique(x, return_inverse=True)
    uy = np.bincount(inverse, weights=y) / np.bincount(inverse)
    if len(ux) < 10:
        raise ValueError(f"{path} has fewer than 10 distinct finite data points")
    return ux, uy


def load_pattern(path: str, wavelength: str | float | None = None, x_unit: str = "2theta") -> Pattern:
    """Load a pattern from .xy/.xye/.dat/.txt/.csv or a pdCIF.

    x_unit is "2theta" (degrees) or "q" (1/Angstrom). For 2theta column files the wavelength
    must be supplied; for pdCIF it is read from the file unless given explicitly. Q data needs
    no wavelength and is returned as 2theta at VIRTUAL_Q_WAVELENGTH.
    """
    ext = os.path.splitext(path)[1].lower()
    lam = parse_wavelength(wavelength)
    if x_unit not in ("2theta", "q"):
        raise ValueError(f"x_unit must be '2theta' or 'q', got {x_unit!r}")

    if ext in (".cif", ".pcif"):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, repo_root)
        from utils.parse_cifs import read_experimental_cif

        with open(path, encoding="utf-8", errors="ignore") as fin:
            text = fin.read()
        if "_diffrn_radiation_wavelength" not in text and "_diffrn_radiation_type" not in text:
            raise ValueError(f"{path} records no wavelength; convert it to a 2theta column file and pass --wavelength")
        result = read_experimental_cif(filepath=path)
        two_theta, intensity, cif_lam = result[6], result[8], float(result[9])
        # Angles from d-spacing columns are derived with the file's wavelength, so an override
        # would silently shift every peak.
        if lam is not None and abs(lam - cif_lam) > 1e-4:
            raise ValueError(f"{path} records wavelength {cif_lam} A; omit --wavelength for pdCIF input")
        x, y = _clean(np.asarray(two_theta, float), np.asarray(intensity, float), path)
        return Pattern(x, y, cif_lam, path)

    if lam is None and x_unit == "2theta":
        raise ValueError(f"A wavelength is required for {path} (e.g. --wavelength CuKa or 1.5406)")

    rows = []
    with open(path, encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            tokens = line.replace(",", " ").replace(";", " ").split()
            if len(tokens) < 2:
                continue
            try:
                rows.append((float(tokens[0]), float(tokens[1])))
            except ValueError:
                continue  # header or comment line
    if len(rows) < 10:
        raise ValueError(f"Could not read a two-column pattern from {path}")
    arr = np.array(rows)
    x, y = _clean(arr[:, 0], arr[:, 1], path)
    if x_unit == "q":
        lam = VIRTUAL_Q_WAVELENGTH
        q_max = 4.0 * np.pi / lam
        if x.min() < 0 or x.max() >= q_max:
            raise ValueError(f"Q values must lie in [0, {q_max:.1f}) 1/Angstrom; got {x.min():.3g}-{x.max():.3g}")
        x = 2.0 * np.degrees(np.arcsin(x * lam / (4.0 * np.pi)))
    elif x.min() < 0 or x.max() >= 180:
        raise ValueError(f"2theta values must lie in [0, 180) degrees; got {x.min():.3g}-{x.max():.3g}")
    return Pattern(x, y, lam, path)


def convert_two_theta(two_theta: np.ndarray, lam_in: float, lam_out: float) -> np.ndarray:
    """Map 2theta between wavelengths through d-spacing. Unreachable angles become NaN."""
    sin_out = np.sin(np.radians(two_theta) / 2.0) * lam_out / lam_in
    with np.errstate(invalid="ignore"):
        return np.where(np.abs(sin_out) <= 1.0, 2.0 * np.degrees(np.arcsin(sin_out)), np.nan)


def _resample_uniform(pattern: Pattern) -> tuple[np.ndarray, np.ndarray]:
    # Use a fine step (10th percentile of the spacings) so locally denser regions keep their
    # resolution, bounded to keep the grid size reasonable.
    x, y = pattern.two_theta, pattern.intensity
    step = max(np.percentile(np.diff(x), 10), (x[-1] - x[0]) / 200_000)
    grid = np.arange(x[0], x[-1], step)
    return grid, np.interp(grid, x, y)


def _subtract_background(y: np.ndarray, window: int) -> np.ndarray:
    # Rolling minimum followed by smoothing: a simple, assumption-light background estimate.
    base = minimum_filter1d(y, size=window, mode="nearest")
    base = uniform_filter1d(base, size=window, mode="nearest")
    return np.clip(y - base, 0.0, None)


def pick_peaks(
    pattern: Pattern,
    max_peaks: int | None = None,
    min_rel_prominence: float = 0.01,
    strip_ka2: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pick peaks from a measured profile.

    Returns (two_theta, intensity, wavelength): peak positions, intensities scaled so the
    strongest peak is 100, sorted by decreasing intensity, and the wavelength the positions
    refer to (the pattern's own). Intensity is an area estimate (background-subtracted height x
    FWHM), which tracks integrated intensities from simulation better than raw heights.

    strip_ka2 merges resolved Cu K-alpha2 satellites into their K-alpha1 parents; merged
    parents are re-expressed at the pattern's wavelength. By default it is enabled for Cu
    wavelengths (1.5406 or 1.5418);
    pass False for monochromated K-alpha1 data. A satellite must sit at the K-alpha2 position
    with 30-75% of the parent's area, and each parent absorbs at most one satellite.
    """
    x, y = _resample_uniform(pattern)
    step = x[1] - x[0]
    is_cu = abs(pattern.wavelength - CU_KA) < 0.003 or abs(pattern.wavelength - CU_KA1) < 0.003
    if strip_ka2 is None:
        strip_ka2 = is_cu
    elif strip_ka2 and not is_cu:
        raise ValueError(f"K-alpha2 stripping only applies to Cu radiation, not {pattern.wavelength} A")

    # Widths below are in degrees at Cu Ka; angular widths scale roughly with wavelength, so
    # Mo, synchrotron, and Q-space (virtual wavelength) data get proportionally narrower windows.
    scale = pattern.wavelength / CU_KA1
    window = max(5, int(round(3.0 * scale / step)))  # background window ~3 deg at Cu
    y_net = _subtract_background(y, window)
    smooth = max(5, int(round(0.06 * scale / step)) | 1)
    if smooth < len(y_net):
        y_net = np.clip(savgol_filter(y_net, smooth, 2), 0.0, None)

    empty = (np.array([]), np.array([]), pattern.wavelength)
    if y_net.max() <= 0:
        return empty
    idx, _ = find_peaks(
        y_net,
        prominence=min_rel_prominence * y_net.max(),
        distance=max(1, int(round(0.05 * scale / step))),
    )
    if len(idx) == 0:
        return empty
    widths = peak_widths(y_net, idx, rel_height=0.5)[0] * step
    pos = x[idx]
    area = y_net[idx] * widths
    lam = pattern.wavelength

    if strip_ka2:
        keep = np.ones(len(pos), dtype=bool)
        parent = np.zeros(len(pos), dtype=bool)
        # Treat the peaks as K-alpha1 lines; strongest parents claim their satellites first.
        expected = convert_two_theta(pos, CU_KA1, CU_KA2)
        for i in np.argsort(-area):
            if not keep[i]:
                continue
            tol = max(0.02, 0.6 * widths[i])
            ratio = area / area[i]
            cand = np.where(
                keep & ~parent & (np.abs(pos - expected[i]) < tol) & (ratio >= 0.3) & (ratio <= 0.75)
            )[0]
            cand = cand[cand != i]
            if len(cand):
                k = cand[np.argmin(np.abs(pos[cand] - expected[i]))]
                keep[k] = False
                area[i] += area[k]
                parent[i] = True
        # A merged parent is a resolved K-alpha1 line: express it at the pattern's wavelength
        # like every unmerged (unresolved) peak, so all positions share one wavelength.
        pos = np.where(parent, convert_two_theta(pos, CU_KA1, lam), pos)
        pos, area = pos[keep], area[keep]

    order = np.argsort(-area)
    pos, area = pos[order], area[order]
    if max_peaks:
        pos, area = pos[:max_peaks], area[:max_peaks]
    return pos, 100.0 * area / area.max(), lam


def write_peaks_csv(path: str, two_theta: np.ndarray, intensity: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        fout.write("2theta,intensity\n")
        for t, i in zip(two_theta, intensity):
            fout.write(f"{t:.4f},{i:.2f}\n")
