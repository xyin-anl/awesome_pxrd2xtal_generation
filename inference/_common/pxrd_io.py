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
        if not any(tag in text for tag in ("_pd_proc_wavelength", "_diffrn_radiation_wavelength", "_diffrn_radiation_type")):
            raise ValueError(f"{path} records no wavelength; convert it to a 2theta column file and pass --wavelength")
        result = read_experimental_cif(filepath=path, require_structure=False)
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


def ka2_asymmetry(pattern: Pattern, min_two_theta: float = 40.0, min_rel_height: float = 0.08) -> tuple[float, int]:
    """Measure whether a Cu profile contains the K-alpha1/K-alpha2 doublet.

    For each strong peak above min_two_theta (where the doublet splits by >= 0.1 deg), compares
    the background-subtracted intensity at the K-alpha2 offset above the peak with the intensity
    the same distance below it, relative to the peak height. Returns (median, number of peaks).
    Single-wavelength profiles give about 0 (at most 0.05); the doublet gives 0.13-0.5.
    """
    x, y = _resample_uniform(pattern)
    step = x[1] - x[0]
    y_net = _subtract_background(y, max(5, int(round(3.0 / step))))
    if y_net.max() <= 0:
        return float("nan"), 0
    idx, _ = find_peaks(y_net, height=min_rel_height * y_net.max(), distance=max(1, int(round(0.05 / step))))
    # Positions are taken as K-alpha1 lines whatever the label says; the offset depends only on angle.
    t = x[idx][x[idx] >= min_two_theta]
    h = y_net[idx][x[idx] >= min_two_theta]
    d = convert_two_theta(t, CU_KA1, CU_KA2) - t
    # A resolved K-alpha2 satellite is itself a peak and would score strongly negative; skip peaks
    # with a stronger signal where their K-alpha1 parent would be.
    satellite = np.interp(convert_two_theta(t, CU_KA2, CU_KA1), x, y_net) > h
    ok = np.isfinite(d) & (t + d <= x[-1]) & ~satellite
    if not ok.any():
        return float("nan"), 0
    score = (np.interp(t + d, x, y_net) - np.interp(t - d, x, y_net))[ok] / h[ok]
    return float(np.median(score)), int(ok.sum())


# Simulated controls: -0.02 to 0.05 without the doublet, 0.19 to 0.50 with it; Ka1-labeled
# benchmark files that contain it: 0.13 to 0.36.
KA2_ASYMMETRY_THRESHOLD = 0.09
KA2_MIN_PEAKS = 3


def pick_peaks(
    pattern: Pattern,
    max_peaks: int | None = None,
    min_rel_prominence: float = 0.01,
    strip_ka2: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pick peaks from a measured profile.

    Returns (two_theta, intensity, wavelength): peak positions, intensities scaled so the
    strongest peak is 100, sorted by decreasing intensity, and the wavelength the positions
    refer to (the pattern's own, or averaged Cu K-alpha when K-alpha2 is stripped). Intensity is an area estimate (background-subtracted height x
    FWHM), which tracks integrated intensities from simulation better than raw heights.

    strip_ka2 merges resolved Cu K-alpha2 satellites into their K-alpha1 parents; merged
    peaks then refer to averaged Cu K-alpha. By default (None) it is enabled for Cu
    data declared at the averaged Cu K-alpha wavelength and for other Cu data whose profile shows the
    doublet (ka2_asymmetry); pass True or False to override. A satellite must sit at the K-alpha2 position
    with 30-75% of the parent's area, and each parent absorbs at most one satellite.
    """
    x, y = _resample_uniform(pattern)
    step = x[1] - x[0]
    is_cu = abs(pattern.wavelength - CU_KA) < 0.003 or abs(pattern.wavelength - CU_KA1) < 0.003
    if strip_ka2 is None:
        # Data declared at the averaged Cu Ka wavelength contain the Ka1/Ka2 doublet. Labels are
        # unreliable the other way round (files declared as Ka1 can still contain Ka2), so other
        # Cu data are stripped when the profile itself shows the doublet.
        strip_ka2 = abs(pattern.wavelength - CU_KA) < 0.0008
        if is_cu and not strip_ka2:
            score, n = ka2_asymmetry(pattern)
            strip_ka2 = n >= KA2_MIN_PEAKS and score >= KA2_ASYMMETRY_THRESHOLD
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
            # Within half the doublet splitting, so a real neighbouring reflection of a broad
            # pattern is not taken for a satellite.
            tol = min(max(0.02, 0.6 * widths[i]), 0.5 * (expected[i] - pos[i]))
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
        # With the doublet present, a picked maximum sits at (or, for an unresolved doublet, close
        # to) the stronger K-alpha1 line, not at the intensity-weighted average: treat every
        # position as K-alpha1 and express it at averaged Cu Ka. On simulated doublet patterns this
        # cut the median position error 2-5x for 0.06-0.1 deg peaks compared with keeping unmerged
        # peaks as they are.
        lam = CU_KA
        pos = convert_two_theta(pos, CU_KA1, lam)
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
