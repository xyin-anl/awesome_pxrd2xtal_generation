"""Helpers for handling cross-wavelength diffraction data transformations."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


HC_IN_KEV_ANGSTROM = 12.398419843320026  # Planck constant * speed of light (keV·Å)


def energy_keV_to_wavelength_A(E_keV: float) -> float:
    """Convert photon energy in keV to wavelength in Ångström."""

    return HC_IN_KEV_ANGSTROM / float(E_keV)


def resolve_input_wavelength(
    *,
    input_energy_keV: Optional[float],
    input_wavelength_A: Optional[float],
    default_wavelength_A: float,
) -> Tuple[float, str]:
    """Infer the wavelength used to collect the input diffraction pattern.

    Returns a tuple of (wavelength_in_A, source_tag) where ``source_tag``
    indicates which configuration entry determined the wavelength.
    """

    if input_wavelength_A is not None:
        return float(input_wavelength_A), "wavelength"

    if input_energy_keV is not None:
        return energy_keV_to_wavelength_A(float(input_energy_keV)), "energy"

    return float(default_wavelength_A), "default"


def warp_to_ref_wavelength(
    twotheta_in_deg: np.ndarray,
    intensity_in: np.ndarray,
    lambda_in_A: float,
    lambda_ref_A: float,
    *,
    lp_match: bool = False,
    lp_ratio_clip: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map an input pattern onto the reference wavelength axis using Bragg's law."""

    twotheta_in_deg = np.asarray(twotheta_in_deg, dtype=float)
    y = np.asarray(intensity_in, dtype=float)

    theta_in = np.deg2rad(twotheta_in_deg / 2.0)
    scale = (lambda_ref_A / lambda_in_A)
    s = scale * np.sin(theta_in)

    valid = np.abs(s) <= 1.0
    if not np.any(valid):
        raise ValueError(
            "No angles can be mapped: (lambda_ref/lambda_in) * sin(theta_in) > 1 everywhere."
        )

    theta_ref = np.arcsin(np.clip(s[valid], -1.0, 1.0))
    twotheta_ref_deg = 2.0 * np.rad2deg(theta_ref)
    y_valid = y[valid]

    if lp_match:
        # Bragg–Brentano, unpolarized: LP(theta) = (1 + cos^2 2θ) / (sin^2 θ * cos θ)
        def LP(th):
            c2t = np.cos(2.0 * th)
            s_ = np.sin(th)
            c_ = np.cos(th)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = (1.0 + c2t * c2t) / (
                    np.maximum(s_ * s_, 1e-12) * np.maximum(c_, 1e-12)
                )
            return out

        LP_in = LP(theta_in[valid])
        LP_ref = LP(theta_ref)
        ratio = LP_ref / np.maximum(LP_in, 1e-12)
        ratio = np.clip(ratio, 0.0, float(lp_ratio_clip))
        y_valid = y_valid * ratio

    return twotheta_ref_deg, y_valid


def nearest_recast(
    x_in: np.ndarray,
    y_in: np.ndarray,
    x_centers: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour assignment of intensities onto a target grid."""

    x_in = np.asarray(x_in, dtype=float)
    y_in = np.asarray(y_in, dtype=float)
    x_centers = np.asarray(x_centers, dtype=float)

    y_out = np.zeros_like(x_centers, dtype=float)
    for x_val, y_val in zip(x_in, y_in):
        idx = np.abs(x_centers - x_val).argmin()
        y_out[idx] = y_val
    return y_out


def area_preserving_rebin(
    x_in: np.ndarray,
    y_in: np.ndarray,
    x_centers: np.ndarray,
    *,
    fill_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rebin onto target bins while conserving integrated intensity."""

    x_in = np.asarray(x_in, dtype=float)
    y_in = np.asarray(y_in, dtype=float)
    x_centers = np.asarray(x_centers, dtype=float)

    order = np.argsort(x_in)
    x = x_in[order]
    y = y_in[order]

    if len(x_centers) < 2:
        raise ValueError("Need at least 2 bin centers for rebinning.")

    step = float(np.diff(x_centers).mean())
    edges = np.concatenate(
        (
            [x_centers[0] - step / 2.0],
            (x_centers[:-1] + x_centers[1:]) / 2.0,
            [x_centers[-1] + step / 2.0],
        )
    )

    x_ext = np.concatenate(([edges[0]], x, [edges[-1]]))
    y_ext = np.concatenate(([y[0]], y, [y[-1]]))
    cum = np.zeros_like(x_ext)
    cum[1:] = np.cumsum(0.5 * (y_ext[1:] + y_ext[:-1]) * np.diff(x_ext))

    cum_at_edges = np.interp(edges, x_ext, cum)
    bin_area = np.diff(cum_at_edges)
    y_out = bin_area / step

    covered = (edges[:-1] < x.max()) & (edges[1:] > x.min())

    if fill_value is None:
        fill_value = float(np.percentile(y_in, 5.0))
    y_out[~covered] = fill_value

    return y_out, covered.astype(np.uint8)
