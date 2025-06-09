# Script to parse experimental CIF files that contain PXRD data
# Inspired by https://github.com/gabeguo/cdvae_xrd/blob/main/process_real_xrds/read_real_xrd.py
# Curated by: Xiangyu Yin (xiangyu-yin.com)

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip

from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.diffraction.xrd import WAVELENGTHS


default_rad_type = "CuKa"
default_min_2theta = 5.0
default_max_2theta = 90.0
possible_2theta_suffixes = [
    "_2theta_corrected",
    "_2theta_scan",
    "_2theta_centroid",
    "_2theta",
]
possible_dspacing_suffixes = [
    "_d_spacing",
]
possible_tof_suffixes = [
    "_time_of_flight",
]
possible_intensity_suffixes = [
    "_intensity_net",
    "_intensity_total",
    "_counts_total",
    "_counts",
    "_pk_height",
    "_intensity",
    "_calc_intensity_total",
    "_calc_intensity_net",
]
possible_bg_suffixes = [
    "_intensity_bkg_calc",
    "_intensity_calc_bkg",
    "_intensity_bkg",
    "_intensity_background",
]


def get_field_value(all_lines, desired_start, is_num=True):
    for i, the_line in enumerate(all_lines):
        if the_line.startswith(desired_start):
            split_line = the_line.split()
            if len(split_line) > 1:
                val = split_line[-1]
                if is_num:
                    try:
                        return float(val)
                    except ValueError:
                        pass
                else:
                    return val
            else:
                ret_val = all_lines[i + 1]
                tokens = ret_val.split()
                for token in tokens:
                    try:
                        return float(token) if is_num else token
                    except ValueError:
                        continue
                if not is_num:
                    return ret_val
                raise ValueError(f"Invalid numeric value for {desired_start}")
    raise ValueError(f"Could not find field '{desired_start}' in CIF lines.")


def find_index_of_xrd_loop(all_lines):
    for i in range(len(all_lines) - 1):
        if all_lines[i].strip() == "loop_" and ("_pd_" in all_lines[i + 1]):
            return i
    raise ValueError("Could not find an XRD data loop (loop_ + _pd_...).")


def find_end_of_xrd(all_lines, start_idx):
    for i in range(start_idx + 1, len(all_lines)):
        line = all_lines[i].strip()
        if (not line) or line.startswith("_") or line.startswith("loop_"):
            return i
    return len(all_lines)


def find_first_by_suffix(field_list, suffix_list):
    lower_field_list = [f.lower() for f in field_list]
    for i, field_name in enumerate(lower_field_list):
        for sfx in suffix_list:
            if field_name.endswith(sfx.lower()):
                return i
    return None


def auto_identify_columns(field_list):
    two_theta_idx = find_first_by_suffix(field_list, possible_2theta_suffixes)
    d_spacing_idx = find_first_by_suffix(field_list, possible_dspacing_suffixes)
    tof_idx = find_first_by_suffix(field_list, possible_tof_suffixes)
    intensity_idx = find_first_by_suffix(field_list, possible_intensity_suffixes)
    bg_idx = find_first_by_suffix(field_list, possible_bg_suffixes)

    return two_theta_idx, d_spacing_idx, tof_idx, intensity_idx, bg_idx


def read_experimental_cif(filepath, plot=False, save_pickle=False, pickle_path=None):

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Could not find file: {filepath}")

    with open(filepath, "r") as fin:
        all_lines = [x.rstrip("\n") for x in fin]

    xrd_loop_start_idx = find_index_of_xrd_loop(all_lines)

    field_list = []
    line_idx = xrd_loop_start_idx + 1
    while line_idx < len(all_lines):
        line = all_lines[line_idx].strip()
        if not line:
            break
        if line.startswith("_pd_"):
            token = line.split()[0]
            field_list.append(token)
            line_idx += 1
        else:
            break

    data_start_idx = line_idx
    while data_start_idx < len(all_lines) and not all_lines[data_start_idx].strip():
        data_start_idx += 1

    data_end_idx = find_end_of_xrd(all_lines, data_start_idx)
    xrd_lines = all_lines[data_start_idx:data_end_idx]

    (two_theta_idx, d_spacing_idx, tof_idx, intensity_idx, bg_idx) = (
        auto_identify_columns(field_list)
    )

    if intensity_idx is None and (
        two_theta_idx is None and d_spacing_idx is None and tof_idx is None
    ):
        two_theta_idx = 0
        intensity_idx = 1

    try:
        rad_type = get_field_value(
            all_lines, "_diffrn_radiation_type", is_num=False
        ).lower()
        if "neutron" in rad_type:
            raise ValueError(
                "Neutron diffraction data is not supported in this script."
            )
    except ValueError:
        pass

    try:
        exp_wavelength = float(
            get_field_value(all_lines, "_diffrn_radiation_wavelength")
        )
    except ValueError:
        try:
            rad_type_raw = get_field_value(
                all_lines, "_diffrn_radiation_type", is_num=False
            )
        except ValueError:
            rad_type_raw = default_rad_type
        if "Cu" in rad_type_raw:
            if "1" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CuKa1"]
            elif "1" in rad_type_raw and "b" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CuKb1"]
            elif "2" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CuKa2"]
            else:
                exp_wavelength = WAVELENGTHS["CuKa"]
        elif "Mo" in rad_type_raw:
            if "1" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["MoKa1"]
            elif "1" in rad_type_raw and "b" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["MoKb1"]
            elif "2" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["MoKa2"]
            else:
                exp_wavelength = WAVELENGTHS["MoKa"]
        elif "Cr" in rad_type_raw:
            if "1" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CrKa1"]
            elif "1" in rad_type_raw and "b" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CrKb1"]
            elif "2" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CrKa2"]
            else:
                exp_wavelength = WAVELENGTHS["CrKa"]
        elif "Fe" in rad_type_raw:
            if "1" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["FeKa1"]
            elif "1" in rad_type_raw and "b" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["FeKb1"]
            elif "2" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["FeKa2"]
            else:
                exp_wavelength = WAVELENGTHS["FeKa"]
        elif "Co" in rad_type_raw:
            if "1" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CoKa1"]
            elif "1" in rad_type_raw and "b" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CoKb1"]
            elif "2" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CoKa2"]
            else:
                exp_wavelength = WAVELENGTHS["CoKa"]
        elif "Ag" in rad_type_raw:
            if "1" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["AgKa1"]
            elif "1" in rad_type_raw and "b" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["AgKb1"]
            elif "2" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["AgKa2"]
            else:
                exp_wavelength = WAVELENGTHS["AgKa"]
        else:
            exp_wavelength = WAVELENGTHS[rad_type_raw]

    if two_theta_idx is not None and len(xrd_lines) > 0:
        try:
            first_val = float(xrd_lines[0].split()[two_theta_idx].split("(")[0])
            last_val = float(xrd_lines[-1].split()[two_theta_idx].split("(")[0])
            exp_2theta_min = min(first_val, last_val)
            exp_2theta_max = max(first_val, last_val)
        except ValueError:
            pass
    else:
        try:
            exp_2theta_min = float(
                get_field_value(all_lines, "_pd_meas_2theta_range_min")
            )
            exp_2theta_max = float(
                get_field_value(all_lines, "_pd_meas_2theta_range_max")
            )
        except ValueError:
            exp_2theta_min = default_min_2theta
            exp_2theta_max = default_max_2theta

    raw_2theta_vals = []
    raw_q_vals = []
    raw_i_vals = []

    fallback_2theta_array = None
    explicit_2theta = two_theta_idx is not None
    if (not explicit_2theta) and (len(xrd_lines) > 1):
        fallback_2theta_array = np.linspace(
            exp_2theta_min, exp_2theta_max, len(xrd_lines)
        )

    for i, line in enumerate(xrd_lines):
        txt = line.strip()
        if not txt:
            continue
        parts = txt.split()
        if len(parts) < 1:
            continue

        if intensity_idx is not None and intensity_idx < len(parts):
            raw_intensity = parts[intensity_idx]
        else:
            raw_intensity = "0.0"

        try:
            intensity_val = float(raw_intensity.split("(")[0])
        except ValueError:
            intensity_val = 0.0

        if bg_idx is not None and bg_idx < len(parts):
            raw_bg = parts[bg_idx]
            try:
                bg_val = float(raw_bg.split("(")[0])
                intensity_val -= bg_val
            except ValueError:
                pass

        if explicit_2theta and two_theta_idx < len(parts):
            try:
                raw_2theta = parts[two_theta_idx]
                two_theta_deg = float(raw_2theta.split("(")[0])
            except ValueError:
                continue
            theta_rad = np.radians(two_theta_deg / 2.0)
            curr_Q = 4.0 * np.pi * np.sin(theta_rad) / exp_wavelength

        elif d_spacing_idx is not None and d_spacing_idx < len(parts):
            try:
                d_val = float(parts[d_spacing_idx].split("(")[0])
            except ValueError:
                continue
            if d_val == 0:
                continue
            curr_Q = 2.0 * np.pi / d_val

            try:
                theta_rad = np.arcsin(exp_wavelength / (2.0 * d_val))
                two_theta_deg = 2.0 * np.degrees(theta_rad)
            except ValueError:
                two_theta_deg = float("nan")

        elif tof_idx is not None and tof_idx < len(parts):
            print(
                "TOF data not supported in this script. Please add your own conversion."
            )
            continue

        else:
            if fallback_2theta_array is not None and i < len(fallback_2theta_array):
                two_theta_deg = fallback_2theta_array[i]
                theta_rad = np.radians(two_theta_deg / 2.0)
                curr_Q = 4.0 * np.pi * np.sin(theta_rad) / exp_wavelength
            else:
                continue

        raw_q_vals.append(curr_Q)
        raw_i_vals.append(intensity_val)
        raw_2theta_vals.append(two_theta_deg)

    if len(raw_q_vals) == 0:
        raise ValueError("No valid Q-intensity data could be parsed from the CIF.")

    raw_q_vals = np.array(raw_q_vals, dtype=float)
    raw_i_vals = np.array(raw_i_vals, dtype=float)
    raw_2theta_vals = np.array(raw_2theta_vals, dtype=float)
    sort_idx = np.argsort(raw_q_vals)
    q_vals = raw_q_vals[sort_idx]
    i_vals = raw_i_vals[sort_idx]
    two_theta_vals = raw_2theta_vals[sort_idx]

    cif_parser = CifParser(filepath)
    structure = cif_parser.get_structures()[0]
    cif_writer = CifWriter(structure)
    cif_str = cif_writer.__str__()

    source_file_name = os.path.basename(filepath)
    pretty_formula = structure.composition.reduced_formula
    frac_coords = structure.frac_coords
    atom_types = structure.atomic_numbers
    sga = SpacegroupAnalyzer(structure)
    spacegroup_number = sga.get_space_group_number()

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(q_vals, i_vals, color="green", label="Experimental XRD")
        plt.title(f"PXRD of {pretty_formula} from {source_file_name}")
        plt.xlabel(r"$Q$ ($\mathrm{\AA}^{-1}$)")
        plt.ylabel("Experimental Intensity")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    result_tuple = (
        source_file_name,
        cif_str,
        pretty_formula,
        frac_coords,
        atom_types,
        spacegroup_number,
        two_theta_vals,
        q_vals,
        i_vals,
        exp_wavelength,
        exp_2theta_min,
        exp_2theta_max,
    )

    if save_pickle:
        if pickle_path is None:
            pickle_path = filepath.replace(".cif", ".pkl.gz")
        with gzip.open(pickle_path, "wb") as f:
            pickle.dump(result_tuple, f)
        print(f"[INFO] Saved results to compressed pickle: {pickle_path}")

    return result_tuple
