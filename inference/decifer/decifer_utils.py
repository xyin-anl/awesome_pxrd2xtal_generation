# PXRD Inference unitilies for deCIFer (https://github.com/frederiklizakjohansen/decifer)
# Curated by: Xiangyu Yin (xiangyu-yin.com)

from typing import Optional, List, Tuple, Union, Dict, Any
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.groups import SpaceGroup

from decifer.tokenizer import Tokenizer
from decifer.utility import (
    replace_symmetry_loop_with_P1,
    extract_space_group_symbol,
    reinstate_symmetry_loop,
    space_group_to_crystal_system,
    space_group_symbol_to_number,
)
from decifer.decifer_model import Decifer, DeciferConfig

global_elements_list = [Element.from_Z(i).symbol for i in range(1, 104)]


def load_model_from_checkpoint(ckpt_path, device):

    # Checkpoint
    checkpoint = torch.load(
        ckpt_path, map_location=device, weights_only=False
    )  # Load checkpoint
    state_dict = checkpoint.get("best_model_state", checkpoint.get("best_model"))

    model_args = checkpoint["model_args"]

    # Map renamed keys
    renamed_keys = {
        "cond_size": "condition_size",
        "condition_with_mlp_emb": "condition",
    }
    for old_key, new_key in renamed_keys.items():
        if old_key in model_args:
            model_args["use_old_model_format"] = True
            warn(
                f"'{old_key}' is deprecated and has been renamed to '{new_key}'. "
                "Please update your checkpoint or configuration files.",
                DeprecationWarning,
                stacklevel=2,
            )
            model_args[new_key] = model_args.pop(old_key)

    # Remove unused keys
    removed_keys = [
        "use_lora",
        "lora_rank",
        "condition_with_cl_emb",
        "cl_model_ckpt",
        "freeze_condition_embedding",
    ]
    for removed_key in removed_keys:
        if removed_key in model_args:
            warn(
                f"'{removed_key}' is no longer used and will be ignored. "
                "Consider removing it from your checkpoint or configuration files.",
                DeprecationWarning,
                stacklevel=2,
            )
            model_args.pop(removed_key)

    # Load the model and checkpoint
    model_config = DeciferConfig(**model_args)
    model = Decifer(model_config).to(device)
    model.device = device

    # Fix the keys of the state dict per CrystaLLM
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    # model.load_state_dict(state_dict)  # Load modified state_dict into the model
    model.load_state_dict(state_dict)
    return model


class DeciferPipeline:
    """
    A pipeline to preprocess experimental data and generate CIF structures using a trained model.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        temperature: float = 1.0,
        max_new_tokens: int = 3000,
    ) -> None:
        # Initialize tokenizer and related token variables
        self.TOKENIZER: Tokenizer = Tokenizer()
        self.VOCAB_SIZE: int = self.TOKENIZER.vocab_size
        self.START_ID: int = self.TOKENIZER.token_to_id["data_"]
        self.PADDING_ID: int = self.TOKENIZER.padding_id
        self.NEWLINE_ID: int = self.TOKENIZER.token_to_id["\n"]
        self.SPACEGROUP_ID: int = self.TOKENIZER.token_to_id[
            "_symmetry_space_group_name_H-M"
        ]
        self.DECODE = self.TOKENIZER.decode
        self.ENCODE = self.TOKENIZER.encode
        self.TOKENIZE = self.TOKENIZER.tokenize_cif

        # Model and generation parameters
        self.model: torch.nn.Module = load_model_from_checkpoint(
            model_path, device=device
        )
        self.temperature: float = temperature
        self.max_new_tokens: int = max_new_tokens

    def _get_space_group_symbols(
        self, crystal_system: str, include: bool = True
    ) -> List[str]:
        """
        Returns a list of space group symbols for the given crystal system.

        Parameters:
            crystal_system (str): The target crystal system.
            include (bool): If True, returns symbols matching the crystal system;
                            if False, returns symbols that do NOT match.

        Returns:
            List[str]: A list of space group symbols with '_sg' appended.
        """
        sg_symbols: List[str] = []
        for number in range(1, 231):
            try:
                sg = SpaceGroup.from_int_number(number)
                symbol = sg.symbol
                is_match = (
                    space_group_to_crystal_system(space_group_symbol_to_number(symbol))
                    == crystal_system
                )
                if (include and is_match) or (not include and not is_match):
                    sg_symbols.append(symbol + "_sg")
            except Exception:
                continue
        return sg_symbols

    def _fix_symmetry_in_cif(self, cif_string: str) -> str:
        """
        Fixes the symmetry of a CIF string.

        Parameters:
            cif_string (str): The raw CIF string.

        Returns:
            str: The CIF string with corrected symmetry.
        """
        # Replace the symmetry loop with a P1 structure
        c = replace_symmetry_loop_with_P1(cif_string)
        sg = extract_space_group_symbol(c)
        # Reinstate the symmetry loop if the space group is not P 1
        return reinstate_symmetry_loop(c, sg) if sg != "P 1" else c

    def _preprocess_data(
        self,
        pxrd_data: pd.DataFrame,
        bg_data: Optional[pd.DataFrame] = None,
        wavelength: Optional[Union[float, str]] = None,
        q_min_crop: float = 1.0,
        q_max_crop: float = 8.0,
        n_points: int = 1000,
    ) -> np.ndarray:
        # 1. compute Q
        if wavelength is not None:
            theta_rad = np.radians(pxrd_data["angle"] / 2.0)
            pxrd_data["Q"] = (4.0 * np.pi / float(wavelength)) * np.sin(theta_rad)
        else:
            pxrd_data["Q"] = pxrd_data["angle"]

        # Adjust cropping boundaries based on available Q values
        actual_q_min = pxrd_data["Q"].min()
        actual_q_max = pxrd_data["Q"].max()
        if actual_q_min > q_min_crop:
            q_min_crop = actual_q_min
        if actual_q_max < q_max_crop:
            q_max_crop = actual_q_max

        # 2. Background subtraction with scaling
        if bg_data is not None:
            if wavelength is not None:
                theta_rad_bg = np.radians(bg_data["angle"] / 2.0)
                bg_data["Q"] = (4.0 * np.pi / float(wavelength)) * np.sin(theta_rad_bg)
            else:
                bg_data["Q"] = bg_data["angle"]
            bg_data.sort_values(by="Q", inplace=True)
            pxrd_data["background_intensity"] = np.interp(
                pxrd_data["Q"], bg_data["Q"], bg_data["intensity"]
            )
            valid = pxrd_data["background_intensity"] > 0
            if valid.any():
                s = (
                    pxrd_data.loc[valid, "intensity"]
                    / pxrd_data.loc[valid, "background_intensity"]
                ).min()
            else:
                s = 1.0
            pxrd_data["scaled_background"] = s * pxrd_data["background_intensity"]
            pxrd_data["intensity_bg"] = (
                pxrd_data["intensity"] - pxrd_data["scaled_background"]
            )
        else:
            pxrd_data["intensity_bg"] = pxrd_data["intensity"]

        # 3. Full signal normalization
        max_val_sel = pxrd_data["intensity_bg"].max(skipna=True)
        min_val_sel = pxrd_data["intensity_bg"].min(skipna=True)
        pxrd_data["intensity_normalized"] = (
            pxrd_data["intensity_bg"] - min_val_sel
        ) / (max_val_sel - min_val_sel)

        # 4. Crop
        pxrd_data_crop = pxrd_data[
            (pxrd_data["Q"] > q_min_crop) & (pxrd_data["Q"] < q_max_crop)
        ].copy()

        # Min max norm
        max_val = pxrd_data_crop["intensity_normalized"].max(skipna=True)
        min_val = pxrd_data_crop["intensity_normalized"].min(skipna=True)
        pxrd_data_crop["intensity_normalized"] = (
            pxrd_data_crop["intensity_normalized"] - min_val
        ) / (max_val - min_val)

        # Add endpoints to zero
        pxrd_data_endpoints = pd.DataFrame(
            {"Q": [q_min_crop, q_max_crop], "intensity_normalized": [0, 0]}
        )
        pxrd_data_crop = pd.concat(
            [pxrd_data_crop[["Q", "intensity_normalized"]], pxrd_data_endpoints],
            ignore_index=True,
        )
        pxrd_data_crop.sort_values(by="Q", inplace=True)

        # 6. Standardize signals onto a common Q grid
        Q_std = np.linspace(0, 10, n_points)
        intensity_original = np.interp(Q_std, pxrd_data["Q"], pxrd_data["intensity"])
        intensity_normalized = np.interp(
            Q_std, pxrd_data_crop["Q"], pxrd_data_crop["intensity_normalized"]
        )
        intensity_crop_normalized = np.interp(
            Q_std, pxrd_data_crop["Q"], pxrd_data_crop["intensity_normalized"]
        )

        return intensity_crop_normalized

    def _conditional_generation(
        self,
        cond_array: Union[torch.Tensor, np.ndarray],
        composition: Optional[str] = None,
        composition_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        spacegroup: Optional[str] = None,
        exclusive_elements: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        crystal_systems: Optional[List[str]] = None,
    ) -> Optional[Tuple[str, Structure]]:
        """
        Generates a CIF string from the provided condition vector and returns the CIF and its corresponding structure.

        Parameters:
            cond_array (Union[torch.Tensor, np.ndarray]): Conditioning array for generation.
            composition (Optional[str]): Optional composition string.
            spacegroup (Optional[str]): Optional space group string.
            do_plot (bool): Flag to indicate whether to plot (unused in current implementation).
            exclusive_elements (Optional[List[str]]): List of elements to exclude.
            temperature (Optional[float]): Temperature for generation.
            max_new_tokens (Optional[int]): Maximum tokens to generate.
            crystal_systems (Optional[List[str]]): List of target crystal systems.

        Returns:
            Optional[Tuple[str, Structure]]: A tuple of the fixed CIF string and the corresponding Structure,
                                               or None if generation fails.
        """
        if self.model is None:
            raise ValueError(
                "Model is not loaded. Please load a model using load_custom_model()."
            )
        if temperature is None:
            temperature = self.temperature
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Ensure condition array is a torch.Tensor and adjust dimensions
        if not isinstance(cond_array, torch.Tensor):
            cond_array = torch.tensor(cond_array)
        cond_array = cond_array.unsqueeze(0).to(self.model.device).float()

        # Determine inactive elements if exclusive elements are provided
        if exclusive_elements is not None:
            inactive_elements_list = [
                el for el in global_elements_list if el not in exclusive_elements
            ]
        else:
            inactive_elements_list = None

        # Determine active space groups if crystal systems are provided
        if crystal_systems is not None:
            active_spacegroups: List[str] = []
            for cs in crystal_systems:
                active_spacegroups.extend(
                    self._get_space_group_symbols(cs, include=True)
                )
        else:
            active_spacegroups = None

        # Create prompt tokens for generation
        prompt = torch.tensor([self.START_ID]).unsqueeze(0).to(self.model.device)
        if composition:
            comp_str = f"data_{composition}\n"
            c_tokens = self.ENCODE(self.TOKENIZE(comp_str))
            prompt = torch.tensor(c_tokens).unsqueeze(0).to(self.model.device)

        # Generate new tokens using the model's custom generate function
        out = (
            self.model.generate_custom(
                idx=prompt,
                max_new_tokens=max_new_tokens,
                cond_vec=cond_array,
                start_indices_batch=[[0]],
                composition_string=composition,
                composition_ranges=composition_ranges,
                spacegroup_string=spacegroup,
                exclude_elements=inactive_elements_list,
                temperature=temperature,
                disable_pbar=False,
                include_spacegroups=active_spacegroups,
            )
            .cpu()
            .numpy()
        )
        cif_raw: str = self.DECODE(out[0])

        try:
            # Fix the symmetry in the generated CIF string and convert it to a Structure object
            cif_fixed = self._fix_symmetry_in_cif(cif_raw)
            structure = Structure.from_str(cif_fixed, fmt="cif")
            return cif_fixed, structure
        except Exception as e:
            return None

    def run(
        self,
        pxrd_data: pd.DataFrame,
        bg_data: Optional[pd.DataFrame] = None,
        wavelength: Optional[Union[float, str]] = None,
        q_min_crop: float = 1.0,
        q_max_crop: float = 8.0,
        n_points: int = 1000,
        n_trials: int = 1,
        composition: Optional[str] = None,
        composition_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        spacegroup: Optional[str] = None,
        exclusive_elements: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        crystal_systems: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Runs multiple generation trials and collects all results.

        Parameters:
            pxrd_data (pd.DataFrame): Experimental data.
            bg_data (Optional[pd.DataFrame]): Background data.
            wavelength (Optional[Union[float, str]]): Wavelength.
            q_min_crop (float): Minimum Q value for cropping.
            q_max_crop (float): Maximum Q value for cropping.
            n_points (int): Number of points for standardization.
            n_trials (int): Number of trials to run.
            composition (Optional[str]): Optional composition string.
            spacegroup (Optional[str]): Optional space group string.
            exclusive_elements (Optional[List[str]]): List of elements to exclude.
            temperature (Optional[float]): Temperature for generation.
            max_new_tokens (Optional[int]): Maximum tokens to generate.
            crystal_systems (Optional[List[str]]): List of target crystal systems.

        Returns:
            Dict[str, Any]: A dictionary containing generation results and experimental signals.
        """
        if self.model is None:
            raise ValueError(
                "Model is not loaded. Please load a model using load_custom_model()."
            )
        if temperature is None:
            temperature = self.temperature
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        pxrd_data_final = self._preprocess_data(
            pxrd_data, bg_data, wavelength, q_min_crop, q_max_crop, n_points
        )

        results: Dict[str, Any] = {
            "gens": [],
            "generation_config": {
                "composition": composition,
                "n_trials": n_trials,
                "exclusive_elements": exclusive_elements,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "crystal_systems": crystal_systems,
                "spacegroup": spacegroup,
            },
            "pxrd_data": pxrd_data,
            "pxrd_data_final": pxrd_data_final,
        }

        # Run the generation trials
        pbar_trials = tqdm(
            total=n_trials,
            desc=f"Running trials",
            leave=True,
            dynamic_ncols=True,
        )
        for _ in range(n_trials):
            gen_out = self._conditional_generation(
                cond_array=pxrd_data_final,
                composition=composition,
                composition_ranges=composition_ranges,
                spacegroup=spacegroup,
                exclusive_elements=exclusive_elements,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                crystal_systems=crystal_systems,
            )
            if gen_out is not None:
                cif_str, struct = gen_out

                results["gens"].append(
                    {
                        "cif_str": cif_str,
                        "struct": struct,
                    }
                )
            pbar_trials.update(1)

        return results
