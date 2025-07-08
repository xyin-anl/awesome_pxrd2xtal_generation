# PXRD Inference script for Uni‑3DAR (https://github.com/dptech-corp/Uni-3DAR)
# Checkpoint: https://huggingface.co/dptech/Uni-3DAR/tree/main
# 1. Clone the Uni-3DAR repo, create a conda environment using the uni3dar_env.yml file
# 2. Obtain the model checkpoint file place it in the top level of the Uni-3DAR repo
# 3. Prepare the experimental PXRD cif file (e.g. wn6225Isup2.rtv.combined.cif)
# 4. Copy the uni3dar_inference.py into the top level of the Uni-3DAR repo and run it
# 5. The script will output the generated structures and their scores
# Inspired by https://github.com/dptech-corp/Uni-3DAR/blob/main/uni3dar/inference.py
# Curated by: Xiangyu Yin (xiangyu-yin.com)

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from unicore import tasks, utils, options
from pymatgen.core.periodic_table import Element


DEFAULT_CFG: Dict[str, Any] = {
    # high‑level definition
    "user_dir": "./uni3dar",
    "task": "uni3dar",
    "task_name": "uni3dar_pxrd",
    "loss": "ar",
    "arch": "uni3dar_sampler",
    # model hyper‑parameters
    "layer": 24,
    "emb_dim": 1024,
    "num_head": 16,
    "merge_level": 8,
    "head_dropout": 0.1,
    # dataloader & batching
    "batch_size": 256,
    "batch_size_valid": 512,
    "num_workers": 8,
    "data_buffer_size": 32,
    "fixed_validation_seed": 11,
    # sampling temperatures / ranking
    "tree_temperature": 0.15,
    "atom_temperature": 0.3,
    "xyz_temperature": 0.3,
    "count_temperature": 1.0,
    "rank_ratio": 0.8,
    "rank_by": "atom+xyz",
    # crystal pxrd specific toggles
    "data_type": "crystal",
    "grid_len": 0.24,
    "xyz_resolution": 0.01,
    "recycle": 1,
    "atom_type_key": "atom_type",
    "atom_pos_key": "atom_pos",
    "lattice_matrix_key": "lattice_matrix",
    "allow_atoms": "all",
    "crystal_pxrd": 4,
    "crystal_pxrd_step": 0.1,
    "crystal_pxrd_noise": 0.1,
    "crystal_component": 1,
    "crystal_component_sqrt": True,
    "crystal_component_noise": 0.1,
    "crystal_pxrd_threshold": 5,
    "max_num_atom": 128,
    # misc
    "seed": 42,
    "bf16": True,
    "gzip": True,
    "ddp_backend": "c10d",
}


def _make_arg_list(cfg: Dict[str, Any]) -> List[str]:
    """Convert a python dict into the argv‑style list expected by unicore."""
    argv: List[str] = ["dummy_data_placeholder"]
    for k, v in cfg.items():
        if k == "data":
            continue
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, (list, tuple)):
            argv.append(flag)
            argv.extend(map(str, v))
        else:
            argv.extend([flag, str(v)])
    return argv


def load_model(
    checkpoint: str,
    *,
    device: Union[str, torch.device] = "cuda",
    overrides: Dict[str, Any] | None = None,
):
    cfg = {**DEFAULT_CFG, **(overrides or {}), "finetune_from_model": checkpoint}
    arg_list = _make_arg_list(cfg)

    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, input_args=arg_list)

    utils.import_user_module(args)
    utils.set_jit_fusion_options()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    task = tasks.setup_task(args)
    model = task.build_model(args)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["ema"]["params"], strict=False)
    if missing or unexpected:
        warnings.warn(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")

    model.eval()
    if str(device) == "cuda" and torch.cuda.is_available():
        model = model.cuda().bfloat16()
    else:
        model = model.float()

    return model


def sample_from_model(
    model,
    cur_data: Dict[str, Any],
    *,
    num_samples: int = 1,
    atom_constraint: np.ndarray | None = None,
) -> Tuple[List, List[float]]:
    """
    Parameters
    ----------
    model : torch.nn.Module
        The sampler returned by `load_model`.
    cur_data : dict
        A single datapoint formatted exactly as Uni‑3DAR expects
        (same fields that used to be read from LMDB).
    num_samples : int, default 1
        How many structures to return.
    atom_constraint : np.ndarray, optional
        List of Z numbers - 1. e.g., [25,25,25,12,12,7,7,7,7,7] for Fe3Al2O5
        Leave `None` for unconstrained sampling.

    Returns
    -------
    crystals : list[ase.Atoms]
        List of generated structures (length == `num_samples`).
    scores   : list[float]
        The internal model score for each returned structure.
    """
    if atom_constraint is not None:
        assert atom_constraint.ndim == 1, "Provide a flat 1D array."

    crystals, scores = [], []
    while len(crystals) < num_samples:
        c, s = model.generate(data=cur_data, atom_constraint=atom_constraint)
        crystals.extend(c)
        scores.extend(s)
    crystals, scores = crystals[:num_samples], scores[:num_samples]

    return crystals, scores


if __name__ == "__main__":
    import sys
    import os

    # Add the root directory to the Python path
    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path.insert(0, root_dir)

    from utils.parse_cifs import read_experimental_cif

    test_cif_path = "../../exp_pxrd_data/pxrdnet/wn6225Isup2.rtv.combined.cif"
    (
        source_file_name,
        cif_str,
        pretty_formula,
        frac_coords,
        atom_Zs,
        spacegroup_number,
        two_theta_vals,
        q_vals,
        i_vals,
        exp_wavelength,
        exp_2theta_min,
        exp_2theta_max,
    ) = read_experimental_cif(
        filepath=test_cif_path,
    )

    # Normalize intensity array between 0 and 100
    intensity_array = 100 * i_vals / np.max(i_vals)
    intensity_array[intensity_array < 0] = 0.0

    # Filter data to keep only values between 15 and 120 degrees
    mask = (two_theta_vals >= 15) & (two_theta_vals <= 120)
    two_theta_vals = two_theta_vals[mask]
    intensity_array = intensity_array[mask]
    atom_type = list(set([Element.from_Z(z).symbol for z in atom_Zs]))
    inference_data = {
        "pxrd_x": two_theta_vals,
        "pxrd_y": intensity_array,
        "atom_type": atom_type,
    }
    atom_constraint = None  # np.array(atom_Zs) - 1

    model = load_model("mp20_pxrd.pt", device="cuda")
    crystals, scores = sample_from_model(
        model, inference_data, atom_constraint=atom_constraint, num_samples=3
    )
    print(crystals)
    print(scores)
