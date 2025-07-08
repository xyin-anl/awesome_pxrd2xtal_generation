# PXRD inference script for PXRDNet (https://github.com/gabeguo/cdvae_xrd)
# Checkpoint: https://huggingface.co/therealgabeguo/cdvae_xrd_sinc10
# 1. Clone the cdvae_xrd repo, create a conda environment using the pxrdnet_env.yml file
# 2. Download the model checkpoint files, and place it in the top level of the cdvae_xrd repo
# 3. Change the project_root environment variable to the path of the cdvae_xrd repo
# 4. Prepare the experimental PXRD cif file (e.g. wn6225Isup2.rtv.combined.cif)
# 5. Copy the pxrdnet_inference.py into the top level of the cdvae_xrd repo and run it
# 6. The script will output the generated structures and their scores

# Curated by: Xiangyu Yin (xiangyu-yin.com)
import os

os.environ["PROJECT_ROOT"] = (
    "/home/shawn/Documents/projects/cdvae_xrd"  # TODO: change to your own path
)


import warnings

warnings.filterwarnings("ignore")
from types import SimpleNamespace
from tqdm import tqdm
from pathlib import Path

import hydra
from hydra import compose
from hydra import initialize_config_dir

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.distributions import MultivariateNormal

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.diffraction.xrd import WAVELENGTHS

from torch_geometric.data import DataLoader
from cdvae.pl_data.dataset import CrystDataset
from cdvae.pl_data.datamodule import worker_init_fn
from cdvae.common.data_utils import build_crystal, build_crystal_graph
from cdvae.common.data_utils import add_scaled_lattice_prop


class CrystDatasetInMemory(CrystDataset):
    def __init__(
        self, *, df: pd.DataFrame, name: str = "in_memory_test_cryst_dataset", **kwargs
    ):
        # Store the dataframe directly
        self._df = df

        # Create a dummy path for the parent class
        self.path = name
        self.name = name

        # Set the dataframe directly instead of reading from pickle
        self.df = df

        # Extract other parameters that the parent class expects
        self.prop = kwargs.get("prop")
        self.niggli = kwargs.get("niggli")
        self.primitive = kwargs.get("primitive")
        self.graph_method = kwargs.get("graph_method")
        self.lattice_scale_method = kwargs.get("lattice_scale_method")
        self.xrd_filter = kwargs.get("xrd_filter")
        self.pdf = kwargs.get("pdf", False)
        self.normalized_pdf = kwargs.get("normalized_pdf", False)
        self.do_not_sinc_gt_xrd = kwargs.get("do_not_sinc_gt_xrd", False)

        # Set XRD filter validation
        assert self.xrd_filter in [
            "gaussian",
            "sinc",
            "both",
        ], "invalid filter requested"

        # Set wavelength and other parameters
        wavesource = kwargs.get("wavesource", "CuKa")
        self.wavelength = WAVELENGTHS[wavesource]
        self.nanomaterial_size = kwargs.get("nanomaterial_size_angstrom", 50)
        self.n_presubsample = kwargs.get("n_presubsample", 4096)
        self.n_postsubsample = kwargs.get("n_postsubsample", 512)

        # Set up sinc filter parameters
        if self.xrd_filter == "sinc" or self.xrd_filter == "both":
            min_2_theta = kwargs.get("min_2_theta", 0)
            max_2_theta = kwargs.get("max_2_theta", 180)

            # compute Q range
            min_theta = min_2_theta / 2
            max_theta = max_2_theta / 2
            Q_min = 4 * np.pi * np.sin(np.radians(min_theta)) / self.wavelength
            Q_max = 4 * np.pi * np.sin(np.radians(max_theta)) / self.wavelength

            # phase shift for sinc filter = half of the signed Q range
            phase_shift = (Q_max - Q_min) / 2

            # compute Qs
            self.Qs = np.linspace(Q_min, Q_max, self.n_presubsample)
            self.Qs_shifted = self.Qs - phase_shift

            self.sinc_filt = self.nanomaterial_size * (
                np.sinc(self.nanomaterial_size * self.Qs_shifted / np.pi) ** 2
            )
        else:
            raise ValueError("Gaussian filter is deprecated. Use sinc filter instead.")

        # Set noise parameters
        self.horizontal_noise_range = kwargs.get(
            "horizontal_noise_range", (1e-2, 1.1e-2)
        )
        self.vertical_noise = kwargs.get("vertical_noise", 1e-3)

        # Process the data directly instead of calling preprocess with a file path
        processed_data = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            crystal_str = row["cif"]
            crystal = build_crystal(
                crystal_str, niggli=self.niggli, primitive=self.primitive
            )
            graph_arrays = build_crystal_graph(crystal, self.graph_method)
            properties = {k: row[k] for k in [self.prop] if k in row.keys()}
            result_dict = {
                "mp_id": row["material_id"],
                "cif": crystal_str,
                "graph_arrays": graph_arrays,
                "spacegroup.number": row["spacegroup.number"],
                "pretty_formula": row["pretty_formula"],
            }
            result_dict.update(properties)
            processed_data.append(result_dict)

        self.cached_data = processed_data

        # Add scaled lattice properties
        add_scaled_lattice_prop(self.cached_data, self.lattice_scale_method)

        # Initialize scalers
        self.lattice_scaler = None
        self.scaler = None

        # smooth XRDs
        for curr_data_dict in tqdm(self.cached_data):
            curr_xrd = curr_data_dict[self.prop]
            curr_xrd = curr_xrd.reshape((self.n_presubsample,))
            curr_data_dict["rawXRD"] = self.sample(
                curr_xrd.numpy()
            )  # need to downsample first
            # have sinc with gaussian filter & sinc w/out gaussian filter
            if self.pdf:
                raise ValueError("not supported")
            else:
                (
                    curr_xrd,
                    sinc_only_xrd,
                    curr_xrd_presubsample,
                    sinc_only_xrd_presubsample,
                ) = self.augment_xrdStrip(
                    curr_xrd,
                    return_both=True,
                    do_not_sinc_gt_xrd=self.do_not_sinc_gt_xrd,
                )
                curr_data_dict[self.prop] = curr_xrd
                curr_data_dict["sincOnly"] = sinc_only_xrd
                curr_data_dict["sincOnlyPresubsample"] = sinc_only_xrd_presubsample
                curr_data_dict["xrdPresubsample"] = curr_xrd_presubsample


def q_i_to_tensor(
    q_vals, i_vals, xrd_vector_dim, desired_wavelength, min_2theta=0, max_2theta=180
):
    lam = WAVELENGTHS[desired_wavelength]
    q_min = 4 * np.pi * np.sin(np.radians(min_2theta / 2)) / lam
    q_max = 4 * np.pi * np.sin(np.radians(max_2theta / 2)) / lam
    xrd_tensor = torch.zeros(xrd_vector_dim)

    for q, i in zip(q_vals, i_vals):
        idx = int((q - q_min) / (q_max - q_min) * xrd_vector_dim)
        if 0 <= idx < xrd_vector_dim:
            xrd_tensor[idx] = max(xrd_tensor[idx], i)

    i_min = max(xrd_tensor.min().item(), 0.0)
    i_max = xrd_tensor.max().item()
    if i_max > i_min:
        xrd_tensor = (xrd_tensor - i_min) / (i_max - i_min)

    return xrd_tensor


def get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms):
    """
    frac_coords: (num_atoms, 3)
    atom_types: (num_atoms)
    lengths: (num_crystals)
    angles: (num_crystals)
    num_atoms: (num_crystals)
    """
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal_array_list.append(
            {
                "frac_coords": cur_frac_coords.detach().cpu().numpy(),
                "atom_types": cur_atom_types.detach().cpu().numpy(),
                "lengths": cur_lengths.detach().cpu().numpy(),
                "angles": cur_angles.detach().cpu().numpy(),
            }
        )
        start_idx = start_idx + num_atom
    return crystal_array_list


def create_materials(
    num_materials,
    frac_coords,
    num_atoms,
    atom_types,
    lengths,
    angles,
):
    # get the crystals
    crystals_list = get_crystals_list(
        frac_coords=frac_coords,
        atom_types=atom_types,
        lengths=lengths,
        angles=angles,
        num_atoms=num_atoms,
    )
    # ret vals
    all_coords = list()
    all_atom_types = list()

    # loop through and process the crystals
    for i in tqdm(range(min(num_materials, len(crystals_list)))):
        curr_crystal = crystals_list[i]
        curr_structure = Structure(
            lattice=Lattice.from_parameters(
                *(curr_crystal["lengths"].tolist() + curr_crystal["angles"].tolist())
            ),
            species=curr_crystal["atom_types"],
            coords=curr_crystal["frac_coords"],
            coords_are_cartesian=False,
        )

        curr_coords = list()
        curr_atom_types = list()

        for site in curr_structure:
            curr_coords.append([site.x, site.y, site.z])
            curr_atom_types.append(Element(site.species_string))

        all_coords.append(np.array(curr_coords))
        all_atom_types.append(curr_atom_types)

    truncated_crystals_list = crystals_list[:num_materials]
    assert len(all_coords) == len(all_atom_types)
    assert len(all_coords) == min(len(num_atoms), num_materials)
    assert len(truncated_crystals_list) == len(all_coords)

    return all_coords, all_atom_types, truncated_crystals_list


def optimize_latent_code(
    model,
    batch,
    target_noisy_xrd,
    num_starting_points,
    lr,
    min_lr,
    l2_penalty,
    num_gradient_steps,
    num_atom_lambda,
    composition_lambda,
    l1_loss,
    z_init=None,
):
    m = MultivariateNormal(
        torch.zeros(model.hparams.hidden_dim).cuda(),
        torch.eye(model.hparams.hidden_dim).cuda(),
    )

    if z_init is None:
        z = torch.randn(
            num_starting_points, model.hparams.hidden_dim, device=model.device
        )
    else:
        z = z_init.detach()
        assert z.shape == (num_starting_points, model.hparams.hidden_dim)

    z.requires_grad = True
    opt = Adam([z], lr=lr)
    total_gradient_steps = num_gradient_steps * (1 + 2 + 4) - 1
    scheduler = CosineAnnealingWarmRestarts(
        opt, num_gradient_steps, T_mult=2, eta_min=min_lr
    )
    model.freeze()
    with tqdm(total=total_gradient_steps, desc="Property opt", unit="steps") as pbar:
        for _ in range(total_gradient_steps):
            opt.zero_grad()
            xrd_loss = (
                F.l1_loss(
                    model.fc_property(z), target_noisy_xrd.broadcast_to(z.shape[0], 512)
                )
                if l1_loss
                else F.mse_loss(
                    model.fc_property(z), target_noisy_xrd.broadcast_to(z.shape[0], 512)
                )
            )
            prob = m.log_prob(z).mean()
            # predict the number of atoms, lattice, composition
            (
                pred_num_atoms,
                _,
                _,
                _,
                pred_composition_per_atom,
            ) = model.decode_stats(
                z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing=False
            )
            num_atom_loss = F.cross_entropy(
                pred_num_atoms, batch.num_atoms.repeat(num_starting_points)
            )
            composition_loss = F.cross_entropy(
                pred_composition_per_atom,
                (batch.atom_types - 1).repeat(num_starting_points),
            )
            pbar.set_postfix_str(
                f"XRD loss: {xrd_loss.item():.3e}; Gaussian log PDF: {prob.item():.3e}; "
                + f"Num atom loss: {num_atom_loss.item():.3e}; Composition loss: {composition_loss.item():.3e}",
                refresh=True,
            )
            # Update the progress bar by one step
            pbar.update(1)
            # calculate total loss: minimize XRD loss, maximize latent code probability (min neg prob)
            total_loss = (
                xrd_loss
                - l2_penalty * prob
                + num_atom_lambda * num_atom_loss
                + composition_lambda * composition_loss
            )

            # backprop through total loss
            total_loss.backward()
            opt.step()
            scheduler.step()
    return z


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

    model_path = os.environ["PROJECT_ROOT"] + "/hydra/singlerun/2025-03-21/mp_20"

    try:
        with initialize_config_dir(str(model_path)):
            cfg = compose(config_name="hparams")
            model = hydra.utils.instantiate(
                cfg.model,
                optim=cfg.optim,
                data=cfg.data,
                logging=cfg.logging,
                _recursive_=False,
            )
            ckpts = list(Path(model_path).glob("*.ckpt"))
            if len(ckpts) > 0:
                ckpt_epochs = np.array(
                    [int(ckpt.parts[-1].split("-")[0].split("=")[1]) for ckpt in ckpts]
                )
                ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
            model = type(model).load_from_checkpoint(
                ckpt,
                strict=False,
                data=model.hparams.data,
                decoder=model.hparams.decoder,
            )
            model.lattice_scaler = torch.load(Path(model_path) / "lattice_scaler.pt")
    except Exception as e:
        print(f"Failed to load model\n {e}")
        exit()

    if torch.cuda.is_available():
        model = model.cuda()

    mask = (two_theta_vals >= 0) & (two_theta_vals <= 180)
    q_vals = q_vals[mask]
    i_vals = i_vals[mask]

    # Ensure q_vals and i_vals have exactly cfg.data.n_presubsample length
    current_length = len(q_vals)
    target_length = cfg.data.n_presubsample

    if current_length > target_length:
        # Subsample if we have more data than needed
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        q_vals = q_vals[indices]
        i_vals = i_vals[indices]
    elif current_length < target_length:
        # Pad if we have less data than needed
        # Repeat the last values to reach target length
        pad_length = target_length - current_length
        q_vals = np.concatenate([q_vals, np.repeat(q_vals[-1], pad_length)])
        i_vals = np.concatenate([i_vals, np.repeat(i_vals[-1], pad_length)])

    # Verify the length is correct
    assert (
        len(q_vals) == cfg.data.n_presubsample
    ), f"Expected {cfg.data.n_presubsample} points, got {len(q_vals)}"
    assert (
        len(i_vals) == cfg.data.n_presubsample
    ), f"Expected {cfg.data.n_presubsample} points, got {len(i_vals)}"

    xrd_tensor = q_i_to_tensor(
        q_vals=q_vals,
        i_vals=i_vals,
        xrd_vector_dim=cfg.data.n_presubsample,
        desired_wavelength=cfg.data.wavesource,
    )

    data_dict = {
        "material_id": source_file_name,
        "pretty_formula": pretty_formula,
        "elements": [Element.from_Z(z).symbol for z in list(set(atom_Zs))],
        "cif": cif_str,
        "spacegroup.number": int(spacegroup_number),
        "xrd": xrd_tensor,
    }
    data_df = pd.DataFrame(
        {
            "material_id": list(),
            "pretty_formula": list(),
            "elements": list(),
            "cif": list(),
            "spacegroup.number": list(),
            "xrd": list(),
        }
    )
    data_df = data_df._append(data_dict, ignore_index=True)

    test_dataset = CrystDatasetInMemory(
        df=data_df,
        prop=cfg.data.prop,
        niggli=cfg.data.niggli,
        primitive=cfg.data.primitive,
        graph_method=cfg.data.graph_method,
        xrd_filter=cfg.data.xrd_filter,
        lattice_scale_method=cfg.data.lattice_scale_method,
        preprocess_workers=1,
        nanomaterial_size_angstrom=cfg.data.nanomaterial_size_angstrom,
        n_presubsample=cfg.data.n_presubsample,
        n_postsubsample=cfg.data.n_postsubsample,
        wavesource=cfg.data.wavesource,
    )

    test_dataset.lattice_scaler = model.lattice_scaler
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )

    num_starting_points = 50  # TODO: change to your desired number of starting points (number of structures to generate)
    lr = 0.1
    min_lr = 1e-4
    l2_penalty = 2e-4
    num_gradient_steps = 5000
    num_atom_lambda = 0.1
    n_step_each = 100
    composition_lambda = 0.1
    l1_loss = True
    step_lr = 1e-4
    min_sigma = 0
    save_traj = False
    disable_bar = False
    z_init = None

    all_structures = []
    for batch in test_loader:
        batch = batch.to(model.device)
        target_xrd = batch.y.reshape(1, cfg.data.n_postsubsample)

        z = optimize_latent_code(
            model=model,
            batch=batch,
            target_noisy_xrd=target_xrd,
            num_starting_points=num_starting_points,
            lr=lr,
            min_lr=min_lr,
            l2_penalty=l2_penalty,
            num_gradient_steps=num_gradient_steps,
            num_atom_lambda=num_atom_lambda,
            composition_lambda=composition_lambda,
            l1_loss=l1_loss,
            z_init=z_init,
        )

        crys_dicts = model.langevin_dynamics(
            z,
            SimpleNamespace(
                n_step_each=n_step_each,
                step_lr=step_lr,
                min_sigma=min_sigma,
                save_traj=save_traj,
                disable_bar=disable_bar,
            ),
            gt_num_atoms=(
                batch.num_atoms.repeat(num_starting_points)
                if num_atom_lambda > 1e-9
                else None
            ),
            gt_atom_types=(
                batch.atom_types.repeat(num_starting_points)
                if composition_lambda > 1e-9
                else None
            ),
        )

        _, _, crystal_list = create_materials(
            num_starting_points,
            crys_dicts["frac_coords"],
            crys_dicts["num_atoms"],
            crys_dicts["atom_types"],
            crys_dicts["lengths"],
            crys_dicts["angles"],
        )

        structures = []
        for crys in crystal_list:
            structure = Structure(
                lattice=Lattice.from_parameters(
                    *(crys["lengths"].tolist() + crys["angles"].tolist())
                ),
                species=crys["atom_types"],
                coords=crys["frac_coords"],
                coords_are_cartesian=False,
            )
            structures.append(structure)
        all_structures.append(structures)

    print(all_structures)
