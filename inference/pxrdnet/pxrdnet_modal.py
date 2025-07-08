import modal
from pymatgen.core import Structure


def cache_model_weights():
    """Download and cache the PXRDNet model weights from Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    # Download the entire model repository
    model_path = snapshot_download(
        repo_id="therealgabeguo/cdvae_xrd_sinc10",
        local_dir="/pxrdnet",
        repo_type="model",
    )

    print(f"Model weights downloaded to: {model_path}")
    return model_path


pxrdnet_image = (
    modal.Image.debian_slim(python_version="3.9.18")
    .pip_install("torch==2.0.0")
    .pip_install(
        ["torch-scatter==2.1.2", "torch-sparse==0.6.18"],
        find_links="https://data.pyg.org/whl/torch-2.0.0%2Bcu117.html",
    )
    .pip_install(
        [
            "hydra-core==1.3.2",
            "matminer==0.9.2",
            "matplotlib==3.8.4",
            "networkx==3.2.1",
            "numpy==1.26.4",
            "omegaconf==2.3.0",
            "p_tqdm==1.4.0",
            "pandas==1.5.3",
            "Pillow==10.3.0",
            "plotly==5.20.0",
            "pymatgen==2023.3.10",
            "pytest==8.2.0",
            "python-dotenv==1.0.1",
            "pytorch_lightning==2.2.0.post0",
            "scikit_learn==1.4.2",
            "scipy==1.13.0",
            "setuptools==68.2.2",
            "SMACT==2.5.5",
            "sympy==1.12",
            "torch_geometric==2.5.3",
            "tqdm==4.66.2",
            "wandb==0.16.3",
            "kaleido==0.2.1",
            "huggingface_hub[hf_transfer]",  # install fast Rust download client
        ]
    )
    .apt_install("git", "git-lfs")
    .run_commands("git clone https://github.com/gabeguo/cdvae_xrd.git /pxrdnet")
    .workdir("/pxrdnet")
    .run_commands("pip install .")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # and enable it
    .run_function(cache_model_weights)
    .env({"PROJECT_ROOT": "/pxrdnet"})
)

app = modal.App("pxrdnet-inference")


def q_i_to_tensor(
    q_vals, i_vals, xrd_vector_dim, desired_wavelength, min_2theta=0, max_2theta=180
):
    import numpy as np
    import torch
    from pymatgen.analysis.diffraction.xrd import WAVELENGTHS

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
    from tqdm import tqdm
    import numpy as np
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.structure import Structure
    from pymatgen.core.periodic_table import Element

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
    from tqdm import tqdm
    import torch
    from torch.distributions import MultivariateNormal
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    import torch.nn.functional as F

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
    for _ in tqdm(range(total_gradient_steps)):
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


@app.function(image=pxrdnet_image, gpu="T4", timeout=1500)
def pxrdnet_pxrd2xtal(
    i_vals,
    q_vals,
    two_theta_vals,
    atom_Zs,
    source_file_name,
    cif_str,
    pretty_formula,
    spacegroup_number,
    num_samples=3,
    model_path="/pxrdnet/mp_20_sinc10",
):
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

    from pymatgen.core.structure import Structure
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.periodic_table import Element
    from pymatgen.analysis.diffraction.xrd import WAVELENGTHS

    from torch_geometric.data import DataLoader
    from cdvae.pl_data.dataset import CrystDataset
    from cdvae.pl_data.datamodule import worker_init_fn
    from cdvae.common.data_utils import build_crystal, build_crystal_graph
    from cdvae.common.data_utils import add_scaled_lattice_prop

    class CrystDatasetInMemory(CrystDataset):
        def __init__(
            self,
            *,
            df: pd.DataFrame,
            name: str = "in_memory_test_cryst_dataset",
            **kwargs,
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
                raise ValueError(
                    "Gaussian filter is deprecated. Use sinc filter instead."
                )

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
            print(f"Loaded model from {ckpt}")
    except Exception as e:
        print(f"Failed to load model\n {e}")
        exit()

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

    num_starting_points = num_samples
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

    return all_structures


@app.local_entrypoint()
def main():
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

    structures = pxrdnet_pxrd2xtal.remote(
        i_vals,
        q_vals,
        two_theta_vals,
        atom_Zs,
        source_file_name,
        cif_str,
        pretty_formula,
        spacegroup_number,
        num_samples=3,
    )
