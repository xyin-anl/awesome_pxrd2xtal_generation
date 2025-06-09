# PXRD Inference script for Crystalyze (https://github.com/ML-PXRD/Crystalyze)
# Checkpoint: https://huggingface.co/shawnyin/Crystalyze/tree/main
# 1. Clone the Crystalyze repo, create a conda environment using the crystalyze_env.yml file
# 2. Download the model_folder file, unzip it, and place it in the top level of the Crystalyze repo
# 3. Prepare the experimental PXRD cif file
# 4. Copy the crystalyze_inference.py and crystalyze_utils.py files into the top level of the Crystalyze repo
# 5. Change the project_root variable to the path of the Crystalyze repo
# 6. Run the script
# Inspired by https://github.com/ML-PXRD/Crystalyze/blob/dev_tsach/scripts/evaluate.py
# Curated by: Xiangyu Yin (xiangyu-yin.com)

import os
import time
from pathlib import Path
from types import SimpleNamespace

import hydra
from hydra import compose, initialize_config_dir

import numpy as np
import torch
from torch_geometric.data import DataLoader

from parse_cifs import read_experimental_cif
from crystalyze_utils import InMemoryCrystDataset, worker_init_fn, reconstructon


project_root = "/home/shawn/Documents/projects/Crystalyze"  # TODO: change this to your project root!!!
os.environ["PROJECT_ROOT"] = project_root
data_root_path = f"{project_root}/inference"
model_path = f"{project_root}/model_folder"
evaluate_file_path = f"{project_root}/evaluate.py"
test_cif_path = "./data/experimental_xrd/wn6225Isup2.rtv.combined.cif"
number_of_guesses = 1
force_num_atoms_flag = False
force_atom_types_flag = False
n_step_each = 100
step_lr = 1e-4
min_sigma = 0
save_traj = True
down_sample_traj_step = 10
disable_bar = False

(
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
) = read_experimental_cif(
    filepath=test_cif_path,
)

# target range 5-90
# target increment 0.01
# target vector dim 8500
# Interpolate over the target range with the target increment (trim extra data and pad missing data with 0)
# Normalize by the max intensity, clip negative values
xrd_array = np.interp(np.linspace(5, 90, 8500), two_theta_vals, i_vals)
xrd_array = xrd_array / np.max(xrd_array)
xrd_array[xrd_array < 0] = 0.0

elements_involved = list(atom_types)
xrd = torch.tensor(xrd_array).type(torch.FloatTensor)
unique_key = source_file_name + "_" + pretty_formula + "_" + str(spacegroup_number)

inference_data = {unique_key: (xrd, elements_involved)}

with initialize_config_dir(model_path):
    cfg = compose(config_name="hparams", overrides=[f"data.root_path={data_root_path}"])

    model = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    model.hparams["data"]["num_augmented_data"] = 0
    model.hparams["data"]["datamodule"]["datasets"]["test"][0]["num_augmented_data"] = 0
    model.hparams["data"]["datamodule"]["datasets"]["val"][0]["num_augmented_data"] = 0
    model.hparams["data"]["datamodule"]["datasets"]["train"]["num_augmented_data"] = 0

    ckpts = list(Path(model_path).glob("*.ckpt"))
    if len(ckpts) > 0:
        ckpt_epochs = np.array(
            [int(ckpt.parts[-1].split("-")[0].split("=")[1]) for ckpt in ckpts]
        )
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
    model = model.load_from_checkpoint(
        ckpt, strict=False, data=model.hparams.data, decoder=model.hparams.decoder
    )
    model.lattice_scaler = torch.load(os.path.join(model_path, "lattice_scaler.pt"))
    model.scaler = torch.load(os.path.join(model_path, "prop_scaler.pt"))

    test_dataset = InMemoryCrystDataset(
        inference_data=inference_data,
        prop=model.hparams["data"]["prop"],
        lattice_scale_method=model.hparams["data"]["datamodule"]["datasets"]["test"][0][
            "lattice_scale_method"
        ],
        max_num_atoms=model.hparams["data"]["max_atoms"],
        scaler=model.scaler,
        lattice_scaler=model.lattice_scaler,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )

    ld_kwargs = SimpleNamespace(
        n_step_each=n_step_each,
        step_lr=step_lr,
        min_sigma=min_sigma,
        save_traj=save_traj,
        disable_bar=disable_bar,
    )
    print("LD kwargs loaded")
    if torch.cuda.is_available():
        model.to("cuda")
        print("Model moved to GPU")
    print("Evaluate model on the reconstruction task.")
    start_time = time.time()
    (
        frac_coords,
        num_atoms,
        atom_types,
        lengths,
        angles,
        all_frac_coords_stack,
        all_atom_types_stack,
        input_data_batch,
    ) = reconstructon(
        test_loader,
        model,
        ld_kwargs,
        number_of_guesses,
        force_num_atoms_flag,
        force_atom_types_flag,
        down_sample_traj_step,
    )
