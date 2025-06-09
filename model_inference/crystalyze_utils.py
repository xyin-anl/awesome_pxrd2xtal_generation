# PXRD Inference utils for Crystalyze (https://github.com/ML-PXRD/Crystalyze)
# Curated by: Xiangyu Yin (xiangyu-yin.com)

import random
import ast
import numpy as np
import pandas as pd

from pymatgen.core import Structure, Lattice, Element
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def create_inference_dataframe(inference_data):
    """
    Create a dataframe for inference data. Note that all columns except for material_id and atomic numbers are filled with dummy data because they are either depreciated features
    or contain information that would be be accessible in the inference process.

    Args:
        inference_data (dict): A dictionary containing inference data.

    Returns:
        pandas.DataFrame: A dataframe with columns 'cif', 'filename', 'material_id', 'atomic_nums',
        'formation_energy_per_atom', 'spacegroup.number', 'xrd_peak_intensities', 'xrd_peak_locations',
        and 'disc_sim_xrd'.

    """

    # initialize an empty dataframe with 'cif' and 'filename' columns
    df = pd.DataFrame(columns=["cif", "filename", "material_id", "atomic_nums"])

    # set the 'material_id' column to the keys of the inference_data dictionary and the 'atomic_numbers' column to the values of the inference_data dictionary
    df["material_id"] = [id for id in inference_data.keys()]
    df["atomic_numbers"] = [data[1] for data in inference_data.values()]

    # the following are data values that were used in R&D but are not relevant for inference (at this point in time)

    # add the 'filename' column and set it to None
    df["filename"] = None

    # add the formation_energy_per_atom and spacegroup.number columns and just set them to 0
    df["formation_energy_per_atom"] = 0
    df["spacegroup.number"] = 0

    # add the cifs to the 'cif' column
    df["cif"] = None

    # make a 'xrd_peak_intensities' and 'xrd_peak_locations' columns where each entry is 256 * [0]
    df["xrd_peak_intensities"] = [256 * [0] for _ in range(len(df))]
    df["xrd_peak_locations"] = [256 * [0] for _ in range(len(df))]
    df["disc_sim_xrd"] = [np.array(256 * [0]) for _ in range(len(df))]

    return df


def create_inference_xrd_data(inference_data):
    """
    Create a dictionary of XRD data to save from the given inference data.

    Parameters:
    inference_data (dict): A dictionary containing inference data, where the keys are IDs and the values are data.

    Returns:
    dict: A dictionary containing XRD data to save, where the keys are IDs and the values are the first element of the data.
    """
    xrd_data_to_save = {id: data[0] for id, data in inference_data.items()}
    return xrd_data_to_save


def create_inference_graph_data(inference_data):
    """
    Create dummy graph data for inference. This graph data does not contain any information about the coordinates or lattice parameters of the graph, it just initializes a
    random unit cell with atoms corresponding to the atom types specified. The types and number of atoms can be used in inference or not used in inference depending on
    the flags input at inference.

    Args:
        inference_data (dict): A dictionary containing inference data.

    Returns:
        dict: A dictionary containing graph data for each ID in the inference data.
    """

    graph_data_dict = {}

    for id, data in inference_data.items():
        _, elements_involved = data

        structure = generate_random_structure(elements_involved)

        pyg_graph_data = generate_pyg_graph(structure)

        graph_data_dict[id] = pyg_graph_data

    return graph_data_dict


def generate_random_structure(atomic_numbers):
    """
    Generates a random crystal structure with the given atomic numbers.

    Parameters:
    atomic_numbers (list): A list of atomic numbers representing the elements in the structure.

    Returns:
    Structure: A random crystal structure object.

    """
    # Generate random lattice parameters
    a = random.uniform(3.0, 10.0)
    b = random.uniform(3.0, 10.0)
    c = random.uniform(3.0, 10.0)
    alpha = random.uniform(70, 110)
    beta = random.uniform(70, 110)
    gamma = random.uniform(70, 110)

    # Create random lattice
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

    # Generate random fractional coordinates for each atom
    coordinates = []
    species = []
    for atomic_number in atomic_numbers:
        species.append(Element.from_Z(atomic_number))
        coordinates.append([random.random(), random.random(), random.random()])

    # Create structure
    structure = Structure(lattice, species, coordinates)

    return structure


def generate_pyg_graph(crystal, graph_method="crystalnn"):
    """
    Generate a PyTorch Geometric graph representation from a crystal structure.

    Args:
        crystal (pymatgen.core.structure.Structure): The crystal structure.
        graph_method (str, optional): The method to generate the graph. Defaults to 'crystalnn'.

    Returns:
        tuple: A tuple containing the following elements:
            - frac_coords (numpy.ndarray): The fractional coordinates of the atoms.
            - atom_types (numpy.ndarray): The atomic numbers of the atoms.
            - lengths (numpy.ndarray): The lengths of the lattice parameters.
            - angles (numpy.ndarray): The angles of the lattice parameters.
            - edge_indices (numpy.ndarray): The indices of the edges in the graph.
            - to_jimages (numpy.ndarray): The translation vectors for periodic images.
            - num_atoms (int): The number of atoms in the crystal structure.
    """

    CrystalNN = local_env.CrystalNN(
        distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False
    )

    if graph_method == "crystalnn":
        crystal_graph = StructureGraph.with_local_env_strategy(crystal, CrystalNN)
    elif graph_method == "none":
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(
        crystal.lattice.matrix, lattice_params_to_matrix(*lengths, *angles)
    )

    edge_indices, to_jimages = [], []
    if graph_method != "none":
        for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms


class InMemoryCrystDataset(Dataset):
    """
    A drop-in replacement for CrystDataset that constructs the necessary
    data tuples from in-memory inference_df, inference_xrd, and inference_graph.
    """

    def __init__(
        self,
        inference_data,
        prop="NA",
        lattice_scale_method="scale_length",
        max_num_atoms=20,
        scaler=None,
        lattice_scaler=None,
    ):
        super().__init__()
        self.prop = prop
        self.lattice_scale_method = lattice_scale_method
        self.max_num_atoms = max_num_atoms
        self.scaler = scaler
        self.lattice_scaler = lattice_scaler

        self.df = create_inference_dataframe(inference_data)
        self.inference_xrd = create_inference_xrd_data(inference_data)
        self.inference_graph = create_inference_graph_data(inference_data)

        # ---------------------------------------------------------------------
        # Convert necessary columns to Python lists just as in 'preprocess()'.
        # Ensure they are all length-256 etc., fill with zeros if needed, etc.
        # Also do the random shuffling of atomic_numbers for parity with preprocess.
        # ---------------------------------------------------------------------
        # random sample the ordering (the original CrystDataset / preprocess does this)
        if "atomic_numbers" in self.df.columns:
            self.df["atomic_numbers"] = self.df["atomic_numbers"].apply(
                lambda x: x if isinstance(x, list) else ast.literal_eval(str(x))
            )
            self.df["atomic_numbers"] = self.df["atomic_numbers"].apply(
                lambda x: random.sample(x, len(x))
            )
        else:
            self.df["atomic_numbers"] = [[] for _ in range(len(self.df))]

        if "xrd_peak_intensities" in self.df.columns:
            self.df["xrd_peak_intensities"] = self.df["xrd_peak_intensities"].apply(
                lambda x: x if isinstance(x, list) else ast.literal_eval(str(x))
            )
        else:
            self.df["xrd_peak_intensities"] = [[0] * 256 for _ in range(len(self.df))]

        if "xrd_peak_locations" in self.df.columns:
            self.df["xrd_peak_locations"] = self.df["xrd_peak_locations"].apply(
                lambda x: x if isinstance(x, list) else ast.literal_eval(str(x))
            )
        else:
            self.df["xrd_peak_locations"] = [[0] * 256 for _ in range(len(self.df))]

        # disc_sim_xrd must be numeric array (size=256)
        if "disc_sim_xrd" in self.df.columns:
            self.df["disc_sim_xrd"] = self.df["disc_sim_xrd"].apply(
                lambda x: (
                    list(x)
                    if isinstance(x, (list, np.ndarray))
                    else [
                        float(val)
                        for val in str(x).replace("[", " ").replace("]", " ").split()
                        if val
                    ]
                )
            )
        else:
            self.df["disc_sim_xrd"] = [[0] * 256 for _ in range(len(self.df))]

        # Create multi-hot
        self.df["atomic_numbers_multi_hot_encoding"] = self.df["atomic_numbers"].apply(
            self._multi_hot_encode
        )

        # ---------------------------------------------------------------------
        # Build up a "cached_data" list of dicts, each containing exactly
        # what CrystDataset/preprocess() would have built for __getitem__.
        # ---------------------------------------------------------------------
        self.cached_data = []
        for i in range(len(self.df)):
            mat_id = self.df["material_id"].iloc[i]

            # If you only want entries with num_atoms <= max_num_atoms, skip otherwise
            atoms_list = self.df["atomic_numbers"].iloc[i]
            if len(atoms_list) > self.max_num_atoms:
                print(
                    f"Skipping {mat_id} because it has {len(atoms_list)} atoms. The max is {self.max_num_atoms}"
                )
                continue

            row_dict = {}
            row_dict["material_id"] = mat_id

            row_dict[self.prop] = (
                self.df[self.prop].iloc[i] if self.prop in self.df.columns else 0.0
            )

            row_dict["atomic_species"] = torch.tensor(
                self.df["atomic_numbers"].iloc[i], dtype=torch.long
            )
            row_dict["xrd_intensities"] = torch.tensor(
                self._pad_or_trim(self.df["xrd_peak_intensities"].iloc[i], 256),
                dtype=torch.float,
            )
            row_dict["xrd_locations"] = torch.tensor(
                self._pad_or_trim(self.df["xrd_peak_locations"].iloc[i], 256),
                dtype=torch.float,
            )
            row_dict["disc_sim_xrd"] = torch.tensor(
                self._pad_or_trim(self.df["disc_sim_xrd"].iloc[i], 256),
                dtype=torch.float,
            )
            row_dict["multi_hot_encoding"] = torch.tensor(
                self.df["atomic_numbers_multi_hot_encoding"].iloc[i], dtype=torch.long
            )

            # The "graph_arrays" is a tuple: (frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms)
            row_dict["graph_arrays"] = self.inference_graph[mat_id]

            # The "pv_xrd" is the full 1D XRD array (e.g., shape = (8500,))
            row_dict["pv_xrd"] = self.inference_xrd[mat_id]

            self.cached_data.append(row_dict)

        # Optionally apply the same "add_scaled_lattice_prop" logic for each item,
        # if your training pipeline expects it.
        if lattice_scale_method in ["scale_length"]:
            self._apply_lattice_scaling()

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        """
        Returns exactly what CrystDataset returns:
           (Data, xrd_intensities, xrd_locations, atomic_species, disc_sim_xrd, pv_xrd, multi_hot_encoding)
        """
        data_dict = self.cached_data[idx]

        # Scale the property if the model expects it
        # (In many inference scenarios, this might just be 0 or not used).
        prop = torch.tensor([data_dict[self.prop]], dtype=torch.float)
        if self.scaler is not None:
            prop = self.scaler.transform(prop)

        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
        ) = data_dict["graph_arrays"]
        xrd_intensities = data_dict["xrd_intensities"]
        xrd_locations = data_dict["xrd_locations"]
        atomic_species = data_dict["atomic_species"]
        disc_sim_xrd = data_dict["disc_sim_xrd"]
        multi_hot = data_dict["multi_hot_encoding"]
        pv_xrd = data_dict["pv_xrd"]

        # 0 out the first 1000 columns of pv_xrd. This makes the actual 2theta range 15-90
        if len(pv_xrd.shape) == 1:
            pv_xrd = pv_xrd.unsqueeze(0)
        pv_xrd[:, :1000] = 0

        # Build the PyG Data object
        data = Data(
            frac_coords=torch.tensor(frac_coords, dtype=torch.float),
            atom_types=torch.tensor(atom_types, dtype=torch.long),
            lengths=torch.tensor(lengths, dtype=torch.float).view(1, -1),
            angles=torch.tensor(angles, dtype=torch.float).view(1, -1),
            edge_index=torch.tensor(edge_indices.T, dtype=torch.long),
            to_jimages=torch.tensor(to_jimages, dtype=torch.long),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
            y=prop.view(1, -1),
        )

        return (
            data,
            xrd_intensities,
            xrd_locations,
            atomic_species,
            disc_sim_xrd,
            pv_xrd,
            multi_hot,
        )

    @staticmethod
    def _multi_hot_encode(atomic_numbers):
        """
        Exactly as in data_utils.py,
        produce a 100-dim multi-hot vector with counts of each Z in [1..100].
        """
        vec = np.zeros(100, dtype=int)
        for num in atomic_numbers:
            if 1 <= num <= 100:
                vec[num - 1] += 1
        return vec

    @staticmethod
    def _pad_or_trim(arr, target_len):
        """
        Utility to ensure the array is length `target_len`.
        If arr is shorter, pad with zeros. If longer, trim.
        """
        arr = list(arr)
        if len(arr) < target_len:
            arr = arr + [0] * (target_len - len(arr))
        elif len(arr) > target_len:
            arr = arr[:target_len]
        return arr

    def _apply_lattice_scaling(self):
        """
        If the training code used "lattice_scale_method='scale_length'", replicate it:
        scale the lengths by (1 / num_atoms^(1/3)).
        """
        for dd in self.cached_data:
            (
                frac_coords,
                atom_types,
                lengths,
                angles,
                edge_indices,
                to_jimages,
                num_atoms,
            ) = dd["graph_arrays"]
            if self.lattice_scale_method == "scale_length":
                lengths = lengths / float(num_atoms) ** (1 / 3)
            # Re-save into the tuple
            dd["graph_arrays"] = (
                frac_coords,
                atom_types,
                lengths,
                angles,
                edge_indices,
                to_jimages,
                num_atoms,
            )


def reconstructon(
    loader,
    model,
    ld_kwargs,
    num_evals,
    force_num_atoms=False,
    force_atom_types=False,
    down_sample_traj_step=1,
):
    """
    reconstruct the crystals in <loader>.
    """
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    input_data_list = []

    for idx, batch in enumerate(loader):
        xrd_int = batch[1]
        xrd_loc = batch[2]
        atom_spec = batch[3]
        disc_sim_xrd = batch[4]
        pv_xrd = batch[5]
        multi_hot_encoding = batch[6]
        batch = batch[0]

        if torch.cuda.is_available():
            xrd_int = xrd_int.cuda()
            xrd_loc = xrd_loc.cuda()
            atom_spec = atom_spec.cuda()
            disc_sim_xrd = disc_sim_xrd.cuda()
            batch = batch.cuda()
            pv_xrd = pv_xrd.cuda()
            multi_hot_encoding = multi_hot_encoding.cuda()

        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        for _ in range(num_evals):
            _, _, z = model.encode(
                None,
                xrd_int,
                xrd_loc,
                atom_spec,
                disc_sim_xrd,
                testing=True,
                pv_xrd=pv_xrd,
                multi_hot_encode=multi_hot_encoding,
            )

            gt_num_atoms = batch.num_atoms if force_num_atoms else None
            gt_atom_types = batch.atom_types if force_atom_types else None
            if gt_num_atoms is not None:
                print("using gt_num_atoms")

            if gt_atom_types is not None:
                print("using gt_atom_types")

            outputs = model.langevin_dynamics(
                z, ld_kwargs, gt_num_atoms, gt_atom_types, atom_spec
            )

            batch_frac_coords.append(outputs["frac_coords"].detach().cpu())
            batch_num_atoms.append(outputs["num_atoms"].detach().cpu())
            batch_atom_types.append(outputs["atom_types"].detach().cpu())
            batch_lengths.append(outputs["lengths"].detach().cpu())
            batch_angles.append(outputs["angles"].detach().cpu())

            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    outputs["all_frac_coords"][::down_sample_traj_step].detach().cpu()
                )
                batch_all_atom_types.append(
                    outputs["all_atom_types"][::down_sample_traj_step].detach().cpu()
                )

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))

        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(torch.stack(batch_all_atom_types, dim=0))

        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)

    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords,
        num_atoms,
        atom_types,
        lengths,
        angles,
        all_frac_coords_stack,
        all_atom_types_stack,
        input_data_batch,
    )
