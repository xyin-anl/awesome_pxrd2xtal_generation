import time
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import torch
from jarvis.core.atoms import Atoms as JarvisAtoms
from jarvis.core.lattice import Lattice
from jarvis.core.atoms import ase_to_atoms


# Default configuration
DEFAULT_CONFIG = {
    "model_name": "knc6/diffractgpt_mistral_chemical_formula",
    "max_seq_length": 2048,
    "dtype": None,
    "load_in_4bit": True,
    "instruction": "Below is a description of a material.",
    "alpaca_prompt": "### Instruction:\n{}\n### Input:\n{}\n### Output:\n{}",
    "output_prompt": " Generate atomic structure description with lattice lengths, angles, coordinates and atom types.",
    "prop": "XRD",
    "intvl": 0.3,
    "x_range": [0, 90],
    "max_new_tokens": 1024,
}


def load_model(
    model_name: str = DEFAULT_CONFIG["model_name"],
    dtype: torch.dtype = DEFAULT_CONFIG["dtype"],
    max_seq_length: int = DEFAULT_CONFIG["max_seq_length"],
    load_in_4bit: bool = DEFAULT_CONFIG["load_in_4bit"],
) -> Tuple[Any, Any]:
    """Load DiffractGPT model and tokenizer.

    Args:
        model_name: HuggingFace model name or local path to model directory
        device: Device to load model on ("cuda" or "cpu")
        dtype: Data type for model weights
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to load in 4-bit quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_name}...")

    from atomgpt.inverse_models.loader import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="auto",
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded with FastLanguageModel")

    return model, tokenizer


def load_xrd_data(
    dat_path: str, formula: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load XRD data from a data file."""
    # Read the data file
    df = pd.read_csv(
        dat_path,
        skiprows=1,
        sep=r"[,\t ]+",
        names=["X", "Y"],
        comment="#",
        engine="python",
    )

    x = df["X"].values
    y = df["Y"].values

    # Normalize intensity
    y = np.array(y, dtype="float")
    y = y / np.max(y)

    # Extract formula from filename if not provided
    if formula is None:
        import os

        formula = os.path.basename(dat_path).split(".")[0]

    return x, y, formula


def baseline_als(y, lam, p, niter=10):
    """ALS baseline correction to remove broad background trends."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def process_xrd(
    x: np.ndarray,
    y: np.ndarray,
    x_range: List[float] = DEFAULT_CONFIG["x_range"],
    intvl: float = DEFAULT_CONFIG["intvl"],
    sigma: float = 0.5,
    tol: float = 0.1,
    background_subs: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process XRD data."""
    y = np.array(y, dtype="float")

    if background_subs:
        background = baseline_als(y, lam=10000, p=0.01)
        y_corrected = y - background
    else:
        y_corrected = y

    # Normalize the corrected spectrum
    y_corrected = y_corrected / np.max(y_corrected)

    # Generate new x-axis values
    x_new = np.arange(x_range[0], x_range[1], intvl)

    # Recast the spectrum onto the new grid
    y_new = np.zeros_like(x_new, dtype=np.float64)

    # Fill the corresponding bins
    for x_val, y_val in zip(x, y_corrected):
        closest_index = np.abs(x_new - x_val).argmin()
        y_new[closest_index] = y_val

    # Apply Gaussian filtering
    y_sharp = gaussian_filter1d(y_new, sigma=sigma, mode="constant")

    # Final normalization
    if np.max(y_sharp) > 0:
        y_sharp = y_sharp / np.max(y_sharp)

    return x_new, y_sharp


def text_to_atoms(response: str) -> JarvisAtoms:
    """Convert model output text to JARVIS atoms structure."""
    tmp_atoms_array = response.strip("</s>").split("\n")
    lat_lengths = np.array(tmp_atoms_array[1].split(), dtype="float")
    lat_angles = np.array(tmp_atoms_array[2].split(), dtype="float")

    lat = Lattice.from_parameters(
        lat_lengths[0],
        lat_lengths[1],
        lat_lengths[2],
        lat_angles[0],
        lat_angles[1],
        lat_angles[2],
    )
    elements = []
    coords = []
    for ii, i in enumerate(tmp_atoms_array):
        if ii > 2 and ii < len(tmp_atoms_array):
            # if ii>2 and ii<len(tmp_atoms_array)-1:
            tmp = i.split()
            if len(tmp) >= 4:  # Add safety check
                elements.append(tmp[0])
                coords.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])

    atoms = JarvisAtoms(
        coords=coords,
        elements=elements,
        lattice_mat=lat.lattice(),
        cartesian=False,
    )
    return atoms


def generate_structure(
    model: Any,
    tokenizer: Any,
    prompt: str,
    alpaca_prompt: str = DEFAULT_CONFIG["alpaca_prompt"],
    instruction: str = DEFAULT_CONFIG["instruction"],
    max_new_tokens: int = DEFAULT_CONFIG["max_new_tokens"],
    device: str = "cuda",
) -> str:
    """Generate atomic structure from prompt using the model."""
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction,
                prompt,
                "",  # output - leave blank for generation
            )
        ],
        return_tensors="pt",
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,  # Deterministic generation
        temperature=1.0,
    )

    response = tokenizer.batch_decode(outputs)[0]
    response = response.split("# Output:")[1].strip("</s>")

    return response


def relax_atoms(
    atoms: JarvisAtoms,
    fmax: float = 0.05,
    nsteps: int = 150,
    constant_volume: bool = False,
    device: str = "cpu",
) -> JarvisAtoms:
    """Relax atoms using ALIGNN-FF calculator.

    Args:
        atoms: JARVIS atoms to relax
        fmax: Force convergence criteria
        nsteps: Maximum number of optimization steps
        constant_volume: Whether to keep volume constant during relaxation
        device: Device for ALIGNN-FF ("cpu" or "cuda")

    Returns:
        Relaxed JARVIS atoms
    """
    try:
        from alignn.ff.ff import AlignnAtomwiseCalculator, default_path
        from ase.optimize import FIRE
        from ase.constraints import ExpCellFilter
    except ImportError:
        print("Warning: ALIGNN-FF not available, skipping relaxation")
        print("Install with: pip install alignn")
        return atoms

    print("Relaxing structure with ALIGNN-FF...")
    t1 = time.time()

    # Initialize calculator
    calculator = AlignnAtomwiseCalculator(path=default_path(), device=device)

    # Convert to ASE atoms
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calculator

    # Apply cell filter for variable cell relaxation
    ase_atoms = ExpCellFilter(ase_atoms, constant_volume=constant_volume)

    # Run optimization
    dyn = FIRE(ase_atoms)
    dyn.run(fmax=fmax, steps=nsteps)

    # Get final energy
    en = ase_atoms.atoms.get_potential_energy()

    # Convert back to JARVIS atoms
    final_atoms = ase_to_atoms(ase_atoms.atoms)

    t2 = time.time()
    print(f"Relaxation completed in {t2-t1:.2f} seconds")
    print(f"Final energy: {en:.4f} eV")

    return final_atoms


def predict_structure(
    model: Any,
    tokenizer: Any,
    formula: str,
    xrd_x: Optional[np.ndarray] = None,
    xrd_y: Optional[np.ndarray] = None,
    config: Dict[str, Any] = DEFAULT_CONFIG,
    relax: bool = False,
    relax_device: str = "cpu",
) -> Optional[JarvisAtoms]:
    """Predict crystal structure from XRD data or property value.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        formula: Chemical formula
        xrd_x: 2-theta values (optional)
        xrd_y: Intensity values (optional)
        prop_val: Property value string (if not using XRD)
        config: Configuration dictionary
        relax: Whether to relax the structure with ALIGNN-FF
        relax_device: Device for ALIGNN-FF relaxation

    Returns:
        JARVIS atoms object (relaxed if requested)
    """
    # Process XRD data
    x_processed, y_processed = process_xrd(
        xrd_x,
        xrd_y,
        x_range=config["x_range"],
        intvl=config["intvl"],
        sigma=0.05,
        background_subs=False,
    )

    # Convert to string format
    y_str = "\n".join(["{0:.2f}".format(x) for x in y_processed])

    # Build the prompt
    prompt = (
        "The chemical formula is "
        + formula
        + " The "
        + config["prop"]
        + " is "
        + y_str
        + ". Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
    )

    # Generate structure
    print(f"Generating structure for {formula}...")
    response = generate_structure(
        model,
        tokenizer,
        prompt,
        alpaca_prompt=config["alpaca_prompt"],
        instruction=config["instruction"],
        max_new_tokens=config["max_new_tokens"],
    )

    # Parse the response
    try:
        jarvis_atoms = text_to_atoms(response)

        if jarvis_atoms and relax:
            # Relax the structure if requested
            jarvis_atoms = relax_atoms(jarvis_atoms, device=relax_device)

        return jarvis_atoms
    except Exception as e:
        print(f"Error parsing structure: {e}")
        print(f"Raw response: {response}")
        return None


if __name__ == "__main__":
    import os

    # Load model
    model, tokenizer = load_model()

    # Check if example file exists
    data_file = "atomgpt/examples/inverse_model_multi/my_data.dat"
    if os.path.exists(data_file):
        # Load XRD data
        xrd_x, xrd_y, formula = load_xrd_data(data_file, formula="LaB6")

        # Predict structure
        jarvis_atoms = predict_structure(
            model,
            tokenizer,
            formula=formula,
            xrd_x=xrd_x,
            xrd_y=xrd_y,
            relax=False,
        )

        print(jarvis_atoms)
