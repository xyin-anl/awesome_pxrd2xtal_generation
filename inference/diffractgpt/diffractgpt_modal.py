import modal
from typing import Tuple, Optional, List, Dict, Any
from jarvis.core.atoms import Atoms as JarvisAtoms


def download_diffractgpt_weights():
    from huggingface_hub import snapshot_download, logging as hf_logging

    hf_logging.set_verbosity_debug()
    model_name = "knc6/diffractgpt_mistral_chemical_formula"
    snapshot_download(repo_id=model_name)


diffractgpt_image = (
    modal.Image.debian_slim(python_version="3.10.17")
    .pip_install(["torch==2.7.0", "torchvision==0.22.0"])
    .pip_install(
        [
            "accelerate==1.7.0",
            "aiohappyeyeballs==2.6.1",
            "aiohttp==3.11.18",
            "aiosignal==1.3.2",
            "annotated-types==0.7.0",
            "ase==3.25.0",
            "async-timeout==5.0.1",
            "attrs==25.3.0",
            "bitsandbytes==0.45.5",
            "black==25.1.0",
            "certifi==2025.4.26",
            "charset-normalizer==3.4.2",
            "click==8.2.1",
            "contourpy==1.3.2",
            "cut-cross-entropy==25.1.1",
            "cycler==0.12.1",
            "datasets==3.6.0",
            "diffusers==0.33.1",
            "dill==0.3.8",
            "docstring-parser==0.16",
            "filelock==3.18.0",
            "fonttools==4.58.0",
            "frozenlist==1.6.0",
            "fsspec==2025.3.0",
            "gguf==0.16.3",
            "hf-transfer==0.1.9",
            "hf-xet==1.1.2",
            "huggingface-hub==0.32.0",
            "idna==3.10",
            "imageio==2.37.0",
            "importlib-metadata==8.7.0",
            "jarvis-tools==2024.10.30",
            "jinja2==3.1.6",
            "joblib==1.5.1",
            "kiwisolver==1.4.8",
            "lazy-loader==0.4",
            "markdown-it-py==3.0.0",
            "markupsafe==3.0.2",
            "matplotlib==3.10.3",
            "mdurl==0.1.2",
            "mpmath==1.3.0",
            "msgspec==0.19.0",
            "multidict==6.4.4",
            "multiprocess==0.70.16",
            "mypy-extensions==1.1.0",
            "networkx==3.4.2",
            "numpy==2.2.6",
            "packaging==25.0",
            "pandas==2.2.3",
            "pathspec==0.12.1",
            "peft==0.15.2",
            "pillow==11.2.1",
            "platformdirs==4.3.8",
            "propcache==0.3.1",
            "protobuf==3.20.3",
            "psutil==7.0.0",
            "pyarrow==20.0.0",
            "pydantic==2.11.5",
            "pydantic-core==2.33.2",
            "pydantic-settings==2.9.1",
            "pygments==2.19.1",
            "pyparsing==3.2.3",
            "python-dateutil==2.9.0.post0",
            "python-dotenv==1.1.0",
            "pytz==2025.2",
            "pyyaml==6.0.2",
            "regex==2024.11.6",
            "requests==2.32.3",
            "rich==14.0.0",
            "safetensors==0.5.3",
            "scikit-image==0.25.2",
            "scikit-learn==1.6.1",
            "scipy==1.15.3",
            "sentencepiece==0.2.0",
            "shtab==1.7.2",
            "six==1.17.0",
            "spglib==2.6.0",
            "sympy==1.14.0",
            "threadpoolctl==3.6.0",
            "tifffile==2025.5.10",
            "tokenizers==0.21.1",
            "tomli==2.2.1",
            "toolz==1.0.0",
            "tqdm==4.67.1",
            "transformers==4.51.3",
            "triton==3.3.0",
            "trl==0.15.2",
            "typeguard==4.4.2",
            "typing-extensions==4.13.2",
            "typing-inspection==0.4.1",
            "typing-inspection==0.4.1",
            "tyro==0.9.21",
            "tzdata==2025.2",
            "urllib3==2.4.0",
            "uv==0.7.8",
            "xformers==0.0.30",
            "xmltodict==0.14.2",
            "xxhash==3.5.0",
            "yarl==1.20.0",
            "zipp==3.21.0",
            "huggingface_hub[hf_transfer]",  # install fast Rust download client
        ]
    )
    .apt_install("git", "git-lfs")
    .run_commands("git clone https://github.com/atomgptlab/atomgpt.git /atomgpt")
    .workdir("/atomgpt")
    .run_commands("pip install .")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_diffractgpt_weights)
)

app = modal.App("diffractgpt-inference")

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


def load_model() -> Tuple[Any, Any]:
    print(f"Loading model from {DEFAULT_CONFIG['model_name']}...")
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )
    from peft import PeftModel, PeftConfig

    LORA_ID = "knc6/diffractgpt_mistral_chemical_formula"
    peft_cfg = PeftConfig.from_pretrained(LORA_ID)
    BASE_ID = peft_cfg.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_ID)

    model = PeftModel.from_pretrained(base_model, LORA_ID)

    model.eval()  # switch to inference mode

    return model, tokenizer


def load_xrd_data(dat_path: str, formula: Optional[str] = None) -> Tuple:
    """Load XRD data from a data file."""
    import numpy as np
    import pandas as pd

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
    import numpy as np
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
    x,
    y,
    x_range: List[float] = DEFAULT_CONFIG["x_range"],
    intvl: float = DEFAULT_CONFIG["intvl"],
    sigma: float = 0.5,
    tol: float = 0.1,
    background_subs: bool = False,
) -> Tuple:
    """Process XRD data."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

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


def text_to_atoms(response: str):
    """Convert model output text to JARVIS atoms structure."""
    import numpy as np
    from jarvis.core.atoms import Atoms as JarvisAtoms
    from jarvis.core.lattice import Lattice

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
    atoms,
    fmax: float = 0.05,
    nsteps: int = 150,
    constant_volume: bool = False,
    device: str = "cpu",
):
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
    import time
    from jarvis.core.atoms import ase_to_atoms

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


@app.function(image=diffractgpt_image, gpu="T4", timeout=1500)
def diffractgpt_pxrd2xtal(
    xrd_x,
    xrd_y,
    formula: str,
    config: Dict[str, Any] = DEFAULT_CONFIG,
    relax: bool = False,
    relax_device: str = "cpu",
):
    """Predict crystal structure from XRD data or property value.

    Args:
        xrd_x: 2-theta values
        xrd_y: Intensity values
        formula: Chemical formula
        config: Configuration dictionary
        relax: Whether to relax the structure with ALIGNN-FF
        relax_device: Device for ALIGNN-FF relaxation

    Returns:
        JARVIS atoms object (relaxed if requested)
    """
    # Load model
    model, tokenizer = load_model()

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


@app.local_entrypoint()
def main():
    import sys
    import os
    import warnings

    warnings.filterwarnings("ignore")

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

    print("pretty_formula", pretty_formula)
    print("atom_Zs", atom_Zs)
    print("spacegroup_number", spacegroup_number)
    print("exp_wavelength", exp_wavelength)

    print("Running DiffractGPT inference test...")
    atoms: JarvisAtoms = diffractgpt_pxrd2xtal.remote(
        two_theta_vals, i_vals, formula=pretty_formula, relax=False, relax_device="cpu"
    )

    print(atoms)
