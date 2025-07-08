# PXRD Inference script for deCIFer (https://github.com/frederiklizakjohansen/decifer)
# Checkpoint: not provided yet, please train your own model or reach out to the authors
# 1. Clone the decifer repo, create a conda environment using the decifer_env.yml file
# 2. Obtain the model checkpoint file place it in the top level of the decifer repo
# 3. Prepare the experimental PXRD cif file (e.g. wn6225Isup2.rtv.combined.cif)
# 4. Copy the decifer_inference.py into the top level of the decifer repo and run it
# 5. The script will output the generated structures and their scores
# Inspired by https://github.com/FrederikLizakJohansen/deCIFer/blob/experimental_data/bin/experimental_pipeline.py
# Curated by: Xiangyu Yin (xiangyu-yin.com)

import pandas as pd
from decifer_utils import DeciferPipeline
from bin.train import TrainConfig  # This is needed for inference, do not remove

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

    pxrd_data = pd.DataFrame(
        {
            "angle": two_theta_vals,
            "intensity": i_vals,
        }
    )

    pipeline = DeciferPipeline(
        model_path="mp_20_model/ckpt.pt",
        device="cuda",
        temperature=1,
        max_new_tokens=3000,
    )

    result = pipeline.run(
        pxrd_data=pxrd_data,
        bg_data=None,
        wavelength=exp_wavelength,
        q_min_crop=0.5,
        q_max_crop=8,
        n_points=1000,
        n_trials=1,
        composition=pretty_formula,  # e.g. "Fe12O18"
        composition_ranges=None,  # e.g. {"Fe": (0, 12), "O": (0, 18)}
        spacegroup=None,  # e.g. R-3c_sg, Ia-3_sg, Pna2_1_sg etc.
        exclusive_elements=None,  # e.g. ['Fe', 'O']
        crystal_systems=None,  # e.g. [1] (triclinic), [2] (monoclinic), [3] (orthorhombic), [4] (tetragonal), [5] (trigonal), [6] (hexagonal), [7] (cubic)
    )

    print(result["gens"][0])
