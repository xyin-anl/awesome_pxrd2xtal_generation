# Reproduces Uni-3DAR's MP-20 PXRD-guided CSP result (paper Table 3: 75.08% top-1 match rate)
# with this wrapper's model loading and generation loop.
# Needs upstream's mp20_data/test.lmdb (from mp20_data.tar.gz on the Hugging Face repo) and, for
# upstream's match_rate_at_k, `pip install smact` in the environment.
# Usage: python reproduce_upstream.py /path/to/mp20_data/test.lmdb [--records 40] [--batch-size 64]

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import os
import pickle
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run as wrapper  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("test_lmdb")
    p.add_argument("--records", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0, help="Record selection seed")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    import lmdb
    from ase import Atoms

    upstream = os.path.join(HERE, ".upstream")
    sys.path.insert(0, upstream)
    model = wrapper.load_model(os.path.join(HERE, ".weights", "mp20_pxrd.pt"), upstream, args.batch_size, 2)
    from uni3dar.data.crystal_data_utils import match_rate_at_k  # after load_model registers the module

    env = lmdb.open(args.test_lmdb, subdir=os.path.isdir(args.test_lmdb), readonly=True, lock=False)
    with env.begin() as txn:
        records = [pickle.loads(gzip.decompress(v)) for _, v in txn.cursor()]
    chosen = np.random.default_rng(args.seed).choice(len(records), args.records, replace=False)

    top1 = top20 = 0
    for i in chosen:
        rec = records[i]
        gt = Atoms(symbols=rec["atom_type"], cell=rec["lattice_matrix"],
                   scaled_positions=np.array(rec["atom_pos"]).reshape(-1, 3), pbc=True)
        target = np.array(sorted(gt.get_atomic_numbers())) - 1
        pool, tries = [], 0
        with contextlib.redirect_stdout(io.StringIO()):
            # upstream inference_crystal_cond with total_n = 1: collect 20 exact-composition samples
            while len(pool) < 20:
                res, score = model.generate(data=rec, atom_constraint=target)
                for atoms, s in zip(res, score):
                    if np.array_equal(np.sort(atoms.get_atomic_numbers()) - 1, target):
                        pool.append((s, atoms))
                    if len(pool) >= 20:
                        break
                tries += 1
                if tries > wrapper.MAX_TRY or (len(pool) / (len(res) + 1e-5) <= 0.1 and tries > 2):
                    break
        preds = [a for _, a in sorted(pool, key=lambda item: item[0])]
        top1 += match_rate_at_k(gt, preds[:1], 1)[0]
        top20 += match_rate_at_k(gt, preds[:20], 20)[0]
    print(f"records={args.records} batch_size={args.batch_size} top1={top1 / args.records:.3f} top20={top20 / args.records:.3f}")


if __name__ == "__main__":
    main()
