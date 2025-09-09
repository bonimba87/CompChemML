#!/usr/bin/env python3
import argparse, pandas as pd
from pathlib import Path
import numpy as np

def read_smi(path, id_1, id_2):
    # DUD-E format: SMILES, number, CHEMBL_ID
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df[[id_1, id_2]]         # keep SMILES and ID
    df.columns = ["smiles","id"]
    return df

def main(actives_smi, decoys_smi, out_csv, n=20, seed=42):
    np.random.seed(seed)
    act = read_smi(actives_smi, 0, 2).drop_duplicates(subset="id")
    dec = read_smi(decoys_smi, 0, 1).drop_duplicates(subset="id")

    n_a = min(n, len(act))
    n_d = min(n, len(dec))

    act_s = act.sample(n_a, random_state=seed).assign(label="active")
    dec_s = dec.sample(n_d, random_state=seed+1).assign(label="decoy")

    panel = pd.concat([act_s, dec_s], ignore_index=True)
    panel = panel[["id","smiles","label"]]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)
    print(f"Wrote {len(panel)} ligands to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--actives", required=True, help="Path to actives_final.smi")
    ap.add_argument("--decoys",  required=True, help="Path to decoys_final.smi")
    ap.add_argument("--out",     default="data/ligands_raw/hivpr_small_panel.csv")
    ap.add_argument("--n",       type=int, default=20)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()
    main(args.actives, args.decoys, args.out, n=args.n, seed=args.seed)
