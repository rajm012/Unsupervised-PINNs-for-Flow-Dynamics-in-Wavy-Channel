#!/usr/bin/env python3

"""
Wrapper to run eval1.py for multiple model checkpoints and reference datasets.
Edit the configuration variables below, then run:
    python eval1_multi.py
"""

import subprocess
import sys
import os
import shlex

eval_script = "eval1.py"

checkpoints = [
    "runs/phi_-30/model_final.pt",
    "runs/phi_-60/model_final.pt",
    "runs/phi_-90/model_final.pt",
    "runs/phi_30/model_final.pt",
    "runs/phi_60/model_final.pt",
    "runs/phi_90/model_final.pt",
    "runs/phi_180/model_final.pt"
]

ref_csvs = [
    "eval_data/cfd_m30.csv",
    "eval_data/cfd_m60.csv",
    "eval_data/cfd_m90.csv",
    "eval_data/cfd_30.csv",
    "eval_data/cfd_60.csv",
    "eval_data/cfd_90.csv",
    "eval_data/cfd_180.csv"
]

out_dirs = [
    "runs/phi_-30/eval",
    "runs/phi_-60/eval",
    "runs/phi_-90/eval",
    "runs/phi_30/eval",
    "runs/phi_60/eval",
    "runs/phi_90/eval",
    "runs/phi_180/eval"
]

hidden_layers = 6
hidden_units = 100
batch_size = 16384
device = "cuda"
python_exe = sys.executable


def main():
    if not (len(checkpoints) == len(ref_csvs) == len(out_dirs)):
        sys.exit(f"Error: checkpoints ({len(checkpoints)}), ref_csvs ({len(ref_csvs)}), "
                 f"and out_dirs ({len(out_dirs)}) must all have the same length.")

    # Base command with fixed arguments
    base_cmd = [
        python_exe, eval_script,
        "--hidden_layers", str(hidden_layers),
        "--hidden_units", str(hidden_units),
        "--batch_size", str(batch_size),
        "--device", device,
    ]

    for ckpt, ref_csv, out_dir in zip(checkpoints, ref_csvs, out_dirs):
        print(f"\n===== Evaluating checkpoint: {ckpt} -> output: {out_dir} =====")
        os.makedirs(out_dir, exist_ok=True)
        cmd = base_cmd + [
            "--checkpoint", ckpt,
            "--ref_csv", ref_csv,
            "--out_dir", out_dir,
        ]
        cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)
        print(f"Executing: {cmd_str}")
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print(f"WARNING: evaluation for {ckpt} exited with code {ret.returncode}")
        else:
            print(f"Finished {ckpt} successfully.")


if __name__ == "__main__":
    main()

