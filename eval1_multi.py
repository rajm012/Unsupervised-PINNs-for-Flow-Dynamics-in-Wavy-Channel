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
    "runs/sym_multi/sym_beta0p5_patience300/model_best.pt",
    "runs/sym_multi/sym_beta0p5_patience500/model_best.pt",
    "runs/sym_multi/sym_beta0p5_patience100/model_best.pt",
    "runs/sym_multi/sym_patience200_beta0.5/model_best.pt",
    "runs/sym_multi/sym_patience300_beta0.5/model_best.pt",
    "runs/sym_multi/sym_patience500_beta0.5/model_best.pt",
    "runs/sym_multi/sym_patience500_beta1.0/model_best.pt",
    "runs/sym_multi/sym_patience200_beta1.0/model_best.pt",
    "runs/sym_multi/sym_patience300_beta1.0/model_best.pt",
    "runs/sym_multi/sym_patience300_beta0.1/model_best.pt",
    "runs/sym_multi/sym_patience500_beta0.1/model_best.pt",
    "runs/sym_multi/sym_patience200_beta0.1/model_best.pt"
]

ref_csvs = [
    "eval_data/cfd_sym.csv",
    "eval_data/cfd_sym.csv",
    "eval_data/cfd_sym.csv",
    "eval_data/cfd_sym.csv",
    "eval_data/cfd_sym.csv",
    "eval_data/cfd_sym.csv",
    "eval_data/cfd_sym.csv",
    "eval_data/cfd_sym.csv",
    "eval_data/cfd_sym.csv",
    "eval_data/cfd_sym.csv",
    "eval_data/cfd_sym.csv",
    "eval_data/cfd_sym.csv"
]

out_dirs = [
    "runs/sym_multi/sym_beta0p5_patience300/eval_best",
    "runs/sym_multi/sym_beta0p5_patience500/eval_best",
    "runs/sym_multi/sym_beta0p5_patience100/eval_best",
    "runs/sym_multi/sym_patience200_beta0.5/eval_best",
    "runs/sym_multi/sym_patience300_beta0.5/eval_best",
    "runs/sym_multi/sym_patience500_beta0.5/eval_best",
    "runs/sym_multi/sym_patience500_beta1.0/eval_best",
    "runs/sym_multi/sym_patience200_beta1.0/eval_best",
    "runs/sym_multi/sym_patience300_beta1.0/eval_best",
    "runs/sym_multi/sym_patience300_beta0.1/eval_best",
    "runs/sym_multi/sym_patience500_beta0.1/eval_best",
    "runs/sym_multi/sym_patience200_beta0.1/eval_best"
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

