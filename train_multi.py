#!/usr/bin/env python3

"""
Wrapper to run train.py for multiple geometries with fixed parameters.
Edit the configuration variables below, then run: python train_multi.py
"""

import subprocess
import sys
import os
import shlex


geom_dirs = [
    # "points/phi_-30",
    # "points/phi_-60",
    # "points/phi_-90",
    # "points/phi_30",
    # "points/phi_60",
    # "points/phi_90",
    "points/phi_180"
]

out_dirs = [
    # "runs/phi_-30",
    # "runs/phi_-60",
    # "runs/phi_-90",
    # "runs/phi_30",
    # "runs/phi_60",
    # "runs/phi_90",
    "runs/phi_180"
]

# Fixed training parameters (as requested)
Re = 10.0
Pr = 0.7
thermal = "none"          # "none", "homogeneous", "nonhomogeneous"
hidden_layers = 6
hidden_units = 100
beta = 2.0
epochs = 20000
batch_size = 16384               # 0 for full batch
lr0 = 1e-3
lr_patience = 200
lr_gamma = 0.8
lr_min = 1e-6
log_every = 10
ckpt_every = 1000
device = "cuda"
seed = 42


python_exe = sys.executable

# ============================================================
# END OF CONFIGURATION
# ============================================================

def main():
    if len(geom_dirs) != len(out_dirs):
        sys.exit(f"Error: geom_dirs ({len(geom_dirs)}) and out_dirs ({len(out_dirs)}) must have the same length.")

    # Base command with all fixed arguments
    base_cmd = [
        python_exe, "train.py",
        "--Re", str(Re),
        "--Pr", str(Pr),
        "--thermal", thermal,
        "--hidden_layers", str(hidden_layers),
        "--hidden_units", str(hidden_units),
        "--beta", str(beta),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr0", str(lr0),
        "--lr_patience", str(lr_patience),
        "--lr_gamma", str(lr_gamma),
        "--lr_min", str(lr_min),
        "--log_every", str(log_every),
        "--ckpt_every", str(ckpt_every),
        "--device", device,
        "--seed", str(seed),
    ]

    for geom_dir, out_dir in zip(geom_dirs, out_dirs):
        print(f"\n===== Running for geometry: {geom_dir} -> output: {out_dir} =====")
        os.makedirs(out_dir, exist_ok=True)
        cmd = base_cmd + ["--geom_dir", geom_dir, "--out_dir", out_dir]
        cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)
        print(f"Executing: {cmd_str}")
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print(f"WARNING: run for {geom_dir} exited with code {ret.returncode}")
        else:
            print(f"Finished {geom_dir} successfully.")


if __name__ == "__main__":
    main()


