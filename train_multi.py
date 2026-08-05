#!/usr/bin/env python3

# """
# Wrapper to run train.py for multiple geometries with fixed parameters.
# Edit the configuration variables below, then run: python train_multi.py
# """

# import subprocess
# import sys
# import os
# import shlex


# geom_dirs = [
#     # "points/phi_-30",
#     # "points/phi_-60",
#     # "points/phi_-90",
#     # "points/phi_30",
#     # "points/phi_60",
#     # "points/phi_90",
#     "points/phi_180"
# ]

# out_dirs = [
#     # "runs/phi_-30",
#     # "runs/phi_-60",
#     # "runs/phi_-90",
#     # "runs/phi_30",
#     # "runs/phi_60",
#     # "runs/phi_90",
#     "runs/phi_180"
# ]

# # Fixed training parameters (as requested)
# Re = 10.0
# Pr = 0.7
# thermal = "none"          # "none", "homogeneous", "nonhomogeneous"
# hidden_layers = 6
# hidden_units = 100
# beta = 2.0
# epochs = 20000
# batch_size = 16384               # 0 for full batch
# lr0 = 1e-3
# lr_patience = 200
# lr_gamma = 0.8
# lr_min = 1e-6
# log_every = 10
# ckpt_every = 1000
# device = "cuda"
# seed = 42


# python_exe = sys.executable

# # ============================================================
# # END OF CONFIGURATION
# # ============================================================

# def main():
#     if len(geom_dirs) != len(out_dirs):
#         sys.exit(f"Error: geom_dirs ({len(geom_dirs)}) and out_dirs ({len(out_dirs)}) must have the same length.")

#     # Base command with all fixed arguments
#     base_cmd = [
#         python_exe, "train.py",
#         "--Re", str(Re),
#         "--Pr", str(Pr),
#         "--thermal", thermal,
#         "--hidden_layers", str(hidden_layers),
#         "--hidden_units", str(hidden_units),
#         "--beta", str(beta),
#         "--epochs", str(epochs),
#         "--batch_size", str(batch_size),
#         "--lr0", str(lr0),
#         "--lr_patience", str(lr_patience),
#         "--lr_gamma", str(lr_gamma),
#         "--lr_min", str(lr_min),
#         "--log_every", str(log_every),
#         "--ckpt_every", str(ckpt_every),
#         "--device", device,
#         "--seed", str(seed),
#     ]

#     for geom_dir, out_dir in zip(geom_dirs, out_dirs):
#         print(f"\n===== Running for geometry: {geom_dir} -> output: {out_dir} =====")
#         os.makedirs(out_dir, exist_ok=True)
#         cmd = base_cmd + ["--geom_dir", geom_dir, "--out_dir", out_dir]
#         cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)
#         print(f"Executing: {cmd_str}")
#         ret = subprocess.run(cmd, check=False)
#         if ret.returncode != 0:
#             print(f"WARNING: run for {geom_dir} exited with code {ret.returncode}")
#         else:
#             print(f"Finished {geom_dir} successfully.")


# if __name__ == "__main__":
#     main()


"""
Wrapper to run train.py for multiple geometries, each with a sweep over
beta and patience, where each combination is a separate run with its own
output folder named as "sym_patience{patience}_beta{beta}".
"""

import subprocess
import sys
import os
import shlex
from itertools import product


geom_dirs = [
    "points/sym"
]

base_out_dir = "runs/sym_multi"
# beta_values = [1.0, 0.5, 0.1, 0.25]
# patience_values = [300, 200, 500]

patience_values=[500]
beta_values=[0.1]

Re = 10.0
Pr = 0.7
thermal = "none"          # "none", "homogeneous", "nonhomogeneous"
hidden_layers = 6
hidden_units = 100
epochs = 20000
batch_size = 16384        # 0 for full batch
lr0 = 1e-3
lr_gamma = 0.8
lr_min = 1e-6
log_every = 10
ckpt_every = 1000
device = "cuda"
seed = 42

train_script = "train.py"
python_exe = sys.executable

def main():
    # Build the base command with fixed arguments (without beta and patience)
    base_cmd = [
        python_exe, train_script,
        "--Re", str(Re),
        "--Pr", str(Pr),
        "--thermal", thermal,
        "--hidden_layers", str(hidden_layers),
        "--hidden_units", str(hidden_units),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr0", str(lr0),
        "--lr_gamma", str(lr_gamma),
        "--lr_min", str(lr_min),
        "--log_every", str(log_every),
        "--ckpt_every", str(ckpt_every),
        "--device", device,
        "--seed", str(seed),
    ]

    # For each geometry (though we have only one in this example)
    for geom_dir in geom_dirs:
        # For each combination of beta and patience
        for beta, patience in product(beta_values, patience_values):
            # Build a descriptive folder name
            # Example: sym_patience300_beta1.0
            geom_name = os.path.basename(os.path.normpath(geom_dir))
            folder_name = f"{geom_name}_patience{patience}_beta{beta}"
            out_dir = os.path.join(base_out_dir, folder_name)

            print(f"\n===== Running: {geom_dir} -> {out_dir} =====")
            os.makedirs(out_dir, exist_ok=True)

            # Build the full command for this combination
            cmd = base_cmd + [
                "--geom_dir", geom_dir,
                "--out_dir", out_dir,
                "--beta", str(beta),
                "--lr_patience", str(patience),
            ]

            cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)
            print(f"Executing: {cmd_str}")

            ret = subprocess.run(cmd, check=False)
            if ret.returncode != 0:
                print(f"WARNING: run for {folder_name} exited with code {ret.returncode}")
            else:
                print(f"Finished {folder_name} successfully.")

if __name__ == "__main__":
    main()


