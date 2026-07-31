
# import torch, time
# from train import (PINN, load_dataset, make_type_masks, precompute_wall_theta, governing_losses, boundary_losses)

# DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Re, Pr, beta, thermal = 10.0, 0.7, 2.0, "none"
# w_thermal = 0.0

# torch.manual_seed(42)
# data = load_dataset("data/sym", DEV)
# is_inlet, is_wall, is_outlet = make_type_masks(data["types"], DEV)
# th_wall = precompute_wall_theta(data["xb"], is_wall, thermal, DEV)
# xc, yc = data["xc"], data["yc"]
# xb, yb, nxb, nyb = data["xb"], data["yb"], data["nxb"], data["nyb"]

# model = PINN(6, 100).to(DEV)
# ck = torch.load("runs/sym_Re10_beta2/checkpoints/epoch_20000.pt",
#                 map_location=DEV, weights_only=False)
# model.load_state_dict(ck["model_state"])

# def total_loss():
#     Lg, _ = governing_losses(model, xc, yc, Re, Pr, w_thermal=w_thermal)
#     LBC, _ = boundary_losses(model, xb, yb, nxb, nyb, is_inlet, is_wall, is_outlet,
#                              th_wall, Re, Pr, thermal_mode=thermal, w_thermal=w_thermal)
#     return Lg + beta * LBC, Lg, LBC

# with torch.no_grad():
#     pass
# Lp0, Lg0, LBC0 = total_loss()
# print(f"[baseline @ frozen ckpt, full-batch] Lp={Lp0.item():.4e} Lg={Lg0.item():.4e} LBC={LBC0.item():.4e}")

# opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # revived LR, full-batch
# t0 = time.time()
# for ep in range(1, 801):
#     opt.zero_grad()
#     Lp, Lg, LBC = total_loss()
#     Lp.backward()
#     opt.step()
#     if ep % 50 == 0 or ep == 1:
#         print(f"[{ep:4d}] Lp={Lp.item():.4e} Lg={Lg.item():.4e} LBC={LBC.item():.4e} ({time.time()-t0:.0f}s)")


# print("DONE_EXP")

# -----------------------------------------------------------------------------------------------------

import torch
import time
from train import (
    PINN, load_dataset, make_type_masks, precompute_wall_theta,
    governing_losses, boundary_losses
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Re, Pr, beta, thermal = 10.0, 0.7, 2.0, "none"
w_thermal = 0.0
BATCH_SIZE = 16384

torch.manual_seed(42)


data = load_dataset("data/sym", DEV)
is_inlet, is_wall, is_outlet = make_type_masks(data["types"], DEV)
th_wall = precompute_wall_theta(data["xb"], is_wall, thermal, DEV)


Nb = data["xb"].shape[0]
th_wall = th_wall.squeeze()
if th_wall.shape[0] != Nb:
    th_wall_full = torch.zeros(Nb, device=DEV)
    th_wall_full[is_wall] = th_wall
    th_wall = th_wall_full

xc, yc = data["xc"], data["yc"]
xb, yb, nxb, nyb = data["xb"], data["yb"], data["nxb"], data["nyb"]

Nc = xc.shape[0]


if Nc >= BATCH_SIZE:
    idx_c = torch.arange(BATCH_SIZE, device=DEV)
else:
    idx_c = torch.randint(0, Nc, (BATCH_SIZE,), device=DEV)

if Nb >= BATCH_SIZE:
    idx_b = torch.arange(BATCH_SIZE, device=DEV)
else:
    idx_b = torch.randint(0, Nb, (BATCH_SIZE,), device=DEV)

xc_b = xc[idx_c]
yc_b = yc[idx_c]
xb_b = xb[idx_b]
yb_b = yb[idx_b]
nxb_b = nxb[idx_b]
nyb_b = nyb[idx_b]
is_inlet_b = is_inlet[idx_b]
is_wall_b = is_wall[idx_b]
is_outlet_b = is_outlet[idx_b]
th_wall_b = th_wall[idx_b]


model = PINN(6, 100).to(DEV)
ck = torch.load(
    "runs/sym_Re10_beta2/checkpoints/epoch_20000.pt",
    map_location=DEV, weights_only=False
)
model.load_state_dict(ck["model_state"])

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = time.time()

for ep in range(1, 801):
    opt.zero_grad()
    Lg, _ = governing_losses(model, xc_b, yc_b, Re, Pr, w_thermal=w_thermal)
    LBC, _ = boundary_losses(
        model, xb_b, yb_b, nxb_b, nyb_b,
        is_inlet_b, is_wall_b, is_outlet_b,
        th_wall_b, Re, Pr,
        thermal_mode=thermal, w_thermal=w_thermal
    )
    Lp = Lg + beta * LBC
    Lp.backward()
    opt.step()

    if ep % 50 == 0 or ep == 1:
        print(f"[{ep:4d}] Lp={Lp.item():.4e} Lg={Lg.item():.4e} LBC={LBC.item():.4e} ({time.time()-t0:.0f}s)")

print("DONE_EXP")

