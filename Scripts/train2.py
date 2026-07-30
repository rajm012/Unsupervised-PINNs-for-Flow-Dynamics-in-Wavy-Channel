
"""
python train.py --geom_dir data/sym --Re 10 --beta_phase1 100 --beta_phase2 2 
--phase1_epochs 5000 --lr_patience_phase1 100 --lr_patience_phase2 20 
--hidden_layers 8 --hidden_units 150 --epochs 25000 
--batch_size 16384 --out_dir runs/sym_Re10_phase
"""


import argparse
import json
import math
import os
import time
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# these are the correct constants used from the paper
L = 8.0
a = 1.0
B = 2.0 * np.pi / L
H_MIN = 12 / 7


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


class PINN(nn.Module):
    OUT_NAMES = ["psi", "p", "sxx", "sxy", "syy", "theta"]
    def __init__(self, hidden_layers=6, hidden_units=100):
        super().__init__()
        layers = [nn.Linear(2, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers += [nn.Linear(hidden_units, len(self.OUT_NAMES))]
        self.net = nn.Sequential(*layers)
        self._xavier_init()

    def _xavier_init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        out = self.net(inp)
        psi, p, sxx, sxy, syy, theta = torch.split(out, 1, dim=1)
        return psi, p, sxx, sxy, syy, theta


def grad(outputs, inputs):
    """d(outputs)/d(inputs), keeping the graph for higher-order derivatives."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]


def velocity_from_psi(model, x, y):
    """u = d(psi)/dy, v = -d(psi)/dx  -- divergence-free by construction."""
    psi, p, sxx, sxy, syy, theta = model(x, y)
    u = grad(psi, y)
    v = -grad(psi, x)
    return u, v, p, sxx, sxy, syy, theta


# ========================================================================
# Governing (collocation) losses -- Eqs. (16)-(21)
# ========================================================================
def governing_losses(model, xc, yc, Re, Pr, w_thermal=1.0):
    xc = xc.clone().requires_grad_(True)
    yc = yc.clone().requires_grad_(True)

    u, v, p, sxx, sxy, syy, theta = velocity_from_psi(model, xc, yc)

    u_x = grad(u, xc)
    u_y = grad(u, yc)
    v_x = grad(v, xc)
    v_y = grad(v, yc)

    sxx_x = grad(sxx, xc)
    sxy_x = grad(sxy, xc)
    sxy_y = grad(sxy, yc)
    syy_y = grad(syy, yc)

    theta_x = grad(theta, xc)
    theta_y = grad(theta, yc)
    theta_xx = grad(theta_x, xc)
    theta_yy = grad(theta_y, yc)

    # Lg1 -- continuity
    Lg1 = torch.mean((u_x + v_y) ** 2)

    # Lg2 -- x-momentum
    res2 = u * u_x + v * u_y - (sxx_x + sxy_y)
    Lg2 = torch.mean(res2 ** 2)

    # Lg3 -- y-momentum
    res3 = u * v_x + v * v_y - (sxy_x + syy_y)
    Lg3 = torch.mean(res3 ** 2)

    # Lg4 -- energy
    res4 = u * theta_x + v * theta_y - (1.0 / (Re * Pr)) * (theta_xx + theta_yy)
    Lg4 = torch.mean(res4 ** 2)

    # Lg5 -- constitutive Cauchy-stress relation
    res5a = sxx + p - (2.0 / Re) * u_x
    res5b = syy + p - (2.0 / Re) * v_y
    res5c = sxy - (1.0 / Re) * (u_y + v_x)
    Lg5 = torch.mean(res5a ** 2 + res5b ** 2 + res5c ** 2)

    Lg = Lg1 + Lg2 + Lg3 + w_thermal * Lg4 + Lg5
    parts = {
        "Lg1_continuity": Lg1.item(),
        "Lg2_xmom": Lg2.item(),
        "Lg3_ymom": Lg3.item(),
        "Lg4_energy": Lg4.item(),
        "Lg5_constitutive": Lg5.item(),
    }
    return Lg, parts


# ========================================================================
# Boundary losses -- Eqs. (22)-(26)
# ========================================================================
def theta_wall_target_np(x_np, mode):
    if mode == "nonhomogeneous":
        return 0.2 * np.cos(math.pi * x_np / L) ** 2
    return np.zeros_like(x_np)


def _mse(t):
    if t.numel() == 0:
        return torch.zeros((), device=t.device, dtype=t.dtype)
    return torch.mean(t ** 2)


def boundary_losses(model, xb, yb, nxb, nyb, is_inlet, is_wall, is_outlet,
                    theta_wall_target, Re, Pr, thermal_mode="homogeneous",
                    w_thermal=1.0):
    if thermal_mode == "none":
        w_thermal = 0.0

    xb = xb.clone().requires_grad_(True)
    yb = yb.clone().requires_grad_(True)

    u, v, p, sxx, sxy, syy, theta = velocity_from_psi(model, xb, yb)

    u_x = grad(u, xb)
    u_y = grad(u, yb)
    v_x = grad(v, xb)
    v_y = grad(v, yb)
    p_x = grad(p, xb)
    p_y = grad(p, yb)
    theta_x = grad(theta, xb)
    theta_y = grad(theta, yb)

    dudn = nxb * u_x + nyb * u_y
    dvdn = nxb * v_x + nyb * v_y
    dthetadn = nxb * theta_x + nyb * theta_y

    # ---- u : Dirichlet at inlet (=1) and wall (=0); Neumann 0 at outlet ----
    Lu = _mse(u[is_inlet] - 1.0) + _mse(u[is_wall] - 0.0) + _mse(dudn[is_outlet])

    # ---- v : Dirichlet 0 at inlet and wall; Neumann 0 at outlet ----
    Lv = _mse(v[is_inlet] - 0.0) + _mse(v[is_wall] - 0.0) + _mse(dvdn[is_outlet])

    # ---- p : Dirichlet p=0 at outlet only ----
    Lp_bc = _mse(p[is_outlet] - 0.0)

    # ---- theta : Dirichlet at inlet + walls, Neumann (=0) at outlet ----
    if thermal_mode == "none":
        Ltheta = torch.zeros((), device=xb.device)
    else:
        theta_inlet_target = 1.0 if thermal_mode == "homogeneous" else 0.0
        Ltheta = (
            _mse(theta[is_inlet] - theta_inlet_target)
            + _mse(theta[is_wall] - theta_wall_target)
            + _mse(dthetadn[is_outlet])
        )

    LBC = Lu + Lv + Lp_bc + w_thermal * Ltheta
    parts = {
        "LBC_u": Lu.item(), "LBC_v": Lv.item(), "LBC_p": Lp_bc.item(),
        "LBC_theta": Ltheta.item() if torch.is_tensor(Ltheta) else Ltheta,
    }
    return LBC, parts


# ========================================================================
# Loss-based learning-rate scheduler -- Eq. (28)
# ========================================================================
class LossBasedScheduler:
    def __init__(self, optimizer, patience=20, gamma=0.8, min_lr=1e-6, rel_tol=1e-4):
        self.optimizer = optimizer
        self.patience = patience
        self.gamma = gamma
        self.min_lr = min_lr
        self.rel_tol = rel_tol
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, loss_value):
        if loss_value < self.best * (1.0 - self.rel_tol):
            self.best = loss_value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            for g in self.optimizer.param_groups:
                g["lr"] = max(self.min_lr, g["lr"] * self.gamma)
            self.bad_epochs = 0

    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ========================================================================
# Data loading
# ========================================================================
def load_dataset(geom_dir, device):
    colloc_df = pd.read_csv(os.path.join(geom_dir, "collocation.csv"))
    bnd_df = pd.read_csv(os.path.join(geom_dir, "boundary.csv"))

    xc = torch.tensor(colloc_df["x"].values, dtype=torch.float32).view(-1, 1).to(device)
    yc = torch.tensor(colloc_df["y"].values, dtype=torch.float32).view(-1, 1).to(device)
    zone = colloc_df["zone"].values

    xb = torch.tensor(bnd_df["x"].values, dtype=torch.float32).view(-1, 1).to(device)
    yb = torch.tensor(bnd_df["y"].values, dtype=torch.float32).view(-1, 1).to(device)
    nxb = torch.tensor(bnd_df["nx"].values, dtype=torch.float32).view(-1, 1).to(device)
    nyb = torch.tensor(bnd_df["ny"].values, dtype=torch.float32).view(-1, 1).to(device)
    types = bnd_df["type"].values

    return {
        "xc": xc, "yc": yc, "zone": zone,
        "xb": xb, "yb": yb, "nxb": nxb, "nyb": nyb, "types": types,
    }


def make_type_masks(types, device):
    is_inlet = torch.tensor(types == "inlet", device=device)
    is_wall = torch.tensor((types == "upper_wall") | (types == "lower_wall"), device=device)
    is_outlet = torch.tensor(types == "outlet", device=device)
    return is_inlet, is_wall, is_outlet


def precompute_wall_theta(xb, is_wall, thermal_mode, device):
    x_wall_np = xb.detach().cpu().numpy().flatten()[is_wall.cpu().numpy()]
    target_np = theta_wall_target_np(x_wall_np, thermal_mode)
    return torch.as_tensor(target_np, dtype=torch.float32, device=device).view(-1, 1)


# ========================================================================
# Continuity residual diagnostic
# ========================================================================
def continuity_residual(model, xc, yc, n_check=2000):
    n_check = min(n_check, xc.shape[0])
    idx = torch.randperm(xc.shape[0])[:n_check]
    x = xc[idx].clone().requires_grad_(True)
    y = yc[idx].clone().requires_grad_(True)
    u, v, p, sxx, sxy, syy, theta = velocity_from_psi(model, x, y)
    u_x = grad(u, x)
    v_y = grad(v, y)
    residual = (u_x + v_y).detach().cpu().numpy()
    return {
        "mean_abs": float(np.mean(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
    }


# ========================================================================
# Training implementation with phase switching
# ========================================================================
def train(args):
    device = torch.device("cpu") if args.device == "cuda" and not torch.cuda.is_available() else torch.device(args.device)
    fallback_warning = None
    if args.device == "cuda" and device.type == "cpu":
        fallback_warning = "[warning] CUDA requested but not available -- falling back to CPU."

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train.log")

    with open(log_path, "a", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            if fallback_warning is not None:
                print(fallback_warning)
            return _train_impl(args, device, ckpt_dir, log_file)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _train_impl(args, device, ckpt_dir, log_file):
    print(f"Logging training output to {os.path.abspath(log_file.name)}")

    data = load_dataset(args.geom_dir, device)
    is_inlet, is_wall, is_outlet = make_type_masks(data["types"], device)
    theta_wall_target = precompute_wall_theta(data["xb"], is_wall, args.thermal, device)

    model = PINN(hidden_layers=args.hidden_layers, hidden_units=args.hidden_units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr0)

    # Initial beta and patience from phase 1
    beta_current = args.beta_phase1
    patience_current = args.lr_patience_phase1
    scheduler = LossBasedScheduler(
        optimizer,
        patience=patience_current,
        gamma=args.lr_gamma,
        min_lr=args.lr_min,
        rel_tol=args.lr_rel_tol
    )

    xc, yc = data["xc"], data["yc"]
    xb, yb, nxb, nyb = data["xb"], data["yb"], data["nxb"], data["nyb"]
    w_thermal = 1.0 if args.thermal != "none" else 0.0

    history = []
    t0 = time.time()
    phase_switched = False

    for epoch in range(1, args.epochs + 1):
        # ---- Switch to phase 2 after phase1_epochs ----
        if epoch == args.phase1_epochs + 1 and not phase_switched:
            beta_current = args.beta_phase2
            patience_current = args.lr_patience_phase2
            scheduler.patience = patience_current
            scheduler.best = float("inf")    # reset best to avoid immediate LR drop
            scheduler.bad_epochs = 0
            phase_switched = True
            print(f"\n[Phase switch] beta={beta_current}, patience={patience_current}\n")

        # ---- Batch selection ----
        if args.batch_size and args.batch_size < xc.shape[0]:
            idx = torch.randperm(xc.shape[0], device=device)[:args.batch_size]
            xc_b, yc_b = xc[idx], yc[idx]
        else:
            xc_b, yc_b = xc, yc

        optimizer.zero_grad()

        Lg, g_parts = governing_losses(model, xc_b, yc_b, args.Re, args.Pr, w_thermal=w_thermal)
        LBC, b_parts = boundary_losses(
            model, xb, yb, nxb, nyb, is_inlet, is_wall, is_outlet,
            theta_wall_target, args.Re, args.Pr,
            thermal_mode=args.thermal, w_thermal=w_thermal,
        )
        Lp = Lg + beta_current * LBC

        Lp.backward()
        optimizer.step()
        scheduler.step(Lp.item())

        # ---- Logging ----
        if epoch % args.log_every == 0 or epoch == 1:
            row = {
                "epoch": epoch,
                "Lp": Lp.item(),
                "Lg": Lg.item(),
                "LBC": LBC.item(),
                "lr": scheduler.current_lr(),
                "beta": beta_current,
                "time_s": time.time() - t0,
            }
            row.update(g_parts)
            row.update(b_parts)
            history.append(row)
            print(
                f"[{epoch:6d}/{args.epochs}] beta={beta_current:.1f}  Lp={Lp.item():.4e}  "
                f"Lg={Lg.item():.4e}  LBC={LBC.item():.4e}  lr={scheduler.current_lr():.2e}"
            )

        # ---- Checkpoint ----
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                    "beta_current": beta_current,
                },
                os.path.join(ckpt_dir, f"epoch_{epoch}.pt"),
            )

    # ---- final diagnostics ----
    diag = continuity_residual(model, xc, yc)
    print(f"\nFinal continuity residual: mean|.|={diag['mean_abs']:.3e}, "
          f"max|.|={diag['max_abs']:.3e}")

    # ---- save history ----
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(args.out_dir, "loss_history.csv"), index=False)
    with open(os.path.join(args.out_dir, "final_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    # ---- loss curve ----
    plt.figure(figsize=(7, 4))
    plt.semilogy(hist_df["epoch"], hist_df["Lp"], label="Lp (total)")
    plt.semilogy(hist_df["epoch"], hist_df["Lg"], label="Lg (governing)")
    plt.semilogy(hist_df["epoch"], hist_df["LBC"], label="LBC (boundary)")
    # vertical line at phase switch
    if args.phase1_epochs < args.epochs:
        plt.axvline(x=args.phase1_epochs, color='k', linestyle='--', alpha=0.5, label='phase switch')
    plt.xlabel("epoch")
    plt.ylabel("loss (log scale)")
    plt.title(f"Training loss -- {os.path.basename(args.geom_dir)}, Re={args.Re}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=200)
    plt.close()

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model_final.pt"))
    print(f"Done. Outputs written to {args.out_dir}")
    return model, hist_df, diag


# ========================================================================
# Argument parser
# ========================================================================
def build_argparser():
    p = argparse.ArgumentParser(description="Train wavy-channel PINN (2025) with phase-switching beta")
    p.add_argument("--geom_dir", type=str, required=True, help="Path to geometry folder, e.g. data/phi_60")
    p.add_argument("--Re", type=float, required=True, help="Reynolds number")
    p.add_argument("--Pr", type=float, default=0.7, help="Prandtl number")
    p.add_argument("--thermal", type=str, default="none",
                   choices=["none", "homogeneous", "nonhomogeneous"],
                   help="Thermal mode (none for hydrodynamics only)")

    # Network architecture
    p.add_argument("--hidden_layers", type=int, default=6, help="Number of hidden layers (Table I)")
    p.add_argument("--hidden_units", type=int, default=100, help="Neurons per hidden layer (Table I)")

    # Beta phase schedule
    p.add_argument("--beta_phase1", type=float, default=100.0, help="Beta during first phase (high to enforce BCs)")
    p.add_argument("--beta_phase2", type=float, default=2.0, help="Beta during second phase (allow physics refinement)")
    p.add_argument("--phase1_epochs", type=int, default=5000, help="Number of epochs for phase 1")

    # LR scheduler parameters (separate patience for each phase)
    p.add_argument("--lr0", type=float, default=1e-3, help="Initial learning rate")
    p.add_argument("--lr_gamma", type=float, default=0.8, help="LR multiplication factor (Eq. 28)")
    p.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate")
    p.add_argument("--lr_rel_tol", type=float, default=1e-4, help="Relative tolerance for LR scheduler")
    p.add_argument("--lr_patience_phase1", type=int, default=100, help="LR patience during phase 1 (keeps LR high)")
    p.add_argument("--lr_patience_phase2", type=int, default=20, help="LR patience during phase 2")

    # Training control
    p.add_argument("--epochs", type=int, default=25000, help="Total epochs")
    p.add_argument("--batch_size", type=int, default=0, help="Mini-batch size (0 = full batch)")
    p.add_argument("--log_every", type=int, default=100, help="Log frequency")
    p.add_argument("--ckpt_every", type=int, default=2000, help="Checkpoint frequency")
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out_dir", type=str, default="runs/default", help="Output directory")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
   
