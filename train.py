
"""
Implements, term-for-term:
  - Mixed-variable network  (x, y) -> (psi, p, sigma_xx, sigma_xy, sigma_yy, theta)
    with u = d(psi)/dy, v = -d(psi)/dx   (Sec. III, Fig. 4)
  - Governing/collocation losses Lg1..Lg5  (Eqs. 16-21)
  - Boundary losses (Dirichlet + Neumann) LBC_u, LBC_v, LBC_p, LBC_theta
    (Eqs. 22-26), driven by the `type`/`nx`/`ny` columns of boundary.csv
  - Total loss Lp = Lg + beta * LBC          (Eq. 27)
  - Loss-based learning-rate scheduler        (Eq. 28)
  - Xavier init, tanh activation, Adam optimizer (Sec. III.B)

Consumes exactly the dataset produced by dataUsedGen.py:
  data/<geom>/collocation.csv   columns: x, y, zone
  data/<geom>/boundary.csv      columns: x, y, nx, ny, type

Smoke test:
    python train_pinn.py --geom_dir data/sym --Re 10 --Pr 0.7 --thermal homogeneous --epochs 50 --batch_size 500 --hidden_layers 3 --hidden_units 32 --log_every 10 --out_dir runs/smoke_test
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
import random

# these are the correct constants used from the paper
L = 8.0
a = 1.0
B = 2.0 * np.pi / L
H_MIN = 12/7


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
    """
    MLP mapping (x, y) -> (psi, p, sigma_xx, sigma_xy, sigma_yy, theta).

    `hidden_layers` / `hidden_units` follow Table I, e.g. 6 x 100 for the
    symmetric/asymmetric wavy channel, 5 x 80 for the plane-channel
    benchmark.
    """

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
# 2. GOVERNING (COLLOCATION) LOSSES -- Eqs. (16)-(21)
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

    # Lg1 -- continuity (Eq. 16)
    Lg1 = torch.mean((u_x + v_y) ** 2)

    # Lg2 -- x-momentum (Eq. 17)
    res2 = u * u_x + v * u_y - (sxx_x + sxy_y)
    Lg2 = torch.mean(res2 ** 2)

    # Lg3 -- y-momentum (Eq. 18); sigma_yx = sigma_xy (symmetric stress)
    res3 = u * v_x + v * v_y - (sxy_x + syy_y)
    Lg3 = torch.mean(res3 ** 2)

    # Lg4 -- energy (Eq. 19)
    res4 = u * theta_x + v * theta_y - (1.0 / (Re * Pr)) * (theta_xx + theta_yy)
    Lg4 = torch.mean(res4 ** 2)

    # Lg5 -- constitutive Cauchy-stress relation (Eq. 20)
    res5a = sxx + p - (2.0 / Re) * u_x
    res5b = syy + p - (2.0 / Re) * v_y
    res5c = sxy - (1.0 / Re) * (u_y + v_x)
    Lg5 = torch.mean(res5a ** 2 + res5b ** 2 + 2*(res5c ** 2))

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
# 3. BOUNDARY LOSSES -- Eqs. (22)-(26)
# ========================================================================
def theta_wall_target_np(x_np, mode):
    """Wall temperature target theta_w(x). Eq. (34) for the non-homogeneous
    case, otherwise a constant 0 (Fig. 3 / Sec. V.B.1)."""
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
    """
    is_inlet / is_wall / is_outlet: precomputed boolean torch tensors (same
    length as xb), built once from the `type` column -- see make_type_masks.
    theta_wall_target: precomputed tensor of theta_w(x) values for the wall
    points only (same length as xb[is_wall]) -- see precompute_wall_theta.
    thermal_mode: 'homogeneous'    -> inlet theta=1, walls theta=0  (Fig. 3)
                  'nonhomogeneous' -> inlet theta=0, walls theta=Eq.(34)
                  'none'           -> theta terms excluded
    """
    if thermal_mode == "none":
        w_thermal = 0.0

    xb = xb.clone().requires_grad_(True)
    yb = yb.clone().requires_grad_(True)

    u, v, p, sxx, sxy, syy, theta = velocity_from_psi(model, xb, yb)

    u_x = grad(u, xb); u_y = grad(u, yb)
    v_x = grad(v, xb); v_y = grad(v, yb)
    p_x = grad(p, xb); p_y = grad(p, yb)
    theta_x = grad(theta, xb); theta_y = grad(theta, yb)

    dudn = nxb * u_x + nyb * u_y
    dvdn = nxb * v_x + nyb * v_y
    dpdn = nxb * p_x + nyb * p_y  # noqa: F841 -- kept for completeness/debugging
    dthetadn = nxb * theta_x + nyb * theta_y

    # ---- u : Dirichlet at inlet (=1) and wall (=0); Neumann 0 at outlet ----
    Lu = _mse(u[is_inlet] - 1.0) + _mse(u[is_wall] - 0.0) + _mse(dudn[is_outlet])

    # ---- v : Dirichlet 0 at inlet and wall; Neumann 0 at outlet ----
    Lv = _mse(v[is_inlet] - 0.0) + _mse(v[is_wall] - 0.0) + _mse(dvdn[is_outlet])

    # ---- p : Dirichlet p=0 at outlet only (Fig. 3) ----
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
        "LBC_u": Lu.item(),  "LBC_v": Lv.item(), "LBC_p": Lp_bc.item(),
        "LBC_theta": Ltheta.item() if torch.is_tensor(Ltheta) else Ltheta,
    }
    return LBC, parts


# ========================================================================
# 4. LOSS-BASED LEARNING-RATE SCHEDULER -- Eq. (28)
# ========================================================================
class LossBasedScheduler:
    """
    eta_{t+1} = max(eta_min, eta_t * gamma)   if loss hasn't improved for
    `patience` consecutive epochs (Eq. 28: patience=20, gamma=0.8,
    eta_min=1e-6 in the paper).
    """

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
# 5. DATA LOADING
# ========================================================================
def load_dataset(geom_dir, device):
    colloc_df = pd.read_csv(os.path.join(geom_dir, "collocation.csv"))
    bnd_df = pd.read_csv(os.path.join(geom_dir, "boundary.csv"))

    xc = torch.tensor(colloc_df["x"].values, dtype=torch.float32).view(-1, 1).to(device)
    yc = torch.tensor(colloc_df["y"].values, dtype=torch.float32).view(-1, 1).to(device)
    zone = colloc_df["zone"].values  # 'core' / 'bl' -- kept for optional diagnostics

    xb = torch.tensor(bnd_df["x"].values, dtype=torch.float32).view(-1, 1).to(device)
    yb = torch.tensor(bnd_df["y"].values, dtype=torch.float32).view(-1, 1).to(device)
    nxb = torch.tensor(bnd_df["nx"].values, dtype=torch.float32).view(-1, 1).to(device)
    nyb = torch.tensor(bnd_df["ny"].values, dtype=torch.float32).view(-1, 1).to(device)
    types = bnd_df["type"].values  # numpy array of strings

    return {
        "xc": xc, "yc": yc, "zone": zone,
        "xb": xb, "yb": yb, "nxb": nxb, "nyb": nyb, "types": types,
    }


def make_type_masks(types, device):
    """Boolean torch masks built once from the `type` column (fixed for the
    whole run -- boundary point identities never change during training)."""
    is_inlet = torch.tensor(types == "inlet", device=device)
    is_wall = torch.tensor((types == "upper_wall") | (types == "lower_wall"), device=device)
    is_outlet = torch.tensor(types == "outlet", device=device)
    return is_inlet, is_wall, is_outlet


def precompute_wall_theta(xb, is_wall, thermal_mode, device):
    """theta_w(x) evaluated once at the wall points (Eq. 34); constant
    throughout training since wall x-coordinates don't change."""
    x_wall_np = xb.detach().cpu().numpy().flatten()[is_wall.cpu().numpy()]
    target_np = theta_wall_target_np(x_wall_np, thermal_mode)
    return torch.as_tensor(target_np, dtype=torch.float32, device=device).view(-1, 1)


# ========================================================================
# 6. MASS-CONSERVATION / RESIDUAL DIAGNOSTIC (Sec. IV.A style check)
# ========================================================================
def continuity_residual(model, xc, yc, n_check=2000):
    """Sample n_check collocation points and report |du/dx + dv/dy| stats,
    mirroring the mass-conservation check in Sec. IV.A / Fig. 8."""
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
# 7. TRAINING LOOP
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
    scheduler = LossBasedScheduler(
        optimizer, patience=args.lr_patience, gamma=args.lr_gamma, min_lr=args.lr_min
    )

    xc, yc = data["xc"], data["yc"]
    xb, yb, nxb, nyb = data["xb"], data["yb"], data["nxb"], data["nyb"]
    w_thermal = 1.0 if args.thermal != "none" else 0.0

    history = []
    t0 = time.time()
    best_loss = float('inf')
    beta = args.beta 
    lookback_buffer = []

    for epoch in range(1, args.epochs + 1):
        if args.batch_size and args.batch_size < xc.shape[0]:
            idx = torch.randperm(xc.shape[0], device=device)[: args.batch_size]
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

        # --- GradNorm or RatioNorm Adaptive Beta Logic ---
        # if args.adaptive_beta:
            # with torch.no_grad():
                # -----------------------------------------------------------
                # # 1. Compute Gradient Norms w.r.t the shared weights (using last layer to save compute)
                # shared_params = list(model.net[-1].parameters())
                # grads_g = torch.autograd.grad(Lg, shared_params, retain_graph=True)
                # grad_norm_g = torch.norm(torch.cat([g.view(-1) for g in grads_g]), p=2)

                # grads_bc = torch.autograd.grad(LBC, shared_params, retain_graph=True)
                # grad_norm_bc = torch.norm(torch.cat([g.view(-1) for g in grads_bc]), p=2)
                
                # # 2. Calculate average gradient norm and average loss
                # avg_grad = (grad_norm_g + grad_norm_bc) / 2.0
                # avg_loss = (Lg.item() + LBC.item()) / 2.0
                
                # # 3. Calculate the Target Gradient Norm for the BC term
                # # A higher alpha forces the gradients to match more strictly
                # alpha = args.gradnorm_alpha
                # target_grad_bc = avg_grad * (LBC.item() / (avg_loss + 1e-8)) ** alpha
                
                # # 4. Compute the target beta required to achieve this gradient norm
                # target_beta = target_grad_bc / (grad_norm_bc + 1e-8)
                
                # # 5. Smoothly update beta via Exponential Moving Average (EMA)
                # beta = (1.0 - args.gradnorm_lr) * beta + args.gradnorm_lr * target_beta
                
                # # 6. Clamp beta to a sensible range to prevent exploding values
                # beta = max(args.beta_min, min(args.beta_max, beta))
                # print(f"Beta for the epoch {epoch} is {beta}")
                # -----------------------------------------------------------
                # ratio = Lg / (LBC + 1e-8)
                # beta = (1.0 - args.beta_update_alpha) * beta + args.beta_update_alpha * ratio
                # beta = max(0.001, min(100.0, beta))
                # print(f"Beta for the epoch {epoch} is {beta}")
                # -----------------------------------------------------------

        # --- ReLoBRaLo Adaptive Beta Logic ---
        if args.relobralo:
            with torch.no_grad():
                 # Compute gradient norms of Lg and LBC w.r.t the last layer (to save compute)
                shared_params = list(model.net[-1].parameters())
                
                grads_g = torch.autograd.grad(Lg, shared_params, retain_graph=True)
                grad_norm_g = torch.norm(torch.cat([g.view(-1) for g in grads_g]), p=2)
                grads_bc = torch.autograd.grad(LBC, shared_params, retain_graph=True)
                grad_norm_bc = torch.norm(torch.cat([g.view(-1) for g in grads_bc]), p=2)
                
                eps = 1e-8
                L_total = Lg + LBC
                G_total = grad_norm_g + grad_norm_bc
                
                # 1. Calculate Ratios
                loss_ratio_g = Lg / (L_total + eps)
                loss_ratio_bc = LBC / (L_total + eps)
                grad_ratio_g = grad_norm_g / (G_total + eps)
                grad_ratio_bc = grad_norm_bc / (G_total + eps)
                
                # 2. Calculate Position Ratios
                pos_ratio_g = loss_ratio_g / (grad_ratio_g + eps)
                pos_ratio_bc = loss_ratio_bc / (grad_ratio_bc + eps)
                
                # 3. Calculate Temperature-scaled Softmax weights
                exp_g = torch.exp(args.relobralo_tau * pos_ratio_g)
                exp_bc = torch.exp(args.relobralo_tau * pos_ratio_bc)
                
                # 4. Target Beta derived from the softmax ratio
                target_beta = (exp_bc / (exp_g + eps)).item()
                
                # 5. Apply Exponential Moving Average smoothing
                new_beta = (1.0 - args.relobralo_smooth) * beta + args.relobralo_smooth * target_beta
                
                # 6. Random Lookback Logic
                if len(lookback_buffer) > 0 and random.random() < args.relobralo_lookback_prob:
                    # Look back into history: sample a random previous beta
                    beta = random.choice(lookback_buffer)
                else:
                    # Standard update
                    beta = new_beta
                    # Append the new beta to the history buffer
                    lookback_buffer.append(beta)
                    if len(lookback_buffer) > args.relobralo_buffer_size:
                        lookback_buffer.pop(0) # Remove the oldest value
                
                # 7. Clamp beta to ensure stability
                beta = max(args.beta_min, min(args.beta_max, beta))
                print(f"Beta for the epoch {epoch} is {beta}")


        Lp = Lg + args.beta * LBC  # Eq. (27)
        Lp_value = Lp.item()

        Lp.backward()
        optimizer.step()
        scheduler.step(Lp_value)
        
        if Lp_value < best_loss:
            best_loss = Lp_value
            torch.save(model.state_dict(), os.path.join(args.out_dir, f'model_best.pt'))
            print(f"[{epoch:6d}] New best model! with Loss= {best_loss:.4e}")

        if epoch % args.log_every == 0 or epoch == 1:
            row = {
                "epoch": epoch, "Lp": Lp.item(), "Lg": Lg.item(), "LBC": LBC.item(),
                "lr": scheduler.current_lr(), "time_s": time.time() - t0,
            }
            row.update(g_parts)
            row.update(b_parts)
            history.append(row)
            print(
                f"[{epoch:6d}/{args.epochs}] Lp={Lp.item():.4e}  Lg={Lg.item():.4e}  "
                f"LBC={LBC.item():.4e}  lr={scheduler.current_lr():.2e}"
            )

        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "args": vars(args)},
                os.path.join(ckpt_dir, f"epoch_{epoch}.pt"),
            )

    # ---- final mass-conservation diagnostic ----
    diag = continuity_residual(model, xc, yc)
    print(f"Final continuity residual: mean|.|={diag['mean_abs']:.3e}, "
          f"max|.|={diag['max_abs']:.3e}")

    # ---- save loss history + plot ----
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(args.out_dir, "loss_history.csv"), index=False)
    with open(os.path.join(args.out_dir, "final_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    plt.figure(figsize=(7, 4))
    plt.semilogy(hist_df["epoch"], hist_df["Lp"], label="Lp (total)")
    plt.semilogy(hist_df["epoch"], hist_df["Lg"], label="Lg (governing)")
    plt.semilogy(hist_df["epoch"], hist_df["LBC"], label="LBC (boundary)")
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


def build_argparser():
    p = argparse.ArgumentParser(description="Train wavy-channel PINN(2025)")
    p.add_argument("--geom_dir", type=str, required=True, help="Path to one geometry folder, e.g. data/phi_60")
    p.add_argument("--Re", type=float, required=True, help="Reynolds number")
    p.add_argument("--Pr", type=float, default=0.7, help="Prandtl number")
    p.add_argument("--thermal", type=str, default="homogeneous", choices=["none", "homogeneous", "nonhomogeneous"], help="Thermal BC mode (Sec. V.B.1 / V.B.2, or 'none' for hydrodynamics-only)")
    p.add_argument("--hidden_layers", type=int, default=6, help="Table I: 6 for wavy, 5 for plane")
    p.add_argument("--hidden_units", type=int, default=100, help="Table I: 100 for wavy, 80 for plane")
    p.add_argument("--beta", type=float, default=10.0, help="Boundary-loss weight (Eq. 27)")
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=16384, help="0 = full-batch (as in the paper)")
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--lr_patience", type=int, default=300)
    p.add_argument("--lr_gamma", type=float, default=0.8)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="runs/default")

    # p.add_argument("--adaptive_beta", action="store_true", help="Enable GradNorm adaptive weighting for beta")
    # p.add_argument("--gradnorm_alpha", type=float, default=0.5, help="GradNorm alpha. 0.5-1.0. Controls sensitivity to loss change rates.")
    # p.add_argument("--gradnorm_lr", type=float, default=0.05, help="EMA smoothing rate for updating beta in GradNorm. Default 0.05.")
    # p.add_argument("--beta_min", type=float, default=0.01, help="Minimum possible value for adaptive beta (clamping).")
    # p.add_argument("--beta_max", type=float, default=100.0, help="Maximum possible value for adaptive beta (clamping).")

    # p.add_argument("--adaptive_beta", action="store_true", help="Enable adaptive weighting for beta (Loss-Ratio balancing)")
    # p.add_argument("--beta_update_alpha", type=float, default=0.05,help="Smoothing factor for beta EMA. Smaller = more stable (default: 0.05)")

    p.add_argument("--relobralo", action="store_true", help="Enable ReLoBRaLo (Relative Loss Balancing with Random Lookback)")
    p.add_argument("--relobralo_tau", type=float, default=1.0, help="Temperature parameter for ReLoBRaLo softmax. Lower = smoother.")
    p.add_argument("--relobralo_smooth", type=float, default=0.05, help="EMA smoothing factor for beta updates in ReLoBRaLo.")
    p.add_argument("--relobralo_lookback_prob", type=float, default=0.2, help="Probability (0-1) to use Random Lookback instead of updating beta.")
    p.add_argument("--relobralo_buffer_size", type=int, default=500, help="Maximum history size for the lookback buffer.")
    p.add_argument("--beta_min", type=float, default=0.001, help="Clamp minimum beta.")
    p.add_argument("--beta_max", type=float, default=100.0, help="Clamp maximum beta.")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)

