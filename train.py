
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

--- PERIODIC MODE (new) ---
If your reference data was generated with a periodic flow condition (fixed
pressure drop Delta_p from inlet to outlet, driving a fully-developed
in-module flow, plus a pinned reference pressure point -- e.g. COMSOL's
"Periodic Flow Condition" + "Pressure Point Constraint"), pass --periodic
and --delta_p. This replaces the inlet-Dirichlet-velocity / outlet-Neumann
/ outlet-p=0 boundary treatment with:
  - u, v (and theta, if thermal != none) matched at paired inlet/outlet
    points (periodicity)
  - p(inlet) - p(outlet) = delta_p at those same paired points (imposed
    pressure drop, NOT p=0 at outlet)
  - p = 0 pinned at the two extreme-y inlet points (the two "left corner"
    points), matching a two-point pressure constraint
The interior governing-equation losses (Lg1..Lg5) are UNCHANGED -- the
Navier-Stokes physics inside the domain doesn't care how the boundary was
driven. Only the boundary loss changes.

Smoke test:
    python train_pinn.py --geom_dir data/sym --Re 10 --Pr 0.7 --thermal homogeneous --epochs 50 --batch_size 500 --hidden_layers 3 --hidden_units 32 --log_every 10 --out_dir runs/smoke_test

Smoke test (periodic):
    python train_pinn.py --geom_dir data/sym --Re 10 --Pr 0.7 --thermal none --periodic --delta_p 13 --epochs 50 --batch_size 500 --hidden_layers 3 --hidden_units 32 --log_every 10 --out_dir runs/smoke_test_periodic
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
#    UNCHANGED for periodic mode: the interior PDEs don't know or care how
#    the boundary is driven.
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
# 3. BOUNDARY LOSSES -- Eqs. (22)-(26)  (non-periodic / original mode)
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
    Original (non-periodic) boundary treatment: fixed inlet velocity,
    zero-gradient outlet velocity, p=0 at outlet (Fig. 3). Use this when
    your reference data was generated with a developing-flow / fixed-inlet
    setup, NOT a periodic flow condition.

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
# 3b. PERIODIC BOUNDARY LOSSES (new) -- for pressure-drop-driven,
#     module-to-module periodic reference data (COMSOL Periodic Flow
#     Condition + Pressure Point Constraint).
# ========================================================================
def build_periodic_pairs(xb, yb, is_inlet, is_outlet):
    """
    Builds matched inlet<->outlet point pairs (by sorting each side on y and
    truncating to the shorter of the two counts) for the periodicity loss,
    plus the two extreme-y inlet points (top-left / bottom-left corners) to
    use as the pinned-pressure reference points.

    This assumes the wall geometry is genuinely periodic with period L
    (y_top(x_min)==y_top(x_max), y_bottom(x_min)==y_bottom(x_max)), which it
    should be if x_max-x_min equals exactly one wavelength -- if not, the
    inlet/outlet cross-sections differ in shape and periodicity is not a
    physically valid boundary condition in the first place.
    """
    xb_in, yb_in = xb[is_inlet], yb[is_inlet]
    xb_out, yb_out = xb[is_outlet], yb[is_outlet]

    order_in = torch.argsort(yb_in.squeeze(-1))
    order_out = torch.argsort(yb_out.squeeze(-1))
    n_pair = min(order_in.shape[0], order_out.shape[0])
    order_in = order_in[:n_pair]
    order_out = order_out[:n_pair]

    xb_in_p, yb_in_p = xb_in[order_in], yb_in[order_in]
    xb_out_p, yb_out_p = xb_out[order_out], yb_out[order_out]

    max_y_mismatch = (yb_in_p - yb_out_p).abs().max().item()
    if max_y_mismatch > 1e-3:
        print(
            f"[warning] inlet/outlet y-pairing mismatch up to {max_y_mismatch:.4f} "
            "-- geometry may not be exactly periodic over [x_min, x_max], or "
            "inlet/outlet point counts in boundary.csv differ. Periodicity loss "
            "will still run, but double-check your wall formulas / regenerate "
            "boundary.csv with equal inlet/outlet counts if this is large."
        )

    # pin points: the two extreme-y inlet points (top-left, bottom-left corners),
    # matching "two corner points on the left" pressure-point constraint.
    y_in_flat = yb_in.squeeze(-1)
    idx_min = torch.argmin(y_in_flat).view(1)
    idx_max = torch.argmax(y_in_flat).view(1)
    pin_idx = torch.cat([idx_min, idx_max])
    xb_pin, yb_pin = xb_in[pin_idx], yb_in[pin_idx]

    return xb_in_p, yb_in_p, xb_out_p, yb_out_p, xb_pin, yb_pin


def periodic_boundary_losses(model, xb_wall, yb_wall, theta_wall_target,
                              xb_in, yb_in, xb_out, yb_out,
                              xb_pin, yb_pin, delta_p,
                              thermal_mode="none", w_thermal=0.0):
    """
    Periodic hydrodynamic boundary treatment:
      - walls: no-slip Dirichlet (u=v=0), theta Dirichlet if thermal != none
      - inlet<->outlet pairs: u, v matched (periodicity); p differs by
        exactly delta_p (p_inlet - p_outlet = delta_p, i.e. pressure drops
        by delta_p across the module in the flow direction)
      - one pinned point (or two, matching a two-point constraint) fixes
        the otherwise-arbitrary additive pressure constant

    NOTE on theta: this pairs theta_inlet == theta_outlet, i.e. treats
    temperature as periodic module-to-module. That's only physically valid
    if there's no net heat addition per module. If you're driving a genuine
    periodically-developing THERMAL case (net heating per module), theta
    needs the same drop/decomposition treatment as pressure -- don't combine
    --periodic with --thermal nonhomogeneous without addressing that first.
    """
    # ---- walls: no-slip (+ theta target) ----
    xb_w = xb_wall.clone().requires_grad_(True)
    yb_w = yb_wall.clone().requires_grad_(True)
    u_w, v_w, p_w, sxx_w, sxy_w, syy_w, theta_w = velocity_from_psi(model, xb_w, yb_w)
    Lu = _mse(u_w - 0.0)
    Lv = _mse(v_w - 0.0)

    # ---- periodicity: matched inlet/outlet pairs ----
    xb_i = xb_in.clone().requires_grad_(True)
    yb_i = yb_in.clone().requires_grad_(True)
    xb_o = xb_out.clone().requires_grad_(True)
    yb_o = yb_out.clone().requires_grad_(True)
    u_i, v_i, p_i, sxx_i, sxy_i, syy_i, theta_i = velocity_from_psi(model, xb_i, yb_i)
    u_o, v_o, p_o, sxx_o, sxy_o, syy_o, theta_o = velocity_from_psi(model, xb_o, yb_o)

    Lu = Lu + _mse(u_i - u_o)
    Lv = Lv + _mse(v_i - v_o)

    # ---- pressure: fixed drop across the module (NOT p=0 at outlet) ----
    Lp_bc = _mse((p_i - p_o) - delta_p)

    # ---- point pin(s): fix the arbitrary additive pressure constant ----
    # pure value match -- no derivative needed, so a plain forward pass is enough.
    _, p_pin, _, _, _, _ = model(xb_pin, yb_pin)
    Lp_bc = Lp_bc + _mse(p_pin - 0.0)

    if thermal_mode == "none":
        Ltheta = torch.zeros((), device=xb_wall.device)
    else:
        Ltheta = _mse(theta_w - theta_wall_target) + _mse(theta_i - theta_o)

    LBC = Lu + Lv + Lp_bc + w_thermal * Ltheta
    parts = {
        "LBC_u": Lu.item(), "LBC_v": Lv.item(), "LBC_p": Lp_bc.item(),
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

    if args.periodic and args.delta_p is None:
        raise ValueError("--periodic requires --delta_p (the imposed pressure drop, "
                          "matching the Delta_p you set in the CFD Periodic Flow Condition).")
    if args.periodic and args.thermal == "nonhomogeneous":
        print("[warning] --periodic with --thermal nonhomogeneous: theta is being treated "
              "as periodic module-to-module (theta_inlet == theta_outlet), which is only "
              "valid if there's no net heat addition per module. Double-check this matches "
              "your CFD thermal setup before trusting these results.")

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

    if args.periodic:
        print(f"Periodic mode: delta_p={args.delta_p}")
        xb_wall, yb_wall = xb[is_wall], yb[is_wall]
        xb_in_p, yb_in_p, xb_out_p, yb_out_p, xb_pin, yb_pin = build_periodic_pairs(
            xb, yb, is_inlet, is_outlet
        )
        theta_wall_target_periodic = theta_wall_target  # same precomputed tensor, still keyed to is_wall order

    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        if args.batch_size and args.batch_size < xc.shape[0]:
            idx = torch.randperm(xc.shape[0], device=device)[: args.batch_size]
            xc_b, yc_b = xc[idx], yc[idx]
        else:
            xc_b, yc_b = xc, yc

        optimizer.zero_grad()

        Lg, g_parts = governing_losses(model, xc_b, yc_b, args.Re, args.Pr, w_thermal=w_thermal)

        if args.periodic:
            LBC, b_parts = periodic_boundary_losses(
                model, xb_wall, yb_wall, theta_wall_target_periodic,
                xb_in_p, yb_in_p, xb_out_p, yb_out_p, xb_pin, yb_pin,
                args.delta_p, thermal_mode=args.thermal, w_thermal=w_thermal,
            )
        else:
            LBC, b_parts = boundary_losses(
                model, xb, yb, nxb, nyb, is_inlet, is_wall, is_outlet,
                theta_wall_target, args.Re, args.Pr,
                thermal_mode=args.thermal, w_thermal=w_thermal,
            )

        Lp = Lg + args.beta * LBC  # Eq. (27)

        Lp.backward()
        optimizer.step()
        scheduler.step(Lp.item())

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
    p.add_argument("--thermal", type=str, default="none",
                    choices=["none", "homogeneous", "nonhomogeneous"],
                    help="Thermal BC mode (Sec. V.B.1 / V.B.2, or 'none' for hydrodynamics-only)")
    p.add_argument("--hidden_layers", type=int, default=6, help="Table I: 6 for wavy, 5 for plane")
    p.add_argument("--hidden_units", type=int, default=100, help="Table I: 100 for wavy, 80 for plane")
    p.add_argument("--beta", type=float, default=10.0, help="Boundary-loss weight (Eq. 27)")
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=0, help="0 = full-batch (as in the paper)")
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--lr_patience", type=int, default=2000)
    p.add_argument("--lr_gamma", type=float, default=0.8)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="runs/default")
    p.add_argument("--periodic", action="store_true",
                    help="Use periodic (pressure-drop-driven) boundary conditions instead of "
                         "fixed-inlet-velocity ones -- matches a COMSOL Periodic Flow Condition "
                         "+ Pressure Point Constraint reference dataset.")
    p.add_argument("--delta_p", type=float, default=None,
                    help="Imposed pressure drop inlet->outlet (required if --periodic). Must "
                         "correspond to the SAME physical flow as --Re -- these are coupled, "
                         "not independent choices.")
    return p



if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)


