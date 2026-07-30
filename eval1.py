

"""
Loads a trained model checkpoint and a reference CSV of (x, y, u, v, p)
[and optionally theta] -- e.g. exported from your CFD solution or a held-out
validation set -- evaluates the PINN at those exact (x, y) points, and
reports the same style of comparison the paper uses in Sec. V (Figs. 13-19,
34): per-field error metrics, parity/scatter plots, and CFD-vs-PINN-vs-error
contour panels.

python eval1.py --checkpoint runs/sym_Re10_beta2/model_final.pt --ref_csv data/eval/cfd_sym.csv --out_dir runs/sym_Re10_beta2/eval
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from train import PINN, velocity_from_psi

# ========================================================================
# Model loading
# ========================================================================
def load_model(checkpoint_path, hidden_layers, hidden_units, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        # periodic checkpoint saved with training args
        saved_args = ckpt.get("args", {})
        hidden_layers = saved_args.get("hidden_layers", hidden_layers)
        hidden_units = saved_args.get("hidden_units", hidden_units)
        state_dict = ckpt["model_state"]
    else:
        # plain state_dict (e.g. model_final.pt) -- caller must supply architecture
        state_dict = ckpt

    model = PINN(hidden_layers=hidden_layers, hidden_units=hidden_units).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, hidden_layers, hidden_units


# ========================================================================
# Inference (batched, autograd enabled only for the u,v derivative)
# ========================================================================
def predict(model, x_np, y_np, device, batch_size=20000):
    """Returns u, v, p, theta as numpy arrays, evaluated at (x_np, y_np)."""
    n = len(x_np)
    u_out = np.empty(n, dtype=np.float64)
    v_out = np.empty(n, dtype=np.float64)
    p_out = np.empty(n, dtype=np.float64)
    theta_out = np.empty(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.tensor(x_np[start:end], dtype=torch.float32, device=device).view(-1, 1)
        y = torch.tensor(y_np[start:end], dtype=torch.float32, device=device).view(-1, 1)
        x.requires_grad_(True)
        y.requires_grad_(True)

        u, v, p, sxx, sxy, syy, theta = velocity_from_psi(model, x, y)

        u_out[start:end] = u.detach().cpu().numpy().flatten()
        v_out[start:end] = v.detach().cpu().numpy().flatten()
        p_out[start:end] = p.detach().cpu().numpy().flatten()
        theta_out[start:end] = theta.detach().cpu().numpy().flatten()

    return u_out, v_out, p_out, theta_out


# ========================================================================
# Metrics
# ========================================================================
def field_metrics(pred, true, name, eps=1e-8):
    """
    MAE / RMSE / max abs error, plus relative L2 error (||pred-true||2 /
    ||true||2). The paper explicitly prefers absolute error over pointwise
    relative error in recirculation zones because near-zero velocities blow
    up the denominator (Sec. V.A) -- we follow that: pointwise relative
    error is NOT reported per-point, only the aggregate relative-L2 norm,
    which is well-behaved even with local near-zero values.
    """
    err = pred - true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_err = float(np.max(abs_err))
    true_norm = np.linalg.norm(true)
    rel_l2 = float(np.linalg.norm(err) / true_norm) if true_norm > eps else float("nan")
    return {
        f"{name}_MAE": mae,
        f"{name}_RMSE": rmse,
        f"{name}_max_abs_err": max_err,
        f"{name}_rel_L2_err": rel_l2,
    }


# ========================================================================
# Plots
# ========================================================================
def contour_compare(x, y, true, pred, field_name, out_path):
    """3-panel CFD / PINN / |error| contour comparison, mirroring Figs. 13-19."""
    triang = tri.Triangulation(x, y)
    abs_err = np.abs(pred - true)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin, vmax = min(true.min(), pred.min()), max(true.max(), pred.max())

    for ax, data, title, cmap, vlim in zip(
        axes,
        [true, pred, abs_err],
        [f"Reference (CFD) {field_name}", f"PINN {field_name}", f"|error| {field_name}"],
        ["jet", "jet", "jet"],
        [(vmin, vmax), (vmin, vmax), (0, abs_err.max() if abs_err.max() > 0 else 1.0)],
    ):
        tpc = ax.tricontourf(triang, data, levels=100, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        fig.colorbar(tpc, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# def parity_plot(true, pred, field_name, out_path):
#     """Predicted vs. true scatter, with the y=x reference line."""
#     lo = min(true.min(), pred.min())
#     hi = max(true.max(), pred.max())
#     plt.figure(figsize=(5, 5))
#     plt.scatter(true, pred, s=3, alpha=0.4)
#     plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
#     plt.xlabel(f"Reference {field_name}")
#     plt.ylabel(f"PINN {field_name}")
#     plt.title(f"Parity plot -- {field_name}")
#     plt.axis("equal")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     plt.close()


# def error_histogram(true, pred, field_name, out_path):
#     err = pred - true
#     plt.figure(figsize=(5, 4))
#     plt.hist(err, bins=60)
#     plt.xlabel(f"error ({field_name}, PINN - reference)")
#     plt.ylabel("count")
#     plt.title(f"Error distribution -- {field_name}")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     plt.close()


# ========================================================================
# Main
# ========================================================================
def evaluate(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warning] CUDA requested but not available -- falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)

    model, hl, hu = load_model(args.checkpoint, args.hidden_layers, args.hidden_units, device)
    print(f"Loaded model ({hl} x {hu}) from {args.checkpoint}")

    ref = pd.read_csv(args.ref_csv)
    required_cols = {"x", "y", "u", "v", "p"}
    missing = required_cols - set(ref.columns)
    if missing:
        raise ValueError(f"Reference CSV is missing required columns: {missing}")

    x = ref["x"].values.astype(np.float64)
    y = ref["y"].values.astype(np.float64)
    u_true = ref["u"].values.astype(np.float64)
    v_true = ref["v"].values.astype(np.float64)
    p_true = ref["p"].values.astype(np.float64)
    has_theta = "theta" in ref.columns
    theta_true = ref["theta"].values.astype(np.float64) if has_theta else None

    u_pred, v_pred, p_pred, theta_pred = predict(model, x, y, device, batch_size=args.batch_size)

    # ---- metrics ----
    metrics = {}
    metrics.update(field_metrics(u_pred, u_true, "u"))
    metrics.update(field_metrics(v_pred, v_true, "v"))
    metrics.update(field_metrics(p_pred, p_true, "p"))
    speed_true = np.sqrt(u_true ** 2 + v_true ** 2)
    speed_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
    metrics.update(field_metrics(speed_pred, speed_true, "speed"))
    if has_theta:
        metrics.update(field_metrics(theta_pred, theta_true, "theta"))

    print(json.dumps(metrics, indent=2))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- per-point predictions + errors, saved for further analysis ----
    out_df = pd.DataFrame({
        "x": x, "y": y,
        "u_true": u_true, "u_pred": u_pred, "u_abs_err": np.abs(u_pred - u_true),
        "v_true": v_true, "v_pred": v_pred, "v_abs_err": np.abs(v_pred - v_true),
        "p_true": p_true, "p_pred": p_pred, "p_abs_err": np.abs(p_pred - p_true),
    })
    if has_theta:
        out_df["theta_true"] = theta_true
        out_df["theta_pred"] = theta_pred
        out_df["theta_abs_err"] = np.abs(theta_pred - theta_true)
    out_df.to_csv(os.path.join(args.out_dir, "predictions_vs_reference.csv"), index=False)

    # ---- plots ----
    fields = [("u", u_true, u_pred), ("v", v_true, v_pred), ("p", p_true, p_pred)]
    if has_theta:
        fields.append(("theta", theta_true, theta_pred))

    for name, true_vals, pred_vals in fields:
        try:
            contour_compare(x, y, true_vals, pred_vals, name,
                             os.path.join(args.out_dir, f"contour_{name}.png"))
        except Exception as e:
            print(f"[warning] contour plot failed for {name} (likely too few / degenerate " f"points for triangulation): {e}")

        # parity_plot(true_vals, pred_vals, name, os.path.join(args.out_dir, f"parity_{name}.png"))
        # error_histogram(true_vals, pred_vals, name, os.path.join(args.out_dir, f"error_hist_{name}.png"))

    print(f"Done. Outputs written to {args.out_dir}")
    return metrics, out_df


def build_argparser():
    p = argparse.ArgumentParser(description="Evaluate a trained wavy-channel PINN against reference CSV data")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model_final.pt or checkpoints/epoch_N.pt")
    p.add_argument("--ref_csv", type=str, required=True, help="CSV with columns x,y,u,v,p (theta optional)")
    p.add_argument("--hidden_layers", type=int, default=6, help="Only used if checkpoint is a plain state_dict (e.g. model_final.pt)")
    p.add_argument("--hidden_units", type=int, default=100, help="Only used if checkpoint is a plain state_dict (e.g. model_final.pt)")
    p.add_argument("--batch_size", type=int, default=16384)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="eval_out")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    evaluate(args)
