

"""
python eval2.py --model_path runs/sym_Re10_beta2/model_final.pt --cfd_data data/eval/cfd_sym.csv --geom_dir data/sym --out_dir eval_sym
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ----------------------------- Geometry (same as generator) ----------------------------
L = 8.0
a = 1.0
B = 2.0 * np.pi / L
H_min = 12/7
x_min = -L/2
x_max = L/2

def y_top(x):
    x_orig = x - x_min
    return a * np.sin(B * x_orig + np.pi/2) + (H_min/2 + a)

def y_bottom(x, phi=0.0):
    x_orig = x - x_min
    return a * np.sin(B * x_orig + np.pi/2 + phi) - (H_min/2 + a)

# ----------------------------- PINN model (must match training) ------------------------------
class PINN(torch.nn.Module):
    def __init__(self, hidden_layers=6, hidden_units=100):
        super().__init__()
        layers = [torch.nn.Linear(2, hidden_units), torch.nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [torch.nn.Linear(hidden_units, hidden_units), torch.nn.Tanh()]
        layers += [torch.nn.Linear(hidden_units, 6)]  # psi, p, sxx, sxy, syy, theta
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        out = self.net(inp)
        psi, p, sxx, sxy, syy, theta = torch.split(out, 1, dim=1)
        return psi, p, sxx, sxy, syy, theta

def velocity_from_psi(model, x, y):
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    psi, p, sxx, sxy, syy, theta = model(x, y)
    grad_psi = torch.autograd.grad(psi, (y, x), grad_outputs=torch.ones_like(psi), create_graph=False)
    u = grad_psi[0]
    v = -grad_psi[1]
    return u.detach(), v.detach(), p.detach()

# ----------------------------- Main evaluation -----------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load CFD data
    cfd_df = pd.read_csv(args.cfd_data)
    # expected columns: x, y, u, v, p (or user can specify)
    x_cfd = cfd_df['x'].values
    y_cfd = cfd_df['y'].values
    u_cfd = cfd_df['u'].values
    v_cfd = cfd_df['v'].values
    p_cfd = cfd_df['p'].values if 'p' in cfd_df.columns else None

    # 2. Load trained model
    model = PINN(hidden_layers=args.hidden_layers, hidden_units=args.hidden_units).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 3. Predict on CFD points (batch-wise)
    batch_size = 10000
    n_points = len(x_cfd)
    u_pred = np.zeros(n_points)
    v_pred = np.zeros(n_points)
    p_pred = np.zeros(n_points)

    for i in range(0, n_points, batch_size):
        x_batch = torch.tensor(x_cfd[i:i+batch_size], dtype=torch.float32, device=device).view(-1, 1)
        y_batch = torch.tensor(y_cfd[i:i+batch_size], dtype=torch.float32, device=device).view(-1, 1)
        u, v, p = velocity_from_psi(model, x_batch, y_batch)
        u_pred[i:i+batch_size] = u.cpu().numpy().flatten()
        v_pred[i:i+batch_size] = v.cpu().numpy().flatten()
        p_pred[i:i+batch_size] = p.cpu().numpy().flatten()

    # 4. Compute error metrics
    def compute_errors(pred, true, name):
        # Mask out near-zero denominators for relative error (optional)
        mask = np.abs(true) > 1e-6
        if np.sum(mask) == 0:
            rel_err = np.nan
        else:
            rel_err = np.mean(np.abs((pred[mask] - true[mask]) / true[mask]))
        l2_err = np.linalg.norm(pred - true) / np.linalg.norm(true)
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true)**2))
        max_abs = np.max(np.abs(pred - true))
        print(f"{name}:")
        print(f"  Mean Relative Error (where |true|>1e-6): {rel_err:.4e}")
        print(f"  L2 Relative Error: {l2_err:.4e}")
        print(f"  MAE: {mae:.4e}")
        print(f"  RMSE: {rmse:.4e}")
        print(f"  Max Absolute Error: {max_abs:.4e}")
        return {'rel': rel_err, 'l2': l2_err, 'mae': mae, 'rmse': rmse, 'max': max_abs}

    print("===== ERROR METRICS =====")
    err_u = compute_errors(u_pred, u_cfd, 'u (axial velocity)')
    err_v = compute_errors(v_pred, v_cfd, 'v (transverse velocity)')
    if p_cfd is not None:
        err_p = compute_errors(p_pred, p_cfd, 'p (pressure)')
    else:
        err_p = None

    # 5. Create comparison plots
    os.makedirs(args.out_dir, exist_ok=True)

    # Scatter plot of predicted vs true for u, v, p
    fig, axes = plt.subplots(1, 3 if p_cfd is not None else 2, figsize=(15, 5))
    if p_cfd is not None:
        ax_u, ax_v, ax_p = axes
    else:
        ax_u, ax_v = axes
        ax_p = None

    ax_u.scatter(u_cfd, u_pred, s=1, alpha=0.5)
    ax_u.plot([u_cfd.min(), u_cfd.max()], [u_cfd.min(), u_cfd.max()], 'r--')
    ax_u.set_xlabel('CFD u')
    ax_u.set_ylabel('PINN u')
    ax_u.set_title(f'u: MAE={err_u["mae"]:.2e}')

    ax_v.scatter(v_cfd, v_pred, s=1, alpha=0.5)
    ax_v.plot([v_cfd.min(), v_cfd.max()], [v_cfd.min(), v_cfd.max()], 'r--')
    ax_v.set_xlabel('CFD v')
    ax_v.set_ylabel('PINN v')
    ax_v.set_title(f'v: MAE={err_v["mae"]:.2e}')

    if p_cfd is not None:
        ax_p.scatter(p_cfd, p_pred, s=1, alpha=0.5)
        ax_p.plot([p_cfd.min(), p_cfd.max()], [p_cfd.min(), p_cfd.max()], 'r--')
        ax_p.set_xlabel('CFD p')
        ax_p.set_ylabel('PINN p')
        ax_p.set_title(f'p: MAE={err_p["mae"]:.2e}')

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'scatter_comparison.png'), dpi=200)
    plt.close()

    # Velocity profiles at selected x locations
    # Use the original x grid from CFD to find indices near chosen x0
    x_locations = np.linspace(x_min, x_max, 5)  # e.g., [-4, -2, 0, 2, 4]
    fig, axes = plt.subplots(1, len(x_locations), figsize=(4*len(x_locations), 4))
    if len(x_locations) == 1:
        axes = [axes]
    for ax, x0 in zip(axes, x_locations):
        # Find points within a narrow band around x0
        tol = 0.1
        mask = np.abs(x_cfd - x0) < tol
        if np.sum(mask) < 10:
            continue
        # Sort by y
        idx = np.argsort(y_cfd[mask])
        y_sel = y_cfd[mask][idx]
        u_cfd_sel = u_cfd[mask][idx]
        u_pred_sel = u_pred[mask][idx]
        ax.plot(u_cfd_sel, y_sel, 'b-', label='CFD', linewidth=2)
        ax.plot(u_pred_sel, y_sel, 'r--', label='PINN', linewidth=2)
        ax.set_title(f'x = {x0:.2f}')
        ax.set_xlabel('u')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'velocity_profiles.png'), dpi=200)
    plt.close()

    # Optional: contour comparison for u
    # We'll create a grid from the CFD points if they are structured, else we can interpolate.
    # Simple approach: use the unique x and y from CFD if it's a structured grid.
    # We'll assume the CFD data is on a rectangular grid; we can try to reshape.
    # If not, we'll skip contour plotting.

    try:
        # Attempt to find unique x and y coordinates and create a structured grid
        x_unique = np.unique(x_cfd)
        y_unique = np.unique(y_cfd)
        if len(x_unique) * len(y_unique) == len(x_cfd):
            # Structured grid
            X = x_unique
            Y = y_unique
            U_cfd_grid = u_cfd.reshape(len(Y), len(X)).T
            V_cfd_grid = v_cfd.reshape(len(Y), len(X)).T
            U_pred_grid = u_pred.reshape(len(Y), len(X)).T
            V_pred_grid = v_pred.reshape(len(Y), len(X)).T
            if p_cfd is not None:
                P_cfd_grid = p_cfd.reshape(len(Y), len(X)).T
                P_pred_grid = p_pred.reshape(len(Y), len(X)).T
            else:
                P_cfd_grid = P_pred_grid = None

            # Plot contours
            fig, axes = plt.subplots(2, 3 if p_cfd is not None else 2, figsize=(18, 10))
            # u
            im1 = axes[0,0].contourf(X, Y, U_cfd_grid, levels=20, cmap='viridis')
            axes[0,0].set_title('CFD u')
            plt.colorbar(im1, ax=axes[0,0])
            im2 = axes[0,1].contourf(X, Y, U_pred_grid, levels=20, cmap='viridis')
            axes[0,1].set_title('PINN u')
            plt.colorbar(im2, ax=axes[0,1])
            im3 = axes[0,2].contourf(X, Y, np.abs(U_cfd_grid - U_pred_grid), levels=20, cmap='hot')
            axes[0,2].set_title('|u error|')
            plt.colorbar(im3, ax=axes[0,2])
            # v
            im4 = axes[1,0].contourf(X, Y, V_cfd_grid, levels=20, cmap='RdBu_r')
            axes[1,0].set_title('CFD v')
            plt.colorbar(im4, ax=axes[1,0])
            im5 = axes[1,1].contourf(X, Y, V_pred_grid, levels=20, cmap='RdBu_r')
            axes[1,1].set_title('PINN v')
            plt.colorbar(im5, ax=axes[1,1])
            im6 = axes[1,2].contourf(X, Y, np.abs(V_cfd_grid - V_pred_grid), levels=20, cmap='hot')
            axes[1,2].set_title('|v error|')
            plt.colorbar(im6, ax=axes[1,2])
            # Add walls
            x_wall = np.linspace(x_min, x_max, 200)
            yt = y_top(x_wall)
            yb = y_bottom(x_wall, args.phi)
            for ax in axes.flatten():
                ax.plot(x_wall, yt, 'k-', linewidth=2)
                ax.plot(x_wall, yb, 'k-', linewidth=2)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_bottom(x_min, args.phi)-0.2, y_top(x_min)+0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, 'contour_comparison.png'), dpi=200)
            plt.close()
    except Exception as e:
        print(f"Contour plotting skipped: {e}")

    # Save predictions to CSV for further analysis
    pred_df = pd.DataFrame({
        'x': x_cfd, 'y': y_cfd,
        'u_pred': u_pred, 'v_pred': v_pred, 'p_pred': p_pred,
        'u_cfd': u_cfd, 'v_cfd': v_cfd
    })
    if p_cfd is not None:
        pred_df['p_cfd'] = p_cfd
    pred_df.to_csv(os.path.join(args.out_dir, 'predictions.csv'), index=False)

    print(f"\nEvaluation complete. Outputs saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--cfd_data", type=str, required=True, help="CSV with CFD results (columns: x,y,u,v,p)")
    parser.add_argument("--geom_dir", type=str, required=True, help="Path to geometry folder (for phi info)")
    parser.add_argument("--phi", type=float, default=0.0, help="Phase shift in degrees (used for wall plotting)")
    parser.add_argument("--out_dir", type=str, default="eval_results", help="Output directory for plots")
    parser.add_argument("--hidden_layers", type=int, default=6, help="Number of hidden layers (must match training)")
    parser.add_argument("--hidden_units", type=int, default=100, help="Number of neurons per layer")
    parser.add_argument("--Re", type=float, default=10, help="Reynolds number (not used, only for labelling)")
    args = parser.parse_args()
    main(args)

