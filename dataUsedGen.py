# # """
# # Collocation / boundary point generator for PINN training on the wavy-channel
# # geometry described in Roy, Mukherjee, Prasad & Nayak,
# # "A computational analysis of flow dynamics and heat transfer in a wavy
# # patterned channel using physics-informed neural networks", Phys. Fluids 37,
# # 043610 (2025).

# # Fixes applied relative to the previous version:
# #   1. Core collocation points are now Latin-Hypercube sampled (paper uses LHS
# #      throughout), not a structured grid.
# #   2. Boundary points are labeled by type (upper_wall / lower_wall / inlet /
# #      outlet) and saved to the CSV.
# #   3. Outward unit wall normals (nx, ny) are computed analytically and saved
# #      for every boundary point (needed for Neumann / WSS / heat-flux terms).
# #   4. The boundary-layer thickness used for near-wall refinement is now a
# #      fraction of the *local* channel height H(x) = y_top(x) - y_bottom(x),
# #      instead of a fixed constant -- so it adapts to the phase shift.
# #   5. Both collocation and boundary point counts are guaranteed to exactly
# #      match the requested N_total (quota-based rejection sampling with a
# #      max-iteration safety guard, rather than an open-ended while loop).
# #   6. Duplicate corner points at x=x_min and x=x_max are removed: wall curves exclude
# #      their endpoints, and the inlet/outlet columns own the corner points.
# # """

# # import numpy as np
# # import os
# # import matplotlib.pyplot as plt
# # from scipy.stats import qmc
# # from scipy.integrate import cumulative_trapezoid


# # x_min = -4.0            # domain start
# # x_max = 4.0             # domain end
# # L = x_max - x_min       # module length (8.0)
# # a = 1.0                 # amplitude
# # B = 2.0 * np.pi / L     # wave number
# # H_min = 12/7            # minimum channel height, by solving the two eqns


# # def y_top(x):
# #     return a * np.sin(B * x + np.pi / 2) + (H_min / 2 + a)


# # def dy_top_dx(x):
# #     return a * B * np.cos(B * x + np.pi / 2)


# # def y_bottom(x, phi=0.0):
# #     return a * np.sin(B * x + 1.5*np.pi + phi) - (H_min / 2 + a)


# # def dy_bot_dx(x, phi=0.0):
# #     return a * B * np.cos(B * x + 1.5*np.pi + phi)


# # def local_height(x, phi=0.0):
# #     """Instantaneous channel height H(x) = y_top(x) - y_bottom(x, phi)."""
# #     return y_top(x) - y_bottom(x, phi)


# # def local_delta_bl(x, phi, frac=0.15):
# #     """Boundary-layer band thickness, scaled by the local channel height
# #     instead of a fixed constant (fix #4)."""
# #     return frac * local_height(x, phi)


# # def wall_normal_top(x):
# #     """Outward normal for the upper wall (points away from the fluid,
# #     i.e. generally +y)."""
# #     dy = dy_top_dx(x)
# #     n = np.column_stack((-dy, np.ones_like(dy)))
# #     n /= np.linalg.norm(n, axis=1, keepdims=True)
# #     return n


# # def wall_normal_bottom(x, phi):
# #     """Outward normal for the lower wall (points away from the fluid,
# #     i.e. generally -y)."""
# #     dy = dy_bot_dx(x, phi)
# #     n = np.column_stack((dy, -np.ones_like(dy)))
# #     n /= np.linalg.norm(n, axis=1, keepdims=True)
# #     return n


# # def sample_curve_arc_length(y_func, dy_func, n_pts, phi=0.0):
# #     """Uniformly samples along a curve's actual perimeter using arc-length
# #     parameterization."""
# #     x_dense = np.linspace(x_min, x_max, 10000)
# #     dy = dy_func(x_dense, phi) if phi != 0.0 else dy_func(x_dense)
# #     ds = np.sqrt(1 + dy ** 2)

# #     s_cum = np.zeros_like(x_dense)
# #     s_cum[1:] = cumulative_trapezoid(ds, x_dense)
# #     total_length = s_cum[-1]

# #     s_targets = np.linspace(0, total_length, n_pts)
# #     x_sampled = np.interp(s_targets, s_cum, x_dense)
# #     y_sampled = y_func(x_sampled, phi) if phi != 0.0 else y_func(x_sampled)

# #     return np.column_stack((x_sampled, y_sampled)), total_length


# # def generate_boundary_points(N_total, phi=0.0):
# #     """Generates labeled boundary points (with outward normals) distributed
# #     proportionally across the four boundary segments (top, bottom, inlet,
# #     outlet), with exact total count and no duplicated corner points."""
# #     _, len_top = sample_curve_arc_length(y_top, dy_top_dx, 200)
# #     _, len_bot = sample_curve_arc_length(y_bottom, dy_bot_dx, 200, phi)
# #     len_inlet = y_top(x_min) - y_bottom(x_min, phi)
# #     len_outlet = y_top(x_max) - y_bottom(x_max, phi)

# #     total_perimeter = len_top + len_bot + len_inlet + len_outlet

# #     n_top = int(round(N_total * (len_top / total_perimeter)))
# #     n_bot = int(round(N_total * (len_bot / total_perimeter)))
# #     n_inlet = int(round(N_total * (len_inlet / total_perimeter)))
# #     n_outlet = N_total - n_top - n_bot - n_inlet
# #     if n_outlet < 1:
# #         raise ValueError("N_total too small for proportional allocation; increase N_total.")


# #     top_full, _ = sample_curve_arc_length(y_top, dy_top_dx, n_top + 2)
# #     top_pts = top_full[1:-1]
# #     top_normals = wall_normal_top(top_pts[:, 0])

# #     bot_full, _ = sample_curve_arc_length(y_bottom, dy_bot_dx, n_bot + 2, phi)
# #     bot_pts = bot_full[1:-1]
# #     bot_normals = wall_normal_bottom(bot_pts[:, 0], phi)

# #     y_in = np.linspace(y_bottom(x_min, phi), y_top(x_min), n_inlet)
# #     inlet_pts = np.column_stack((np.full(n_inlet, x_min), y_in))
# #     inlet_normals = np.tile([-1.0, 0.0], (n_inlet, 1))

# #     y_out = np.linspace(y_bottom(x_max, phi), y_top(x_max), n_outlet)
# #     outlet_pts = np.column_stack((np.full(n_outlet, x_max), y_out))
# #     outlet_normals = np.tile([1.0, 0.0], (n_outlet, 1))

# #     pts = np.vstack((top_pts, bot_pts, inlet_pts, outlet_pts))
# #     normals = np.vstack((top_normals, bot_normals, inlet_normals, outlet_normals))
# #     labels = (
# #         ["upper_wall"] * len(top_pts)
# #         + ["lower_wall"] * len(bot_pts)
# #         + ["inlet"] * len(inlet_pts)
# #         + ["outlet"] * len(outlet_pts)
# #     )
# #     assert pts.shape[0] == N_total, "Boundary point count mismatch"
# #     return pts, normals, np.array(labels)


# # def generate_collocation_points(N_total, phi=0.0, bl_thickness_frac=0.20, bl_ratio=0.50, seed=42, max_iter=500):
# #     """
# #     Generates internal points using Latin Hypercube Sampling everywhere
# #     (matching the paper), split into:
# #       - Zone 1 (core): points farther than the local BL band from any wall.
# #       - Zone 2 (boundary layer): points within a height-adaptive band of
# #         any of the 4 boundaries.

# #     Both zone counts are filled to their EXACT target via quota-based
# #     rejection sampling in batches, with a max_iter safety cap so the loop
# #     cannot spin forever if delta_bl becomes very small (fix #7).
# #     """
# #     rng_seed = seed
# #     sampler = qmc.LatinHypercube(d=2, seed=rng_seed)

# #     n_bl_target = int(N_total * bl_ratio)
# #     n_core_target = N_total - n_bl_target

# #     core_pts = []
# #     bl_pts = []

# #     batch_size = max(N_total, 5000)
# #     it = 0
# #     while (len(core_pts) < n_core_target or len(bl_pts) < n_bl_target) and it < max_iter:
# #         it += 1
# #         samples = sampler.random(n=batch_size)
# #         x_cand = x_min + samples[:, 0] * (x_max - x_min)
# #         yb_cand = y_bottom(x_cand, phi)
# #         yt_cand = y_top(x_cand)
# #         y_cand = yb_cand + samples[:, 1] * (yt_cand - yb_cand)

# #         delta = local_delta_bl(x_cand, phi, bl_thickness_frac)

# #         d_top = yt_cand - y_cand
# #         d_bot = y_cand - yb_cand
# #         d_in = x_cand - x_min
# #         d_out = x_max - x_cand

# #         is_bl = (
# #             (d_top <= delta)
# #             | (d_bot <= delta)
# #             | (d_in <= delta * 0.5)
# #             | (d_out <= delta * 0.5)
# #         )

# #         if len(core_pts) < n_core_target:
# #             core_cand = np.column_stack((x_cand[~is_bl], y_cand[~is_bl]))
# #             need = n_core_target - len(core_pts)
# #             core_pts.extend(core_cand[:need].tolist())

# #         if len(bl_pts) < n_bl_target:
# #             bl_cand = np.column_stack((x_cand[is_bl], y_cand[is_bl]))
# #             need = n_bl_target - len(bl_pts)
# #             bl_pts.extend(bl_cand[:need].tolist())

# #     if len(core_pts) < n_core_target or len(bl_pts) < n_bl_target:
# #         print(
# #             f"  [warning] hit max_iter={max_iter} before filling quotas "
# #             f"(core: {len(core_pts)}/{n_core_target}, "
# #             f"bl: {len(bl_pts)}/{n_bl_target}). "
# #             "Consider raising bl_thickness_frac, batch_size, or max_iter."
# #         )

# #     core_pts = np.array(core_pts[:n_core_target])
# #     bl_pts = np.array(bl_pts[:n_bl_target])

# #     zone_labels = np.array(["core"] * len(core_pts) + ["bl"] * len(bl_pts))
# #     all_pts = np.vstack((core_pts, bl_pts))
# #     return all_pts, zone_labels


# # def plot_geometry_check(phis_deg, out_dir):
# #     x_dense = np.linspace(x_min, x_max, 400)

# #     fig, axes = plt.subplots(3, 3, figsize=(12, 12))
# #     axes = axes.ravel()

# #     for ax, phi_deg in zip(axes, phis_deg):
# #         phi = np.radians(phi_deg)
# #         ax.plot(x_dense, y_top(x_dense), "k-")
# #         ax.plot(x_dense, y_bottom(x_dense, phi), "k-")
# #         ax.set_title(f"$\\phi$ = {phi_deg}°")
# #         ax.set_aspect("equal", adjustable="box")

# #     for ax in axes[len(phis_deg):]:
# #         ax.axis("off")

# #     plt.tight_layout()
# #     plt.savefig(os.path.join(out_dir, "geometry_check.png"), dpi=200)
# #     plt.close()


# # if __name__ == "__main__":
# #     phis_deg = [0, 30, 60, 90, 180, -30, -60, -90]
# #     labels = ["sym" if p == 0 else f"phi_{p}" for p in phis_deg]

# #     out_dir = "points"      
# #     os.makedirs(out_dir, exist_ok=True)
# #     plot_geometry_check(phis_deg, out_dir)

# #     for phi_deg, label in zip(phis_deg, labels):
# #         phi = np.radians(phi_deg)

# #         if phi_deg == 0:
# #             n_colloc = 62000
# #             n_bound = 8000
# #         else:
# #             n_colloc = 89600
# #             n_bound = 9000

# #         print(f"Generating {label}: colloc={n_colloc}, boundary={n_bound}")

# #         colloc, zone_labels = generate_collocation_points(n_colloc, phi, seed=42)
# #         boundary, normals, bnd_labels = generate_boundary_points(n_bound, phi)

# #         subdir = os.path.join(out_dir, label)
# #         os.makedirs(subdir, exist_ok=True)

# #         colloc_out = np.column_stack((colloc, zone_labels))
# #         np.savetxt(
# #             os.path.join(subdir, "collocation.csv"),
# #             colloc_out,
# #             delimiter=",",
# #             header="x,y,zone",
# #             comments="",
# #             fmt="%s",
# #         )

# #         boundary_out = np.column_stack((boundary, normals, bnd_labels))
# #         np.savetxt(
# #             os.path.join(subdir, "boundary.csv"),
# #             boundary_out,
# #             delimiter=",",
# #             header="x,y,nx,ny,type",
# #             comments="",
# #             fmt="%s",
# #         )

# #         colloc_f = colloc.astype(float)
# #         plt.figure(figsize=(10, 6))
# #         is_core = zone_labels == "core"

# #         plt.scatter(colloc_f[is_core][::5, 0], colloc_f[is_core][::5, 1],
# #             s=0.5, c="g", alpha=0.4, label="core (subsampled)")

# #         plt.scatter(colloc_f[~is_core][::2, 0], colloc_f[~is_core][::2, 1],
# #             s=0.5, c="g", alpha=0.5, label="boundary layer (subsampled)")

# #         boundary_f = boundary.astype(float)
# #         plt.scatter(boundary_f[:, 0], boundary_f[:, 1], s=2, c="r", label="boundary")
# #         plt.title(f"Channel Geometry: {label}")
# #         plt.axis("equal")
# #         plt.legend(loc="upper right", fontsize=7)
# #         plt.tight_layout()
# #         plt.savefig(os.path.join(subdir, "collocation.png"), dpi=200)
# #         plt.close()

# #     print("All point sets generated successfully!")
    


# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # ==============================================================================
# # Geometry Wall Profiles
# # ==============================================================================
# def get_wall_bounds(x, mode="symmetric", L=8.0):
#     """
#     Returns (y_lower, y_upper) boundaries at coordinate x.
#     L = channel length (default 8.0, x in [-4, 4])
#     """
#     if mode == "symmetric":
#         # Height ranges from H_min = 12/7 (~1.714) at ends to H_max = 5.6 at x=0
#         h_min = 12.0 / 7.0
#         h_max = 5.6
#         h = h_min + (h_max - h_min) * (np.cos(np.pi * x / L) ** 2)
#         y_upper = 0.5 * h
#         y_lower = -0.5 * h

#     elif mode == "asymmetric":
#         # Sinusoidal wavy channel
#         a = 1.0
#         b = 0.9
#         y_upper = a + b * np.cos(2.0 * np.pi * x / L)
#         y_lower = -a + b * np.sin(2.0 * np.pi * x / L)
        
#     return y_lower, y_upper


# # ==============================================================================
# # Point Sampling Generator
# # ==============================================================================
# def generate_dataset(
#     mode="symmetric",
#     L=8.0,
#     x_min=-4.0,
#     x_max=4.0,
#     # Core Grid parameters
#     nx_core=40,
#     ny_core=30,
#     # Boundary Layer Refinement parameters
#     n_candidate_dense=60000,
#     strip_width_x=0.35,      # Width of dense strip near inlet/outlet
#     strip_width_y=0.22,      # Thickness of dense strip near top/bottom walls
#     # Surface Boundary points count
#     n_wall_pts=400,
#     n_inlet_pts=100,
#     n_outlet_pts=100,
#     seed=42
# ):
#     np.random.seed(seed)

#     # --------------------------------------------------------------------------
#     # 1. CORE INTERNAL POINTS (Regular Grid)
#     # --------------------------------------------------------------------------
#     x_grid = np.linspace(x_min, x_max, nx_core)
#     y_grid = np.linspace(-3.0, 3.0, ny_core)
#     X_g, Y_g = np.meshgrid(x_grid, y_grid)
#     X_flat, Y_flat = X_g.flatten(), Y_g.flatten()

#     y_low_g, y_up_g = get_wall_bounds(X_flat, mode=mode, L=L)
    
#     # Keep grid points strictly inside the interior (slightly away from walls)
#     margin = 0.05
#     core_mask = (Y_flat >= y_low_g + margin) & (Y_flat <= y_up_g - margin)
#     x_core = X_flat[core_mask]
#     y_core = Y_flat[core_mask]

#     # --------------------------------------------------------------------------
#     # 2. DENSE REFINEMENT STRIPS (Random Sampling in Boundary Layers)
#     # --------------------------------------------------------------------------
#     x_rand = np.random.uniform(x_min, x_max, n_candidate_dense)
#     y_rand = np.random.uniform(-3.0, 3.0, n_candidate_dense)

#     y_low_r, y_up_r = get_wall_bounds(x_rand, mode=mode, L=L)

#     # Condition A: Must be physically inside the channel
#     inside_channel = (y_rand >= y_low_r) & (y_rand <= y_up_r)

#     # Condition B: Must lie within a boundary strip
#     near_inlet      = (x_rand <= x_min + strip_width_x)
#     near_outlet     = (x_rand >= x_max - strip_width_x)
#     near_upper_wall = (y_rand >= y_up_r - strip_width_y)
#     near_lower_wall = (y_rand <= y_low_r + strip_width_y)

#     # Union (OR) of all boundary strips -> Creates natural corner overlaps
#     strip_mask = inside_channel & (near_inlet | near_outlet | near_upper_wall | near_lower_wall)

#     x_strip = x_rand[strip_mask]
#     y_strip = y_rand[strip_mask]

#     # Combine Core + Refinement Strip -> Internal Collocation Points
#     xc = np.concatenate([x_core, x_strip])
#     yc = np.concatenate([y_core, y_strip])

#     # --------------------------------------------------------------------------
#     # 3. BOUNDARY SURFACE POINTS
#     # --------------------------------------------------------------------------
#     # Upper Wall
#     x_wall_u = np.linspace(x_min, x_max, n_wall_pts // 2)
#     _, y_wall_u = get_wall_bounds(x_wall_u, mode=mode, L=L)

#     # Lower Wall
#     x_wall_l = np.linspace(x_min, x_max, n_wall_pts // 2)
#     y_wall_l, _ = get_wall_bounds(x_wall_l, mode=mode, L=L)

#     xb_wall = np.concatenate([x_wall_u, x_wall_l])
#     yb_wall = np.concatenate([y_wall_u, y_wall_l])
#     type_wall = np.array(["upper_wall"] * (n_wall_pts // 2) + ["lower_wall"] * (n_wall_pts // 2))

#     # Inlet
#     xb_inlet = np.full(n_inlet_pts, x_min)
#     y_low_in, y_up_in = get_wall_bounds(x_min, mode=mode, L=L)
#     yb_inlet = np.linspace(y_low_in, y_up_in, n_inlet_pts)
#     type_inlet = np.array(["inlet"] * n_inlet_pts)

#     # Outlet
#     xb_outlet = np.full(n_outlet_pts, x_max)
#     y_low_out, y_up_out = get_wall_bounds(x_max, mode=mode, L=L)
#     yb_outlet = np.linspace(y_low_out, y_up_out, n_outlet_pts)
#     type_outlet = np.array(["outlet"] * n_outlet_pts)

#     # Combine Boundary Arrays
#     xb = np.concatenate([xb_wall, xb_inlet, xb_outlet])
#     yb = np.concatenate([yb_wall, yb_inlet, yb_outlet])
#     btypes = np.concatenate([type_wall, type_inlet, type_outlet])

#     # Compute Outward Normal Vectors (nx, ny)
#     nxb = np.zeros_like(xb)
#     nyb = np.zeros_like(yb)

#     # Inlet normals pointing left (-1, 0)
#     nxb[btypes == "inlet"] = -1.0
#     nyb[btypes == "inlet"] = 0.0

#     # Outlet normals pointing right (+1, 0)
#     nxb[btypes == "outlet"] = 1.0
#     nyb[btypes == "outlet"] = 0.0

#     # Wall normals via numerical derivatives
#     dx = 1e-4
#     # Upper Wall
#     mask_u = (btypes == "upper_wall")
#     x_u = xb[mask_u]
#     _, y_u_plus = get_wall_bounds(x_u + dx, mode=mode, L=L)
#     _, y_u_minus = get_wall_bounds(x_u - dx, mode=mode, L=L)
#     dydx_u = (y_u_plus - y_u_minus) / (2 * dx)
#     norm_u = np.sqrt(dydx_u**2 + 1.0)
#     nxb[mask_u] = -dydx_u / norm_u
#     nyb[mask_u] = 1.0 / norm_u

#     # Lower Wall
#     mask_l = (btypes == "lower_wall")
#     x_l = xb[mask_l]
#     y_l_plus, _ = get_wall_bounds(x_l + dx, mode=mode, L=L)
#     y_l_minus, _ = get_wall_bounds(x_l - dx, mode=mode, L=L)
#     dydx_l = (y_l_plus - y_l_minus) / (2 * dx)
#     norm_l = np.sqrt(dydx_l**2 + 1.0)
#     nxb[mask_l] = dydx_l / norm_l
#     nyb[mask_l] = -1.0 / norm_l

#     return (xc, yc), (xb, yb, nxb, nyb, btypes)


# # ==============================================================================
# # Save and Plot
# # ==============================================================================
# def save_and_plot():
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

#     modes = ["symmetric", "asymmetric"]
#     sublabels = ["(a)", "(b)"]

#     for i, mode in enumerate(modes):
#         out_dir = f"data/{mode}"
#         os.makedirs(out_dir, exist_ok=True)

#         (xc, yc), (xb, yb, nxb, nyb, btypes) = generate_dataset(mode=mode)

#         # Save Collocation CSV
#         df_colloc = pd.DataFrame({"x": xc, "y": yc, "zone": "internal"})
#         df_colloc.to_csv(os.path.join(out_dir, "collocation.csv"), index=False)

#         # Save Boundary CSV
#         df_bnd = pd.DataFrame({"x": xb, "y": yb, "nx": nxb, "ny": nyb, "type": btypes})
#         df_bnd.to_csv(os.path.join(out_dir, "boundary.csv"), index=False)

#         # Plotting to match Paper Figure
#         ax = axes[i]
        
#         # Internal points (green dots)
#         ax.scatter(xc, yc, s=1.2, c="green", alpha=0.6, label="internal points")

#         # Boundary points
#         mask_w = (btypes == "upper_wall") | (btypes == "lower_wall")
#         mask_in = (btypes == "inlet")
#         mask_out = (btypes == "outlet")

#         ax.scatter(xb[mask_w], yb[mask_w], s=4, c="red", label="wall points")
#         ax.scatter(xb[mask_in], yb[mask_in], s=4, c="blue", label="inlet points")
#         ax.scatter(xb[mask_out], yb[mask_out], s=4, c="black", label="outlet points")

#         ax.set_aspect("equal")
#         ax.axis("off")
#         ax.set_title(sublabels[i], y=-0.15, fontsize=14)
#         ax.legend(loc="upper right", markerscale=2, fontsize=9)

#     plt.tight_layout()
#     plt.savefig("replicated_point_distribution.png", bbox_inches="tight")
#     plt.show()

# if __name__ == "__main__":
#     save_and_plot()



"""
Collocation / boundary point generator for PINN training on the wavy-channel
geometry (Roy, Mukherjee, Prasad & Nayak, Phys. Fluids 37, 043610 (2025)).

Changes vs. previous version (PAPER-FIX):
  * Boundary-layer refinement now uses a CONSTANT strip thickness (does NOT
    scale with local channel height H(x) and does NOT depend on curvature).
  * Wall curves are sampled WITH their endpoints (x_min and x_max).
  * Inlet/outlet columns are sampled WITH their endpoints (y_bottom and y_top).
  * This produces the overlapping corner points visible in the paper figures
    (each of the 4 corners is owned by BOTH a wall AND inlet/outlet).
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.integrate import cumulative_trapezoid


# ----------------------------- geometry -------------------------------------
x_min = -4.0
x_max = 4.0
L = x_max - x_min
a = 1.0
B = 2.0 * np.pi / L
H_min = 12.0 / 7.0

# PAPER-FIX #1: constant boundary-layer thickness (independent of x, of
# curvature, and of local channel height). Tune this single number to match
# the strip width seen in the paper figure.
DELTA_BL = 0.35          # same band along upper wall, lower wall, inlet, outlet
DELTA_BL_IO = 0.35       # x-direction band at inlet/outlet (same value)


def y_top(x):
    return a * np.sin(B * x + np.pi / 2) + (H_min / 2 + a)


def dy_top_dx(x):
    return a * B * np.cos(B * x + np.pi / 2)


def y_bottom(x, phi=0.0):
    return a * np.sin(B * x + 1.5 * np.pi + phi) - (H_min / 2 + a)


def dy_bot_dx(x, phi=0.0):
    return a * B * np.cos(B * x + 1.5 * np.pi + phi)


def wall_normal_top(x):
    dy = dy_top_dx(x)
    n = np.column_stack((-dy, np.ones_like(dy)))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n


def wall_normal_bottom(x, phi):
    dy = dy_bot_dx(x, phi)
    n = np.column_stack((dy, -np.ones_like(dy)))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n


def sample_curve_arc_length(y_func, dy_func, n_pts, phi=0.0, include_endpoints=True):
    """Arc-length-uniform sampling. By default KEEPS endpoints so that
    corner points are shared with the inlet/outlet columns (PAPER-FIX #2)."""
    x_dense = np.linspace(x_min, x_max, 10000)
    dy = dy_func(x_dense, phi) if phi != 0.0 else dy_func(x_dense)
    ds = np.sqrt(1 + dy ** 2)

    s_cum = np.zeros_like(x_dense)
    s_cum[1:] = cumulative_trapezoid(ds, x_dense)
    total_length = s_cum[-1]

    if include_endpoints:
        s_targets = np.linspace(0.0, total_length, n_pts)
    else:
        # interior-only sampling (old behaviour)
        seg = total_length / (n_pts + 1)
        s_targets = seg + np.arange(n_pts) * seg

    x_sampled = np.interp(s_targets, s_cum, x_dense)
    y_sampled = y_func(x_sampled, phi) if phi != 0.0 else y_func(x_sampled)
    return np.column_stack((x_sampled, y_sampled)), total_length


# --------------------------- boundary points --------------------------------
def generate_boundary_points(N_total, phi=0.0):
    """Wall points sampled WITH endpoints, inlet/outlet columns sampled WITH
    endpoints -> corners are overlapped (PAPER-FIX #2)."""
    _, len_top = sample_curve_arc_length(y_top, dy_top_dx, 200)
    _, len_bot = sample_curve_arc_length(y_bottom, dy_bot_dx, 200, phi)
    len_inlet  = y_top(x_min) - y_bottom(x_min, phi)
    len_outlet = y_top(x_max) - y_bottom(x_max, phi)

    total_perimeter = len_top + len_bot + len_inlet + len_outlet
    n_top    = max(1, int(round(N_total * (len_top    / total_perimeter))))
    n_bot    = max(1, int(round(N_total * (len_bot    / total_perimeter))))
    n_inlet  = max(1, int(round(N_total * (len_inlet  / total_perimeter))))
    n_outlet = N_total - n_top - n_bot - n_inlet
    if n_outlet < 1:
        raise ValueError("N_total too small for proportional allocation.")

    # walls include their endpoints (corners belong to walls too)
    top_pts, _ = sample_curve_arc_length(y_top, dy_top_dx, n_top,
                                         include_endpoints=True)
    top_normals = wall_normal_top(top_pts[:, 0])

    bot_pts, _ = sample_curve_arc_length(y_bottom, dy_bot_dx, n_bot, phi,
                                         include_endpoints=True)
    bot_normals = wall_normal_bottom(bot_pts[:, 0], phi)

    # inlet/outlet columns include their endpoints (corners belong to them too)
    y_in = np.linspace(y_bottom(x_min, phi), y_top(x_min), n_inlet)
    inlet_pts = np.column_stack((np.full(n_inlet, x_min), y_in))
    inlet_normals = np.tile([-1.0, 0.0], (n_inlet, 1))

    y_out = np.linspace(y_bottom(x_max, phi), y_top(x_max), n_outlet)
    outlet_pts = np.column_stack((np.full(n_outlet, x_max), y_out))
    outlet_normals = np.tile([1.0, 0.0], (n_outlet, 1))

    pts     = np.vstack((top_pts, bot_pts, inlet_pts, outlet_pts))
    normals = np.vstack((top_normals, bot_normals, inlet_normals, outlet_normals))
    labels  = (["upper_wall"] * len(top_pts)
               + ["lower_wall"] * len(bot_pts)
               + ["inlet"]      * len(inlet_pts)
               + ["outlet"]     * len(outlet_pts))
    # Note: total count == n_top + n_bot + n_inlet + n_outlet == N_total,
    # but the 4 corner coordinates appear in TWO arrays each (overlap).
    return pts, normals, np.array(labels)


# --------------------------- collocation points -----------------------------
def generate_collocation_points(N_total, phi=0.0,
                                bl_thickness=DELTA_BL,
                                bl_thickness_io=DELTA_BL_IO,
                                bl_ratio=0.30, seed=42, max_iter=500):
    """
    Latin-Hypercube sampling in the wavy channel, split into core + near-wall
    band. The near-wall band has CONSTANT thickness (PAPER-FIX #1):
        * DELTA_BL    in the y-direction (distance to upper/lower wall)
        * DELTA_BL_IO in the x-direction (distance to inlet/outlet)
    Neither depends on x, on curvature, or on local channel height.
    """
    sampler = qmc.LatinHypercube(d=2, seed=seed)

    n_bl_target   = int(N_total * bl_ratio)
    n_core_target = N_total - n_bl_target

    core_pts, bl_pts = [], []
    batch_size = max(N_total, 5000)
    it = 0
    while (len(core_pts) < n_core_target or len(bl_pts) < n_bl_target) and it < max_iter:
        it += 1
        samples = sampler.random(n=batch_size)
        x_cand = x_min + samples[:, 0] * (x_max - x_min)
        yb_cand = y_bottom(x_cand, phi)
        yt_cand = y_top(x_cand)
        y_cand  = yb_cand + samples[:, 1] * (yt_cand - yb_cand)

        # PAPER-FIX #1: constant deltas (no x dependence, no curvature dependence)
        d_top = yt_cand - y_cand
        d_bot = y_cand - yb_cand
        d_in  = x_cand - x_min
        d_out = x_max - x_cand

        is_bl = ((d_top <= bl_thickness)
                 | (d_bot <= bl_thickness)
                 | (d_in  <= bl_thickness_io)
                 | (d_out <= bl_thickness_io))

        if len(core_pts) < n_core_target:
            core_cand = np.column_stack((x_cand[~is_bl], y_cand[~is_bl]))
            need = n_core_target - len(core_pts)
            core_pts.extend(core_cand[:need].tolist())

        if len(bl_pts) < n_bl_target:
            bl_cand = np.column_stack((x_cand[is_bl], y_cand[is_bl]))
            need = n_bl_target - len(bl_pts)
            bl_pts.extend(bl_cand[:need].tolist())

    if len(core_pts) < n_core_target or len(bl_pts) < n_bl_target:
        print(f"  [warning] max_iter={max_iter} reached "
              f"(core {len(core_pts)}/{n_core_target}, bl {len(bl_pts)}/{n_bl_target})")

    core_pts = np.array(core_pts[:n_core_target])
    bl_pts   = np.array(bl_pts[:n_bl_target])
    zone_labels = np.array(["core"] * len(core_pts) + ["bl"] * len(bl_pts))
    return np.vstack((core_pts, bl_pts)), zone_labels


# ----------------------------- plotting -------------------------------------
def plot_geometry_check(phis_deg, out_dir):
    x_dense = np.linspace(x_min, x_max, 400)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    for ax, phi_deg in zip(axes, phis_deg):
        phi = np.radians(phi_deg)
        ax.plot(x_dense, y_top(x_dense), "k-")
        ax.plot(x_dense, y_bottom(x_dense, phi), "k-")
        ax.set_title(f"$\\phi$ = {phi_deg}°")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[len(phis_deg):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "geometry_check.png"), dpi=200)
    plt.close()


# ------------------------------- main ---------------------------------------
if __name__ == "__main__":
    phis_deg = [0, 30, 60, 90, 180, -30, -60, -90]
    labels   = ["sym" if p == 0 else f"phi_{p}" for p in phis_deg]

    out_dir = "points"
    os.makedirs(out_dir, exist_ok=True)
    plot_geometry_check(phis_deg, out_dir)

    for phi_deg, label in zip(phis_deg, labels):
        phi = np.radians(phi_deg)
        if phi_deg == 0:
            n_colloc, n_bound = 62000, 8000
        else:
            n_colloc, n_bound = 89600, 9000

        print(f"Generating {label}: colloc={n_colloc}, boundary={n_bound}")

        colloc, zone_labels = generate_collocation_points(n_colloc, phi, seed=42)
        boundary, normals, bnd_labels = generate_boundary_points(n_bound, phi)

        subdir = os.path.join(out_dir, label)
        os.makedirs(subdir, exist_ok=True)

        np.savetxt(os.path.join(subdir, "collocation.csv"),
                   np.column_stack((colloc, zone_labels)),
                   delimiter=",", header="x,y,zone", comments="", fmt="%s")

        np.savetxt(os.path.join(subdir, "boundary.csv"),
                   np.column_stack((boundary, normals, bnd_labels)),
                   delimiter=",", header="x,y,nx,ny,type", comments="", fmt="%s")

        # ---- paper-style plot: coloured by boundary type ----
        colloc_f = colloc.astype(float)
        plt.figure(figsize=(10, 6))
        is_core = zone_labels == "core"
        plt.scatter(colloc_f[is_core][::5, 0],  colloc_f[is_core][::5, 1],
                    s=0.5, c="g", alpha=0.4, label="core (subsampled)")
        plt.scatter(colloc_f[~is_core][::2, 0], colloc_f[~is_core][::2, 1],
                    s=0.5, c="g", alpha=0.7, label="boundary layer (subsampled)")

        bnd_f = boundary.astype(float)
        cmap = {"upper_wall": "red", "lower_wall": "red",
                "inlet": "blue", "outlet": "cyan"}
        for t in ["upper_wall", "lower_wall", "inlet", "outlet"]:
            m = bnd_labels == t
            plt.scatter(bnd_f[m, 0], bnd_f[m, 1], s=2,
                        c=cmap[t], label=t)
        plt.title(f"Channel point distribution: {label}")
        plt.axis("equal")
        plt.legend(loc="upper right", fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, "collocation.png"), dpi=200)
        plt.close()

    print("All point sets generated successfully!")
    
    
