
"""
Collocation / boundary point generator for PINN training on the wavy-channel
geometry described in Roy, Mukherjee, Prasad & Nayak,
"A computational analysis of flow dynamics and heat transfer in a wavy
patterned channel using physics-informed neural networks", Phys. Fluids 37,
043610 (2025).

  * build a dense polyline of the *entire*
    boundary (top wall + outlet segment + bottom wall + inlet segment),
    put it in a KD-tree, and classify a candidate point as "boundary
    layer" using its true nearest-neighbor Euclidean distance to that
    polyline against a single constant threshold `bl_thickness`. This is
    a genuine constant-width offset band: it automatically follows wall
    curvature correctly and rounds the four corners smoothly (no overlap
    artifacts), matching Fig. 6.

Everything else (LHS everywhere, exact point counts via quota-based
rejection sampling, boundary labels, analytic outward normals, no
duplicated corner points) is unchanged from the previous version.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree


x_min = -4.0            # domain start
x_max = 4.0             # domain end
L = x_max - x_min       # module length (8.0)
a = 1.0                 # amplitude
B = 2.0 * np.pi / L     # wave number
H_min = 12 / 7          # minimum channel height


def y_top(x):
    return a * np.sin(B * x + np.pi / 2) + (H_min / 2 + a)


def dy_top_dx(x):
    return a * B * np.cos(B * x + np.pi / 2)


def y_bottom(x, phi=0.0):
    return a * np.sin(B * x + 1.5 * np.pi + phi) - (H_min / 2 + a)


def dy_bot_dx(x, phi=0.0):
    return a * B * np.cos(B * x + 1.5 * np.pi + phi)


def local_height(x, phi=0.0):
    """Instantaneous channel height H(x) = y_top(x) - y_bottom(x, phi)."""
    return y_top(x) - y_bottom(x, phi)


def wall_normal_top(x):
    """Outward normal for the upper wall (points away from the fluid)."""
    dy = dy_top_dx(x)
    n = np.column_stack((-dy, np.ones_like(dy)))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n


def wall_normal_bottom(x, phi):
    """Outward normal for the lower wall (points away from the fluid)."""
    dy = dy_bot_dx(x, phi)
    n = np.column_stack((dy, -np.ones_like(dy)))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n


def sample_curve_arc_length(y_func, dy_func, n_pts, phi=0.0):
    """Uniformly samples along a curve's actual perimeter using arc-length
    parameterization."""
    x_dense = np.linspace(x_min, x_max, 10000)
    dy = dy_func(x_dense, phi) if phi != 0.0 else dy_func(x_dense)
    ds = np.sqrt(1 + dy ** 2)

    s_cum = np.zeros_like(x_dense)
    s_cum[1:] = cumulative_trapezoid(ds, x_dense)
    total_length = s_cum[-1]

    s_targets = np.linspace(0, total_length, n_pts)
    x_sampled = np.interp(s_targets, s_cum, x_dense)
    y_sampled = y_func(x_sampled, phi) if phi != 0.0 else y_func(x_sampled)

    return np.column_stack((x_sampled, y_sampled)), total_length


def build_boundary_polyline(phi, n_per_segment=3000):
    """Dense point sampling of the full closed boundary (top wall, outlet,
    bottom wall, inlet) used only to measure true nearest-boundary
    distance -- not saved to any CSV."""
    x_dense = np.linspace(x_min, x_max, n_per_segment)
    top_pts = np.column_stack((x_dense, y_top(x_dense)))
    bot_pts = np.column_stack((x_dense, y_bottom(x_dense, phi)))

    y_in_dense = np.linspace(y_bottom(x_min, phi), y_top(x_min), n_per_segment)
    inlet_pts = np.column_stack((np.full(n_per_segment, x_min), y_in_dense))

    y_out_dense = np.linspace(y_bottom(x_max, phi), y_top(x_max), n_per_segment)
    outlet_pts = np.column_stack((np.full(n_per_segment, x_max), y_out_dense))

    return np.vstack((top_pts, bot_pts, inlet_pts, outlet_pts))


def build_boundary_tree(phi, n_per_segment=3000):
    poly = build_boundary_polyline(phi, n_per_segment)
    return cKDTree(poly)


def distance_to_boundary(points_xy, tree):
    """True nearest-neighbor Euclidean distance from each point to the
    boundary polyline (dense enough that vertex-distance closely
    approximates true perpendicular segment-distance)."""
    dist, _ = tree.query(points_xy, k=1)
    return dist


def generate_boundary_points(N_total, phi=0.0):
    _, len_top = sample_curve_arc_length(y_top, dy_top_dx, 200)
    _, len_bot = sample_curve_arc_length(y_bottom, dy_bot_dx, 200, phi)
    len_inlet = y_top(x_min) - y_bottom(x_min, phi)
    len_outlet = y_top(x_max) - y_bottom(x_max, phi)

    total_perimeter = len_top + len_bot + len_inlet + len_outlet

    n_top = int(round(N_total * (len_top / total_perimeter)))
    n_bot = int(round(N_total * (len_bot / total_perimeter)))
    n_inlet = int(round(N_total * (len_inlet / total_perimeter)))
    n_outlet = N_total - n_top - n_bot - n_inlet
    if n_outlet < 1:
        raise ValueError("N_total too small for proportional allocation; increase N_total.")

    top_full, _ = sample_curve_arc_length(y_top, dy_top_dx, n_top + 2)
    top_pts = top_full[1:-1]
    top_normals = wall_normal_top(top_pts[:, 0])

    bot_full, _ = sample_curve_arc_length(y_bottom, dy_bot_dx, n_bot + 2, phi)
    bot_pts = bot_full[1:-1]
    bot_normals = wall_normal_bottom(bot_pts[:, 0], phi)

    y_in = np.linspace(y_bottom(x_min, phi), y_top(x_min), n_inlet)
    inlet_pts = np.column_stack((np.full(n_inlet, x_min), y_in))
    inlet_normals = np.tile([-1.0, 0.0], (n_inlet, 1))

    y_out = np.linspace(y_bottom(x_max, phi), y_top(x_max), n_outlet)
    outlet_pts = np.column_stack((np.full(n_outlet, x_max), y_out))
    outlet_normals = np.tile([1.0, 0.0], (n_outlet, 1))

    pts = np.vstack((top_pts, bot_pts, inlet_pts, outlet_pts))
    normals = np.vstack((top_normals, bot_normals, inlet_normals, outlet_normals))
    labels = (
        ["upper_wall"] * len(top_pts)
        + ["lower_wall"] * len(bot_pts)
        + ["inlet"] * len(inlet_pts)
        + ["outlet"] * len(outlet_pts)
    )
    assert pts.shape[0] == N_total, "Boundary point count mismatch"
    return pts, normals, np.array(labels)



def generate_collocation_points(N_total, phi=0.0, bl_thickness=0.25, bl_ratio=0.40, seed=42, max_iter=500, tree_points_per_segment=3000):
    """
    Latin-Hypercube samples the full domain, then splits into:
      - Zone 1 (core): points whose true nearest-boundary distance exceeds
        `bl_thickness` (a CONSTANT absolute offset -- not scaled by local
        channel height, since Fig. 6's band is visually constant-width).
      - Zone 2 (boundary layer): points within `bl_thickness` of the
        boundary, measured as true 2D Euclidean distance to the nearest
        point on the full boundary polyline (top/bottom walls + inlet +
        outlet) -- correctly follows wall curvature and rounds corners.

    `bl_thickness` is in the same absolute units as x, y (e.g. 0.25 means
    a band roughly a quarter of a unit-length wide, hugging the wall all
    the way around). Tune it to match how thick the dense band looks in
    Fig. 6 relative to the channel's overall size.
    """
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    tree = build_boundary_tree(phi, n_per_segment=tree_points_per_segment)

    n_bl_target = int(N_total * bl_ratio)
    n_core_target = N_total - n_bl_target

    core_pts = []
    bl_pts = []

    batch_size = max(N_total, 5000)
    it = 0
    while (len(core_pts) < n_core_target or len(bl_pts) < n_bl_target) and it < max_iter:
        it += 1
        samples = sampler.random(n=batch_size)
        x_cand = x_min + samples[:, 0] * (x_max - x_min)
        yb_cand = y_bottom(x_cand, phi)
        yt_cand = y_top(x_cand)
        y_cand = yb_cand + samples[:, 1] * (yt_cand - yb_cand)

        cand_xy = np.column_stack((x_cand, y_cand))
        dist = distance_to_boundary(cand_xy, tree)
        is_bl = dist <= bl_thickness

        if len(core_pts) < n_core_target:
            core_cand = cand_xy[~is_bl]
            need = n_core_target - len(core_pts)
            core_pts.extend(core_cand[:need].tolist())

        if len(bl_pts) < n_bl_target:
            bl_cand = cand_xy[is_bl]
            need = n_bl_target - len(bl_pts)
            bl_pts.extend(bl_cand[:need].tolist())

    if len(core_pts) < n_core_target or len(bl_pts) < n_bl_target:
        print(
            f"  [warning] hit max_iter={max_iter} before filling quotas "
            f"(core: {len(core_pts)}/{n_core_target}, "
            f"bl: {len(bl_pts)}/{n_bl_target}). "
            "Consider lowering bl_thickness, raising batch_size, or raising max_iter."
        )

    core_pts = np.array(core_pts[:n_core_target])
    bl_pts = np.array(bl_pts[:n_bl_target])

    zone_labels = np.array(["core"] * len(core_pts) + ["bl"] * len(bl_pts))
    all_pts = np.vstack((core_pts, bl_pts))
    return all_pts, zone_labels


# ========================================================================
# Geometry sanity-check plot
# ========================================================================
def plot_geometry_check(phis_deg, out_dir):
    x_dense = np.linspace(x_min, x_max, 400)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    for ax, phi_deg in zip(axes, phis_deg):
        phi = np.radians(phi_deg)
        ax.plot(x_dense, y_top(x_dense), "k-")
        ax.plot(x_dense, y_bottom(x_dense, phi), "k-")
        ax.set_title(f"$\\phi$ = {phi_deg}\u00b0")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[len(phis_deg):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "geometry_check.png"), dpi=200)
    plt.close()


# ========================================================================
# Main Execution
# ========================================================================
if __name__ == "__main__":
    phis_deg = [0, 30, 60, 90, 180, -30, -60, -90]
    labels = ["sym" if p == 0 else f"phi_{p}" for p in phis_deg]

    out_dir = "points"
    os.makedirs(out_dir, exist_ok=True)
    plot_geometry_check(phis_deg, out_dir)

    for phi_deg, label in zip(phis_deg, labels):
        phi = np.radians(phi_deg)

        if phi_deg == 0:
            n_colloc = 62000
            n_bound = 8000
        else:
            n_colloc = 89600
            n_bound = 9000

        print(f"Generating {label}: colloc={n_colloc}, boundary={n_bound}")

        colloc, zone_labels = generate_collocation_points(n_colloc, phi, seed=42)
        boundary, normals, bnd_labels = generate_boundary_points(n_bound, phi)

        subdir = os.path.join(out_dir, label)
        os.makedirs(subdir, exist_ok=True)

        colloc_out = np.column_stack((colloc, zone_labels))
        np.savetxt(
            os.path.join(subdir, "collocation.csv"),
            colloc_out, delimiter=",", header="x,y,zone", comments="", fmt="%s",
        )

        boundary_out = np.column_stack((boundary, normals, bnd_labels))
        np.savetxt(
            os.path.join(subdir, "boundary.csv"),
            boundary_out, delimiter=",", header="x,y,nx,ny,type", comments="", fmt="%s",
        )

        colloc_f = colloc.astype(float)
        plt.figure(figsize=(10, 6))
        is_core = zone_labels == "core"
        plt.scatter(colloc_f[is_core][::5, 0], colloc_f[is_core][::5, 1],
                    s=0.5, c="g", alpha=0.4, label="core (subsampled)")
        plt.scatter(colloc_f[~is_core][::2, 0], colloc_f[~is_core][::2, 1],
                    s=0.5, c="g", alpha=0.5, label="boundary layer (subsampled)")
        boundary_f = boundary.astype(float)
        plt.scatter(boundary_f[:, 0], boundary_f[:, 1], s=2, c="r", label="boundary")
        plt.title(f"Channel Geometry: {label}")
        plt.axis("equal")
        plt.legend(loc="upper right", fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, "collocation.png"), dpi=200)
        plt.close()

    print("All point sets generated successfully!")

