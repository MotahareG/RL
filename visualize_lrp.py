# visualize_lrp.py  –  New file
# Visualises LRP solutions: collision heatmap, depot, customers,
# facilities (open/closed), and decoded routes.
# Call visualize_solution() from train_lrp.py after inference.

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless backend for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os


# ---------------------------------------------------------------------------
# Bounding-box metadata (must match data_generator_lrp.py)
# ---------------------------------------------------------------------------
BBOX = ((34.0, -118.5), (34.4, -118.0))  # (lat_min, lon_min), (lat_max, lon_max)


def _denorm(pts_norm):
    """Convert normalised [0,1]² coordinates back to (lat, lon)."""
    lat_min, lon_min = BBOX[0]
    lat_max, lon_max = BBOX[1]
    lat = pts_norm[:, 0] * (lat_max - lat_min) + lat_min
    lon = pts_norm[:, 1] * (lon_max - lon_min) + lon_min
    return np.stack([lat, lon], axis=1)     # columns: (lat, lon) = (y, x) in map


# ---------------------------------------------------------------------------
# Core visualisation
# ---------------------------------------------------------------------------
def visualize_solution(
    coords,               # (n_nodes, 2) normalised coordinates
    node_type,            # (n_nodes,)   0=depot, 1=customer, 2=facility
    decoded_route,        # list or 1-D array of node indices (the decode sequence)
    opened_facility_mask, # (n_nodes,) or (n_nodes,1) – 1 if facility opened, 0 otherwise
    collision_points,     # (N_coll, 2) normalised collision coordinates
    lambda_matrix=None,   # (n_nodes, n_nodes) optional, for edge-safety overlay
    out_path="lrp_solution.png",
    title="LRP Solution",
):
    """
    Produce a publication-quality figure showing:
      • Background heatmap of collision density
      • Collision point scatter
      • Depot (black star)
      • Customers (blue circles)
      • Closed facilities (grey triangles)
      • Opened facilities (red triangles, annotated)
      • Decoded route (arrows connecting nodes in sequence)
      • Optional: unsafe edges highlighted in orange

    Parameters
    ----------
    coords : ndarray (n_nodes, 2)
        Normalised [0,1]² node positions.  Column 0 = lat-dim, Column 1 = lon-dim.
    decoded_route : array-like of int
        Sequence of node indices as decoded by the agent.
    opened_facility_mask : array-like of float
        1.0 for each facility node that was selected/opened, 0.0 otherwise.
    collision_points : ndarray (N, 2)
        Normalised collision locations.
    out_path : str
        File path for the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 9))

    # ── 1. Collision density heatmap ──────────────────────────────────────
    if len(collision_points) > 0:
        from scipy.stats import gaussian_kde
        try:
            kde   = gaussian_kde(collision_points.T, bw_method=0.08)
            xx, yy = np.mgrid[0:1:200j, 0:1:200j]
            zz    = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(200, 200)
            ax.contourf(xx, yy, zz, levels=15, cmap='Reds', alpha=0.30)
            ax.contour( xx, yy, zz, levels=5,  colors='red', linewidths=0.3, alpha=0.4)
        except Exception:
            pass  # KDE can fail if too few unique points
        ax.scatter(
            collision_points[:, 1], collision_points[:, 0],
            c='red', s=6, alpha=0.25, label='AV accident', zorder=2
        )

    # ── 2. Risk-circle annotation (optional λ overlay) ───────────────────
    if lambda_matrix is not None:
        n = coords.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if lambda_matrix[i, j] == 0:
                    xs = [coords[i, 1], coords[j, 1]]
                    ys = [coords[i, 0], coords[j, 0]]
                    ax.plot(xs, ys, color='orange', lw=0.5, alpha=0.35, zorder=1)

    # ── 3. Draw the decoded route ─────────────────────────────────────────
    route = list(decoded_route)
    if len(route) > 1:
        palette = cm.tab10(np.linspace(0, 1, 10))
        seg_color_idx = 0
        seg_start = 0

        for step_idx, node in enumerate(route):
            if node == 0 and step_idx > 0:
                # Segment ends at depot → draw segment in one colour
                seg_nodes = route[seg_start: step_idx + 1]
                _draw_segment(ax, seg_nodes, coords, palette[seg_color_idx % 10])
                seg_color_idx += 1
                seg_start = step_idx

        # Draw remaining segment (if last action isn't depot return)
        if seg_start < len(route) - 1:
            _draw_segment(ax, route[seg_start:], coords, palette[seg_color_idx % 10])

    # ── 4. Nodes ─────────────────────────────────────────────────────────
    node_type  = np.array(node_type).flatten()
    opened_fac = np.array(opened_facility_mask).flatten()
    n_nodes    = len(coords)

    for idx in range(n_nodes):
        x, y = coords[idx, 1], coords[idx, 0]   # lon → x-axis, lat → y-axis
        ntype = node_type[idx] if idx < len(node_type) else -1

        if ntype == 0:  # depot
            ax.scatter(x, y, marker='*', c='black', s=400, zorder=6,
                       edgecolors='gold', linewidths=1.5)
            ax.annotate('Depot', (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=7, color='black')

        elif ntype == 1:  # customer
            ax.scatter(x, y, marker='o', c='steelblue', s=60, zorder=5,
                       edgecolors='navy', linewidths=0.8)
            ax.annotate(str(idx), (x, y), xytext=(4, 4),
                        textcoords='offset points', fontsize=6, color='navy')

        elif ntype == 2:  # facility
            is_open = (idx < len(opened_fac)) and (opened_fac[idx] > 0.5)
            color   = 'crimson' if is_open else 'dimgrey'
            marker  = '^' if is_open else 'v'
            size    = 180 if is_open else 80
            ax.scatter(x, y, marker=marker, c=color, s=size, zorder=5,
                       edgecolors='black', linewidths=0.8)
            label = f'F{idx}' + (' ✓' if is_open else '')
            ax.annotate(label, (x, y), xytext=(4, 4),
                        textcoords='offset points', fontsize=6,
                        color='crimson' if is_open else 'grey')

    # ── 5. Legend & cosmetics ─────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='none',
                       label='─── Nodes ───'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                   markersize=12, label='Depot'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                   markersize=8,  label='Customer'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='crimson',
                   markersize=10, label='Facility (opened)'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='dimgrey',
                   markersize=8,  label='Facility (closed)'),
        mpatches.Patch(facecolor='none', edgecolor='none',
                       label='─── Edges ───'),
        mpatches.Patch(facecolor='red',    alpha=0.3, label='Collision density'),
        plt.Line2D([0], [0], color='orange', lw=1.5, label='Unsafe edge (λ=0)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
              framealpha=0.85, ncol=1)

    # Axis labels in lat/lon
    lat_ticks = np.linspace(0, 1, 5)
    lon_ticks = np.linspace(0, 1, 5)
    lat_min, lon_min = BBOX[0]
    lat_max, lon_max = BBOX[1]
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels([f"{lon_min + t*(lon_max - lon_min):.2f}°W"
                        for t in lon_ticks], fontsize=7)
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels([f"{lat_min + t*(lat_max - lat_min):.2f}°N"
                        for t in lat_ticks], fontsize=7)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude",  fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved → {out_path}")


def _draw_segment(ax, seg_nodes, coords, color):
    """Draw a single route segment (list of node indices) with arrows."""
    for k in range(len(seg_nodes) - 1):
        i, j = seg_nodes[k], seg_nodes[k + 1]
        x0, y0 = coords[i, 1], coords[i, 0]
        x1, y1 = coords[j, 1], coords[j, 0]
        ax.annotate(
            '', xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle='->', color=color, lw=1.5,
                connectionstyle='arc3,rad=0.05'
            ),
            zorder=3
        )


# ---------------------------------------------------------------------------
# Batch visualisation helper (called from train_lrp.py)
# ---------------------------------------------------------------------------
def visualize_batch(
    input_data_np,    # (batch, n_nodes, 4) numpy array
    routes_np,        # (decode_len, batch, 1) or (decode_len, batch) numpy array
    opened_fac_np,    # (batch, n_nodes) numpy array
    collision_points, # (N, 2) normalised numpy array
    lambda_matrix,    # (n_nodes, n_nodes) numpy array or None
    out_dir="logs/figures",
    n_samples=5,
    step=0,
):
    """
    Save one figure per sample (up to n_samples) from a batch evaluation.
    """
    os.makedirs(out_dir, exist_ok=True)

    batch_size = input_data_np.shape[0]
    n_samples  = min(n_samples, batch_size)

    for b in range(n_samples):
        coords    = input_data_np[b, :, :2]          # (n_nodes, 2)
        node_type = input_data_np[b, :, 3]           # (n_nodes,)

        # Flatten route: shape (decode_len,)
        route_b = routes_np[:, b].flatten().astype(int)

        opened = opened_fac_np[b] if opened_fac_np is not None else np.zeros(len(coords))

        # Compute total route cost for the title
        route_cost = _compute_route_length(coords, route_b)
        n_open     = int(np.sum(opened > 0.5))
        ttl        = f"Step {step} | Sample {b} | Route cost {route_cost:.3f} | Open facilities: {n_open}"

        out_path = os.path.join(out_dir, f"step{step:06d}_sample{b:02d}.png")
        visualize_solution(
            coords            = coords,
            node_type         = node_type,
            decoded_route     = route_b,
            opened_facility_mask = opened,
            collision_points  = collision_points,
            lambda_matrix     = lambda_matrix,
            out_path          = out_path,
            title             = ttl,
        )

    print(f"  Saved {n_samples} route figures to {out_dir}/")


def _compute_route_length(coords, route):
    """Euclidean route length along the decoded sequence."""
    total = 0.0
    for k in range(len(route) - 1):
        i, j   = route[k], route[k + 1]
        total += np.linalg.norm(coords[i] - coords[j])
    return total
