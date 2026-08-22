# visualize_routes.py
# ==============================================================================
# Changes vs original:
#   1. FIX  – _draw_edge_safety: edge is only "infeasible-violated" when
#             lambda_ij=0 AND neither endpoint is an opened facility.
#             Previously coloured ALL lambda=0 edges orange, including legitimate
#             ones that are allowed because an opened station is at i or j.
#   2. FIX  – opened_facilities passed into _draw_edge_safety so it can apply
#             the full model.pdf constraint: F_i + F_j >= 1 - lambda_ij
#   3. NEW  – save_convergence_plot() function for training diagnostics
#   4. KEEP – all existing functions (plot_risk_map, plot_lrp_solution,
#             visualise_after_training) unchanged except the edge-colouring fix.
# ==============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


# ==============================================================================
# Tour colour palette  (Improvement 1)
# ==============================================================================
# Twelve perceptually distinct colours.  Green (#2ca02c) and orange (#ff7f0e)
# are intentionally excluded because they are reserved for the safety-status
# overlay (allowed-unsafe and violated edges respectively).
_TOUR_PALETTE = [
    '#1f77b4',  # blue          — tour 1
    '#9467bd',  # purple        — tour 2
    '#8c564b',  # brown         — tour 3
    '#e377c2',  # pink/magenta  — tour 4
    '#17becf',  # teal/cyan     — tour 5
    '#bcbd22',  # olive         — tour 6
    '#7f7f7f',  # grey          — tour 7
    '#aec7e8',  # light blue    — tour 8
    '#c5b0d5',  # lavender      — tour 9
    '#c49c94',  # tan           — tour 10
    '#f7b6d2',  # light pink    — tour 11
    '#dbdb8d',  # light olive   — tour 12
]


def _segment_tours(route):
    """
    Split a flat decode sequence into depot-to-depot sub-tours.

    The routes TensorArray does NOT include a leading depot entry — route[0]
    is the first real action (a customer or facility node).  This function
    therefore never skips route[0]; every node is processed in order.

    Empty depot-bounces (D → D with no intermediate nodes) are filtered out.

    Returns
    -------
    list[list[int]]
        Each inner list is one sub-tour including depot endpoints, e.g.
        [0, 3, 7, 12, 0].  Guaranteed len ≥ 3 (depot + ≥1 node + depot)
        except for the last tour which may not close at the depot.
    """
    tours   = []
    current = [0]           # every tour starts implicitly at the depot

    for node in route:
        node = int(node)
        if node == 0:
            # Depot return — close the current tour if it has real nodes
            if len(current) > 1:
                current.append(0)
                tours.append(current)
            current = [0]   # next tour also starts at depot
        else:
            current.append(node)

    # Last segment may not have returned to depot — include it if non-trivial
    if len(current) > 1:
        tours.append(current)

    # Drop any remaining D→D empty artefacts (only depot, no real nodes)
    return [t for t in tours if any(n != 0 for n in t[1:])]


# ==============================================================================
# 1.  Risk map
# ==============================================================================

def plot_risk_map(dataGen, save_path='risk_map.png', resolution=120):
    """
    KDE risk surface heatmap with raw collision points overlaid.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    risk_grid = dataGen.get_risk_map(resolution=resolution)
    im = ax.imshow(risk_grid, origin='lower', cmap='YlOrRd',
                   extent=[0, 1, 0, 1], alpha=0.8, vmin=0, vmax=1)

    pts = dataGen.collision_points
    ax.scatter(pts[:, 1], pts[:, 0], s=6, c='black', alpha=0.5,
               zorder=3, label='Collision point')

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Normalised collision risk (KDE)', fontsize=10)

    bb = dataGen.bounding_box
    ax.set_xlabel(f'Longitude  [{bb[0][1]:.2f} to {bb[1][1]:.2f}]', fontsize=10)
    ax.set_ylabel(f'Latitude   [{bb[0][0]:.2f} to {bb[1][0]:.2f}]', fontsize=10)
    ax.set_title('AV Collision Risk Map  (CA AV Collision 2019-2024)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'[vis] Risk map saved -> {save_path}')


# ==============================================================================
# 2.  LRP solution plot
# ==============================================================================

def plot_lrp_solution(coords, node_type, routes, opened_facilities,
                      lambda_mat=None, save_path='lrp_solution.png',
                      instance_title='LRP Solution',
                      collision_points=None):
    """
    Plot the decoded solution for one problem instance.

    Edge colouring  (Improvement 1 — per-tour distinct colours):
      - Each depot-to-depot tour uses one distinct colour from _TOUR_PALETTE.
      - Safety status overrides the tour colour for unsafe edges:
          Green solid  : lambda_ij=0, station at endpoint → allowed
          Orange dashed: lambda_ij=0, no station → violated

    Args:
        coords            : np.ndarray [n_nodes, 2]  normalised (lat, lon) in [0,1]
        node_type         : np.ndarray [n_nodes]     0=depot 1=customer 2=facility
        routes            : list[int] flat decode sequence for one sample
        opened_facilities : np.ndarray [n_nodes] binary (1 = facility opened)
        lambda_mat        : np.ndarray [n_nodes, n_nodes] or None
        save_path         : output file path
        instance_title    : figure title
        collision_points  : np.ndarray [N, 2] normalised collision locations,
                            or None.
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    # ── Collision scatter ─────────────────────────────────────────────────────
    if collision_points is not None and len(collision_points) > 0:
        ax.scatter(
            collision_points[:, 1],
            collision_points[:, 0],
            s=12, c='#e74c3c', alpha=0.25, zorder=1, marker='o',
            label='Collision point')

    # ── Edges (Improvement 1: per-tour colours) ───────────────────────────────
    tour_legend_entries = _draw_edges(ax, coords, routes, lambda_mat,
                                      opened_facilities)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    nt_flat = node_type.squeeze() if node_type.ndim > 1 else node_type
    of_flat = opened_facilities.squeeze() if (
        opened_facilities is not None and opened_facilities.ndim > 1
    ) else opened_facilities

    for i in range(coords.shape[0]):
        nt = int(nt_flat[i])
        x  = coords[i, 1]
        y  = coords[i, 0]

        if nt == 0:
            ax.scatter(x, y, s=220, c='gold', edgecolors='black',
                       linewidths=1.5, zorder=6, marker='*')
            ax.annotate('Depot', (x, y), textcoords='offset points',
                        xytext=(6, 6), fontsize=8, color='black', weight='bold')

        elif nt == 1:
            ax.scatter(x, y, s=70, c='steelblue', edgecolors='white',
                       linewidths=0.8, zorder=5, marker='o')
            ax.annotate(str(i), (x, y), textcoords='offset points',
                        xytext=(4, 4), fontsize=7, color='steelblue')

        else:
            opened = (of_flat is not None) and (of_flat[i] > 0.5)
            color  = '#d62728' if opened else '#9e9e9e'
            marker = '^'       if opened else 'v'
            ms     = 140       if opened else 100
            label  = f'F{i}' + (' \u2713' if opened else '')
            ax.scatter(x, y, s=ms, c=color, edgecolors='white',
                       linewidths=0.8, zorder=6, marker=marker)
            ax.annotate(label, (x, y), textcoords='offset points',
                        xytext=(5, -12), fontsize=7, color=color, weight='bold')

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = []

    # Collision points (if present)
    if collision_points is not None and len(collision_points) > 0:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                   alpha=0.5, markersize=6, label='Collision point'))

    # Node-type entries
    legend_elements += [
        Line2D([0],[0], marker='*', color='w', markerfacecolor='gold',
               markeredgecolor='black', markersize=13, label='Depot'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='steelblue',
               markersize=9, label='Customer'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='#d62728',
               markersize=11, label='Opened station'),
        Line2D([0],[0], marker='v', color='w', markerfacecolor='#9e9e9e',
               markersize=10, label='Closed candidate'),
    ]

    # Per-tour colour entries  (Improvement 1)
    if tour_legend_entries:
        legend_elements.append(
            mpatches.Patch(color='none', label='─── Tours ───'))
        for t_idx, (t_color, n_cust_in_tour) in enumerate(tour_legend_entries):
            legend_elements.append(
                Line2D([0], [0], color=t_color, lw=2.2,
                       label=f'Tour {t_idx + 1}  ({n_cust_in_tour} stops)'))

    # Safety-status entries (always shown)
    legend_elements += [
        mpatches.Patch(color='none', label='─── Edge safety ───'),
        Line2D([0],[0], color='#2ca02c', lw=1.8,
               label='Unsafe edge, station at endpoint (allowed)'),
        Line2D([0],[0], color='#ff7f0e', lw=1.8, ls='--',
               label='Unsafe edge, no station (violated)'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=7.5,
              framealpha=0.9)

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('Normalised longitude', fontsize=10)
    ax.set_ylabel('Normalised latitude',  fontsize=10)
    ax.set_title(instance_title, fontsize=12, weight='bold')
    ax.grid(True, alpha=0.25, linestyle=':')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'[vis] LRP solution saved -> {save_path}')


# ==============================================================================
# 3.  Convergence plot  (NEW)
# ==============================================================================

def save_convergence_plot(history, save_path='convergence.png',
                          smooth_window=50):
    """
    Three-panel convergence plot saved to disk.

    Panels:
      Top   : Mean batch reward  (higher = better; should rise during training)
      Middle: Actor loss         (REINFORCE policy gradient loss)
      Bottom: Critic loss        (value-network MSE)

    Args:
        history     : dict with keys 'step', 'reward', 'actor_loss',
                      'critic_loss'  — one value per training step.
        save_path   : output .png path.
        smooth_window: rolling-mean window size for the smoothed overlay line.
                       Set to 1 to disable smoothing.
    """
    steps       = np.array(history['step'],        dtype=float)
    rewards     = np.array(history['reward'],       dtype=float)
    actor_loss  = np.array(history['actor_loss'],   dtype=float)
    critic_loss = np.array(history['critic_loss'],  dtype=float)

    def _smooth(x, w):
        """Simple uniform rolling mean; returns same length as input."""
        if w <= 1 or len(x) < w:
            return x.copy()
        kernel = np.ones(w) / w
        # Use 'valid' convolution and pad edges with original values
        smoothed = np.convolve(x, kernel, mode='same')
        # Fix the boundary artefacts: keep raw values where window is incomplete
        half = w // 2
        smoothed[:half]  = x[:half]
        smoothed[-half:] = x[-half:]
        return smoothed

    reward_sm     = _smooth(rewards,     smooth_window)
    actor_loss_sm = _smooth(actor_loss,  smooth_window)
    critic_loss_sm= _smooth(critic_loss, smooth_window)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle('Training Convergence', fontsize=14, weight='bold', y=0.98)

    # Colour palette
    RAW_ALPHA = 0.25
    C = ['#1f77b4', '#d62728', '#2ca02c']   # blue, red, green

    # ── Panel 1 : Reward ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(steps, rewards,    color=C[0], alpha=RAW_ALPHA, lw=0.8, label='Per-step')
    ax.plot(steps, reward_sm,  color=C[0], lw=2.0,          label=f'Smoothed (w={smooth_window})')
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.set_ylabel('Mean Reward  (−Total Cost)', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle=':')
    # Annotate final value
    if len(rewards):
        ax.annotate(f'Final: {reward_sm[-1]:.3f}',
                    xy=(steps[-1], reward_sm[-1]),
                    xytext=(-60, 10), textcoords='offset points',
                    fontsize=8, color=C[0],
                    arrowprops=dict(arrowstyle='->', color=C[0], lw=1.0))

    # ── Panel 2 : Actor loss ─────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(steps, actor_loss,    color=C[1], alpha=RAW_ALPHA, lw=0.8)
    ax.plot(steps, actor_loss_sm, color=C[1], lw=2.0, label=f'Smoothed (w={smooth_window})')
    ax.set_ylabel('Actor Loss', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle=':')

    # ── Panel 3 : Critic loss ────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(steps, critic_loss,    color=C[2], alpha=RAW_ALPHA, lw=0.8)
    ax.plot(steps, critic_loss_sm, color=C[2], lw=2.0, label=f'Smoothed (w={smooth_window})')
    ax.set_ylabel('Critic Loss  (MSE)', fontsize=10)
    ax.set_xlabel('Training Step', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'[vis] Convergence plot saved -> {save_path}')


# ==============================================================================
# 4.  Helpers – edge drawing  (FIXED)
# ==============================================================================

def _draw_edges(ax, coords, route, lambda_mat, opened_facilities):
    """
    Draw route arrows with per-tour distinct colours  (Improvement 1).

    Each depot-to-depot sub-tour is assigned one colour from _TOUR_PALETTE.
    Safety status may override the tour colour on individual edges:

        lambda_ij = 1                        → tour colour  (safe)
        lambda_ij = 0, F_i=1 or F_j=1       → green solid  (allowed)
        lambda_ij = 0, F_i=0 and F_j=0      → orange dashed (violated)

    Parameters
    ----------
    ax                : matplotlib Axes
    coords            : (n_nodes, 2) normalised [lat, lon]
    route             : flat list/array of node indices (no leading depot)
    lambda_mat        : (n_nodes, n_nodes) or None
    opened_facilities : (n_nodes,) float binary or None

    Returns
    -------
    list of (hex_color, n_stops) — one entry per tour, used to build the legend.
    n_stops counts every non-depot node in the tour.
    """
    if not route:
        return []

    of = (opened_facilities.squeeze()
          if opened_facilities is not None and opened_facilities.ndim > 1
          else opened_facilities)

    _arrow_base = dict(arrowstyle='->', mutation_scale=14)
    tours = _segment_tours(route)
    tour_legend_info = []

    for t_idx, tour in enumerate(tours):
        t_color = _TOUR_PALETTE[t_idx % len(_TOUR_PALETTE)]
        n_stops = sum(1 for n in tour if n != 0)
        tour_legend_info.append((t_color, n_stops))

        for k in range(len(tour) - 1):
            prev, node = tour[k], tour[k + 1]
            if prev == node:
                continue

            x0, y0 = coords[prev, 1], coords[prev, 0]
            x1, y1 = coords[node, 1], coords[node, 0]

            if lambda_mat is None:
                color, ls, lw = t_color, '-', 1.5
            else:
                lam_ij = int(lambda_mat[prev, node])
                if lam_ij == 1:
                    # Safe edge — use this tour's colour
                    color, ls, lw = t_color, '-', 1.5
                else:
                    # Unsafe edge — safety status overrides tour colour
                    has_station = (of is not None) and (
                        of[prev] > 0.5 or of[node] > 0.5)
                    if has_station:
                        color, ls, lw = '#2ca02c', '-', 1.8   # allowed
                    else:
                        color, ls, lw = '#ff7f0e', '--', 1.8  # violated

            ax.annotate(
                '', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(**_arrow_base, color=color,
                                linestyle=ls, lw=lw))

    return tour_legend_info


# ==============================================================================
# 5.  Convenience wrapper (called from train_lrp.py at end of training)
# ==============================================================================

def visualise_after_training(dataGen, agent, sess, env, args, log_dir='.'):
    """
    Generates risk map, clustering map (Improvement 2), and solution figures
    after training.  Called from train_lrp.py — the convergence plot is saved
    separately by save_convergence_plot() directly in the training loop.
    """
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    fig_dir = os.path.join(log_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    n_cust = args.get('n_cust', 10)
    n_fac  = args.get('n_fac',  5)   # task-specific value from args (Improvement 2)

    # Risk map
    try:
        plot_risk_map(dataGen,
                      save_path=os.path.join(fig_dir, 'risk_map.png'))
    except Exception as e:
        print(f'[vis] Risk map failed: {e}')

    # ── Clustering / candidate-station map  (Improvement 2) ──────────────────
    # Uses the same n_fac as the current task so the number of K-means clusters
    # always matches the candidate-station pool the agent was trained with.
    try:
        from visualize_clustering import visualize_risk_and_candidates
        csv_path = args.get('collision_csv_path', 'CA_AV_Collision_2019-2024.csv')
        visualize_risk_and_candidates(
            collision_csv_path=csv_path,
            n_fac=n_fac,
            html_save_path=os.path.join(fig_dir, 'clustering_map.html'),
            static_save_path=os.path.join(fig_dir,
                                          f'clustering_map_n{n_fac}.png'))
        print(f'[vis] Clustering map generated with n_fac={n_fac}')
    except Exception as e:
        print(f'[vis] Clustering map failed: {e}')

    # Solution plot
    try:
        inst = dataGen.generate_instance(n_cust, n_fac, seed=0,
                                         compute_lambda=True)
        feat = inst['features'][np.newaxis, ...]
        lam  = inst['lambda'].astype(np.float32)[np.newaxis, ...]

        R, v, _, actions, idxs, batch, _, routes = sess.run(
            agent.val_summary_greedy,
            feed_dict={agent.env.input_data:  feat,
                       agent.env.lambda_ph:   lam,
                       agent.decodeStep.dropout: 0.0})

        route_flat = routes[:, 0, 0].tolist()

        n_nodes   = inst['coords'].shape[0]
        opened_np = np.zeros(n_nodes, dtype=np.float32)
        nt_flat   = inst['node_type'].squeeze()
        for nid in route_flat:
            if nt_flat[int(nid)] == 2:
                opened_np[int(nid)] = 1.0

        plot_lrp_solution(
            coords=inst['coords'],
            node_type=nt_flat,
            routes=route_flat,
            opened_facilities=opened_np,
            lambda_mat=inst['lambda'],
            save_path=os.path.join(fig_dir, 'lrp_solution.png'),
            instance_title=f'LRP Solution  (cost={-float(R[0]):.3f})',
            collision_points=dataGen.collision_points)
    except Exception as e:
        print(f'[vis] Solution plot failed: {e}')
