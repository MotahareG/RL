# plot_routes_journal.py
# =========================================================================
# Journal-quality route visualisation for AV-LRP.
#
# Drop-in replacement cell: run after sess.run(agent.val_summary_greedy).
# Produces one clean PDF + PNG per instance, plus an optional multi-panel
# grid figure for the paper.
#
# Usage (inside your training notebook / eval cell):
#
#   from plot_routes_journal import plot_instance, plot_grid
#
#   # After sess.run:
#   R, v, logprobs, actions, idxs, batch, _, routes = sess.run(
#       agent.val_summary_greedy,
#       feed_dict={...})
#
#   plot_grid(
#       features     = features,       # (batch, n_nodes, 4)
#       routes_raw   = routes,          # (decode_len, batch, 1)
#       rewards      = R,               # (batch,) negative costs
#       n_instances  = 4,               # how many to show
#       save_path    = "route_grid.pdf",
#       fac_cost     = 10.0,
#   )
# =========================================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe


# ── colour palette ───────────────────────────────────────────────────────
# 6 route colours: distinct, print-safe, colourblind-friendly.
ROUTE_COLOURS = [
    '#2563EB',   # blue
    '#DC2626',   # red
    '#059669',   # emerald
    '#9333EA',   # purple
    '#D97706',   # amber
    '#0891B2',   # cyan
]

# Node colours
_CLR_DEPOT    = '#111111'
_CLR_CUST     = '#374151'
_CLR_STATION  = '#DC2626'
_CLR_CUST_LBL = '#4B5563'


# ── helper: split decode sequence → sub-tours ────────────────────────────

def _segment_tours(route):
    """
    Split a flat decode sequence into depot-to-depot sub-tours.
    Returns list of lists, each starting and ending at depot (0).
    Filters out facility-only visits and empty bounces.
    """
    tours, current = [], [0]

    for node in route:
        node = int(node)
        if node == 0:
            if len(current) > 1:
                current.append(0)
                tours.append(current)
            current = [0]
        else:
            current.append(node)

    if len(current) > 1:
        tours.append(current)

    # keep only tours with at least one customer
    return [t for t in tours if any(n != 0 for n in t[1:])]


# ── helper: classify nodes from features array ──────────────────────────

def _parse_instance(features_1, route_flat, reward, fac_cost):
    """
    Parse a single instance from the batch tensors.

    features_1 : (n_nodes, 4) — columns: x, y, demand, node_type
    route_flat : list[int] — flat decode sequence
    reward     : float — negative total cost
    fac_cost   : float — per-facility opening cost

    Returns dict with everything needed to plot.
    """
    coords    = features_1[:, :2]             # (n_nodes, 2)
    demands   = features_1[:, 2]              # (n_nodes,)
    node_type = features_1[:, 3].astype(int)  # 0 depot, 1 cust, 2 fac

    # identify opened facilities from the route
    opened_set = set()
    for nid in route_flat:
        nid = int(nid)
        if node_type[nid] == 2:
            opened_set.add(nid)

    tours = _segment_tours(route_flat)

    # count real customer-serving tours (ignore facility-only tours)
    customer_tours = []
    for tour in tours:
        has_cust = any(node_type[n] == 1 for n in tour if n != 0)
        if has_cust:
            customer_tours.append(tour)

    total_cost = -float(reward)
    n_open     = len(opened_set)
    routing    = total_cost - n_open * fac_cost

    return dict(
        coords=coords, demands=demands, node_type=node_type,
        opened=opened_set, tours=customer_tours,
        total_cost=total_cost, n_open=n_open,
        routing_cost=routing, n_vehicles=len(customer_tours),
    )


# =========================================================================
# SINGLE-INSTANCE PLOT
# =========================================================================

def plot_instance(features_1, route_flat, reward, fac_cost=10.0,
                  instance_id=0, ax=None, save_path=None,
                  show_demand=True, label_size=7):
    """
    Plot one AV-LRP solution in journal style.

    If ax is provided, draws into that axes (for grid layout).
    If save_path is provided (and ax is None), saves standalone figure.
    """
    info = _parse_instance(features_1, route_flat, reward, fac_cost)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))

    coords    = info['coords']
    node_type = info['node_type']
    demands   = info['demands']
    tours     = info['tours']
    opened    = info['opened']
    n_nodes   = len(coords)

    # Axis mapping: features store (lat, lon), so col 0 → Y, col 1 → X.
    # This matches visualize_routes.py convention.
    def _xy(node_id):
        return coords[node_id, 1], coords[node_id, 0]

    # ── routes ───────────────────────────────────────────────────────────
    for t_idx, tour in enumerate(tours):
        clr = ROUTE_COLOURS[t_idx % len(ROUTE_COLOURS)]

        # draw edges with small direction arrows
        for k in range(len(tour) - 1):
            i, j = tour[k], tour[k + 1]
            x0, y0 = _xy(i)
            x1, y1 = _xy(j)

            ax.annotate(
                '', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=clr,
                    lw=1.4,
                    mutation_scale=10,
                    shrinkA=4, shrinkB=4,
                ),
                zorder=2,
            )

    # ── depot ────────────────────────────────────────────────────────────
    depot_mask = node_type == 0
    ax.scatter(
        coords[depot_mask, 1], coords[depot_mask, 0],
        s=160, c=_CLR_DEPOT, marker='s', zorder=5,
        edgecolors='white', linewidths=1.0,
    )

    # ── open stations ────────────────────────────────────────────────────
    for nid in opened:
        sx, sy = _xy(nid)
        ax.scatter(
            sx, sy,
            s=120, c=_CLR_STATION, marker='D', zorder=5,
            edgecolors='white', linewidths=0.8,
        )
        ax.annotate(
            f'S{nid}', (sx, sy),
            textcoords='offset points', xytext=(5, -10),
            fontsize=label_size - 1, color=_CLR_STATION,
            fontweight='bold',
        )

    # ── customers ────────────────────────────────────────────────────────
    cust_mask = node_type == 1
    cust_ids  = np.where(cust_mask)[0]

    ax.scatter(
        coords[cust_mask, 1], coords[cust_mask, 0],
        s=50, c='white', marker='o', zorder=4,
        edgecolors=_CLR_CUST, linewidths=1.2,
    )

    for nid in cust_ids:
        x, y = _xy(nid)
        if show_demand:
            lbl = f'{nid}({int(demands[nid])})'
        else:
            lbl = str(nid)
        ax.annotate(
            lbl, (x, y),
            textcoords='offset points', xytext=(4, 4),
            fontsize=label_size, color=_CLR_CUST_LBL,
            path_effects=[
                pe.withStroke(linewidth=2, foreground='white'),
            ],
        )

    # ── title ────────────────────────────────────────────────────────────
    station_txt = f'Stations = {info["n_open"]}' if info['n_open'] > 0 else ''
    parts = [f'Instance {instance_id}',
             f'Cost = {info["total_cost"]:.3f}',
             f'Vehicles = {info["n_vehicles"]}']
    if station_txt:
        parts.append(station_txt)
    ax.set_title('    '.join(parts), fontsize=9, fontweight='medium', pad=8)

    # ── axis styling ─────────────────────────────────────────────────────
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylabel('y', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal')

    # light frame, no grid
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#D1D5DB')
    ax.tick_params(colors='#9CA3AF', width=0.5)

    if standalone:
        # add legend at bottom
        _add_legend(fig, info['n_vehicles'])
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'  Saved → {save_path}')
        plt.close(fig)

    return info


# =========================================================================
# MULTI-INSTANCE GRID
# =========================================================================

def plot_grid(features, routes_raw, rewards, n_instances=4,
              save_path='route_grid.pdf', fac_cost=10.0,
              show_demand=True, cols=2):
    """
    Plot multiple instances in a clean grid layout.

    features   : (batch, n_nodes, 4) numpy
    routes_raw : (decode_len, batch, 1) numpy
    rewards    : (batch,) numpy — negative total costs
    n_instances: how many instances to plot (from the front of the batch)
    save_path  : output path (.pdf recommended for journals)
    fac_cost   : per-facility opening cost
    cols       : grid columns (2 for two-column paper, 3 for wider)
    """
    n = min(n_instances, features.shape[0])
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4.5 * cols, 4.5 * rows),
        squeeze=False,
    )

    max_vehicles = 0

    for idx in range(n):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        feat_i = features[idx]                       # (n_nodes, 4)
        route_i = routes_raw[:, idx].flatten().astype(int).tolist()
        reward_i = float(rewards[idx])

        info = plot_instance(
            feat_i, route_i, reward_i,
            fac_cost=fac_cost,
            instance_id=idx,
            ax=ax,
            show_demand=show_demand,
        )
        max_vehicles = max(max_vehicles, info['n_vehicles'])

    # hide unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    # shared legend at bottom
    _add_legend(fig, max_vehicles)

    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Grid saved → {save_path}')


# =========================================================================
# LEGEND
# =========================================================================

def _add_legend(fig, n_tours):
    """
    Compact horizontal legend pinned to the bottom of the figure.
    Shows: depot, customer, open station, and one entry per tour colour.
    """
    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=_CLR_DEPOT,
               markeredgecolor='white', markersize=9, label='Depot'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor=_CLR_CUST, markeredgewidth=1.2,
               markersize=8, label='Customer'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=_CLR_STATION,
               markeredgecolor='white', markersize=8, label='Open station'),
    ]

    for t in range(min(n_tours, len(ROUTE_COLOURS))):
        handles.append(
            Line2D([0], [0], color=ROUTE_COLOURS[t], lw=1.8,
                   label=f'Vehicle {t + 1}')
        )

    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=len(handles),
        fontsize=8,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.4,
    )


# =========================================================================
# STANDALONE TEST (run on the uploaded LRP10 data if available)
# =========================================================================

if __name__ == '__main__':
    print('Self-test with synthetic LRP10 data...')

    rng = np.random.RandomState(42)
    n_cust, n_fac = 10, 5
    n_nodes = 1 + n_cust + n_fac  # 16

    # synthetic features: (n_nodes, 4) = x, y, demand, node_type
    # convention: col 0 = lat-like (y), col 1 = lon-like (x)
    coords = rng.rand(n_nodes, 2)
    coords[0] = [0.5, 0.5]  # depot at centre

    demands = np.zeros(n_nodes)
    demands[1:n_cust+1] = rng.randint(1, 10, n_cust)

    node_type = np.zeros(n_nodes)
    node_type[1:n_cust+1] = 1
    node_type[n_cust+1:]  = 2

    features = np.stack([coords[:, 0], coords[:, 1], demands, node_type], axis=1)

    # synthetic route: 3 sub-tours; facility 11 visited in tour 2
    route = [1, 2, 3, 0, 11, 4, 5, 6, 7, 0, 8, 9, 10, 0]
    reward = -22.5  # includes facility cost 10

    # test single plot
    plot_instance(
        features, route, reward,
        fac_cost=10.0, instance_id=0,
        save_path='/home/claude/test_single.png',
        show_demand=True,
    )

    # test grid
    batch_feat = np.tile(features[np.newaxis, :, :], (4, 1, 1))
    for b in range(1, 4):
        batch_feat[b, 1:, :2] += rng.randn(n_nodes - 1, 2) * 0.03

    routes_raw = np.array(route)[:, np.newaxis, np.newaxis]
    routes_raw = np.tile(routes_raw, (1, 4, 1))

    rewards = np.array([-22.5, -23.1, -22.8, -23.3])

    plot_grid(
        batch_feat, routes_raw, rewards,
        n_instances=4,
        save_path='/home/claude/test_grid.png',
        fac_cost=10.0,
    )
    print('Done.')
