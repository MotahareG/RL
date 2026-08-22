# task_specific_params.py
# ==============================================================================
# KEY DESIGN CLARIFICATION (aligns with model.pdf):
#
#   n_fac  = number of CANDIDATE facility locations (K-means cluster centres
#             from collision data).  It is the UPPER BOUND on how many
#             assistance stations could be built — NOT a fixed count.
#
#   The RL agent decides at inference time which subset (0 … n_fac) to actually
#   open, by choosing whether to visit each candidate node.  Opening a station
#   incurs cost C_j; the agent learns to open only what safety requires.
#
#   This implements the model.pdf objective:
#       Min  Σ_j C_j · F_j  +  Σ_{i,j,v} C_AV · d_ij · x_ijv
#   subject to:  F_i + F_j ≥ 1 − λ_ij   (safety constraint)
#
#   decode_len upper bound:
#       n_cust (customer visits)
#     + n_fac  (at most all candidates opened as waypoints)
#     + ceil(n_cust / n_comp) (depot returns between sub-tours)
#     + 2 (safety buffer)
#     ≈ n_cust + n_fac + n_cust//4 + 2
# ==============================================================================

from collections import namedtuple

TaskTSP = namedtuple('TaskTSP', ['task_name', 'input_dim', 'n_nodes', 'decode_len'])

TaskVRP = namedtuple('TaskVRP', [
    'task_name', 'input_dim', 'n_nodes', 'n_cust',
    'decode_len', 'capacity', 'demand_max'])

TaskLRP = namedtuple('TaskLRP', [
    'task_name', 'input_dim', 'n_nodes', 'n_cust', 'n_fac',
    'decode_len', 'capacity_per_compartment', 'num_compartments',
    'demand_max', 'facility_opening_cost'])

task_lst = {}

# ── TSP ───────────────────────────────────────────────────────────────────────
for size, nn in [(10, 10), (20, 20), (50, 50), (100, 100)]:
    task_lst[f'tsp{size}'] = TaskTSP('tsp', 2, nn, nn)

# ── VRP ───────────────────────────────────────────────────────────────────────
for size, nn, nc, dl, cap in [
        (10,  11, 10, 16,  20),
        (20,  21, 20, 30,  30),
        (50,  51, 50, 70,  40),
        (100,101,100,140,  50)]:
    task_lst[f'vrp{size}'] = TaskVRP('vrp', 3, nn, nc, dl, cap, 9)

# ── LRP ───────────────────────────────────────────────────────────────────────
# input_dim  = 4   (x, y, demand, node_type)
# n_nodes    = 1 + n_cust + n_fac   (depot + customers + candidate stations)
#
# n_fac: number of K-means cluster centres derived from the collision CSV.
#   These are the CANDIDATE locations where a station MAY be built.
#   The agent chooses 0 … n_fac of them to open at run time.
#
# decode_len: n_cust  (serve every customer)
#           + n_fac   (visit every candidate at most once)
#           + n_cust//4 + 1  (depot returns: one per sub-tour of ≤4 customers)
#           + 2       (safety buffer)
#
# facility_opening_cost: C_j / H in the revised model (post-defence).
#   C_j = 10 (station construction cost), H = 3650 (planning horizon in days).
#   Amortised per-period cost = 10 / 3650 ≈ 0.00274.
#   Old constraint C1 (F_i+F_j ≥ 1-λ_ij) is removed; safety is enforced
#   only through constraint (2): x_ijv ≤ F_i+F_j+λ_ij (masking in RL).

lrp_configs = [
    # (tag,   n_cust, n_fac)   ← n_fac = candidate pool size (from K-means)
    # lrp3 و lrp8 برای مقایسه با روش دقیق GAMS اضافه شدند
    # n_fac مقادیر جدید با interpolate خطی بین نقاط لنگر موجود محاسبه شد:
    #   lrp10→5, lrp20→8, lrp50→12
    ('lrp3',     3,   3),
    ('lrp8',     8,   4),
    ('lrp10',   10,   5),
    ('lrp15',   15,   6),   # interpolated: 5 + (15-10)/(20-10)*(8-5) = 6.5 → 6
    ('lrp20',   20,   8),
    ('lrp25',   25,   9),   # interpolated: 8 + (25-20)/(50-20)*(12-8) = 8.67 → 9
    ('lrp30',   30,   9),   # interpolated: 8 + (30-20)/(50-20)*(12-8) = 9.33 → 9
    ('lrp35',   35,  10),   # interpolated: 8 + (35-20)/(50-20)*(12-8) = 10.0 → 10
    ('lrp40',   40,  11),   # interpolated: 8 + (40-20)/(50-20)*(12-8) = 10.67 → 11
    ('lrp45',   45,  11),   # interpolated: 8 + (45-20)/(50-20)*(12-8) = 11.33 → 11
    ('lrp50',   50,  12),
    ('lrp100', 100,  20),
]

for tag, nc, nf in lrp_configs:
    task_lst[tag] = TaskLRP(
        task_name               = 'lrp',
        input_dim               = 4,
        n_nodes                 = 1 + nc + nf,
        n_cust                  = nc,
        n_fac                   = nf,          # CANDIDATE pool — agent opens a subset
        decode_len              = nc + nf + nc // 4 + 3,
        capacity_per_compartment= 10.0,        # float required for tf.fill
        num_compartments        = 4,
        demand_max              = 10,
        facility_opening_cost   = 10.0 / 3650.0, # C_j/H (revised model: 10/3650 ≈ 0.00274)
    )
