# lrp_env.py
# ==============================================================================
# Single-phase routing aligned with model.pdf.
#
# BUG FIX — Feature-type-based node classification (v3)
# ──────────────────────────────────────────────────────
# Previous versions used hardcoded INDEX RANGES to classify nodes:
#   is_fac  = (fac_start <= idx <= fac_end)   where fac_start = n_cust + 1
#   is_cust = (1 <= idx <= n_cust)
#
# This broke for lrp20+ when the data generator placed facility nodes at
# indices < n_cust (e.g. nodes 11-18 instead of 21-28), causing the F→F
# mask to silently fail: current_node=13 was a facility in the FEATURES
# (type=2) but < fac_start=21, so at_facility=False and F→F was allowed.
#
# FIX: ALL node classification now uses self.node_type_bb — a [batch_beam,
# n_nodes] tensor derived from input_data[:,:,3] (the node_type feature
# channel fed at sess.run time).  This is always consistent with the data
# generator regardless of n_cust or n_fac.
#
# Mask rules (6 total):
#   1. Already-visited customers
#   2. Customers whose demand exceeds all compartment capacities
#   3. λ safety constraint (risky edges need open station at endpoint)
#   4. Already-opened facilities (re-visiting has no effect)
#   5. Facility → facility transitions (feature-type-based — FIXED)
#   6. Depot gating
# ==============================================================================

import collections
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ==============================================================================
# NOTE (post-defense revision):
#   The old C1 constraint  F_i + F_j >= 1 - λ_ij  has been REMOVED.
#   Facilities are now opened only when actually needed for traversal
#   (constraint 2:  x_ijv <= F_i + F_j + λ_ij).
#   Therefore compute_mandatory_facilities / compute_batch_mandatory are
#   no longer needed and have been deleted.
# ==============================================================================



class State(collections.namedtuple(
        "State", ("load", "demand", "d_sat", "mask"))):
    pass


class Env(object):

    def __init__(self, args):
        self.args               = args
        self.n_nodes            = args["n_nodes"]
        self.n_cust             = args["n_cust"]
        self.n_fac              = args.get("n_fac",
                                    args["n_nodes"] - args["n_cust"] - 1)
        self.input_dim          = args["input_dim"]
        self.n_comp             = args.get("n_comp", 4)
        self.capacity_per_comp  = float(args.get("capacity_per_compartment", 10.0))
        self.total_actions      = self.n_nodes

        # ── Store representative λ for the get_default_lambda_batch() helper ──
        # This is the λ computed from the seed-42 instance; used to fill the
        # placeholder when per-instance matrices are unavailable.
        self._default_lambda_np = args.get("lambda_mat", None)

        # ── Replace tf.constant with a placeholder ────────────────────────────
        # Shape [None, n_nodes, n_nodes] — a separate matrix per batch element.
        # Fed at sess.run() time via feed_dict={env.lambda_ph: batch_lambda}.
        self.lambda_ph = tf.placeholder(
            tf.float32,
            shape=[None, self.n_nodes, self.n_nodes],
            name='lambda_ph')

        node_type = args.get("node_type", None)
        if node_type is not None:
            nt = np.array(node_type)
            if nt.ndim == 2:
                nt = nt.squeeze(-1)
            self.base_node_type = tf.constant(nt, tf.float32)
        else:
            self.base_node_type = None

        self.depot_idx      = 0
        self.customer_start = 1
        self.customer_end   = self.n_cust
        self.fac_start      = self.n_cust + 1
        self.fac_end        = self.n_nodes - 1


        self.input_data = tf.placeholder(
            tf.float32, shape=[None, self.n_nodes, self.input_dim])

        # ── Gain matrix placeholder (reward shaping) ──────────────────────────
        # Shape [None, n_nodes, n_nodes] — g̅_ij (normalized gains for risky edges).
        # Fed at sess.run() time.  Default zeros when not supplied.
        self.gain_ph = tf.placeholder_with_default(
            tf.zeros([1, self.n_nodes, self.n_nodes]),
            shape=[None, self.n_nodes, self.n_nodes],
            name='gain_ph')

        self.input_pnt       = self.input_data[:, :, :2]
        self.demand_full     = self.input_data[:, :, 2]
        self.input_node_type = self.input_data[:, :, 3]

    # ── Feed-dict helper ──────────────────────────────────────────────────────
    def get_default_lambda_batch(self, batch_size):
        """
        Returns a (batch_size, n_nodes, n_nodes) numpy array by tiling the
        representative λ matrix stored in args['lambda_mat'].  Use this when
        per-instance λ is unavailable (e.g. for the old fixed-λ training).
        """
        lam = self._default_lambda_np
        if lam is None:
            return np.ones((batch_size, self.n_nodes, self.n_nodes),
                           dtype=np.float32)
        return np.tile(np.array(lam, dtype=np.float32)[np.newaxis],
                       (batch_size, 1, 1))


    # =========================================================================
    # RESET
    # =========================================================================
    def reset(self, beam_width=1):
        self.beam_width = beam_width
        batch_size_dyn  = tf.shape(self.input_pnt)[0]
        self.batch_beam = batch_size_dyn * beam_width

        # ── Per-instance λ from placeholder ───────────────────────────────────
        # lambda_ph: [batch, n_nodes, n_nodes]  (fed per batch at sess.run time)
        # For beam search (width W > 1): each instance is replicated W times.
        #   [B, N, N] → expand → [B, 1, N, N] → tile → [B, W, N, N]
        #             → reshape → [B*W, N, N]
        if beam_width == 1:
            self.lambda_mat = self.lambda_ph   # [batch, N, N]
        else:
            lam_exp = tf.tile(
                tf.expand_dims(self.lambda_ph, 1),   # [B, 1, N, N]
                [1, beam_width, 1, 1])               # [B, W, N, N]
            self.lambda_mat = tf.reshape(
                lam_exp,
                tf.stack([batch_size_dyn * beam_width,
                          self.n_nodes, self.n_nodes]))  # [B*W, N, N]

        # ── Per-instance node types from features ─────────────────────────────
        # self.input_node_type = input_data[:, :, 3]  →  [batch, n_nodes]
        # For beam search (width W > 1): replicate so shape = [batch*W, n_nodes].
        # This tensor drives ALL node classification in _apply_action and
        # _build_mask, replacing the old hardcoded index-range checks
        # (fac_start, customer_end, etc.) which failed for lrp20+.
        if beam_width == 1:
            self.node_type_bb = self.input_node_type           # [batch, n_nodes]
        else:
            nt = tf.tile(
                tf.expand_dims(self.input_node_type, 1),       # [B, 1, n_nodes]
                [1, beam_width, 1])                            # [B, W, n_nodes]
            self.node_type_bb = tf.reshape(
                nt, tf.stack([batch_size_dyn * beam_width, self.n_nodes]))

        if self.base_node_type is not None:
            self.node_type = tf.tile(
                tf.expand_dims(self.base_node_type, 0),
                tf.stack([self.batch_beam, 1]))
        else:
            self.node_type = None

        # ── Opened facilities: all start closed ─────────────────────────────
        # Revised model (post-defence): C1 constraint removed.
        # No mandatory pre-opening.  The agent opens facilities on-demand
        # when it needs to traverse an unsafe edge (constraint 2 / Rule 3).
        self.opened_facilities = tf.zeros(
            [self.batch_beam, self.n_nodes], tf.float32)

        self.visited_customers = tf.zeros(
            [self.batch_beam, self.n_nodes], tf.float32)

        # All compartments full at the start of each episode.
        self.comp_loads = tf.fill(
            tf.stack([self.batch_beam, self.n_comp]),
            tf.constant(self.capacity_per_comp, tf.float32))

        # All compartments free (undedicated) at the start of each episode.
        # Shape [batch_beam, n_comp].  1.0 = free, 0.0 = dedicated to a customer.
        # A compartment becomes dedicated the moment it accepts a customer's demand
        # and cannot be used by any other customer until the vehicle returns to the
        # depot (where comp_free is reset to all-ones alongside comp_loads).
        self.comp_free = tf.ones(
            tf.stack([self.batch_beam, self.n_comp]), tf.float32)

        self.current_node = tf.fill(
            [self.batch_beam], tf.constant(self.depot_idx, tf.int64))

        self.route_step = tf.zeros([self.batch_beam], tf.int32)

        self.demand = tf.clip_by_value(
            tf.tile(self.demand_full, [beam_width, 1]),
            0.0, self.capacity_per_comp)

        self.d_sat = tf.zeros([self.batch_beam, self.n_nodes], tf.float32)
        self.mask  = self._build_mask()

        return State(self.comp_loads, self.demand, self.d_sat, self.mask)

    # =========================================================================
    # STEP
    # =========================================================================
    def step(self, idx, beam_parent=None):
        if beam_parent is not None:
            (idx,
             self.current_node,
             self.opened_facilities, self.visited_customers,
             self.comp_loads, self.route_step,
             self.demand, self.d_sat,
             self.comp_free) = self._gather_beam_state(idx, beam_parent)

        if len(idx.get_shape()) > 1:
            idx = tf.squeeze(idx, axis=1)
        idx = tf.cast(idx, tf.int64)

        self._apply_action(idx)
        self.mask = self._build_mask()

        return State(self.comp_loads, self.demand, self.d_sat, self.mask)

    # =========================================================================
    # APPLY ACTION
    # =========================================================================
    def _apply_action(self, idx):
        node_one_hot = tf.one_hot(idx, self.n_nodes, dtype=tf.float32)

        # ── Feature-based node classification ────────────────────────────────
        # Look up the node_type feature (channel 3) for the chosen action.
        # Values: 0.0 = depot, 1.0 = customer, 2.0 = candidate station.
        # Using feature-based typing (not hardcoded index ranges) makes the
        # mask correct for all problem sizes (lrp10, lrp20, lrp50, lrp100)
        # regardless of how the data generator assigns node indices.
        type_i   = tf.gather(self.node_type_bb,
                             tf.cast(idx, tf.int32), batch_dims=1)  # [B]
        is_depot = tf.equal(idx, tf.cast(self.depot_idx, tf.int64))
        is_cust  = tf.equal(type_i, 1.0)   # customer  (type 1 in features)
        is_fac   = tf.equal(type_i, 2.0)   # facility  (type 2 in features)

        # ── Facility: open candidate station (safety infrastructure) ─────────
        # Incurs cost C_j once.  No compartment change.
        self.opened_facilities = tf.where(
            tf.tile(tf.expand_dims(is_fac, -1), [1, self.n_nodes]),
            tf.maximum(self.opened_facilities, node_one_hot),
            self.opened_facilities)

        # ── Customer: serve demand from the first FREE compartment that fits ────
        #
        # Dedicated-compartment semantics (new behaviour):
        #   1. A compartment is eligible only when it is currently FREE
        #      (comp_free == 1) AND has remaining capacity >= demand_i.
        #   2. The first eligible compartment (lowest index) is chosen via argmax
        #      on the boolean eligibility mask.
        #   3. After assignment, comp_free for that slot is set to 0, making it
        #      dedicated to this customer.  No other customer can use it for the
        #      remainder of this sub-tour.
        #   4. When the vehicle returns to the depot, comp_free is reset to all-ones
        #      so every compartment is available again for the next sub-tour.
        demand_i = tf.reduce_sum(self.demand * node_one_hot, axis=1)  # [B]

        # Eligibility: compartment must be free AND large enough
        comp_free_bool = tf.cast(self.comp_free, tf.bool)                    # [B, n_comp]
        can_fit_load   = self.comp_loads >= tf.expand_dims(demand_i, 1)      # [B, n_comp]
        eligible       = tf.logical_and(can_fit_load, comp_free_bool)        # [B, n_comp]

        # argmax over float cast: first True wins (True=1 > False=0)
        comp_idx     = tf.cast(
            tf.argmax(tf.cast(eligible, tf.float32), axis=1), tf.int32)      # [B]
        comp_one_hot = tf.one_hot(comp_idx, self.n_comp, dtype=tf.float32)   # [B, n_comp]
        deduct       = comp_one_hot * tf.expand_dims(demand_i, -1)           # [B, n_comp]

        is_cust_exp = tf.tile(tf.expand_dims(is_cust, -1), [1, self.n_comp])

        # Deduct the customer's demand from the chosen compartment
        self.comp_loads = tf.where(
            is_cust_exp,
            self.comp_loads - deduct,
            self.comp_loads)

        # Mark the chosen compartment as dedicated (set its comp_free slot to 0)
        self.comp_free = tf.where(
            is_cust_exp,
            self.comp_free * (1.0 - comp_one_hot),
            self.comp_free)

        self.visited_customers = tf.where(
            tf.tile(tf.expand_dims(is_cust, -1), [1, self.n_nodes]),
            tf.maximum(self.visited_customers, node_one_hot),
            self.visited_customers)

        d_sat_delta = node_one_hot * tf.expand_dims(demand_i, -1)
        self.d_sat = tf.where(
            tf.tile(tf.expand_dims(is_cust, -1), [1, self.n_nodes]),
            tf.maximum(self.d_sat, d_sat_delta),
            self.d_sat)

        # ── Depot: refill all compartments and release all dedications ──────────
        is_depot_exp = tf.tile(tf.expand_dims(is_depot, -1), [1, self.n_comp])

        self.comp_loads = tf.where(
            is_depot_exp,
            tf.fill(tf.shape(self.comp_loads),
                    tf.constant(self.capacity_per_comp, tf.float32)),
            self.comp_loads)

        # Release all compartment dedications on depot return so the next
        # sub-tour starts with a full set of free compartments.
        self.comp_free = tf.where(
            is_depot_exp,
            tf.ones_like(self.comp_free),
            self.comp_free)

        self.current_node = idx
        self.route_step   = self.route_step + 1

    # =========================================================================
    # MASK  (1 = blocked, 0 = allowed)
    # =========================================================================
    def _build_mask(self):
        """
        Constructs a [batch_beam, n_nodes] binary mask  (1 = blocked, 0 = allowed).

        All node classification is now FEATURE-TYPE-BASED using node_type_bb
        (channel 3 of the fed feature tensor: 0=depot, 1=customer, 2=station).
        This replaces the previous hardcoded index-range checks
        (fac_start, n_cust, etc.) which broke for lrp20+ because the data
        generator's node layout did not always match the env's index ranges.

        Six rules (applied in order):
          1. Already-visited customers.
          2. Customer nodes whose demand exceeds ALL compartment capacities.
          3. λ safety constraint (risky edges blocked unless station opens them).
          4. Already-opened facility nodes (re-visiting wastes a decode step).
          5. Facility → facility transitions (must serve a customer between
             facility visits — prevents station-chaining for all problem sizes).
          6. Depot gating (return only when no useful service work remains).
        """
        # ── Precompute per-node type masks ────────────────────────────────────
        # node_type_bb: [batch_beam, n_nodes], values {0.0, 1.0, 2.0}
        is_cust_node = tf.equal(self.node_type_bb, 1.0)   # [B, n_nodes] bool
        is_fac_node  = tf.equal(self.node_type_bb, 2.0)   # [B, n_nodes] bool
        cust_mask_f  = tf.cast(is_cust_node, tf.float32)  # float version
        fac_mask_f   = tf.cast(is_fac_node,  tf.float32)  # float version

        # Start fully open
        mask = tf.zeros([self.batch_beam, self.n_nodes], tf.float32)

        # ── Rule 1: Already-visited customers ─────────────────────────────────
        mask = tf.maximum(mask, self.visited_customers)

        # ── Rule 2: Customers beyond all FREE compartment capacities ─────────────
        # A customer is only serviceable if there exists at least one compartment
        # that is BOTH free (undedicated) AND has remaining load >= demand.
        # Dedicated compartments (comp_free == 0) are excluded even if they
        # have residual capacity — they belong to their assigned customer only.
        demand_exp    = tf.expand_dims(self.demand, 2)          # [B, n_nodes, 1]
        loads_exp     = tf.expand_dims(self.comp_loads, 1)      # [B, 1,       n_comp]
        comp_free_exp = tf.expand_dims(self.comp_free,  1)      # [B, 1,       n_comp]
        free_and_enough = tf.logical_and(
            loads_exp >= demand_exp,                            # [B, n_nodes, n_comp]
            tf.cast(comp_free_exp, tf.bool))                    # broadcast over n_nodes
        can_serve = tf.reduce_any(free_and_enough, axis=2)      # [B, n_nodes]
        # Block only CUSTOMER nodes that cannot be served; depot and facility
        # nodes always have demand=0 so can_serve is trivially True for them.
        cap_mask = tf.cast(
            tf.logical_and(tf.logical_not(can_serve), is_cust_node),
            tf.float32)
        mask = tf.maximum(mask, cap_mask)

        # ── Rule 3: λ safety constraint ───────────────────────────────────────
        # Depot is permanently open (F_depot = 1).
        depot_col      = tf.fill([self.batch_beam],
                                 tf.constant(self.depot_idx, tf.int32))
        depot_one_hot  = tf.one_hot(depot_col, self.n_nodes, dtype=tf.float32)
        effective_open = tf.minimum(1.0, self.opened_facilities + depot_one_hot)

        current    = tf.cast(self.current_node, tf.int32)
        lambda_row = tf.gather(self.lambda_mat,  current, batch_dims=1)
        cur_open   = tf.gather(effective_open,   current, batch_dims=1)
        cur_open_t = tf.tile(tf.expand_dims(cur_open, 1), [1, self.n_nodes])
        edge_safe  = tf.minimum(1.0, lambda_row + cur_open_t + effective_open)
        mask       = tf.maximum(mask, 1.0 - edge_safe)

        # ── Rule 4: Block already-opened facilities ─────────────────────────
        # Re-visiting an opened facility wastes a decode step.
        fac_reopen_mask = self.opened_facilities * fac_mask_f
        mask = tf.maximum(mask, fac_reopen_mask)

        # ── Rule 5: Facility → facility transitions ───────────────────────────
        # If the vehicle is currently AT a facility (type=2 in features), block
        # ALL other facility nodes so the next action must be a customer or depot.
        #
        # KEY FIX: detection is now feature-based, not index-based.
        # Previously: current_node >= fac_start (= n_cust+1).
        # For lrp20 this meant checking >= 21, but data-generator facilities
        # could be at nodes 11-18 — so the check silently failed and F→F
        # transitions were not blocked.
        # Now: we look up node_type_bb for the current node; if it equals 2.0
        # (facility), we block all nodes whose type is also 2.0.
        current_type = tf.gather(self.node_type_bb,
                                 tf.cast(self.current_node, tf.int32),
                                 batch_dims=1)              # [batch_beam]
        at_facility  = tf.equal(current_type, 2.0)         # [batch_beam] bool

        # Expand to block every facility column in the rows where at_facility=True
        at_fac_col = tf.tile(
            tf.expand_dims(tf.cast(at_facility, tf.float32), 1),
            [1, self.n_nodes])                              # [batch_beam, n_nodes]
        mask = tf.maximum(mask, at_fac_col * fac_mask_f)

        # ── Fix B: مسیریابی مستقیم مشتری‌به‌مشتری وقتی ممکن است ───────────────
        # علت ساب‌تور اضافه (مشکل دوم):
        #   policy یاد گرفته همیشه از F11 عبور کند چون λ-embedding آن را جذاب
        #   می‌کند. حتی وقتی یال C_a→C_b ایمن است (λ=1)، agent به F11 می‌رود.
        #   این باعث اتلاف decode step می‌شود.
        #
        # اصلاح: وقتی agent روی یک customer است و حداقل یک customer دیگر مستقیم
        #   قابل دسترس است (بعد از اعمال Rules 1-5)، همه facility‌ها را block کن.
        #   → مسیر C_a→F11→C_b تبدیل می‌شود به C_a→C_b وقتی یال ایمن است.
        #   → Fix B برای نتیجه بهتر نیاز به retrain دارد؛ بدون retrain هم
        #     عملکرد را بهبود می‌دهد.
        #
        # current_type از Rule 5 قبلاً محاسبه شده — reuse می‌کنیم.
        at_customer_fb  = tf.equal(current_type, 1.0)          # [B] — آیا روی customer هستیم؟
        avail_custs_fb  = tf.logical_and(                       # customer‌های accessible (mask=0)
            tf.equal(mask, 0.0), is_cust_node)                  # [B, n_nodes]
        has_direct_cust = tf.reduce_any(avail_custs_fb, axis=1) # [B]
        force_direct    = tf.logical_and(at_customer_fb, has_direct_cust)  # [B]
        force_direct_exp = tf.tile(
            tf.expand_dims(tf.cast(force_direct, tf.float32), 1),
            [1, self.n_nodes])                                   # [B, n_nodes]
        mask = tf.maximum(mask, force_direct_exp * fac_mask_f)

        # ── Rule 6: Depot gating ──────────────────────────────────────────────
        # depot مجاز است فقط وقتی:
        #   (a) همه مشتریان سرویس گرفته‌اند، یا
        #   (b) وسیله نقلیه هیچ مشتری باقی‌مانده‌ای را نمی‌تواند سرویس دهد.
        remaining_demand = (self.demand
                            * cust_mask_f
                            * (1.0 - self.visited_customers))  # [B, n_nodes]
        all_served = tf.less_equal(
            tf.reduce_sum(remaining_demand, axis=1), 0.0)

        # آیا کامپارتمان آزادی هست که مشتری باقی‌مانده‌ای را سرویس دهد؟
        unvisited_cust = cust_mask_f * (1.0 - self.visited_customers)  # [B, n_nodes]
        dem_exp2       = tf.expand_dims(remaining_demand, 2)   # [B, n_nodes, 1]
        loads_exp2     = tf.expand_dims(self.comp_loads, 1)    # [B, 1,       n_comp]
        comp_free_exp2 = tf.expand_dims(self.comp_free,  1)    # [B, 1,       n_comp]
        can_fit2 = tf.logical_and(
            tf.greater_equal(loads_exp2, dem_exp2),
            tf.cast(comp_free_exp2, tf.bool))
        useful_per_node = tf.logical_and(
            tf.reduce_any(can_fit2, axis=2),
            tf.cast(unvisited_cust > 0.5, tf.bool))            # [B, n_nodes]
        vehicle_useful  = tf.reduce_any(useful_per_node, axis=1)

        # ── Fix A: وقتی vehicle پر است، facility‌ها را block کن ─────────────
        # علت ساب‌تور اضافه (مشکل اصلی):
        #   وقتی comp_free=[F,F,F,F] (همه کامپارتمان پر)، depot مجاز است
        #   اما F11 هم مجاز است. policy به جای depot→F11 می‌رود که یک decode
        #   step اضافه می‌خورد و C8 از محدوده decode خارج می‌شود.
        #
        #   بدون Fix A:
        #     C4→F11→depot (9 steps در sub-tour1)، C8 از decode_len=17 خارج می‌شود
        #   با Fix A:
        #     C4→depot (9-1=8 steps)، C8 جا می‌شود → 2 vehicles ✓
        #
        #   این Fix را می‌توان بدون retrain اعمال کرد — فقط depot خروج مستقیم.
        vehicle_exhausted     = tf.logical_not(vehicle_useful)  # [B]
        vehicle_exhausted_exp = tf.tile(
            tf.expand_dims(tf.cast(vehicle_exhausted, tf.float32), 1),
            [1, self.n_nodes])                                   # [B, n_nodes]
        mask = tf.maximum(mask, vehicle_exhausted_exp * fac_mask_f)

        # depot gating (Rule 6 اصلی)
        depot_allowed = tf.logical_or(all_served,
                                      tf.logical_not(vehicle_useful))
        depot_blocked = tf.cast(tf.logical_not(depot_allowed), tf.float32)

        mask = tf.concat([
            tf.expand_dims(depot_blocked, 1), mask[:, 1:]
        ], axis=1)

        return mask

    # =========================================================================
    # BEAM-SEARCH STATE GATHER
    # =========================================================================
    def _gather_beam_state(self, idx, beam_parent):
        batch_size_dyn   = tf.shape(self.current_node)[0] // self.beam_width
        beam_seq         = tf.expand_dims(
            tf.tile(tf.cast(tf.range(batch_size_dyn), tf.int64),
                    [self.beam_width]), 1)
        batched_beam_idx = (beam_seq
                            + tf.cast(batch_size_dyn, tf.int64) * beam_parent)

        def _g(t):
            return tf.gather_nd(t, batched_beam_idx)

        return (_g(idx),
                _g(self.current_node),
                _g(self.opened_facilities), _g(self.visited_customers),
                _g(self.comp_loads), _g(self.route_step),
                _g(self.demand), _g(self.d_sat),
                _g(self.comp_free))
