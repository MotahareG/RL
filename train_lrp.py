# train_lrp.py  —  Fixed-Dataset Edition (Nazari et al. NeurIPS 2018)
# ==============================================================================
# Training paradigm change:
#
#   BEFORE (online):
#     for step in range(n_train):
#         batch = dataGen.get_train_next()   # fresh random batch every step
#         agent.run_train_step()
#
#   NOW (fixed dataset, Nazari §4):
#     dataGen.generate_fixed_dataset(n_instances)   # generate ONCE
#     for epoch in range(n_epochs):
#         for batch_feat, batch_lam in dataGen.get_epoch_batches(batch_size):
#             agent.run_train_step(features=batch_feat, lambda_data=batch_lam)
#
# Why this matters:
#   • Stable gradient signal — same instances seen multiple times.
#   • Per-instance λ — every instance has its own λ matrix derived from its
#     specific customer positions (not a single shared representative matrix).
#   • Epoch-level shuffling prevents order memorisation.
#   • Directly comparable to Nazari et al. results.
#
# New CLI flags (see configs.py):
#   --n_instances 10000    number of pre-generated training instances
#   --n_epochs    100      training epochs over the fixed dataset
#   --use_fixed_dataset    True | False  (default True)
# ==============================================================================

from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time, os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = os.path.isdir('/content/drive')

from configs            import ParseParams
from lrp_env            import Env
from data_generator_lrp import LRPDataGenerator
from attention_agent    import RLAgent
from visualize_routes   import save_convergence_plot
import misc_utils as utils

try:
    from vrp_attention import AttentionVRPActor, AttentionVRPCritic
except ImportError:
    try:
        from VRP.vrp_attention import AttentionVRPActor, AttentionVRPCritic
    except ImportError:
        print("[ERROR] Cannot import vrp_attention.")
        sys.exit(1)


# ==============================================================================
# Reward function  (revised model — curriculum learning)
# ==============================================================================
def lrp_reward_func(
        actions,
        opened_facilities     = None,
        facility_opening_cost = 10.0 / 3650.0):
    """
    R(τ) = −( routing_cost + facility_cost )

    facility_opening_cost can be a TF placeholder — fed differently
    during training (high, e.g. 1.0) vs evaluation (real: C_j/H ≈ 0.00274).
    """
    sample   = tf.stack(actions, axis=0)                         # [T, B, 2]
    shifted  = tf.concat([tf.expand_dims(sample[-1], 0),
                          sample[:-1]], axis=0)                  # [T, B, 2]
    route_lengths = tf.reduce_sum(
        tf.sqrt(tf.reduce_sum(tf.square(shifted - sample), axis=2)),
        axis=0)                                                  # [B]

    fac_cost = tf.constant(0.0)
    if opened_facilities is not None:
        n_opened = tf.reduce_sum(opened_facilities, axis=1)      # [B]
        fac_cost = n_opened * facility_opening_cost               # works with both float & TF tensor

    return -(route_lengths + fac_cost)




# ==============================================================================
# 2-opt local search  (post-processing — standard in RL+CO literature)
#
# References:
#   Croes (1958) "A method for solving traveling-salesman problems"
#   Kool et al. (2019) "Attention, Learn to Solve Routing Problems!"
#   da Costa et al. (2020) "Learning 2-opt Heuristics for TSP via Deep RL"
# ==============================================================================

def _tour_length(tour, coords):
    """Euclidean length of a closed tour given as list of node indices."""
    c = 0.0
    for k in range(len(tour) - 1):
        dx = coords[tour[k], 0] - coords[tour[k+1], 0]
        dy = coords[tour[k], 1] - coords[tour[k+1], 1]
        c += np.sqrt(dx*dx + dy*dy)
    return c


def _two_opt_tour(tour, coords, lam=None, max_iter=200):
    """
    Intra-tour 2-opt: reverse sub-segments to shorten a single tour.
    Respects safety constraint if lam is provided.
    tour: [depot, c1, c2, ..., cn, depot]
    Returns (improved_tour, improved_length).
    """
    best = list(tour)
    best_cost = _tour_length(best, coords)
    n = len(best)

    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Reverse segment best[i..j]
                cand = best[:i] + best[i:j+1][::-1] + best[j+1:]

                # Safety check: only new edges (i-1,i) and (j,j+1) matter
                if lam is not None:
                    a, b = cand[i-1], cand[i]
                    c, d = cand[j],   cand[j+1]
                    if lam[a, b] == 0 or lam[c, d] == 0:
                        continue

                c_cost = _tour_length(cand, coords)
                if c_cost < best_cost - 1e-10:
                    best = cand
                    best_cost = c_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best, best_cost


def _parse_tours(route_seq, depot=0):
    """Split a flat decode sequence into individual vehicle tours."""
    tours = []
    cur = [depot]
    for nid in route_seq:
        nid = int(nid)
        cur.append(nid)
        if nid == depot and len(cur) > 2:
            tours.append(cur)
            cur = [depot]
    if len(cur) > 2:
        cur.append(depot)
        tours.append(cur)
    return tours if tours else [[depot, depot]]


def apply_two_opt(route_seq, coords, lam=None):
    """
    Apply 2-opt to every sub-tour in a route sequence.
    Returns total improved cost.
    """
    tours = _parse_tours(route_seq)
    total = 0.0
    for tour in tours:
        if len(tour) <= 3:          # depot→x→depot — nothing to improve
            total += _tour_length(tour, coords)
        else:
            _, tc = _two_opt_tour(tour, coords, lam)
            total += tc
    return total


def train():
    args, prt = ParseParams()

    prt.print_out("=" * 70)
    prt.print_out("LRP-RL  —  Fixed-Dataset Training  (Nazari et al. paradigm)")
    if IN_COLAB:
        prt.print_out("[Colab] T4 GPU detected.")
    prt.print_out("=" * 70)

    args.setdefault('facility_opening_cost', 10.0 / 3650.0)  # C_j/H (real model cost for eval)
    args.setdefault('facility_opening_cost_train', 1.0)  # curriculum: high cost for training
    args.setdefault('is_lrp', True)
    args['task_name'] = 'lrp'

    use_fixed = args.get('use_fixed_dataset', True)
    n_instances = args.get('n_instances', 10000)
    n_epochs    = args.get('n_epochs',    100)

    # Figures / checkpoint dirs
    if IN_COLAB:
        figures_dir = '/content/figures'
        os.makedirs(figures_dir, exist_ok=True)
    else:
        figures_dir = os.path.join(args['log_dir'], 'figures')
        os.makedirs(figures_dir, exist_ok=True)
    args['figures_dir'] = figures_dir

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess   = tf.Session(config=config)

    # ------------------------------------------------------------------
    # 1. Data generator — Stage 1 (KDE) + Stage 2 (K-means)
    # ------------------------------------------------------------------
    prt.print_out("\n[1/6] Initialising data generator ...")
    collision_path = args.get('collision_csv_path', 'CA_AV_Collision_2019-2024.csv')

    dataGen = LRPDataGenerator(
        args=args,
        collision_csv_path=collision_path,
        bounding_box=((32.5, -124.5), (42.0, -114.0)),
        grid_size=200)
    prt.print_out(f"  {len(dataGen.collision_points)} collision points loaded.")
    args['collision_points'] = dataGen.collision_points

    # ------------------------------------------------------------------
    # 2. Representative λ for args (used by visualisation / env helper)
    # ------------------------------------------------------------------
    prt.print_out("\n[2/6] Computing representative λ matrix (seed=42) ...")
    prt.print_out("  (per-instance λ computed for every training instance separately)")
    rep_inst = dataGen.generate_instance(
        args.get('n_cust', 10),
        args.get('n_fac',   5),
        seed=42, compute_lambda=True)
    args['lambda_mat'] = rep_inst['lambda']
    args['node_type']  = rep_inst['node_type']
    lam = args['lambda_mat']

    # ── λ statistics (C1 constraint removed in revised model) ──────────
    n   = lam.shape[0]
    risky = int((lam == 0).sum()) // 2
    total = n * (n - 1) // 2
    prt.print_out(f"  Representative λ: {risky}/{total} risky ({100*risky/total:.1f}%)")
    prt.print_out(f"  Facility cost (C_j/H): {args.get('facility_opening_cost'):.6f}")

    # ------------------------------------------------------------------
    # 3. Fixed dataset generation  (Stage 3: per-instance λ)
    # ------------------------------------------------------------------
    if use_fixed:
        prt.print_out(f"\n[3/6] Pre-generating fixed dataset ({n_instances:,} instances) ...")
        t0 = time.time()
        dataGen.generate_fixed_dataset(
            n_instances=n_instances,
            seed=args['random_seed'],
            n_lambda_samples=10,
            verbose=True)
        prt.print_out(f"  Dataset generated in {time.time()-t0:.1f}s.")

        steps_per_epoch = dataGen.steps_per_epoch
        total_steps     = n_epochs * steps_per_epoch
        prt.print_out(f"  Steps/epoch: {steps_per_epoch}  |  Total steps: {total_steps:,}")
    else:
        prt.print_out("\n[3/6] Using online (fresh-batch-per-step) mode.")
        total_steps     = args['n_train']
        steps_per_epoch = None
        n_epochs        = 1   # single "epoch" = n_train steps

    args['n_train'] = total_steps  # keep for log_interval compatibility

    # ------------------------------------------------------------------
    # 4. Environment
    # ------------------------------------------------------------------
    prt.print_out("\n[4/6] Creating environment ...")
    env = Env(args)
    prt.print_out("[OK] Environment ready.")

    # ------------------------------------------------------------------
    # 5. Agent
    # ------------------------------------------------------------------
    prt.print_out("\n[5/6] Building RL agent ...")
    agent = RLAgent(
        args=args, prt=prt, env=env, dataGen=dataGen,
        reward_func=lrp_reward_func,
        clAttentionActor=AttentionVRPActor,
        clAttentionCritic=AttentionVRPCritic,
        is_train=args['is_train'])
    prt.print_out("[OK] Agent ready.")

    # ------------------------------------------------------------------
    # 6. Initialise TF variables
    # ------------------------------------------------------------------
    prt.print_out("\n[6/6] Initialising TF variables ...")
    agent.Initialize(sess)
    prt.print_out("[OK] Variables initialised.")

    # ------------------------------------------------------------------
    # Training loop — epoch-based (fixed dataset) or step-based (online)
    # ------------------------------------------------------------------
    prt.print_out("\n" + "=" * 70)
    if use_fixed:
        prt.print_out(
            f"Fixed-Dataset Training\n"
            f"  Instances  : {n_instances:,}\n"
            f"  Epochs     : {n_epochs}\n"
            f"  Batch size : {args['batch_size']}\n"
            f"  Steps/epoch: {steps_per_epoch}\n"
            f"  Total steps: {total_steps:,}\n"
            f"  Candidates : {args.get('n_fac',5)}  (agent opens 0…{args.get('n_fac',5)})\n"
            f"  Facility cost (train): {args['facility_opening_cost_train']}\n"
            f"  Facility cost (eval) : {args['facility_opening_cost']:.6f}")
    else:
        prt.print_out(
            f"Online Training  ({total_steps:,} steps, fresh batch per step)")
    prt.print_out("=" * 70)

    history = {
        'step': [], 'reward': [], 'actor_loss': [], 'critic_loss': [],
        'epoch_reward': [], 'epoch_idx': []
    }

    global_step  = 0
    start_time   = time.time()
    log_t        = time.time()

    # ── EPOCH LOOP ────────────────────────────────────────────────────────────
    for epoch in range(n_epochs):

        epoch_rewards = []
        epoch_t       = time.time()

        if use_fixed:
            batch_iter = dataGen.get_epoch_batches(args['batch_size'], shuffle=True)
        else:
            def _online_batches():
                for _ in range(total_steps):
                    yield dataGen.get_train_next()
            batch_iter = _online_batches()

        # ── BATCH LOOP ─────────────────────────────────────────────────────────
        for batch_features, batch_lambda in batch_iter:

            try:
                summary = agent.run_train_step(
                    features=batch_features,
                    lambda_data=batch_lambda)
            except Exception as e:
                prt.print_out(f"[ERROR] step {global_step}: {e}")
                global_step += 1
                continue

            R_mean      = float(np.mean(summary[6])) if len(summary) > 6 else 0.0
            v_mean      = float(np.mean(summary[7])) if len(summary) > 7 else 0.0
            actor_loss  = float(summary[2])           if len(summary) > 2 else 0.0
            critic_loss = float(summary[3])           if len(summary) > 3 else 0.0

            epoch_rewards.append(R_mean)
            history['step'].append(global_step)
            history['reward'].append(R_mean)
            history['actor_loss'].append(actor_loss)
            history['critic_loss'].append(critic_loss)

            # ── Logging ───────────────────────────────────────────────────────
            if global_step % args['log_interval'] == 0 and global_step > 0:
                elapsed = time.time() - log_t
                prt.print_out(
                    f"\n[Epoch {epoch+1}/{n_epochs} | Step {global_step}]  "
                    f"R={R_mean:.4f}  v={v_mean:.4f}  "
                    f"a_loss={actor_loss:.5f}  c_loss={critic_loss:.5f}  "
                    f"t={elapsed:.1f}s")
                log_t = time.time()

            # ── Evaluation ────────────────────────────────────────────────────
            if global_step % args['test_interval'] == 0 and global_step > 0:
                prt.print_out(f"\n--- Eval @ step {global_step} (epoch {epoch+1}) ---")
                try:
                    agent.inference(args['infer_type'])
                except Exception as e:
                    prt.print_out(f"[ERROR] eval: {e}")

            # ── Checkpoint ────────────────────────────────────────────────────
            if global_step % args['save_interval'] == 0 and global_step > 0:
                ckpt = agent.saver.save(
                    sess, os.path.join(args['model_dir'], 'model.ckpt'),
                    global_step=global_step)
                prt.print_out(f"  Checkpoint saved: {ckpt}")

            global_step += 1

        # ── End-of-epoch summary ──────────────────────────────────────────────
        epoch_elapsed = time.time() - epoch_t
        epoch_avg     = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        history['epoch_reward'].append(epoch_avg)
        history['epoch_idx'].append(epoch + 1)

        prt.print_out(
            f"\n{'─'*60}\n"
            f"Epoch {epoch+1}/{n_epochs} complete | "
            f"avg_R = {epoch_avg:.4f} | "
            f"steps = {len(epoch_rewards)} | "
            f"time = {epoch_elapsed:.1f}s\n"
            f"{'─'*60}")

    # ------------------------------------------------------------------
    # Final evaluation, timing, and save
    # ------------------------------------------------------------------
    training_elapsed = time.time() - start_time

    prt.print_out("\n" + "=" * 70)
    prt.print_out("Training complete — running post-training evaluation + timing ...")
    prt.print_out("=" * 70)

    # ── Save checkpoint ────────────────────────────────────────────────────────
    final = agent.saver.save(
        sess, os.path.join(args['model_dir'], 'final_model.ckpt'))
    prt.print_out(f"[OK] Final model: {final}")

    # ── Build test set ─────────────────────────────────────────────────────────
    prt.print_out("\n  Building test set ...")
    features, lambda_data = dataGen.get_test_all()
    n_test = features.shape[0]
    prt.print_out(f"  Test set: {n_test} instances  shape={features.shape}")

    # ── Warm-up pass (excluded from timing) ───────────────────────────────────
    prt.print_out("  Warm-up pass (not timed)...")
    sess.run(
        agent.val_summary_greedy,
        feed_dict={agent.env.input_data:     features[:args['batch_size']],
                   agent.env.lambda_ph:      lambda_data[:args['batch_size']],
                   agent.decodeStep.dropout: 0.0})

    # ── Batch inference — 5 runs for stable average ────────────────────────────
    N_REPEATS   = 5
    batch_times = []
    last_R      = None
    prt.print_out(f"  Batch inference ({n_test} instances × {N_REPEATS} runs)...")
    for _ in range(N_REPEATS):
        t0 = time.time()
        R, v, logprobs, actions, idxs, batch, _, routes = sess.run(
            agent.val_summary_greedy,
            feed_dict={agent.env.input_data:     features,
                       agent.env.lambda_ph:      lambda_data,
                       agent.decodeStep.dropout: 0.0})
        batch_times.append(time.time() - t0)
        last_R = R

    last_R       = np.concatenate(np.split(np.expand_dims(last_R, 1), 1, axis=0), 1)
    last_R       = np.amin(last_R, 1, keepdims=False)
    batch_avg_s  = float(np.mean(batch_times))
    per_inst_ms  = batch_avg_s / n_test * 1000.0

    # ── Single-instance timing — 100 runs ─────────────────────────────────────
    N_SINGLE     = 100
    single_ms    = []
    prt.print_out(f"  Single-instance timing ({N_SINGLE} runs)...")
    for i in range(N_SINGLE):
        t0 = time.time()
        sess.run(
            agent.val_summary_greedy,
            feed_dict={agent.env.input_data:     features[i:i+1],
                       agent.env.lambda_ph:      lambda_data[i:i+1],
                       agent.decodeStep.dropout: 0.0})
        single_ms.append((time.time() - t0) * 1000.0)

    s_avg = float(np.mean(single_ms))
    s_std = float(np.std(single_ms))

    # ── Print summary ──────────────────────────────────────────────────────────
    training_hms = time.strftime('%H:%M:%S', time.gmtime(training_elapsed))
    sep = "=" * 70

    prt.print_out(f"\n{sep}")
    prt.print_out("RESULTS SUMMARY")
    prt.print_out(sep)
    prt.print_out(f"  Training time          : {training_hms}  ({training_elapsed:.1f}s total)")
    prt.print_out(f"  Epochs completed       : {n_epochs}")
    prt.print_out(f"  Training instances     : {n_instances:,}")
    prt.print_out(f"")
    prt.print_out(f"  Test set size          : {n_test} instances")
    prt.print_out(f"  Avg reward (greedy)    : {float(np.mean(last_R)):.4f}")
    prt.print_out(f"  Std reward             : {float(np.std(last_R)):.4f}")
    prt.print_out(f"")
    prt.print_out(f"  Batch inference time   : {batch_avg_s:.3f}s  ({per_inst_ms:.3f} ms/instance)")
    prt.print_out(f"  Single-instance time   : {s_avg:.3f} ± {s_std:.3f} ms")
    prt.print_out(f"")
    prt.print_out(f"  ► For thesis comparison table:")
    prt.print_out(f"      Training time      : {training_hms}")
    prt.print_out(f"      Inference time     : {s_avg:.1f} ± {s_std:.1f} ms  (greedy, single instance)")
    prt.print_out(f"      Avg reward (test)  : {float(np.mean(last_R)):.4f}")
    prt.print_out(sep)

    # ==================================================================
    # Beam Search evaluation
    # ==================================================================
    beam_w = args.get('beam_width', 10)
    prt.print_out(f"\n{'='*70}")
    prt.print_out(f"BEAM SEARCH evaluation (beam_width={beam_w})")
    prt.print_out(f"{'='*70}")

    # Beam search — batch inference
    prt.print_out(f"  Running beam search on {n_test} instances ...")
    t0_beam = time.time()
    R_beam, _, _, _, _, _, _, routes_beam = sess.run(
        agent.val_summary_beam,
        feed_dict={agent.env.input_data:     features,
                   agent.env.lambda_ph:      lambda_data,
                   agent.decodeStep.dropout: 0.0})
    beam_time = time.time() - t0_beam

    # Beam returns [n_test * beam_w] results — pick best per instance
    R_beam_2d = R_beam.reshape(n_test, beam_w)
    best_beam_idx = np.argmax(R_beam_2d, axis=1)          # best beam per inst
    R_beam_best   = R_beam_2d[np.arange(n_test), best_beam_idx]

    avg_greedy = float(np.mean(-last_R))
    avg_beam   = float(np.mean(-R_beam_best))
    prt.print_out(f"  Beam search done in {beam_time:.1f}s")
    prt.print_out(f"  Greedy avg cost : {avg_greedy:.4f}")
    prt.print_out(f"  Beam   avg cost : {avg_beam:.4f}")
    prt.print_out(f"  Improvement     : {avg_greedy - avg_beam:.4f} ({(avg_greedy-avg_beam)/avg_greedy*100:.1f}%)")

    # ==================================================================
    # SAMPLING evaluation  (Kool et al. 2019 approach)
    #   Run the stochastic policy N times, keep the best per instance.
    #   Same trained model, no retraining — just smarter inference.
    # ==================================================================
    N_SAMPLES = 128
    prt.print_out(f"\n{'='*70}")
    prt.print_out(f"SAMPLING evaluation ({N_SAMPLES} stochastic samples per instance)")
    prt.print_out(f"{'='*70}")

    best_R_sample      = np.full(n_test, -1e9)
    best_routes_sample = np.zeros_like(routes)

    t0_sample = time.time()
    for s in range(N_SAMPLES):
        R_s, _, _, _, _, _, _, routes_s = sess.run(
            agent.train_summary,
            feed_dict={agent.env.input_data:     features,
                       agent.env.lambda_ph:      lambda_data,
                       agent.decodeStep.dropout: 0.0})
        improved = R_s > best_R_sample
        for idx_i in np.where(improved)[0]:
            best_routes_sample[:, idx_i, :] = routes_s[:, idx_i, :]
        best_R_sample[improved] = R_s[improved]
        if (s + 1) % 32 == 0:
            prt.print_out(f"    {s+1}/{N_SAMPLES} — best avg so far: "
                          f"{float(np.mean(-best_R_sample)):.4f}")

    sample_time = time.time() - t0_sample
    avg_sample  = float(np.mean(-best_R_sample))
    prt.print_out(f"  Done in {sample_time:.1f}s")
    prt.print_out(f"  Greedy : {avg_greedy:.4f}")
    prt.print_out(f"  Sample : {avg_sample:.4f}")
    prt.print_out(f"  Improve: {avg_greedy - avg_sample:.4f} "
                  f"({(avg_greedy-avg_sample)/avg_greedy*100:.1f}%)")

    # ==================================================================
    # Facility Enumeration — greedy + beam
    # ==================================================================
    n_fac     = args.get('n_fac', 8)
    fac_start = args['n_cust'] + 1
    real_cf   = args['facility_opening_cost']
    n_combos  = 2 ** n_fac

    prt.print_out(f"\n{'='*70}")
    prt.print_out(f"FACILITY ENUMERATION — {n_combos} combos × greedy")
    prt.print_out(f"{'='*70}")

    # ── Greedy enumeration (all test instances) ───────────────────────────
    best_costs_greedy_enum = np.full(n_test, 1e9)
    best_config_greedy     = [[] for _ in range(n_test)]

    for combo in range(n_combos):
        open_facs = [fac_start + f for f in range(n_fac) if combo & (1 << f)]
        n_open    = len(open_facs)
        lam_mod   = lambda_data.copy()
        for fj in open_facs:
            lam_mod[:, fj, :] = 1
            lam_mod[:, :, fj] = 1

        R_c, _, _, _, _, _, _, _ = sess.run(
            agent.val_summary_greedy,
            feed_dict={agent.env.input_data:     features,
                       agent.env.lambda_ph:      lam_mod,
                       agent.decodeStep.dropout: 0.0})
        costs = -R_c + n_open * real_cf
        improved = costs < best_costs_greedy_enum
        best_costs_greedy_enum[improved] = costs[improved]
        for i in np.where(improved)[0]:
            best_config_greedy[int(i)] = open_facs[:]

    avg_greedy_enum = float(np.mean(best_costs_greedy_enum))
    prt.print_out(f"  Greedy+Enum avg cost: {avg_greedy_enum:.4f}")

    # ── Beam + enumeration (instance 0 only — for GAMS comparison) ────────
    prt.print_out(f"\n  Beam+Enum for instance 0 (GAMS comparison) ...")
    feat_0 = features[0:1]
    lam_0  = lambda_data[0]

    best_cost_beam_enum = 1e9
    best_config_beam    = []
    best_routes_beam_0  = None

    for combo in range(n_combos):
        open_facs = [fac_start + f for f in range(n_fac) if combo & (1 << f)]
        n_open    = len(open_facs)
        lam_mod_0 = lam_0.copy()
        for fj in open_facs:
            lam_mod_0[fj, :] = 1
            lam_mod_0[:, fj] = 1

        R_b, _, _, _, _, _, _, rts_b = sess.run(
            agent.val_summary_beam,
            feed_dict={agent.env.input_data:     feat_0,
                       agent.env.lambda_ph:      lam_mod_0[None],
                       agent.decodeStep.dropout: 0.0})
        best_r = float(np.max(R_b))
        cost   = -best_r + n_open * real_cf
        if cost < best_cost_beam_enum:
            best_cost_beam_enum = cost
            best_config_beam    = open_facs[:]
            best_beam_idx_0     = int(np.argmax(R_b))
            best_routes_beam_0  = rts_b[:, best_beam_idx_0, 0]
            best_lam_0          = lam_mod_0

    # ==================================================================
    # 2-OPT LOCAL SEARCH  (applied to all methods)
    # ==================================================================
    prt.print_out(f"\n{'='*70}")
    prt.print_out(f"2-OPT LOCAL SEARCH — improving all solutions")
    prt.print_out(f"{'='*70}")
    coords_all = features[:, :, :2]  # [n_test, n_nodes, 2]

    # ── 2-opt on greedy routes ─────────────────────────────────────────────
    prt.print_out(f"  Greedy + 2-opt ...")
    t0_2opt = time.time()
    greedy_2opt_costs = np.zeros(n_test)
    for i in range(n_test):
        greedy_2opt_costs[i] = apply_two_opt(
            routes[:, i, 0], coords_all[i], lam=lambda_data[i])
    avg_greedy_2opt = float(np.mean(greedy_2opt_costs))
    prt.print_out(f"    {avg_greedy:.4f} → {avg_greedy_2opt:.4f}"
                  f"  (Δ = {avg_greedy - avg_greedy_2opt:.4f},"
                  f" {(avg_greedy-avg_greedy_2opt)/avg_greedy*100:.1f}%)"
                  f"  [{time.time()-t0_2opt:.1f}s]")

    # ── 2-opt on beam routes ───────────────────────────────────────────────
    prt.print_out(f"  Beam + 2-opt ...")
    t0_2opt = time.time()
    beam_2opt_costs = np.zeros(n_test)
    # routes_beam shape: [decode_len, n_test*beam_w, 1]
    for i in range(n_test):
        bi = int(best_beam_idx[i])
        flat_idx = i * beam_w + bi
        beam_2opt_costs[i] = apply_two_opt(
            routes_beam[:, flat_idx, 0], coords_all[i], lam=lambda_data[i])
    avg_beam_2opt = float(np.mean(beam_2opt_costs))
    prt.print_out(f"    {avg_beam:.4f} → {avg_beam_2opt:.4f}"
                  f"  (Δ = {avg_beam - avg_beam_2opt:.4f},"
                  f" {(avg_beam-avg_beam_2opt)/avg_beam*100:.1f}%)"
                  f"  [{time.time()-t0_2opt:.1f}s]")

    # ── 2-opt on beam+enum for instance 0 ─────────────────────────────────
    inst0_beam_enum_2opt = apply_two_opt(
        best_routes_beam_0, coords_all[0], lam=best_lam_0)
    inst0_beam_enum_2opt += len(best_config_beam) * real_cf
    prt.print_out(f"  Beam+Enum+2opt inst 0: {best_cost_beam_enum:.4f} → {inst0_beam_enum_2opt:.4f}")

    # ── 2-opt on sampling routes ───────────────────────────────────────────
    prt.print_out(f"  Sampling + 2-opt ...")
    t0_s2 = time.time()
    sample_2opt_costs = np.zeros(n_test)
    for i in range(n_test):
        sample_2opt_costs[i] = apply_two_opt(
            best_routes_sample[:, i, 0], coords_all[i], lam=lambda_data[i])
    avg_sample_2opt = float(np.mean(sample_2opt_costs))
    prt.print_out(f"    {avg_sample:.4f} → {avg_sample_2opt:.4f}"
                  f"  [{time.time()-t0_s2:.1f}s]")

    # ── Sampling + enum + 2-opt for instance 0 ────────────────────────────
    prt.print_out(f"\n  Sampling+Enum+2opt for instance 0 ...")
    N_SAMPLES_ENUM = 128
    best_cost_sample_enum_0 = 1e9
    best_config_sample_0    = []

    for combo in range(n_combos):
        open_facs = [fac_start + f for f in range(n_fac) if combo & (1 << f)]
        n_open    = len(open_facs)
        lam_mod_0 = lam_0.copy()
        for fj in open_facs:
            lam_mod_0[fj, :] = 1
            lam_mod_0[:, fj] = 1

        # Sample N times with this facility config
        best_r_this = -1e9
        best_rt_this = None
        for _ in range(N_SAMPLES_ENUM):
            R_s, _, _, _, _, _, _, rts_s = sess.run(
                agent.train_summary,
                feed_dict={agent.env.input_data:     feat_0,
                           agent.env.lambda_ph:      lam_mod_0[None],
                           agent.decodeStep.dropout: 0.0})
            if R_s[0] > best_r_this:
                best_r_this = R_s[0]
                best_rt_this = rts_s[:, 0, 0].copy()

        # Apply 2-opt
        cost_2opt = apply_two_opt(best_rt_this, coords_all[0], lam=lam_mod_0)
        total = cost_2opt + n_open * real_cf
        if total < best_cost_sample_enum_0:
            best_cost_sample_enum_0 = total
            best_config_sample_0    = open_facs[:]

    prt.print_out(f"  Best: {best_cost_sample_enum_0:.4f}  config: "
                  f"{', '.join(f'F{f}' for f in best_config_sample_0) if best_config_sample_0 else 'none'}")

    # ── Final comparison table ─────────────────────────────────────────────
    inst0_greedy      = float(-last_R[0])
    inst0_beam        = float(-R_beam_best[0])
    inst0_greedy_enum = float(best_costs_greedy_enum[0])
    inst0_greedy_2opt = float(greedy_2opt_costs[0])
    inst0_beam_2opt   = float(beam_2opt_costs[0])
    inst0_sample      = float(-best_R_sample[0])
    inst0_sample_2opt = float(sample_2opt_costs[0])

    prt.print_out(f"\n{'='*70}")
    prt.print_out(f"FINAL COMPARISON TABLE")
    prt.print_out(f"{'='*70}")
    prt.print_out(f"")
    prt.print_out(f"  {'Method':<35} {'Avg (1000)':>10} {'Inst 0':>10}")
    prt.print_out(f"  {'─'*57}")
    prt.print_out(f"  {'RL greedy':<35} {avg_greedy:>10.4f} {inst0_greedy:>10.4f}")
    prt.print_out(f"  {'RL greedy + 2-opt':<35} {avg_greedy_2opt:>10.4f} {inst0_greedy_2opt:>10.4f}")
    prt.print_out(f"  {'RL beam (w={})'.format(beam_w):<35} {avg_beam:>10.4f} {inst0_beam:>10.4f}")
    prt.print_out(f"  {'RL beam + 2-opt':<35} {avg_beam_2opt:>10.4f} {inst0_beam_2opt:>10.4f}")
    prt.print_out(f"  {'RL sampling (×{})'.format(N_SAMPLES):<35} {avg_sample:>10.4f} {inst0_sample:>10.4f}")
    prt.print_out(f"  {'RL sampling + 2-opt':<35} {avg_sample_2opt:>10.4f} {inst0_sample_2opt:>10.4f}")
    prt.print_out(f"  {'RL greedy + enum':<35} {avg_greedy_enum:>10.4f} {inst0_greedy_enum:>10.4f}")
    prt.print_out(f"  {'RL beam + enum':<35} {'—':>10} {best_cost_beam_enum:>10.4f}")
    prt.print_out(f"  {'RL beam + enum + 2-opt':<35} {'—':>10} {inst0_beam_enum_2opt:>10.4f}")
    prt.print_out(f"  {'RL sample+enum+2opt (×{})'.format(N_SAMPLES_ENUM):<35} {'—':>10} {best_cost_sample_enum_0:>10.4f}")
    prt.print_out(f"")
    prt.print_out(f"  Instance 0 — best config: "
                  f"{', '.join(f'F{f}' for f in best_config_sample_0) if best_config_sample_0 else 'none'}")
    prt.print_out(sep)

    # ── Save timing to CSV ─────────────────────────────────────────────────────
    import csv as _csv
    timing_path = os.path.join(args['log_dir'], 'results_summary.csv')
    with open(timing_path, 'w', newline='') as _f:
        _w = _csv.writer(_f)
        _w.writerow(['metric', 'value', 'unit'])
        _w.writerow(['task',             args['task'],                            ''])
        _w.writerow(['n_cust',           args.get('n_cust', ''),                  ''])
        _w.writerow(['n_fac',            args.get('n_fac', ''),                   ''])
        _w.writerow(['n_train',          n_instances,                             'instances'])
        _w.writerow(['n_epochs',         n_epochs,                                ''])
        _w.writerow(['training_time_s',  f'{training_elapsed:.1f}',              's'])
        _w.writerow(['training_time',    training_hms,                            'HH:MM:SS'])
        _w.writerow(['n_test',           n_test,                                  'instances'])
        _w.writerow(['avg_reward',       f'{float(np.mean(last_R)):.6f}',        ''])
        _w.writerow(['std_reward',       f'{float(np.std(last_R)):.6f}',         ''])
        _w.writerow(['batch_time_s',     f'{batch_avg_s:.4f}',                   's'])
        _w.writerow(['per_inst_ms',      f'{per_inst_ms:.4f}',                   'ms'])
        _w.writerow(['single_mean_ms',   f'{s_avg:.4f}',                         'ms'])
        _w.writerow(['single_std_ms',    f'{s_std:.4f}',                         'ms'])
    prt.print_out(f"[OK] Results saved: {timing_path}")

    # ── Convergence plot ───────────────────────────────────────────────────────
    conv_path = os.path.join(args['figures_dir'], 'convergence.png')
    save_convergence_plot(history, save_path=conv_path)
    prt.print_out(f"[OK] Convergence plot: {conv_path}")

    # Save training history for multi-scale comparison plot
    import json as _json
    hist_path = os.path.join(args['log_dir'], 'training_history.json')
    with open(hist_path, 'w') as _f:
        _json.dump(history, _f)
    prt.print_out(f"[OK] Training history: {hist_path}")

    sess.close()


if __name__ == "__main__":
    train()
