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
# Reward function  (model.pdf aligned — NO violation_penalty)
# ==============================================================================
def lrp_reward_func(
        actions,
        opened_facilities     = None,
        facility_opening_cost = 10.0 / 3650.0):
    """
    R(τ) = −( routing_cost + facility_cost )

    دقیقاً برابر با منفی تابع هدف مدل اصلاح‌شده:
        min  Σ_j (C_j/H)·F_j  +  Σ_{i,j,v} c_AV·d_ij·x_ijv

    حیث C_j = 10, H = 3650 → C_j/H ≈ 0.00274

    اجرای قیدها کاملاً از طریق masking انجام می‌شود:
      • قید ۲ (ایمنی مسیریابی): Rule 3 — x_ijv ≤ F_i+F_j+λ_ij
      • قید ۴ (همه مشتریان): Rule 6 — depot فقط وقتی vehicle_useful=False
      • قید ۹ (اختصاصی محفظه): Rule 2 — ظرفیت و آزاد بودن محفظه
    """
    # ── هزینه routing ─────────────────────────────────────────────────────────
    sample   = tf.stack(actions, axis=0)                         # [T, B, 2]
    shifted  = tf.concat([tf.expand_dims(sample[-1], 0),
                          sample[:-1]], axis=0)                  # [T, B, 2]
    route_lengths = tf.reduce_sum(
        tf.sqrt(tf.reduce_sum(tf.square(shifted - sample), axis=2)),
        axis=0)                                                  # [B]

    # ── هزینه باز کردن facility ───────────────────────────────────────────────
    fac_cost = tf.constant(0.0)
    if opened_facilities is not None:
        n_opened = tf.reduce_sum(opened_facilities, axis=1)
        if isinstance(facility_opening_cost, (int, float)):
            fac_cost = tf.cast(n_opened, tf.float32) * float(facility_opening_cost)
        else:
            fac_cost = tf.reduce_sum(
                opened_facilities * facility_opening_cost, axis=1)

    return -(route_lengths + fac_cost)


def train():
    args, prt = ParseParams()

    prt.print_out("=" * 70)
    prt.print_out("LRP-RL  —  Fixed-Dataset Training  (Nazari et al. paradigm)")
    if IN_COLAB:
        prt.print_out("[Colab] T4 GPU detected.")
    prt.print_out("=" * 70)

    args.setdefault('facility_opening_cost', 10.0 / 3650.0)  # C_j/H (revised model)
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
            f"  Candidates : {args.get('n_fac',5)}  (agent opens 0…{args.get('n_fac',5)})")
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
            # Online mode: yield fresh batches for n_train steps
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
