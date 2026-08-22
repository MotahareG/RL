"""
run_inference_timing.py
=======================
Standalone inference + timing script.
Called by Cell 7 of LRP_Colab_Github.ipynb via:
    !python run_inference_timing.py --task=lrp10 --log_dir=/content/logs/lrp10-xxx/model

What it does:
  1. Rebuilds the identical model graph used during training
  2. Restores weights from the final checkpoint
  3. Generates a fresh 1000-instance test set using the same pipeline
  4. Runs warm-up, batch timing, and single-instance timing
  5. Prints results and saves inference_timing.csv next to the checkpoint

Usage:
  python run_inference_timing.py --task=lrp10 --log_dir=/content/logs/lrp10-.../model
"""

import os, sys, time, csv, glob, argparse
import numpy as np

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--task',        default='lrp10')
parser.add_argument('--log_dir',     default='',
                    help='Path to model dir (contains .ckpt files). '
                         'If empty, uses the latest /content/logs/lrp* directory.')
parser.add_argument('--csv',         default='CA_AV_Collision_2019-2024.csv')
parser.add_argument('--test_size',   type=int, default=1000)
parser.add_argument('--n_repeats',   type=int, default=5,
                    help='Repeat batch inference N times for stable average')
parser.add_argument('--n_single',    type=int, default=100,
                    help='Number of single-instance runs for per-instance timing')
parser.add_argument('--gpu',         default='0')
cli = parser.parse_args()

# ── GPU ───────────────────────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = cli.gpu

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ── project imports ───────────────────────────────────────────────────────────
from task_specific_params import task_lst
from configs              import initialize_task_settings
from data_generator_lrp   import LRPDataGenerator
from lrp_env               import Env
from attention_agent       import RLAgent
import misc_utils as utils

try:
    from vrp_attention import AttentionVRPActor, AttentionVRPCritic
except ImportError:
    from VRP.vrp_attention import AttentionVRPActor, AttentionVRPCritic

# ── find checkpoint ───────────────────────────────────────────────────────────
if cli.log_dir:
    model_dir = cli.log_dir
else:
    runs = sorted(glob.glob(f'/content/logs/{cli.task}*/model'))
    if not runs:
        runs = sorted(glob.glob(f'/content/logs/lrp*/model'))
    if not runs:
        sys.exit('[ERROR] No checkpoint found. Train the model first (Cell 5).')
    model_dir = runs[-1]

ckpt = tf.train.latest_checkpoint(model_dir)
if ckpt is None:
    sys.exit(f'[ERROR] No checkpoint in {model_dir}')

print(f'Checkpoint : {ckpt}')
print(f'Task       : {cli.task}')

# ── build args dict (mirrors train_lrp.py exactly) ───────────────────────────
args = {
    'task':                   cli.task,
    'task_name':              'lrp',
    'is_lrp':                 True,
    'is_train':               False,
    'batch_size':             cli.test_size,
    'test_size':              cli.test_size,
    'agent_type':             'attention',
    'embedding_dim':          128,
    'hidden_dim':             128,
    'n_process_blocks':       3,
    'rnn_layers':             1,
    'forget_bias':            1.0,
    'n_glimpses':             0,
    'tanh_exploration':       10.0,
    'use_tanh':               False,
    'mask_glimpses':          True,
    'mask_pointer':           True,
    'dropout':                0.0,
    'beam_width':             10,
    'infer_type':             'batch',
    'random_seed':            24601,
    'facility_opening_cost':  10.0 / 3650.0,  # C_j/H (revised model)
    'load_path':              model_dir,
    'log_dir':                os.path.dirname(model_dir),
    'model_dir':              model_dir,
    'figures_dir':            '/content/figures',
    'stdout_print':           True,
}
# fill task-specific fields (n_cust, n_fac, n_nodes, decode_len, …)
args = initialize_task_settings(args, cli.task)

# ── representative lambda (needed for Env initialisation) ────────────────────
print('\nInitialising data generator...')
dataGen = LRPDataGenerator(
    args=args,
    collision_csv_path=cli.csv,
    bounding_box=((32.5, -124.5), (42.0, -114.0)),
    grid_size=200)
args['collision_points'] = dataGen.collision_points

rep = dataGen.generate_instance(
        args['n_cust'], args['n_fac'], seed=42, compute_lambda=True)
args['lambda_mat'] = rep['lambda']
args['node_type']  = rep['node_type']

# ── session + model ───────────────────────────────────────────────────────────
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

out_file = open(os.path.join(os.path.dirname(model_dir), 'timing_log.txt'), 'w')
prt = utils.printOut(out_file, stdout_print=True)

env   = Env(args)
agent = RLAgent(
    args=args, prt=prt, env=env, dataGen=dataGen,
    reward_func=None,
    clAttentionActor=AttentionVRPActor,
    clAttentionCritic=AttentionVRPCritic,
    is_train=False)

agent.Initialize(sess)
print(f'Weights restored from: {ckpt}')

# ── build test set ────────────────────────────────────────────────────────────
print(f'\nBuilding test set ({cli.test_size} instances)...')
features, lambda_data = dataGen.get_test_all()
n_test = features.shape[0]
print(f'Test set shape: {features.shape}')

# ── warm-up (excluded from timing) ───────────────────────────────────────────
print('\nWarm-up pass (not timed)...')
sess.run(agent.val_summary_greedy,
         feed_dict={agent.env.input_data:     features[:args['batch_size']],
                    agent.env.lambda_ph:      lambda_data[:args['batch_size']],
                    agent.decodeStep.dropout: 0.0})
print('  Warm-up done.')

# ── batch inference ───────────────────────────────────────────────────────────
print(f'\nBatch inference — {n_test} instances × {cli.n_repeats} runs...')
batch_times, last_R = [], None
for rep in range(cli.n_repeats):
    t0 = time.time()
    R, v, logprobs, actions, idxs, batch, _, routes = sess.run(
        agent.val_summary_greedy,
        feed_dict={agent.env.input_data:     features,
                   agent.env.lambda_ph:      lambda_data,
                   agent.decodeStep.dropout: 0.0})
    batch_times.append(time.time() - t0)
    last_R = R

last_R = np.concatenate(np.split(np.expand_dims(last_R, 1), 1, axis=0), 1)
last_R = np.amin(last_R, 1, keepdims=False)
batch_avg_s  = np.mean(batch_times)
per_inst_ms  = batch_avg_s / n_test * 1000

# ── single-instance timing ────────────────────────────────────────────────────
print(f'Single-instance timing — {cli.n_single} runs...')
single_ms = []
for i in range(cli.n_single):
    t0 = time.time()
    sess.run(agent.val_summary_greedy,
             feed_dict={agent.env.input_data:     features[i:i+1],
                        agent.env.lambda_ph:      lambda_data[i:i+1],
                        agent.decodeStep.dropout: 0.0})
    single_ms.append((time.time() - t0) * 1000)

s_avg = np.mean(single_ms)
s_std = np.std(single_ms)
s_min = np.min(single_ms)
s_max = np.max(single_ms)

# ── report ────────────────────────────────────────────────────────────────────
task_label = f"{cli.task.upper()}  ({args['n_cust']} customers, {args['n_fac']} facilities)"
sep = '═' * 62

print(f'\n{sep}')
print(f'POST-TRAINING EVALUATION — {task_label}')
print(sep)
print(f'\n  Solution quality  ({n_test} test instances, greedy):')
print(f'    Avg reward : {np.mean(last_R):.4f}')
print(f'    Std        : {np.std(last_R):.4f}')
print(f'\n  Batch inference  ({n_test} instances, avg of {cli.n_repeats} runs):')
print(f'    Total time : {batch_avg_s:.3f} s')
print(f'    Per-instance: {per_inst_ms:.3f} ms')
print(f'\n  Single-instance inference  ({cli.n_single} runs):')
print(f'    Mean : {s_avg:.3f} ms')
print(f'    Std  : {s_std:.3f} ms')
print(f'    Min  : {s_min:.3f} ms')
print(f'    Max  : {s_max:.3f} ms')
print(f'\n{"─"*62}')
print(f'  ► For thesis comparison table:')
print(f'    Inference time : {s_avg:.1f} ± {s_std:.1f} ms  (single-instance, greedy)')
print(f'    Avg reward     : {np.mean(last_R):.4f}')
print(sep)

# ── save CSV ──────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(model_dir), 'inference_timing.csv')
with open(out_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric', 'value', 'unit'])
    w.writerow(['task',              cli.task,                    ''])
    w.writerow(['n_cust',            args['n_cust'],              ''])
    w.writerow(['n_fac',             args['n_fac'],               ''])
    w.writerow(['n_test',            n_test,                      'instances'])
    w.writerow(['avg_reward',        f'{np.mean(last_R):.6f}',    ''])
    w.writerow(['std_reward',        f'{np.std(last_R):.6f}',     ''])
    w.writerow(['batch_total_s',     f'{batch_avg_s:.4f}',        's'])
    w.writerow(['per_inst_batch_ms', f'{per_inst_ms:.4f}',        'ms'])
    w.writerow(['single_mean_ms',    f'{s_avg:.4f}',              'ms'])
    w.writerow(['single_std_ms',     f'{s_std:.4f}',              'ms'])
    w.writerow(['single_min_ms',     f'{s_min:.4f}',              'ms'])
    w.writerow(['single_max_ms',     f'{s_max:.4f}',              'ms'])

print(f'\n  Timing CSV saved → {out_path}')
sess.close()
