import argparse
import misc_utils as utils
import os
from task_specific_params import task_lst
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('true', '1', 'yes')


def initialize_task_settings(args, task):

    try:
        task_params = task_lst[task]
    except KeyError:
        raise Exception('Task is not implemented.')

    for name, value in task_params._asdict().items():
        args[name] = value

    # اگر decode_len مشخص نشده باشد مقدار پیش‌فرض task استفاده می‌شود
    if args.get('decode_len') is None and hasattr(task_params, 'decode_len'):
        args['decode_len'] = task_params.decode_len

    return args


def ParseParams():

    parser = argparse.ArgumentParser(
        description="Neural Combinatorial Optimization with RL"
    )

    # ======================
    # Data
    # ======================
    parser.add_argument('--task', default='vrp10',
                        help="Select task e.g. tsp10, vrp10, lrp10")

    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size in training')

    parser.add_argument('--n_train', default=260000, type=int,
                        help='Number of training steps (used when use_fixed_dataset=False)')

    parser.add_argument('--n_instances', default=10000, type=int,
                        help='[Fixed-Dataset] Pre-generated training instances (e.g. 10000)')

    parser.add_argument('--n_epochs', default=100, type=int,
                        help='[Fixed-Dataset] Training epochs over the fixed dataset')

    parser.add_argument('--use_fixed_dataset', default=True, type=str2bool,
                        help='Use Nazari-style pre-generated fixed dataset (recommended: True)')

    parser.add_argument('--test_size', default=1000, type=int,
                        help='Number of problems in test set')

    # ======================
    # Network
    # ======================
    parser.add_argument('--agent_type', default='attention',
                        help="attention | pointer")

    parser.add_argument('--forget_bias', default=1.0, type=float,
                        help="Forget bias for BasicLSTMCell.")

    parser.add_argument('--embedding_dim', default=128, type=int,
                        help='Dimension of input embedding')

    parser.add_argument('--hidden_dim', default=128, type=int,
                        help='Dimension of hidden layers')

    parser.add_argument('--n_process_blocks', default=3, type=int,
                        help='Process blocks in critic')

    parser.add_argument('--rnn_layers', default=1, type=int,
                        help='Number of LSTM layers')

    parser.add_argument('--decode_len', default=None, type=int,
                        help='Decoder time steps')

    parser.add_argument('--n_glimpses', default=0, type=int,
                        help='Number of glimpses')

    parser.add_argument('--tanh_exploration', default=10., type=float,
                        help='Exploration scaling')

    parser.add_argument('--use_tanh', type=str2bool, default=True)

    parser.add_argument('--mask_glimpses', type=str2bool, default=True)

    parser.add_argument('--mask_pointer', type=str2bool, default=True)

    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout probability')

    # ======================
    # Training
    # ======================
    parser.add_argument('--is_train', default=True, type=str2bool,
                        help="Run training")

    parser.add_argument('--actor_net_lr', default=1e-4, type=float,
                        help="Actor learning rate")

    parser.add_argument('--critic_net_lr', default=1e-4, type=float,
                        help="Critic learning rate")

    parser.add_argument('--random_seed', default=24601, type=int)

    parser.add_argument('--max_grad_norm', default=2.0, type=float,
                        help='Gradient clipping')

    parser.add_argument('--entropy_coeff', default=0.0, type=float,
                        help='Entropy regularization')

    # ======================
    # Inference
    # ======================
    parser.add_argument('--infer_type', default='batch',
                        help='single | batch')

    parser.add_argument('--beam_width', default=10, type=int)

    # ======================
    # Misc
    # ======================
    parser.add_argument('--stdout_print', default=True, type=str2bool)

    parser.add_argument('--gpu', default='0', type=str,
                        help="GPU number")

    parser.add_argument('--log_interval', default=200, type=int)

    parser.add_argument('--test_interval', default=200, type=int)

    parser.add_argument('--save_interval', default=10000, type=int)

    parser.add_argument('--log_dir', type=str, default='logs')

    parser.add_argument('--data_dir', type=str, default='data')

    parser.add_argument('--model_dir', type=str, default='')

    parser.add_argument('--load_path', type=str, default='',
                        help='Load trained model')

    parser.add_argument('--disable_tqdm', default=True, type=str2bool)

    # ======================
    # Parse
    # ======================
    args, unknown = parser.parse_known_args()
    args = vars(args)

    # ======================
    # Learning rate aliases
    # ======================
    args['learning_rate'] = args['actor_net_lr']
    args['max_epochs'] = args['n_train']

    # ======================
    # Log directories
    # ======================
    args['log_dir'] = "{}/{}-{}".format(
        args['log_dir'],
        args['task'],
        utils.get_time()
    )

    if args['model_dir'] == '':
        args['model_dir'] = os.path.join(args['log_dir'], 'model')

    os.makedirs(args['log_dir'], exist_ok=True)
    os.makedirs(args['model_dir'], exist_ok=True)

    # ======================
    # Logging file
    # ======================
    out_file = open(
        os.path.join(args['log_dir'], 'results.txt'),
        'w+'
    )

    prt = utils.printOut(out_file, args['stdout_print'])

    # ======================
    # GPU
    # ======================
    os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']

    # ======================
    # Task specific params
    # ======================
    args = initialize_task_settings(args, args['task'])

    # ======================
    # Print config
    # ======================
    for key, value in sorted(args.items()):
        prt.print_out("{}: {}".format(key, value))

    return args, prt
