import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
import os

from embeddings  import LinearEmbedding
from decode_step import RNNDecodeStep
from visualize_routes import plot_lrp_solution
# C1 constraint removed in revised model — no mandatory facility computation


class RLAgent(object):

    def __init__(self,
                 args,
                 prt,
                 env,
                 dataGen,
                 reward_func,
                 clAttentionActor,
                 clAttentionCritic,
                 is_train=True,
                 _scope=''):

        self.args             = args
        self.prt              = prt
        self.env              = env
        self.dataGen          = dataGen
        self.reward_func      = reward_func
        self.clAttentionCritic= clAttentionCritic
        self.is_lrp           = args.get('is_lrp', False)

        self.embedding = LinearEmbedding(args['embedding_dim'],
                                         _scope=_scope + 'Actor')
        self.decodeStep = RNNDecodeStep(
            clAttentionActor,
            args['hidden_dim'],
            use_tanh         = args['use_tanh'],
            tanh_exploration = args['tanh_exploration'],
            n_glimpses       = args['n_glimpses'],
            mask_glimpses    = args['mask_glimpses'],
            mask_pointer     = args['mask_pointer'],
            forget_bias      = args['forget_bias'],
            rnn_layers       = args['rnn_layers'],
            _scope           = 'Actor')

        self.decoder_input = tf.get_variable(
            'decoder_input', [1, 1, args['embedding_dim']],
            initializer=tf.initializers.glorot_uniform())

        start_time = time.time()
        if is_train:
            self.train_summary      = self.build_model(decode_type="stochastic")
            self.train_step         = self.build_train_step()
            self.val_summary_greedy = self.build_model(decode_type="greedy")
            self.val_summary_beam   = self.build_model(decode_type="beam_search")
        else:
            self.val_summary_greedy = self.build_model(decode_type="greedy")
            self.val_summary_beam   = self.build_model(decode_type="beam_search")

        model_time = time.time() - start_time
        self.prt.print_out("Model built in {:.1f}s".format(model_time))

        self.saver = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    # =========================================================================
    # BUILD MODEL
    # =========================================================================
    def build_model(self, decode_type="greedy"):

        args       = self.args
        env        = self.env
        batch_size = tf.shape(env.input_pnt)[0]   # dynamic

        input_pnt       = env.input_pnt
        encoder_emb_inp = self.embedding(input_pnt)

        beam_width = args['beam_width'] if decode_type == 'beam_search' else 1
        env.reset(beam_width)

        BatchSequence = tf.expand_dims(
            tf.cast(tf.range(batch_size * beam_width), tf.int64), 1)

        actions_tmp, logprobs, probs, idxs = [], [], [], []

        start_idx    = env.depot_idx
        idx          = start_idx * tf.ones([batch_size * beam_width, 1], dtype=tf.int64)
        action       = tf.squeeze(
            tf.tile(tf.expand_dims(input_pnt[:, start_idx], 1), [beam_width, 1, 1]), 1)

        initial_h    = tf.zeros([batch_size * beam_width, args['hidden_dim']])
        initial_c    = tf.zeros([batch_size * beam_width, args['hidden_dim']])
        decoder_state= (initial_h, initial_c)

        if args['task_name'] == 'tsp':
            decoder_input = tf.tile(
                self.decoder_input, [batch_size * beam_width, 1, 1])
        else:   # vrp or lrp
            decoder_input = tf.tile(
                tf.expand_dims(encoder_emb_inp[:, env.depot_idx], 1),
                [beam_width, 1, 1])

        context = tf.tile(encoder_emb_inp, [beam_width, 1, 1])
        routes  = tf.TensorArray(dtype=tf.int64, size=args['decode_len'])

        for i in range(args['decode_len']):

            logit, prob, logprob, decoder_state = self.decodeStep.step(
                decoder_input, context, env, decoder_state)

            beam_parent = None

            # ── Action selection ───────────────────────────────────────────
            if decode_type == 'greedy':
                idx    = tf.expand_dims(tf.argmax(prob, 1), 1)
                routes = routes.write(i, idx)

            elif decode_type == 'stochastic':
                def my_multinomial():
                    prob_idx     = tf.stop_gradient(prob)
                    prob_idx_cum = tf.cumsum(prob_idx, 1)
                    rand_uni     = tf.tile(
                        tf.random_uniform([batch_size * beam_width, 1]),
                        [1, env.total_actions])
                    sorted_ind = tf.cast(
                        tf.tile(tf.expand_dims(tf.range(env.total_actions), 0),
                                [batch_size * beam_width, 1]), tf.int64)
                    tmp = (tf.multiply(
                               tf.cast(tf.greater(prob_idx_cum, rand_uni), tf.int64),
                               sorted_ind)
                           + 10000 * tf.cast(
                               tf.greater_equal(rand_uni, prob_idx_cum), tf.int64))
                    idx_ = tf.expand_dims(tf.argmin(tmp, 1), 1)
                    return tmp, idx_

                tmp, idx   = my_multinomial()
                bad_sample = tf.cast(
                    tf.reduce_sum(tf.cast(
                        tf.greater(tf.reduce_sum(tmp, 1),
                                   (10000 * env.total_actions) - 1), tf.int32)),
                    tf.bool)
                tmp, idx   = tf.cond(bad_sample, my_multinomial, lambda: (tmp, idx))
                routes     = routes.write(i, idx)

            elif decode_type == 'beam_search':
                if i == 0:
                    batchBeamSeq = tf.expand_dims(
                        tf.tile(tf.cast(tf.range(batch_size), tf.int64),
                                [beam_width]), 1)
                    beam_path, log_beam_probs = [], []
                    log_beam_prob = tf.log(
                        tf.split(prob, num_or_size_splits=beam_width, axis=0)[0])
                else:
                    log_beam_prob = tf.log(prob) + log_beam_probs[-1]
                    log_beam_prob = tf.concat(
                        tf.split(log_beam_prob, beam_width, axis=0), 1)

                topk_val, topk_ind = tf.nn.top_k(log_beam_prob, beam_width)
                topk_val = tf.transpose(tf.reshape(tf.transpose(topk_val), [1, -1]))
                topk_ind = tf.transpose(tf.reshape(tf.transpose(topk_ind), [1, -1]))
                idx         = tf.cast(topk_ind % env.total_actions, tf.int64)
                beam_parent = tf.cast(topk_ind // env.total_actions, tf.int64)
                batchedBeamIdx = batchBeamSeq + tf.cast(batch_size, tf.int64) * beam_parent
                prob  = tf.gather_nd(prob, batchedBeamIdx)
                beam_path.append(beam_parent)
                log_beam_probs.append(topk_val)
                routes = routes.write(i, idx)

            state = env.step(idx, beam_parent)

            batched_idx   = tf.concat([BatchSequence, idx], 1)
            decoder_input = tf.expand_dims(
                tf.gather_nd(tf.tile(encoder_emb_inp, [beam_width, 1, 1]),
                             batched_idx), 1)

            logprob_ = tf.log(tf.clip_by_value(
                tf.gather_nd(prob, batched_idx), 1e-10, 1.0))

            probs.append(prob)
            idxs.append(idx)
            logprobs.append(logprob_)
            action = tf.gather_nd(
                tf.tile(input_pnt, [beam_width, 1, 1]), batched_idx)
            actions_tmp.append(action)

        final_routes = routes.stack()

        if decode_type == 'beam_search':
            tmplst, tmpind = [], [BatchSequence]
            for k in reversed(range(len(actions_tmp))):
                tmplst = [tf.gather_nd(actions_tmp[k], tmpind[-1])] + tmplst
                tmpind.append(tf.gather_nd(
                    batchBeamSeq + tf.cast(batch_size, tf.int64) * beam_path[k],
                    tmpind[-1]))
            actions = tmplst
        else:
            actions = actions_tmp

        # ── Curriculum learning: facility cost placeholder ──────────────────
        # Training: feed high cost (e.g. 1.0) so agent learns WHERE to open.
        # Evaluation: defaults to real C_j/H ≈ 0.00274.
        self.fac_cost_ph = tf.placeholder_with_default(
            float(self.args.get('facility_opening_cost', 10.0 / 3650.0)),
            shape=[], name='fac_cost_ph')

        R = self.reward_func(
                actions,
                env.opened_facilities,
                self.fac_cost_ph)

        # Critic (only built for stochastic / training)
        v = tf.constant(0.0)
        if decode_type == 'stochastic':
            with tf.variable_scope("Critic"):
                with tf.variable_scope("Encoder"):
                    init_state    = tf.zeros(
                        [args['rnn_layers'], 2, batch_size, args['hidden_dim']])
                    l             = tf.unstack(init_state, axis=0)
                    rnn_tuple     = tuple([(l[k][0], l[k][1])
                                           for k in range(args['rnn_layers'])])
                    hy            = rnn_tuple[0][1]

                with tf.variable_scope("Process"):
                    for i in range(args['n_process_blocks']):
                        proc    = self.clAttentionCritic(
                            args['hidden_dim'], _name="P" + str(i))
                        e, logit = proc(hy, encoder_emb_inp, env)
                        prob_c   = tf.nn.softmax(logit)
                        hy       = tf.squeeze(
                            tf.matmul(tf.expand_dims(prob_c, 1), e), 1)

                with tf.variable_scope("Linear"):
                    d1 = tf.keras.layers.Dense(
                        args['hidden_dim'], activation=tf.nn.relu, name='L1')
                    d2 = tf.keras.layers.Dense(1, name='L2')
                    v  = tf.squeeze(d2(d1(hy)), 1)

        return (R, v, logprobs, actions, idxs, env.input_pnt, probs, final_routes)

    # =========================================================================
    # BUILD TRAIN STEP
    # =========================================================================
    def build_train_step(self):

        args = self.args
        R, v, logprobs, actions, idxs, batch, probs, routes = self.train_summary

        v_nograd = tf.stop_gradient(v)
        R        = tf.stop_gradient(R)

        # FIX: original was (R - v_nograd) * logπ → minimising this REDUCES reward.
        # REINFORCE: minimise -(R - baseline) * log π
        actor_loss  = tf.reduce_mean(-(R - v_nograd) * tf.add_n(logprobs))
        critic_loss = tf.losses.mean_squared_error(R, v)

        actor_optim  = tf.train.AdamOptimizer(args['actor_net_lr'])
        critic_optim = tf.train.AdamOptimizer(args['critic_net_lr'])

        actor_gv  = actor_optim.compute_gradients(
            actor_loss,
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor'))
        critic_gv = critic_optim.compute_gradients(
            critic_loss,
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic'))

        clip_agv = [(tf.clip_by_norm(g, args['max_grad_norm']), v_)
                    for g, v_ in actor_gv  if g is not None]
        clip_cgv = [(tf.clip_by_norm(g, args['max_grad_norm']), v_)
                    for g, v_ in critic_gv if g is not None]

        actor_step  = actor_optim.apply_gradients(clip_agv)
        critic_step = critic_optim.apply_gradients(clip_cgv)

        return [actor_step, critic_step,
                actor_loss, critic_loss,
                actor_gv, critic_gv,
                R, v, logprobs, probs, actions, idxs]

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    def Initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.load_model()

    def load_model(self):
        load_path = self.args.get('load_path', '')
        if load_path:
            latest = tf.train.latest_checkpoint(load_path)
            if latest:
                self.saver.restore(self.sess, latest)

    # =========================================================================
    # EVALUATE SINGLE  (one-by-one through test set)
    # =========================================================================
    def evaluate_single(self, eval_type='greedy'):
        start_time = time.time()
        avg_reward = []

        summary = (self.val_summary_greedy if eval_type == 'greedy'
                   else self.val_summary_beam)
        self.dataGen.reset()

        for step in range(self.dataGen.n_problems):
            data = self.dataGen.get_test_next()
            # get_test_next() returns (features, lambda_data) tuple.
            if isinstance(data, (tuple, list)) and len(data) == 2:
                features, lambda_data = data
            else:
                features  = data
                lambda_data = self.env.get_default_lambda_batch(features.shape[0])

            R, v, logprobs, actions, idxs, batch, _, routes = self.sess.run(
                summary,
                feed_dict={self.env.input_data:    features,
                           self.env.lambda_ph:     lambda_data,
                           self.decodeStep.dropout: 0.0})

            if eval_type == 'greedy':
                avg_reward.append(R)
                R_ind0 = 0
            elif eval_type == 'beam_search':
                R      = np.concatenate(
                    np.split(np.expand_dims(R, 1), self.args['beam_width'], axis=0), 1)
                R_val  = np.amin(R, 1, keepdims=False)
                R_ind0 = np.argmin(R, 1)[0]
                avg_reward.append(R_val)

            if step % int(self.args['log_interval']) == 0:
                example_input  = [list(batch[0, i, :]) for i in range(self.env.n_nodes)]
                example_output = [list(action[R_ind0 * np.shape(batch)[0]])
                                  for action in actions]
                self.prt.print_out(
                    '\nVal-Step of {}: {}'.format(eval_type, step))
                self.prt.print_out(
                    '\nExample test output: {}'.format(example_output))
                self.prt.print_out(
                    '\nExample test reward: {} - best: {}'.format(R[0], R_ind0))

        elapsed = time.time() - start_time
        self.prt.print_out(
            '\nValidation avg_reward: {}'.format(np.mean(avg_reward)))
        self.prt.print_out(
            'Validation reward std: {}'.format(np.sqrt(np.var(avg_reward))))
        self.prt.print_out(
            "Finished evaluation in %s." %
            time.strftime("%H:%M:%S", time.gmtime(elapsed)))

    # =========================================================================
    # EVALUATE BATCH  (all test instances in one sess.run)
    # =========================================================================
    def evaluate_batch(self, eval_type='greedy'):
        # FIX: removed self.env.reset() — the graph is already built with the
        # correct tensors from build_model(). Calling reset() here only creates
        # orphaned tensors that no graph references; it does not change the
        # running computation and caused confusion.

        summary    = (self.val_summary_greedy if eval_type == 'greedy'
                      else self.val_summary_beam)
        beam_width = self.args['beam_width'] if eval_type == 'beam_search' else 1

        data = self.dataGen.get_test_all()
        # get_test_all() returns (features, lambda_data) since the LRP generator
        # carries per-instance λ matrices.
        if isinstance(data, (tuple, list)) and len(data) == 2:
            features, lambda_data = data
        else:
            features  = data
            lambda_data = self.env.get_default_lambda_batch(features.shape[0])

        t0   = time.time()
        R, v, logprobs, actions, idxs, batch, _, routes = self.sess.run(
            summary,
            feed_dict={self.env.input_data:    features,
                       self.env.lambda_ph:     lambda_data,
                       self.decodeStep.dropout: 0.0})

        R = np.concatenate(
            np.split(np.expand_dims(R, 1), beam_width, axis=0), 1)
        R = np.amin(R, 1, keepdims=False)

        elapsed = time.time() - t0
        self.prt.print_out(
            '  Greedy avg reward: {:.4f}  std: {:.4f}'.format(
                np.mean(R), np.sqrt(np.var(R))))

        # ── Route figures ─────────────────────────────────────────────────────
        self._print_route_analysis(routes, features, n_samples=3)
        self._save_route_figures(features, routes, batch, R, args=self.args)

        return R, routes, batch

    def _print_route_analysis(self, routes, data, n_samples=5):
        """Print human-readable route tours for debugging."""
        print("\n" + "="*70)
        print("ROUTE TOUR ANALYSIS")
        print("="*70)
        
        for sample_idx in range(min(n_samples, routes.shape[1])):
            route = routes[:, sample_idx, 0]  # [decode_len]
            node_types = data[sample_idx, :, 3]  # [n_nodes]
            
            print(f"\nSample {sample_idx + 1}:")
            print("-" * 70)
            
            # Extract tours (depot-to-depot segments)
            tours = []
            current_tour = [0]  # Start with depot
            
            for node_id in route[:]: 
                node_id = int(node_id)
                current_tour.append(node_id)
                
                if node_id == 0:  # Returned to depot
                    tours.append(current_tour)
                    current_tour = [0]
            
            # If route doesn't end at depot, still add the tour
            if len(current_tour) > 1:
                tours.append(current_tour)
            
            # Print each tour
            for tour_idx, tour in enumerate(tours, 1):
                # Separate customers and facilities
                # FIX: use actual n_cust from env instead of hardcoded 10.
                # For lrp20: n_cust=20, so nodes 11-20 are customers, not stations.
                n_cust    = self.env.n_cust              # e.g. 20 for lrp20
                fac_start = self.env.n_cust + 1          # e.g. 21 for lrp20
                fac_end   = self.env.n_nodes - 1         # e.g. 28 for lrp20
                customers  = [n for n in tour if 1 <= n <= n_cust]
                facilities = [n for n in tour if fac_start <= n <= fac_end]
                
                # Build tour string
                tour_str = " → ".join([
                    "D"      if n == 0        else
                    f"C{n}"  if n <= n_cust   else
                    f"F{n}"
                    for n in tour
                ])
                
                print(f"  Tour {tour_idx}: {tour_str}")
                print(f"    Customers visited: {len(customers)} {customers}")
                if facilities:
                    print(f"    Stations used: {facilities}")
            
            # Summary
            n_cust    = self.env.n_cust
            fac_start = self.env.n_cust + 1
            fac_end   = self.env.n_nodes - 1
            total_customers = sum(len([n for n in t if 1 <= n <= n_cust]) for t in tours)
            total_stations  = len(set(n for t in tours for n in t if fac_start <= n <= fac_end))
            print(f"\n  SUMMARY: {len(tours)} tours, {total_customers}/{n_cust} customers, {total_stations} stations opened")

    def _save_route_figures(self, data, routes, batch, R, args, n_samples=5):
        """Save per-sample route figures using visualize_routes."""
        try:
            # figures_dir is set by train_lrp.py — on Colab it is /content/figures/
            # (fast local NVMe); locally it lives inside log_dir/figures/.
            fig_dir = args.get('figures_dir',
                               os.path.join(args.get('log_dir', '.'), 'figures'))
            os.makedirs(fig_dir, exist_ok=True)
            step_str = str(args.get('n_train', 0)).zfill(8)

            n_save = min(n_samples, data.shape[0])
            # routes shape: [decode_len, batch_size, 1]
            for s in range(n_save):
                route_flat = routes[:, s, 0].tolist()

                coords    = data[s, :, :2]
                node_type = data[s, :, 3]
                n_nodes   = coords.shape[0]

                # Infer opened facilities: route-visited + mandatory pre-opened
                opened = np.zeros(n_nodes, dtype=np.float32)
                for nid in route_flat:
                    if int(node_type[int(nid)]) == 2:
                        opened[int(nid)] = 1.0

                # Lambda matrix and collision points stored in env args
                lam = args.get('lambda_mat', None)
                collision_pts = args.get('collision_points', None)

                save_path = os.path.join(
                    fig_dir, f'step{step_str}_sample{s:02d}.png')

                plot_lrp_solution(
                    coords            = coords,
                    node_type         = node_type,
                    routes            = route_flat,
                    opened_facilities = opened,
                    lambda_mat        = lam,
                    save_path         = save_path,
                    instance_title    = (
                        f'Step {args.get("n_train",0)} | Sample {s+1} | '
                        f'Route cost {-float(R[s]):.3f} | '
                        f'Open facilities: {int(opened.sum())}'),
                    collision_points  = collision_pts)
                print(f'  Figure saved -> {save_path}')

            print(f'  Saved {n_save} route figures to {fig_dir}/')
        except Exception as e:
            print(f'  [WARNING] Route figure generation failed: {e}')

    # =========================================================================
    # INFERENCE ENTRY POINT
    # =========================================================================
    def inference(self, infer_type='batch'):
        if infer_type == 'batch':
            self.evaluate_batch('greedy')
            if hasattr(self, 'val_summary_beam'):
                self.evaluate_batch('beam_search')
        elif infer_type == 'single':
            self.evaluate_single('greedy')
            if hasattr(self, 'val_summary_beam'):
                self.evaluate_single('beam_search')
        self.prt.print_out("#" * 66)

    # =========================================================================
    # TRAINING STEP
    # =========================================================================
    def run_train_step(self, features=None, lambda_data=None):
        if features is None or lambda_data is None:
            features, lambda_data = self.dataGen.get_train_next()

        if lambda_data is None:
            batch_size = features.shape[0]
            lambda_data = self.env.get_default_lambda_batch(batch_size)

        # Curriculum learning: feed high facility cost during training
        train_fac_cost = self.args.get('facility_opening_cost_train', 1.0)

        return self.sess.run(
            self.train_step,
            feed_dict={self.env.input_data:    features,
                       self.env.lambda_ph:     lambda_data,
                       self.decodeStep.dropout: self.args['dropout'],
                       self.fac_cost_ph:       train_fac_cost})
