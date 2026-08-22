import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class AttentionVRPActor(object):
    """
    Attention module for the actor network.

    Fix applied: original code computed load-demand via
        tf.tile(tf.expand_dims(load, 1), [1, max_time]) - demand
    where `load` had shape [batch_beam] derived from env.comp_loads.
    If comp_loads was built with a static Python-int batch size (e.g. 128),
    then load.shape = [128] and the tile produced [128, max_time].
    At evaluation time (1000 test samples), demand.shape = [1000, 16] →
    incompatible shapes → crash.

    Fix: use tf.reduce_sum(..., keepdims=True) → shape [batch, 1], then let
    TensorFlow broadcast [batch, 1] - [batch, n_nodes] = [batch, n_nodes].
    This is safe regardless of whether batch dim is static or dynamic.
    """

    def __init__(self, dim, use_tanh=False, C=10, _name='Attention', _scope=''):
        self.use_tanh = use_tanh
        self._scope   = _scope

        with tf.variable_scope(_scope + _name):
            self.v = tf.get_variable('v', [1, dim],
                                     initializer=tf.initializers.glorot_uniform())
            self.v = tf.expand_dims(self.v, 2)

        safe_name = (_scope + _name).replace('/', '_')

        self.emb_d      = tf.keras.layers.Conv1D(dim, 1, name=safe_name + '_emb_d')
        self.emb_ld     = tf.keras.layers.Conv1D(dim, 1, name=safe_name + '_emb_ld')
        self.emb_lambda = tf.keras.layers.Conv1D(dim, 1, name=safe_name + '_emb_lambda')

        self.project_d      = tf.keras.layers.Conv1D(dim, 1, name=safe_name + '_proj_d')
        self.project_ld     = tf.keras.layers.Conv1D(dim, 1, name=safe_name + '_proj_ld')
        self.project_lambda = tf.keras.layers.Conv1D(dim, 1, name=safe_name + '_proj_lambda')
        self.project_query  = tf.keras.layers.Dense(dim,    name=safe_name + '_proj_q')
        self.project_ref    = tf.keras.layers.Conv1D(dim, 1, name=safe_name + '_proj_ref')

        self.C    = C
        self.tanh = tf.nn.tanh

    def __call__(self, query, ref, env):

        demand   = env.demand                              # [batch, n_nodes]
        max_time = tf.shape(demand)[1]

        # ── Demand embedding ──────────────────────────────────────────────────
        d = self.project_d(self.emb_d(tf.expand_dims(demand, 2)))

        # ── Load-demand embedding ─────────────────────────────────────────────
        # FIX: use keepdims=True so load.shape = [batch, 1], then broadcast.
        # Original: tf.tile(tf.expand_dims(load, 1), [1, max_time]) - demand
        #           → static [128, n] vs dynamic [1000, n] → shape error.
        load     = tf.reduce_sum(env.comp_loads, axis=1, keepdims=True)  # [batch, 1]
        ld_input = load - demand                                          # [batch, n_nodes]
        ld       = self.project_ld(self.emb_ld(tf.expand_dims(ld_input, 2)))

        # ── Lambda (safety) embedding ─────────────────────────────────────────
        if env.lambda_mat is not None:
            current    = tf.cast(env.current_node, tf.int32)
            lambda_row = tf.gather(env.lambda_mat, current, batch_dims=1)  # [batch, n_nodes]
            lambda_emb = self.project_lambda(
                self.emb_lambda(tf.expand_dims(lambda_row, -1)))
        else:
            lambda_emb = 0

        # ── Attention ─────────────────────────────────────────────────────────
        e          = self.project_ref(ref)
        q          = self.project_query(query)
        expanded_q = tf.tile(tf.expand_dims(q, 1), [1, max_time, 1])
        v_view     = tf.tile(self.v, [tf.shape(e)[0], 1, 1])

        u = tf.squeeze(
            tf.matmul(self.tanh(expanded_q + e + d + ld + lambda_emb), v_view), 2)
        u = u / tf.sqrt(tf.cast(tf.shape(e)[-1], tf.float32))

        logits = self.C * self.tanh(u) if self.use_tanh else u

        if hasattr(env, 'mask'):
            logits = logits - 1e9 * env.mask

        return e, logits


class AttentionVRPCritic(object):

    def __init__(self, dim, use_tanh=False, C=10, _name='Attention', _scope=''):
        self.use_tanh = use_tanh

        with tf.variable_scope(_scope + _name):
            self.v = tf.get_variable('v', [1, dim],
                                     initializer=tf.initializers.glorot_uniform())
            self.v = tf.expand_dims(self.v, 2)

        safe_name = (_scope + _name).replace('/', '_')

        self.emb_d        = tf.keras.layers.Conv1D(dim, 1, name=safe_name + '_emb_d')
        self.project_d    = tf.keras.layers.Conv1D(dim, 1, name=safe_name + '_proj_d')
        self.project_query= tf.keras.layers.Dense(dim,    name=safe_name + '_proj_q')
        self.project_ref  = tf.keras.layers.Conv1D(dim, 1, name=safe_name + '_proj_e')

        self.C    = C
        self.tanh = tf.nn.tanh

    def __call__(self, query, ref, env):

        demand   = env.input_data[:, :, -1]
        max_time = tf.shape(demand)[1]

        d          = self.project_d(self.emb_d(tf.expand_dims(demand, 2)))
        e          = self.project_ref(ref)
        q          = self.project_query(query)
        expanded_q = tf.tile(tf.expand_dims(q, 1), [1, max_time, 1])
        v_view     = tf.tile(self.v, [tf.shape(e)[0], 1, 1])

        u      = tf.squeeze(
            tf.matmul(self.tanh(expanded_q + e + d), v_view), 2)
        logits = self.C * self.tanh(u) if self.use_tanh else u

        return e, logits
