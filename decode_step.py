# decode_step.py
# ==============================================================================
# Fix: DropoutWrapper is not available with Keras 3 (Colab TF 2.x + Keras 3).
#
#      Root cause: tf.compat.v1.nn.rnn_cell.DropoutWrapper was an internal
#      TF1 shim that Keras 3 removed completely.  Any call to it raises:
#        AttributeError: `DropoutWrapper` is not available with Keras 3.
#
#      Fix: Remove DropoutWrapper entirely from make_cell().
#      Replacement: _apply_dropout() applies the identical transformation
#      (dropout on LSTM input and output) manually in get_logit_op().
#
#      tf.nn.dropout(x, rate=r) — TF2/Keras3 API — is used inside a
#      tf.cond so that rate=0.0 (inference) truly produces a no-op and
#      avoids any potential divide-by-zero inside the dropout kernel.
#
#      MultiRNNCell is also wrapped in try/except because Keras 3 may
#      remove it in a future minor update (rnn_layers=1 is the default
#      so this path is not taken in normal use).
# ==============================================================================

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.nn.rnn_cell import LSTMCell


class DecodeStep(object):
    def __init__(self, ClAttention, hidden_dim, use_tanh=False, tanh_exploration=10.,
                 n_glimpses=0, mask_glimpses=True, mask_pointer=True, _scope=''):
        self.hidden_dim       = hidden_dim
        self.use_tanh         = use_tanh
        self.tanh_exploration = tanh_exploration
        self.n_glimpses       = n_glimpses
        self.mask_glimpses    = mask_glimpses
        self.mask_pointer     = mask_pointer
        self._scope           = _scope
        self.BIGNUMBER        = 100000.

        self.glimpses = []
        for i in range(self.n_glimpses):
            self.glimpses.append(
                ClAttention(hidden_dim, use_tanh=False,
                            _scope=self._scope, _name=f"Glimpse{i}"))
        self.pointer = ClAttention(
            hidden_dim, use_tanh=use_tanh, C=tanh_exploration,
            _scope=self._scope, _name="Decoder/Attention")

    def get_logit_op(self, decoder_inp, context, env, decoder_state=None):
        hy = decoder_inp
        for i in range(self.n_glimpses):
            ref, logit = self.glimpses[i](hy, context, env)
            if self.mask_glimpses:
                logit -= self.BIGNUMBER * env.mask[:, :env.n_nodes]
            prob = tf.nn.softmax(logit)
            hy   = tf.squeeze(tf.matmul(tf.expand_dims(prob, 1), ref), 1)
        _, logit = self.pointer(hy, context, env)
        if self.mask_pointer:
            logit -= self.BIGNUMBER * env.mask[:, :env.n_nodes]
        return logit, None

    def step(self, decoder_inp, context, env, decoder_state=None):
        logit, _  = self.get_logit_op(decoder_inp, context, env, decoder_state)
        logprob   = tf.nn.log_softmax(logit)
        prob      = tf.exp(logprob)
        return logit, prob, logprob, decoder_state


class RNNDecodeStep(DecodeStep):
    def __init__(self, ClAttention, hidden_dim, use_tanh=False, tanh_exploration=10.,
                 n_glimpses=0, mask_glimpses=True, mask_pointer=True,
                 forget_bias=1.0, rnn_layers=1, _scope=''):
        super(RNNDecodeStep, self).__init__(
            ClAttention, hidden_dim, use_tanh, tanh_exploration,
            n_glimpses, mask_glimpses, mask_pointer, _scope)

        self.forget_bias = forget_bias
        self.rnn_layers  = rnn_layers

        # Dropout rate fed at sess.run time:
        #   0.0           = no dropout (inference / evaluation)
        #   args['dropout'] (e.g. 0.1) = dropout during training
        self.dropout_placeholder = tf.placeholder_with_default(
            0.0, shape=(), name='dropout')

        # ── FIX: bare LSTMCell, no DropoutWrapper ─────────────────────────────
        # DropoutWrapper is not available with Keras 3.
        # Dropout is now applied manually in _apply_dropout() below,
        # producing identical behaviour to DropoutWrapper's
        # input_keep_prob / output_keep_prob.
        def make_cell():
            return LSTMCell(hidden_dim, name='lstm_cell')

        if rnn_layers > 1:
            # MultiRNNCell may also be removed in a future Keras 3 update;
            # guard with try/except. (rnn_layers=1 by default, so this
            # branch is not taken in normal configurations.)
            try:
                cells     = [make_cell() for _ in range(rnn_layers)]
                self.cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)
            except AttributeError:
                self.cell = make_cell()
        else:
            self.cell = make_cell()

    # ── Dropout helper ────────────────────────────────────────────────────────
    def _apply_dropout(self, x):
        """
        Apply dropout to tensor x.

        Uses tf.nn.dropout(x, rate=r) — the TF2/Keras3 API where r is the
        fraction of values to DROP (not keep_prob as in TF1).

        Wrapped in tf.cond so that when dropout_placeholder=0.0 (inference)
        the operation is a strict no-op, avoiding any divide-by-zero that
        tf.nn.dropout can produce when rate=0 in some TF builds.
        """
        return tf.cond(
            tf.greater(self.dropout_placeholder, 0.0),
            true_fn=lambda: tf.nn.dropout(x, rate=self.dropout_placeholder),
            false_fn=lambda: x
        )

    # ── Decode step ───────────────────────────────────────────────────────────
    def get_logit_op(self, decoder_inp, context, env, decoder_state):
        decoder_inp = tf.squeeze(decoder_inp, axis=1)

        # Input dropout — equivalent to DropoutWrapper input_keep_prob
        decoder_inp = self._apply_dropout(decoder_inp)

        output, new_state = self.cell(decoder_inp, decoder_state)

        # Output dropout — equivalent to DropoutWrapper output_keep_prob
        hy = self._apply_dropout(output)

        for i in range(self.n_glimpses):
            ref, logit = self.glimpses[i](hy, context, env)
            if self.mask_glimpses:
                logit -= self.BIGNUMBER * env.mask[:, :env.n_nodes]
            prob = tf.nn.softmax(logit)
            hy   = tf.squeeze(tf.matmul(tf.expand_dims(prob, 1), ref), 1)

        _, logit = self.pointer(hy, context, env)
        if self.mask_pointer:
            logit -= self.BIGNUMBER * env.mask[:, :env.n_nodes]

        return logit, new_state

    @property
    def dropout(self):
        """Used by attention_agent.py in feed_dict as decodeStep.dropout."""
        return self.dropout_placeholder
