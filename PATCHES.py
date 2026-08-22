# misc_utils.py  –  Minimal fix patch
# Only print_time() is changed (TypeError: float + string).
# All other functions left as-is.

# -----------------------------------------------------------------------
# Original buggy line (line ~45 in the original):
#
#   def print_time(self, s, start_time):
#       self.print_out("%s, time %ds, %s." % (
#           s, (time.time() - start_time) +"  " +str(time.ctime())   # ← BUG
#       ))
#
# Fix:
#   def print_time(self, s, start_time):
#       self.print_out("%s, time %ds, %s." % (
#           s, int(time.time() - start_time), str(time.ctime())       # ← FIXED
#       ))
#       return time.time()
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# PATCH FOR attention_agent.py  (do not create a new file — apply inline)
# -----------------------------------------------------------------------
# Bug 7 fix: embed full feature vector for LRP (not just x,y).
# In build_model(), change the two lines:
#
#   BEFORE:
#       input_pnt        = env.input_pnt
#       encoder_emb_inp  = self.embedding(input_pnt)
#
#   AFTER:
#       # For LRP, embed all 4 features (x, y, demand, node_type)
#       if args.get('task_name') == 'lrp':
#           embedding_input = env.input_data    # [batch, n_nodes, 4]
#       else:
#           embedding_input = env.input_pnt     # [batch, n_nodes, 2]
#       input_pnt        = env.input_pnt        # keep for action coords
#       encoder_emb_inp  = self.embedding(embedding_input)
#
# Also fix the LSTM state initialisation for multi-layer RNN:
#
#   BEFORE:
#       initial_h    = tf.zeros([batch_size * beam_width, args['hidden_dim']])
#       initial_c    = tf.zeros([batch_size * beam_width, args['hidden_dim']])
#       decoder_state = (initial_h, initial_c)
#
#   AFTER:
#       bb = batch_size * beam_width
#       h0 = tf.zeros([bb, args['hidden_dim']])
#       c0 = tf.zeros([bb, args['hidden_dim']])
#       if args['rnn_layers'] == 1:
#           decoder_state = tf.nn.rnn_cell.LSTMStateTuple(h0, c0)
#       else:
#           decoder_state = tuple([
#               tf.nn.rnn_cell.LSTMStateTuple(
#                   tf.zeros([bb, args['hidden_dim']]),
#                   tf.zeros([bb, args['hidden_dim']])
#               )
#               for _ in range(args['rnn_layers'])
#           ])
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# PATCH FOR task_specific_params.py
# -----------------------------------------------------------------------
# decode_len should be long enough to cover multiple trips.
# Rule of thumb: decode_len = n_cust * 2  (covers facility selections + customers)
#
#   lrp10:  decode_len = 20  (was 10)
#   lrp20:  decode_len = 40  (was 20)
#   lrp50:  decode_len = 100 (was 50)
#   lrp100: decode_len = 200 (was 100)
# -----------------------------------------------------------------------
