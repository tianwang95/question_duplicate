import sys
sys.path.append('../')
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import utils
from custom.layers import self_attention
from utils import Mode


class LSTMKway(object):
    def __init__(self,
                 rnn_dim,
                 k_choices,
                 num_dense=0,
                 use_dense=False,
                 use_self_attention=False,
                 stack_size=1,
                 use_bilstm=False):

        self.dim = rnn_dim
        self.k_choices = k_choices
        self.num_dense = num_dense
        self.use_dense = use_dense
        self.use_self_attention = use_self_attention
        self.stack_size = stack_size
        self.use_bilstm = use_bilstm
        assert(stack_size > 0 or (stack_size >= 0 and use_self_attention),
               "Stack size invalid. Got a stack size of {}".format(self.stack_size))
        if self.stack_size == 0:
            print("WARNING: Stack size of 0 is being used! This means no LSTM operation is actually performed")

    """
    Inputs = [b x L x h (q1), b x k x L x h (choices)]
    """
    def __call__(self, inputs):
        assert(len(inputs) == 2)
        assert(inputs[1].get_shape()[1] == self.k_choices)
        choices_shape = inputs[1].get_shape().as_list()
        with tf.name_scope("calc_lengths"):
            choices = tf.reshape(inputs[1],
                                 [-1, choices_shape[2], choices_shape[3]])
            lengths = [utils.length(inputs[0]), utils.length(choices)]

        with tf.name_scope("lstm_stack"):
            q1_output = None
            choices_output = None
            curr_inputs = [inputs[0], choices]
            if self.use_bilstm:
                # BILSTM implementation
                for i in range(self.stack_size):
                    with tf.variable_scope("lstm_q1_choices_{}".format(i)) as scope:
                        lstm_cell_fw = LSTMCell(self.dim)
                        lstm_cell_bw = LSTMCell(self.dim)
                        q1_output, _ = tf.nn.bidirectional_dynamic_rnn(
                            lstm_cell_fw,
                            lstm_cell_bw,
                            curr_inputs[0],
                            dtype=tf.float32,
                            sequence_length=lengths[0]
                        )
                        scope.reuse_variables()
                        choices_output, _ = tf.nn.bidirectional_dynamic_rnn(
                            lstm_cell_fw,
                            lstm_cell_bw,
                            curr_inputs[1],
                            dtype=tf.float32,
                            sequence_length=lengths[1]
                        )
                        q1_output = tf.concat(q1_output, -1)
                        choices_output = tf.concat(choices_output, -1)
                        curr_inputs = [q1_output, choices_output]
                        print('stack {}'.format(i))
            else:
                # stacked lstm implementation
                for i in range(self.stack_size):
                    with tf.variable_scope("lstm_q1_choices_{}".format(i)) as scope:
                        lstm_cell = LSTMCell(self.dim)
                        q1_output, _ = tf.nn.dynamic_rnn(
                            lstm_cell,
                            curr_inputs[0],
                            dtype=tf.float32,
                            sequence_length=lengths[0]
                        )

                        scope.reuse_variables()

                        choices_output, _ = tf.nn.dynamic_rnn(
                            lstm_cell,
                            curr_inputs[1],
                            dtype=tf.float32,
                            sequence_length=lengths[1]
                        )

                        curr_inputs = [q1_output, choices_output]
                        print('stack {}'.format(i))

        # Check if stack size is 0
        if q1_output is None and choices_output is None:
            q1_output = inputs[0]
            choices_output = choices

        # batch_size x dim
        if self.use_self_attention:
            with tf.variable_scope('self_attention_var') as scope:
                q1_last = self_attention(q1_output)
                scope.reuse_variables()
                choices_last = self_attention(choices_output)
        else:
            with tf.name_scope("extract_final"):
                q1_last = utils.last_relevant(q1_output, lengths[0])
                choices_last = utils.last_relevant(choices_output, lengths[1])

        if not self.use_dense:
            with tf.name_scope("cosine_dist"):
                q1_last_norm = tf.nn.l2_normalize(q1_last, 1)
                choices_last_norm = tf.nn.l2_normalize(choices_last, 1)
                q1_last_norm_rpt = tf.reshape(tf.tile(q1_last_norm, [1, self.k_choices]),
                                              [-1, q1_last_norm.get_shape().as_list()[1]])
                cosine_dist = tf.reduce_sum(tf.multiply(q1_last_norm_rpt, choices_last_norm), axis=1, keep_dims=True)
                output = tf.reshape(cosine_dist, [-1, self.k_choices])

            return output, q1_last_norm
        else:
            with tf.name_scope("concat"):
                q1_last_rpt = tf.reshape(tf.tile(q1_last, [1, self.k_choices]), [-1, q1_last.get_shape().as_list()[1]])
                # choices_last  (batch size * 15) x dim
                merged_last = tf.concat([q1_last_rpt, choices_last], -1)
                # (batch size * 15) x 2 * dim

            with tf.name_scope("dense"):
                """
                for i in range(self.num_dense):
                    merged_last = tf.layers.dense(merged_last, 2 * self.dim,
                                                  activation=tf.nn.relu, name="dense_{}".format(i))
                                                  """
                merged_last = tf.layers.dense(merged_last, 2 * merged_last.get_shape().as_list()[-1],
                                              activation=tf.nn.relu, name="dense_hidden")
                merged_last_condensed = tf.layers.dense(merged_last, 1, activation=tf.nn.relu, name="final_dense")
                output = tf.reshape(merged_last_condensed, [-1, self.k_choices])
            return output, q1_last
