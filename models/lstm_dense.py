import sys
sys.path.append('../')
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import utils


class LSTMDense(object):
    def __init__(self,
                 rnn_dim,
                 k_choices,
                 num_dense=0):

        self.dim = rnn_dim
        self.k_choices = k_choices
        self.num_dense = num_dense
    
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

        with tf.variable_scope("lstm_q1_choices") as scope:
            lstm_cell = LSTMCell(self.dim)
            q1_output, _ = tf.nn.dynamic_rnn(
                lstm_cell,
                inputs[0],
                dtype=tf.float32,
                sequence_length=lengths[0]
            )

            scope.reuse_variables()

            choices_output, _ = tf.nn.dynamic_rnn(
                lstm_cell,
                choices,
                dtype=tf.float32,
                sequence_length=lengths[1]
            )

        # batch_size x dim
        with tf.name_scope("extract_final"):
            q1_last = utils.last_relevant(q1_output, lengths[0])
            choices_last = utils.last_relevant(choices_output, lengths[1])

        with tf.name_scope("concat"):
            q1_last_rpt = tf.reshape(tf.tile(q1_last, [1, self.k_choices]), [-1, q1_last.get_shape().as_list()[1]])
            # choices_last  (batch size * 15) x dim
            merged_last = tf.concat([q1_last_rpt, choices_last], -1)
            # (batch size * 15) x 2 * dim

        with tf.name_scope("dense"):
            for i in range(self.num_dense):
                merged_last = tf.layers.dense(merged_last, 2 * self.dim,
                                              activation=tf.nn.relu, name="dense_{}".format(i))

            merged_last_condensed = tf.layers.dense(merged_last, 1, activation=tf.nn.relu, name="final_dense")
            output = tf.reshape(merged_last_condensed, [-1, self.k_choices])

        return output, q1_last
