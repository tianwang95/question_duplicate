import sys
sys.path.append('../')
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import utils
from utils import Mode

class LSTMCosine(object):

    def __init__(self,
                 rnn_dim,
                 k_choices,
                 num_dense = 0):

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

        with tf.name_scope("dense"):
            for i in range(self.num_dense):
                q1_last = tf.layers.dense(q1_last, self.dim, activation=tf.nn.relu, name="dense_{}".format(i))
                choices_last = tf.layers.dense(choices_last, self.dim, activation=tf.nn.relu,
                                               name="dense_{}".format(i), reuse=True)

        with tf.name_scope("cosine_dist"):
            q1_last_norm = tf.nn.l2_normalize(q1_last, 1)
            choices_last_norm = tf.nn.l2_normalize(choices_last, 1)
            q1_last_norm_rpt = tf.reshape(tf.tile(q1_last_norm, [1, self.k_choices]), [-1, q1_last_norm.get_shape().as_list()[1]])
            cosine_dist = tf.reduce_sum(tf.multiply(q1_last_norm_rpt, choices_last_norm), axis=1, keep_dims=True)
            output = tf.reshape(cosine_dist, [-1, self.k_choices])

        return output, q1_last_norm
