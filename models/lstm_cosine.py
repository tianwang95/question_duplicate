import sys
sys.path.append('../')
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import utils
from utils import Mode

class LSTMCosine(object):

    def __init__(self,
                 input_length,
                 rnn_dim,
                 word_dim):

        self.word_dim = word_dim
        self.input_length = input_length
        self.dim = rnn_dim
    
    """
    Inputs = [b x L x h (q1), b x L x h (q2)]
    """
    def __call__(self, inputs):
        assert(len(inputs) == 2)
        with tf.name_scope("calc_lengths"):
            lengths = [utils.length(inputs[0]), utils.length(inputs[1])]

        with tf.variable_scope("lstm_q1_q2") as scope:
            lstm_cell = LSTMCell(self.dim)
            q1_output, _ = tf.nn.dynamic_rnn(
                lstm_cell,
                inputs[0],
                dtype = tf.float32,
                sequence_length = lengths[0] 
            )
            
            scope.reuse_variables()

            q2_output, _ = tf.nn.dynamic_rnn(
                lstm_cell,
                inputs[1],
                dtype = tf.float32,
                sequence_length = lengths[1]
            )

        # batch_size x dim
        with tf.name_scope("extract_final"):
            q1_last = utils.last_relevant(q1_output, lengths[0])
            q2_last = utils.last_relevant(q2_output, lengths[1])

        with tf.name_scope("cosine_dist"):
            q1_last_norm = tf.nn.l2_normalize(q1_last, 1)
            q2_last_norm = tf.nn.l2_normalize(q2_last, 1)
            cosine_dist = tf.reduce_sum(tf.multiply(q1_last_norm, q2_last_norm), axis=1, keep_dims=True)
            output = tf.concat([tf.multiply(-1.0, cosine_dist), cosine_dist], axis=1)

        return output
