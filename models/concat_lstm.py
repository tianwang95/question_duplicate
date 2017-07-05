import sys
sys.path.append('../')
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import utils
from utils import Mode

class ConcatLSTM(object):

    def __init__(self,
                 input_length,
                 rnn_dim,
                 word_dim,
                 num_hidden = 0):

        self.word_dim = word_dim
        self.input_length = input_length
        self.dim = rnn_dim
        self.num_hidden = num_hidden
    
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

        with tf.name_scope("merge"):
            merged = tf.concat([q1_last, q2_last], axis=1)

        with tf.name_scope("feed_forward_layers"):
            for i in range(self.num_hidden):
                merged = tf.layers.dense(merged, self.dim * 2, activation=tf.nn.relu, name="dense_{}".format(i))

        with tf.name_scope("final_dense"):
            output = tf.layers.dense(merged, 2)

        return output