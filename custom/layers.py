import tensorflow as tf

"""
Self attention. Takes sequence output of rnn and re-weights the individual timesteps before summing
----
Expects tensors of batch_size x length x dim
"""


def self_attention(input):
    with tf.name_scope("self_attention"):
        input_shape = input.get_shape()
        w1 = tf.get_variable("w1", shape=[input_shape[-1], 1])
        input_reshaped = tf.reshape(input, [-1, input_shape[-1]]) # (batch_size * length) * dim
        multiplied = tf.matmul(input_reshaped, w1) # (batch_size * length) * 1
        attention_raw = tf.reshape(multiplied, [-1, input_shape[1]]) # batch_size * length
        attention_weights = tf.nn.softmax(attention_raw)
        reweight_input = input * tf.expand_dims(attention_weights, -1)
        output = tf.reduce_sum(reweight_input, 1)
        return output

