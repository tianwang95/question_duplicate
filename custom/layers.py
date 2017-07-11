import tensorflow as tf

"""
Self attention. Takes sequence output of rnn and re-weights the individual timesteps before summing
----
Expects tensors of batch_size x length x dim
"""


def self_attention(input_tensor, name="w1", init_weights=None, dtype=tf.float32):
    with tf.name_scope("self_attention"):
        input_shape = input_tensor.get_shape().as_list()
        if init_weights is not None:
            assert(input_shape[-1] == init_weights.shape[0])
            assert(init_weights.shape[1] == 1)
            w1 = tf.get_variable(name, initializer=tf.constant_initializer(init_weights, dtype=dtype), shape=init_weights.shape)
        else:
            w1 = tf.get_variable(name, shape=[input_shape[-1], 1])
        input_reshaped = tf.reshape(input_tensor, [-1, input_shape[-1]]) # (batch_size * length) * dim
        multiplied = tf.matmul(input_reshaped, w1) # (batch_size * length) * 1
        attention_raw = tf.reshape(multiplied, [-1, input_shape[1]]) # batch_size * length
        attention_weights = tf.nn.softmax(attention_raw)
        reweight_input = input_tensor * tf.expand_dims(attention_weights, -1)
        output = tf.reduce_sum(reweight_input, 1)
        return output
