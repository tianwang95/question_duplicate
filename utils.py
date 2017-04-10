import tensorflow as tf
from enum import Enum

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices = 1)
    length = tf.cast(length, tf.int32)
    return length

"""
Output: batch_size x L x dim
Lenght: batch_size (vector of lengths)
"""
def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

class Mode(Enum):
    TRAIN = 1
    INFER = 2
    EVAL = 3