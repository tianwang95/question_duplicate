from keras import backend as K
from keras.layers import GRU
from keras.engine.topology import Layer
import numpy as np
from custom.utils import get_time_index
import tensorflow as tf

class InitStateGRU(GRU):
    def __init__(self, *args, **kwargs):
        super(InitStateGRU, self).__init__(*args, **kwargs)

    def get_initial_states(self, x):
        return [self.init_state]

    def call(self, inputs, mask=None):
        self.init_state = inputs[1]
        return super(InitStateGRU, self).call(inputs[0], mask[0])

    def build(self, input_shape):
        return super(InitStateGRU, self).build(input_shape[0])

"""
Inputs:
    Y       s x L x k
    h_N     s x k
Weights:
    W_y     k x k
    W_h     k x k
    w       k
Operations:
    M = tanh(W_y * y + W_h * y)                 s x k x L
    a = softmax(w.T * M)                        s x L
    r = Y* a.T                                  s x k
    h* = tanh(W_p * r + W_x * h_N)              s x k
"""
class Attention(Layer):
    def __init__(self,
                 W_y_regularizer = None,
                 W_h_regularizer = None,
                 w_regularizer = None,
                 W_p_regularizer = None,
                 W_x_regularizer = None,
                 **kwargs):
        self.W_y_regularizer = W_y_regularizer
        self.W_h_regularizer = W_h_regularizer
        self.w_regularizer = w_regularizer
        self.W_p_regularizer = W_p_regularizer
        self.W_x_regularizer = W_x_regularizer
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.L = input_shape[0][1]
        self.k = input_shape[0][2]
        self.output_dim = self.k

        self.W_y = self.add_weight((self.k, self.k),
                                   initializer='glorot_uniform',
                                   regularizer=self.W_y_regularizer,
                                   trainable = True)
        self.W_h = self.add_weight((self.k, self.k),
                                   initializer='glorot_uniform',
                                   regularizer=self.W_y_regularizer,
                                   trainable = True)
        self.W_p = self.add_weight((self.k, self.k),
                                   initializer='glorot_uniform',
                                   regularizer=self.W_p_regularizer,
                                   trainable = True)
        self.W_x = self.add_weight((self.k, self.k),
                                   initializer='glorot_uniform',
                                   regularizer=self.W_x_regularizer,
                                   trainable = True)
        self.w = self.add_weight((self.k, 1),
                                 initializer='glorot_uniform',
                                 regularizer=self.w_regularizer,
                                 trainable = True)
        self.built = True
    
    def call(self, inputs, mask=None):
        Y = inputs[0]
        h_N = inputs[1]
        h_N_rpt = K.repeat(h_N, self.L)

        M = K.tanh(K.dot(Y, self.W_y) + K.dot(h_N_rpt, self.W_h))
        print("M = ", M)
        # Mask
        if mask != None and mask[0] != None:
            y_mask = mask[0]
            if y_mask.dtype != tf.bool:
                y_mask = tf.cast(y_mask, tf.bool)
            if len(y_mask.get_shape()) == 2:
                y_mask = K.expand_dims(y_mask)
            tiled_y_mask = tf.tile(y_mask, [1, 1, tf.shape(M)[2]])
            M = tf.where(tiled_y_mask, x=M, y=K.zeros_like(M))
        a = K.softmax(K.dot(M, self.w))
        print("a = ", a)
        r = K.squeeze(K.batch_dot(Y, a, [[1, 2], [1, 2]]), 2)
        print("r = ", r)
        h_star = K.tanh(K.dot(r, self.W_p) + K.dot(h_N, self.W_x))
        print("h_star = ", h_star)
        return h_star

    def compute_mask(self, inputs, input_mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return input_shape[1]

"""
Gets the last time index of the passed in tensor
"""
class GetLastIndex(Layer):
        def __init__(self, **kwargs):
            self.supports_masking = True
            super(GetLastIndex, self).__init__(**kwargs)

        def build(self, inputs, input_mask=None):
            self.built = True

        def call(self, x, mask=None):
            return get_time_index(x, K.shape(x)[1] - 1)

        def compute_mask(self, inputs, input_mask=None):
            return None

"""
P, Q    s x L x d
X1 = ReLU(W_a * P^T + b_a)      h x L x s
h_b = ReLU(W_b * h_a + b_b)     h x L x s
"""
class DecomposableAttention(Layer):
    def __init__(self, hidden_size = 2, **kwargs):
        self.supports_masking = True
        super(DecomposableAttention, self).__init__(**kwargs)

    """
    Set up weights for feed-forward neural network
    """
    def build(self, inputs, input_mask = None):
        self.built = True

