import keras.backend as K
import tensorflow as tf

def flip(t):
    neg_ones = K.ones_like(t) * -1
    return K.pow(neg_ones, t) + t

def binary_precision(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(tf.multiply(y_true, y_pred))
    fp = K.sum(tf.multiply(flip(y_true), y_pred))
    return tp / (tp + fp + K.epsilon())

def binary_recall(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(tf.multiply(y_true, y_pred))
    fn = K.sum(tf.multiply(y_true, flip(y_pred)))
    return tp / (tp + fn + K.epsilon())

def binary_f1(y_true, y_pred):
    p = binary_precision(y_true, y_pred)
    r = binary_recall(y_true, y_pred)
    score = 2 * (p * r) / (p + r + K.epsilon())
    return score
