import sys
sys.path.append('../')
import unittest
import tensorflow as tf
import numpy as np
from custom.layers import self_attention


class TestLayers(unittest.TestCase):

    def test_self_attention(self):
        input_sequence = np.asarray([[[7, 6, 3, 9, 9],
                                      [8, 3, 7, 9, 5],
                                      [5, 6, 4, 5, 4],
                                      [4, 6, 6, 6, 5]],
                                     [[2, 6, 6, 3, 8],
                                      [2, 1, 4, 4, 4],
                                      [2, 8, 3, 8, 5],
                                      [1, 4, 5, 7, 6]],
                                     [[6, 2, 8, 2, 3],
                                      [3, 2, 2, 3, 8],
                                      [3, 3, 4, 3, 7],
                                      [5, 2, 7, 5, 9]]], dtype=np.float32)

        weights = np.expand_dims(np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32), -1)

        with tf.name_scope('test_input'):
            input_tensor = tf.placeholder(tf.float32, [None, 4, 5])

        output = self_attention(input_tensor, init_weights=weights)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            result = sess.run(output, feed_dict={input_tensor: input_sequence})

        correct = np.asarray([[7.00958157,  5.40108204,  3.97519779,  8.78310871,  7.91672659],
                              [1.71367991,  6.05550909,  4.63603306,  5.90809727,  6.32120323],
                              [4.81426954,  2.05898523,  6.62419224,  4.7357316,  8.72052574]], dtype=np.float32)

        self.assertTrue(np.all(np.isclose(result, correct)))


if __name__ == '__main__':
    unittest.main()
