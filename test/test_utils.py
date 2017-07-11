import sys
sys.path.append('../')
import unittest
import tensorflow as tf
import numpy as np
from utils import accuracy


class TestUtils(unittest.TestCase):

    def test_accuracy_exact_match(self):
        prediction = tf.get_variable("prediction", [6], initializer=tf.constant_initializer(np.asarray([1, 0, 1, 1, 0, 1], np.int32)))
        gold = tf.get_variable("gold", [6], initializer=tf.constant_initializer(np.asarray([1, 0, 1, 1, 0, 1], np.int32)))
        acc = accuracy(gold, prediction)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            result = sess.run(acc)

        self.assertEqual(result, 1.0)

if __name__ == '__main__':
    unittest.main()
