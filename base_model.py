import tensorflow as tf
import utils
from utils import Mode
import os
import numpy as np
import time

"""
Trains a tf model on some generator
"""
class BaseModel(object):

    def __init__(self, data, embed_weights, model, train_embed = False, log_dir = None, save_dir = None, save_freq = None, k_choices = None):
        self.data = data
        self.input_length = data.max_sentence_length
        self.word_dim = embed_weights.shape[1]
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.save_freq = save_freq if save_freq else (int(data.train_count / data.batch_size) + (0 if data.train_count % data.batch_size == 0 else 1))
        print("save_freq\t{}".format(self.save_freq))
        self.k_choices = k_choices

        # inputs
        with tf.name_scope('inputs'):
            q1_input = tf.placeholder(tf.int32, [None, data.max_sentence_length], name = "x_1")
            q2_input = None
            if self.k_choices == None:
                q2_input = tf.placeholder(tf.int32, [None, data.max_sentence_length], name = "x_2")
            else:
                q2_input = tf.placeholder(tf.int32, [None, self.k_choices, data.max_sentence_length], name = "x_2")
            self.inputs = [q1_input, q2_input]

        with tf.name_scope('targets'):
            self.y_ = tf.placeholder(tf.int32, [None], name = "y_")

        # embed
        with tf.name_scope('embedding'):
            embedding = tf.Variable(tf.constant(embed_weights, name="InitEmbedWeights"), trainable=train_embed, name="EmbedWeights")
            q1_embed = tf.nn.embedding_lookup(embedding, q1_input)
            q2_embed = tf.nn.embedding_lookup(embedding, q2_input)

        # build the heart of the model
        with tf.name_scope('model'):
            self.output = model([q1_embed, q2_embed])

        # get the actual prediction
        with tf.name_scope('prediction'):
            self.output_softmax = tf.nn.softmax(self.output)
            self.y = tf.cast(tf.argmax(self.output_softmax, axis=1), tf.int32)

        with tf.name_scope('metrics'):
            # compute loss
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=self.output))

            # compute accuracy
            self.accuracy = utils.accuracy(self.y_, self.y)

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged_summary = tf.summary.merge_all()

    def feed_dict(self, batch):
        return {self.inputs[0]: batch[0], self.inputs[1]: batch[1], self.y_: batch[2]}
        
    def train(self):
        with tf.Session() as sess:
            # aggregate summaries and create file writers
            self.train_writer = None
            self.val_writer = None
            if self.log_dir != None:
                self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', sess.graph)
                self.val_writer = tf.summary.FileWriter(self.log_dir + '/val')
            self.saver = tf.train.Saver()

            tf.global_variables_initializer().run()
            
            iteration = 0
            dev_generator = self.data.dev_generator()

            # timer
            start_time = time.time()
            for train_batch in self.data.train_generator():
                # train step
                if self.train_writer != None:
                    summary, _= sess.run([self.merged_summary, self.train_step], feed_dict = self.feed_dict(train_batch))
                    self.train_writer.add_summary(summary, iteration)
                else:
                    sess.run(self.train_step, feed_dict = self.feed_dict(train_batch))

                # eval 1 dev minibatch every 10 train minibatches
                if self.val_writer != None and iteration % 10 == 0:
                    summary = sess.run(self.merged_summary, feed_dict = self.feed_dict(next(dev_generator)))
                    self.val_writer.add_summary(summary, iteration)

                # save every self.save_freq examples
                if iteration % self.save_freq == self.save_freq - 1:
                    print("Iteration: {}\tTime: {}".format(iteration, time.time() - start_time))
                    loss, acc = self.evaluate(self.data.dev_generator(loop = False), sess)
                    if self.save_dir != None:
                        self.saver.save(sess, os.path.join(self.save_dir, "model-acc:{}-loss:{}.ckpt".format(acc, loss)), global_step = iteration)

                # update everything  
                iteration += 1

    def evaluate(self, generator, sess, name = 'evaluation'):
        loss_total = 0.0
        accuracy_total = 0.0
        mistakes = []

        count = 0

        for batch in generator:
            loss, accuracy, y = sess.run([self.loss, self.accuracy, self.y], feed_dict = self.feed_dict(batch))
            loss_total += loss * batch[0].shape[0]
            accuracy_total += accuracy * batch[0].shape[0]

            # find mistakes
            is_correct = np.equal(batch[2], y).tolist()
            curr_mistakes = [self.data.point_to_words((batch[0][i], batch[1][i], batch[2][i])) for i in range(batch[0].shape[0]) if not is_correct[i]]
            mistakes += curr_mistakes

            count += batch[0].shape[0]

        loss_agg = loss_total / count
        accuracy_agg = accuracy_total / count

        print("Loss:\t{}".format(loss_agg))
        print("Accuracy:\t{}".format(accuracy_agg))
        ### save evaluation results
        if self.save_dir != None:
            mistake_file = os.path.join(self.save_dir, "{}-acc:{}-loss:{}.tsv".format(name, accuracy_agg, loss_agg))
            with open(mistake_file, 'w+') as f:
                for mistake in mistakes:
                    f.write("{}\t{}\t{}\n".format(mistake[0], mistake[1], mistake[2]))

        return loss_agg, accuracy_agg

