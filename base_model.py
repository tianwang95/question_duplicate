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

    def __init__(self, data, embed_weights, model, train_embed = False, log_dir = None, save_dir = None, save_freq = None, k_choices = None, use_pos = False):
        self.data = data
        self.input_length = data.max_sentence_length
        self.word_dim = embed_weights.shape[1]
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.save_freq = save_freq if save_freq else (int(data.train_count / data.batch_size) + (0 if data.train_count % data.batch_size == 0 else 1))
        print("save_freq\t{}".format(self.save_freq))
        self.k_choices = k_choices
        self.use_pos = use_pos

        # inputs
        with tf.name_scope('inputs'):
            self.inputs = []
            # Q1 Input
            self.inputs.append(tf.placeholder(tf.int32, [None, data.max_sentence_length], name = "x_1"))
            if self.k_choices == None:
                # Q2 Input
                self.inputs.append(tf.placeholder(tf.int32, [None, data.max_sentence_length], name = "x_2"))
            else:
                # Choices Input
                self.inputs.append(tf.placeholder(tf.int32, [None, self.k_choices, data.max_sentence_length], name = "x_2"))
                if self.use_pos:
                    # Q1 POS
                    self.inputs.append(tf.placeholder(tf.int32, [None, data.max_sentence_length], name = "x_1_pos"))
                    # Choices POS
                    self.inputs.append(tf.placeholder(tf.int32, [None, self.k_choices, data.max_sentence_length], name = "x_2_pos"))

        with tf.name_scope('targets'):
            self.y_ = tf.placeholder(tf.int32, [None], name = "y_")

        # embed
        with tf.name_scope('embedding'):
            embedding = tf.get_variable("EmbedWeights",
                                        initializer=tf.constant_initializer(embed_weights, dtype=tf.float32),
                                        trainable=train_embed, shape=embed_weights.shape)
            q1_embed = tf.nn.embedding_lookup(embedding, self.inputs[0])
            q2_embed = tf.nn.embedding_lookup(embedding, self.inputs[1])
            print(q1_embed.get_shape())
            print(q2_embed.get_shape())
            if self.use_pos:
                q1_embed = tf.concat([q1_embed, tf.one_hot(self.inputs[2], len(data.pos_id_to_tag), dtype=tf.float32)], -1)
                q2_embed = tf.concat([q2_embed, tf.one_hot(self.inputs[3], len(data.pos_id_to_tag), dtype=tf.float32)], -1)
                print(q1_embed.get_shape())
                print(q2_embed.get_shape())

        # build the heart of the model
        with tf.name_scope('model'):
            returned = model([q1_embed, q2_embed])
            self.output = None
            self.question_embedding = None
            if len(returned) > 1:
                self.output = returned[0]
                self.question_embedding = tf.identity(returned[1], name='question_embedding')
            else:
                self.output = returned

        # get the actual prediction
        with tf.name_scope('prediction'):
            self.output_softmax = tf.nn.softmax(self.output)
            self.y = tf.cast(tf.argmax(self.output_softmax, axis=1), tf.int32)
            self.y = tf.identity(self.y, name='prediction_index')

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

        # Print import tensor names
        print("Input Q1 Node:\t{}".format(self.inputs[0].name))
        if self.question_embedding != None:
            print("Q1 Embed Node:\t{}".format(self.question_embedding))

    def feed_dict(self, batch):
        if self.use_pos:
            return {self.inputs[0]: batch[0],
                    self.inputs[1]: batch[1],
                    self.y_: batch[2],
                    self.inputs[2]: batch[3],
                    self.inputs[3]: batch[4]}
        else:
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

            # Training
            tf.global_variables_initializer().run()
            dev_generator = self.data.dev_generator()
            iteration = 0
            save_count = 0
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
                    loss, acc = self.evaluate(self.data.dev_generator(loop = False), sess, save_count)
                    if self.save_dir != None:
                        self.saver.save(sess, os.path.join(self.save_dir, "model{}-acc:{:.3f}-loss:{:.3f}.ckpt".format(save_count, acc, loss)), global_step = iteration)
                    save_count += 1

                # update everything  
                iteration += 1

    def evaluate(self, generator, sess, iteration, name = 'evaluation'):
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
            curr_mistakes = [list(self.data.point_to_words((batch[0][i], batch[1][i], batch[2][i]))) + [y[i]] for i in range(batch[0].shape[0]) if not is_correct[i]]
            mistakes += curr_mistakes

            count += batch[0].shape[0]

        loss_agg = loss_total / count
        accuracy_agg = accuracy_total / count

        print("Loss:\t{}".format(loss_agg))
        print("Accuracy:\t{}".format(accuracy_agg))
        ### save evaluation results
        if self.save_dir != None:
            mistake_file = os.path.join(self.save_dir, "{}-{}-acc:{:.3f}-loss:{:.3f}.tsv".format(name, iteration, accuracy_agg, loss_agg))
            with open(mistake_file, 'w+') as f:
                for mistake in mistakes:
                    f.write("\t".join(str(x) for x in mistake) + "\n")

        return loss_agg, accuracy_agg
