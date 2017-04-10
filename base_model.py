import tensorflow as tf
import utils
from utils import Mode

"""
Trains a tf model on some generator
"""
class BaseModel(object):

    def __init__(self, data, embed_weights, model, train_embed = True, log_dir = None, save_dir = None):
        self.data = data
        self.input_length = data.max_sentence_length
        self.word_dim = embed_weights.shape[1]
        self.log_dir = log_dir
        self.save_dir = save_dir

        # inputs
        with tf.name_scope('inputs'):
            q1_input = tf.placeholder(tf.int32, [None, data.max_sentence_length], name = "x_1")
            q2_input = tf.placeholder(tf.int32, [None, data.max_sentence_length], name = "x_2")
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
            self.output = model([q1_embed, q2_embed], [utils.length(q1_embed), utils.length(q2_embed)])

        # get the actual prediction
        with tf.name_scope('prediction'):
            self.y = tf.argmax(tf.nn.softmax(self.output), axis=1)

        with tf.name_scope('metrics'):
            # compute loss
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=self.output))

            # compute accuracy
            self.accuracy, _ = tf.metrics.accuracy(self.y_, self.y)

            # compute f1
            self.precision, _ = tf.metrics.precision(self.y_, self.y)
            self.recall, _ = tf.metrics.recall(self.y_, self.y)
            self.f1 = 2.0 * (self.precision * self.recall) / (self.precision + self.recall)

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('f1', self.f1)
        self.merged_summary = tf.summary.merge_all()
        
    def train(self):
        def feed_dict(batch):
            return {self.inputs[0]: batch[0][0], self.inputs[1]: batch[0][1], self.y_: batch[1]}

        with tf.Session() as sess:
            # aggregate summaries and create file writers
            self.train_writer = None
            self.val_writer = None
            if self.log_dir != None:
                self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', sess.graph)
                self.val_writer = tf.summary.FileWriter(self.log_dir + '/val')

            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            
            iteration = 0
            dev_generator = self.data.dev_generator()
            for train_batch in self.data.train_generator():
                if self.val_writer != None and iteration % 10 == 0:
                    summary = sess.run(self.merged_summary, feed_dict = feed_dict(next(dev_generator)))
                    self.val_writer.add_summary(summary, iteration)
                if self.train_writer != None:
                    summary, _ = sess.run([self.merged_summary, self.train_step], feed_dict = feed_dict(train_batch))
                    self.train_writer.add_summary(summary, iteration)
                else:
                    sess.run(self.train_step, feed_dict = feed_dict(train_batch))
                iteration += 1