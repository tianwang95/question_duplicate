import csv
import contextlib
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
import random

def glove2dict(src_filename):
    with open(src_filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)
        return {line[0]: np.array(list(map(float, line[1: ])),
                                  dtype='float32') for line in reader}

class Data:
    def __init__(self, train_file, dev_file, test_file, embed_dim = 50, batch_size=32, randomize = False, limit = None, delim_questions = False):
        print("Reading Glove vectors... ")
        self.glove = glove2dict('dataset/glove.6B.{}d.txt'.format(embed_dim)) # Glove dictionary
        self.batch_size = batch_size
        self.randomize = randomize
        self.embed_dim = embed_dim
        self.delim_questions = delim_questions
        self.delim_char = '<DELIM>'
        self.limit = limit

        print("Creating embedding matrix for Keras... ")
        self.vocab_size = len(self.glove)          # Number of tokens
        # List of all tokens
        self.vocabulary = (["<PAD>", "<UNK>"]
                          + sorted(list(self.glove.keys()))
                          + ([self.delim_char] if self.delim_questions else []))

        self.token_to_ind = defaultdict(lambda: 1)              # Map from token to index in above list
        self.embedding_matrix = np.zeros((len(self.vocabulary), self.embed_dim), dtype='float32')
        for i, token in enumerate(self.vocabulary):
            self.token_to_ind[token] = i
            embedding = self.glove[token] if token in self.glove else self.random_vector()
            self.embedding_matrix[i] = embedding

        # make sure padding goes to all zeros
        self.embedding_matrix[0] = np.zeros((self.embed_dim), dtype='float32')

        print("Iterating over question samples to extract metadata...")
        # Iterate over Quora questions to extract metadata

        self.train = []
        self.dev = []
        self.test = []
        self.max_sentence_length = 0    # Maximum sentence length
        
        self.read_split(train_file, self.train)
        self.read_split(dev_file, self.dev)
        self.read_split(test_file, self.test)

        self.pad_dataset(self.train)
        self.pad_dataset(self.dev)
        self.pad_dataset(self.test)

        self.train_count = len(self.train)
        self.dev_count = len(self.dev)
        self.test_count = len(self.test)

        print("Data generator initialized")

    def read_split(self, filename, dataset):
        count = 0
        for is_duplicate, q1_text, q2_text, pair_id in Data.quora_dataset(filename):
            if self.limit != None and count >= self.limit:
                break;
            q1_tokens = q1_text.split()
            q2_tokens = q2_text.split()
            if self.delim_questions:
                q2_tokens = [self.delim_char] + q2_tokens
            self.max_sentence_length = max(len(q1_tokens), self.max_sentence_length)
            self.max_sentence_length = max(len(q2_tokens), self.max_sentence_length)

            q1 = self.transform_to_embed(q1_tokens)
            q2 = self.transform_to_embed(q2_tokens)
            dataset.append((q1, q2, is_duplicate))
            count += 1

    def transform_to_embed(self, tokens):
        return np.asarray([self.token_to_ind[token.lower()] for token in tokens], dtype='int32')

    def pad_array(self, array):
        padded = np.zeros((self.max_sentence_length,), dtype='int32')
        padded[:array.shape[0]] = array
        return padded

    def pad_dataset(self, dataset):
        for i in range(len(dataset)):
            q1 = self.pad_array(dataset[i][0])
            q2 = self.pad_array(dataset[i][1])
            dataset[i] = (q1, q2, dataset[i][2])

    # Get a random vector
    def random_vector(self):
        return np.random.uniform(low=-0.05, high=0.05, size=(self.embed_dim,)).astype('float32') 

    @staticmethod
    def quora_dataset(filename):
        with open(filename, 'r') as f:
            question_tsv = csv.reader(f, delimiter = '\t')
            next(question_tsv)
            for row in question_tsv:
                yield int(row[0]), row[1], row[2], int(row[3])

    # skip: % of samples to skip.
    # total: % of samples to generate.
    def __partial_generator(self, dataset):
        if (self.randomize):
            random.shuffle(dataset)
        for sample in dataset:
            yield sample

    # Batches samples from a generator.
    def batch_generator(self, sample_generator):
        batch_X1 = []
        batch_X2 = []
        batch_y = []
        for x1, x2, y in sample_generator:
            batch_X1.append(x1)
            batch_X2.append(x2)
            batch_y.append(y)
            if len(batch_y) >= self.batch_size:
                yield([np.vstack(batch_X1), np.vstack(batch_X2)], np.squeeze(np.vstack(batch_y)))
                batch_X1 = []
                batch_X2 = []
                batch_y = []
        if len(batch_y) > 0:
            yield([np.vstack(batch_X1), np.vstack(batch_X2)], np.squeeze(np.vstack(batch_y)))

    # Generator for training data points
    def train_generator(self):
        while True:
            for batch in self.batch_generator(self.__partial_generator(self.train)):
                yield batch

    # Generator for test data points
    def test_generator(self):
        while True:
            for batch in self.batch_generator(self.__partial_generator(self.test)):
                yield batch

    # Generator for dev data points
    def dev_generator(self):
        while True:
            for batch in self.batch_generator(self.__partial_generator(self.dev)):
                yield batch


if __name__ == '__main__':
    data = Data('dataset/raw/train.tsv', 'dataset/raw/dev.tsv', 'dataset/raw/test.tsv', limit = 100)
    generator = data.dev_generator()
    print(data.dev_count)
    print(next(generator))
    print(next(generator))
    print(next(generator))
