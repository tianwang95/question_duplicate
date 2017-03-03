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
    def __init__(self, filename, embed_dim = 50, training=0.75, test=.2, batch_size=32, randomize = False, limit = None, delim_questions = False, dev_mode = False):
        self.filename = filename
        print("Reading Glove vectors... ")
        self.glove = glove2dict('dataset/glove.6B.{}d.txt'.format(embed_dim)) # Glove dictionary
        self.batch_size = batch_size
        self.randomize = randomize
        self.embed_dim = embed_dim
        self.delim_questions = delim_questions
        self.delim_char = '<::>'

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

        print("Iterating over question samples to extract metadata...")
        # Iterate over Quora questions to extract metadata
        self.questions = {}             # Map from id -> tokenized text
        self.questions_embed = {}      # Map from id -> embeded array
        self.max_sentence_length = 0    # Maximum sentence length
        self.dataset = []               # list of (q1_id, q2_id, is_duplicate)
        
        count = 0
        for q1_id, q2_id, q1_text, q2_text, is_duplicate in Data.quora_dataset(filename):
            if dev_mode and count >= 10000:
                break;
            self.add_question(q1_id, q1_text)
            self.add_question(q2_id, q2_text)
            self.dataset.append((q1_id, q2_id, is_duplicate))
            count += 1
        
        if self.delim_questions:
            self.max_sentence_length += 1
            
        for q_id, sentence in self.questions.items():
            self.transform_to_embed(q_id, sentence)

        self.number_samples = len(self.dataset)         # Total number of question pair samples
        self.train_start = 0
        self.train_count = int(len(self.dataset) * training)
        self.test_start = self.train_count
        self.test_count = int(len(self.dataset) * test)
        self.dev_start = self.test_start + self.test_count
        self.dev_count = len(self.dataset) - self.train_count - self.test_count

        if limit:
            self.train_count = min(self.train_count, limit)
            self.dev_count = min(self.dev_count, limit)

        print("Data generator initialized")

    def add_question(self, q_id, text, prepend_delim = False):
        sentence = word_tokenize(text)
        self.max_sentence_length = max(self.max_sentence_length, len(sentence))
        self.questions[q_id] = sentence

    def transform_to_embed(self, q_id, tokens):
        X = self.pad_array(np.asarray([self.token_to_ind[token.lower()] for token in tokens], dtype='int32'))
        self.questions_embed[q_id] = X

    def pad_array(self, array):
        padded = np.zeros((self.max_sentence_length,), dtype='int32')
        padded[:array.shape[0]] = array
        return padded

    # Get a random vector
    def random_vector(self):
        return np.random.uniform(low=-0.05, high=0.05, size=(self.embed_dim,)).astype('float32') 

    @staticmethod
    def quora_dataset(filename):
        with open(filename, 'r') as f:
            question_tsv = csv.reader(f, delimiter = '\t')
            next(question_tsv)
            for row in question_tsv:
                yield int(row[1]), int(row[2]), row[3], row[4], int(row[5])

    # skip: % of samples to skip.
    # total: % of samples to generate.
    def __partial_generator(self, start, count):
        samples = self.dataset[start : start + count]
        while True:
            if (self.randomize):
                samples = random.shuffle(samples)
            for sample in samples:
                X1 = self.questions_embed[sample[0]]
                X2 = self.questions_embed[sample[1]]
                if self.delim_questions:
                    X2 = np.hstack([self.token_to_ind[self.delim_char], X2[:-1]])
                y = sample[2]
                yield(X1, X2, y)

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
                yield([np.vstack(batch_X1), np.vstack(batch_X2)], np.vstack(batch_y))
                batch_X1 = []
                batch_X2 = []
                batch_y = []

    # Generator for training data points
    def train_generator(self):
        return self.batch_generator(self.__partial_generator(
            self.train_start, self.train_count))

    # Generator for test data points
    def test_generator(self):
        return self.batch_generator(self.__partial_generator(
            self.test_start, self.test_count))

    # Generator for dev data points
    def dev_generator(self):
        return self.batch_generator(self.__partial_generator(
            self.dev_start, self.dev_count))

if __name__ == '__main__':
    data = Data('dataset/raw/quora_duplicate_questions.tsv')
    generator = data.train_generator()
    print(next(generator))
    print(next(generator))
    print(next(generator))
