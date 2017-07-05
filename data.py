import csv
import contextlib
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
import random
import os

def glove2dict(src_filename):
    with open(src_filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)
        return {line[0]: np.array(list(map(float, line[1: ])),
                                  dtype='float32') for line in reader}

def transform_to_embed(token_to_ind, tokens):
        return np.asarray([token_to_ind[token.lower()] for token in tokens], dtype='int32')

def pad_array(array, length):
    padded = np.zeros((length,), dtype='int32')
    padded[:array.shape[0]] = array
    return padded

# Get a random vector
def random_vector(dim):
    return np.random.uniform(low=-0.05, high=0.05, size=(dim,)).astype('float32') 

def indices_to_words(vocab, array):
    return " ".join([vocab[index] for index in array if index != 0])

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

        if not self.randomize:
            random.seed(42)

        print("Creating embedding matrix...")
        self.vocab_size = len(self.glove)          # Number of tokens
        # List of all tokens
        self.vocabulary = (["<PAD>", "<UNK>"]
                          + sorted(list(self.glove.keys()))
                          + ([self.delim_char] if self.delim_questions else []))

        self.token_to_ind = defaultdict(lambda: 1)              # Map from token to index in above list
        self.embedding_matrix = np.zeros((len(self.vocabulary), self.embed_dim), dtype='float32')
        for i, token in enumerate(self.vocabulary):
            self.token_to_ind[token] = i
            embedding = self.glove[token] if token in self.glove else random_vector(self.embed_dim)
            self.embedding_matrix[i] = embedding

        # make sure padding goes to all zeros
        self.embedding_matrix[0] = np.zeros((self.embed_dim), dtype='float32')

        print("Iterating over question samples...")

        self.max_sentence_length = 0    # Maximum sentence length
        
        self.train = self.read_split(train_file)
        self.dev = self.read_split(dev_file)
        self.test = self.read_split(test_file)

        self.pad_dataset(self.train)
        self.pad_dataset(self.dev)
        self.pad_dataset(self.test)

        self.train_count = len(self.train)
        self.dev_count = len(self.dev)
        self.test_count = len(self.test)

        print("Data generator initialized")

    def read_split(self, filename):
        count = 0
        dataset = []
        for is_duplicate, q1_text, q2_text, pair_id in Data.quora_dataset(filename):
            if self.limit != None and count >= self.limit:
                break;
            q1_tokens = q1_text.split()
            q2_tokens = q2_text.split()
            if self.delim_questions:
                q2_tokens = [self.delim_char] + q2_tokens
            self.max_sentence_length = max(len(q1_tokens), self.max_sentence_length)
            self.max_sentence_length = max(len(q2_tokens), self.max_sentence_length)

            q1 = transform_to_embed(self.token_to_ind, q1_tokens)
            q2 = transform_to_embed(self.token_to_ind, q2_tokens)
            dataset.append((q1, q2, is_duplicate))
            count += 1

        return dataset

    def pad_dataset(self, dataset):
        for i in range(len(dataset)):
            q1 = pad_array(dataset[i][0], self.max_sentence_length)
            q2 = pad_array(dataset[i][1], self.max_sentence_length)
            dataset[i] = (q1, q2, dataset[i][2])
    
    @staticmethod
    def quora_dataset(filename):
        with open(filename, 'r') as f:
            question_tsv = csv.reader(f, delimiter = '\t')
            for row in question_tsv:
                yield int(row[0]), row[1], row[2], int(row[3])

    
    def point_to_words(self, data_point):
        return indices_to_words(self.vocabulary, data_point[0]), indices_to_words(self.vocabulary, data_point[1]), data_point[2]

    # skip: % of samples to skip.
    # total: % of samples to generate.
    def __partial_generator(self, dataset, randomize = False):
        if not randomize:
            random.seed(42)
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
                yield np.vstack(batch_X1), np.vstack(batch_X2), np.squeeze(np.vstack(batch_y))
                batch_X1 = []
                batch_X2 = []
                batch_y = []
        if len(batch_y) > 0:
            yield np.vstack(batch_X1), np.vstack(batch_X2), np.squeeze(np.vstack(batch_y))
            
    # Generator for training data points
    def train_generator(self, loop = True):
        while True:
            for batch in self.batch_generator(self.__partial_generator(self.train, self.randomize)):
                yield batch
            if not loop:
                break

    # Generator for test data points
    def test_generator(self, loop = True):
        while True:
            for batch in self.batch_generator(self.__partial_generator(self.test)):
                yield batch
            if not loop:
                break

    # Generator for dev data points
    def dev_generator(self, loop = True):
        while True:
            for batch in self.batch_generator(self.__partial_generator(self.dev)):
                yield batch
            if not loop:
                break

class DataKWay(object):
    def __init__(self, questions_file, train_file, dev_file, test_file, k_choices, embed_dim = 50, batch_size=32, randomize = False, limit = None, random_distractors = 0):
        assert k_choices > 1, "Number of choices must be greater than 1"
        assert k_choices <= 15, "Number of choices cannot exceed 15 for this dataset"
        print("Reading Glove vectors... ")
        self.glove = glove2dict('dataset/glove.6B.{}d.txt'.format(embed_dim)) # Glove dictionary
        self.batch_size = batch_size
        self.randomize = randomize
        self.embed_dim = embed_dim
        self.delim_char = '<DELIM>'
        self.limit = limit
        self.k_choices = k_choices
        self.random_distractors = random_distractors

        if not self.randomize:
            random.seed(42)

        print("Creating embedding matrix... ")
        self.vocab_size = len(self.glove)          # Number of tokens
        # List of all tokens
        self.vocabulary = (["<PAD>", "<UNK>"]
                          + sorted(list(self.glove.keys())))

        self.token_to_ind = defaultdict(lambda: 1)              # Map from token to index in above list
        self.embedding_matrix = np.zeros((len(self.vocabulary), self.embed_dim), dtype='float32')
        for i, token in enumerate(self.vocabulary):
            self.token_to_ind[token] = i
            embedding = self.glove[token] if token in self.glove else random_vector(self.embed_dim)
            self.embedding_matrix[i] = embedding

        # make sure padding goes to all zeros
        self.embedding_matrix[0] = np.zeros((self.embed_dim), dtype='float32')

        print("Iterating over question samples...")
        # Iterate over Quora questions to extract metadata

        self.max_sentence_length = 0    # Maximum sentence length

        self.id_to_question, self.id_to_pos, self.pos_id_to_tag = self.read_questions(questions_file)
        
        self.train = self.read_split(train_file)
        self.dev = self.read_split(dev_file)
        self.test = self.read_split(test_file)

        self.train_count = len(self.train)
        self.dev_count = len(self.dev)
        self.test_count = len(self.test)

        print("K-way Data generator initialized")

    def read_questions(self, filename):
        id_to_pos_tag = []
        pos_tag_to_id = {}
        def get_pos_tag_id(tag):
            if tag not in pos_tag_to_id:
                pos_tag_to_id[tag] = len(id_to_pos_tag)
                id_to_pos_tag.append(tag)
            return pos_tag_to_id[tag]

        def embed_pos_tags(tags):
            return np.asarray([get_pos_tag_id(tag) for tag in tags], dtype='int32')

        with open(filename, 'r') as f:
            question_tsv = list(csv.reader(f, delimiter = '\t'))
            id_to_question = [None for _ in range(len(question_tsv))]
            id_to_pos = [None for _ in range(len(question_tsv))]
            for row in question_tsv:
                q_id = int(row[0])
                tokens = row[1].split()
                pos = row[2].split()
                self.max_sentence_length = max(self.max_sentence_length, len(tokens))
                id_to_question[q_id] = transform_to_embed(self.token_to_ind, tokens)
                id_to_pos[q_id] = embed_pos_tags(pos)

        for q_id in range(len(id_to_question)):
            id_to_question[q_id] = pad_array(id_to_question[q_id], self.max_sentence_length)
            id_to_pos[q_id] = pad_array(id_to_pos[q_id], self.max_sentence_length)

        return id_to_question, id_to_pos, id_to_pos_tag

    def read_split(self, filename):
        split = []
        count = 0
        with open(filename, 'r') as f:
            tsv = csv.reader(f, delimiter = '\t')
            for row in tsv:
                if self.limit and count >= self.limit:
                    break
                whole_point = [int(x) for x in row]
                split.append(whole_point)
                count += 1
        return split

    def point_to_words(self, point):
        return tuple([indices_to_words(self.vocabulary, point[0])] + [indices_to_words(self.vocabulary, point[1][i]) for i in range(point[1].shape[0])] + [point[2]])

    def get_random_questions(self, count, ignore_list):
        choices = []
        while len(choices) < count:
            choice = random.choice(range(len(self.id_to_question)))
            if choice not in choices and choice not in ignore_list:
                choices.append(choice)
        return choices

    def _partial_generator(self, dataset, randomize = False):
        random.shuffle(dataset)
        for point in dataset:
            q1_id = point[0]
            q2_id = point[1]
            choice_ids = random.sample(point[2:], self.k_choices - 1 - self.random_distractors) + self.get_random_questions(self.random_distractors, [q1_id, q2_id]) + [q2_id]
            random.shuffle(choice_ids)
            choices = [self.id_to_question[q_id] for q_id in choice_ids]
            choices_pos = [self.id_to_pos[q_id] for q_id in choice_ids]
            yield self.id_to_question[q1_id], np.vstack(choices), choice_ids.index(q2_id), self.id_to_pos[q1_id], np.vstack(choices_pos)

    def batch_generator(self, partial_generator):
        x1 = []
        x2 = []
        y = []
        x1_pos = []
        x2_pos = []
        for q1, choices, index, q1_pos, choices_pos in partial_generator:
            x1.append(q1)
            x2.append(choices)
            y.append(index)
            x1_pos.append(q1_pos)
            x2_pos.append(choices_pos)
            if len(y) >= self.batch_size:
                yield np.vstack(x1), np.asarray(x2), np.asarray(y, dtype='int32'), np.vstack(x1_pos), np.asarray(x2_pos)
                x1 = []
                x2 = []
                y = []
                x1_pos = []
                x2_pos = []
        if len(y) > 0:
            yield np.vstack(x1), np.asarray(x2), np.asarray(y, dtype='int32'), np.vstack(x1_pos), np.asarray(x2_pos)

    def train_generator(self, loop = True):
        while True:
            for batch in self.batch_generator(self._partial_generator(self.train, self.randomize)):
                yield batch
            if not loop:
                break

    def dev_generator(self, loop = True):
        while True:
            for batch in self.batch_generator(self._partial_generator(self.dev)):
                yield batch
            if not loop:
                break

    def test_generator(self, loop = True):
        while True:
            for batch in self.batch_generator(self._partial_generator(self.test)):
                yield batch
            if not loop:
                break

if __name__ == '__main__':
    data_dir = '/mnt/disks/main/question_duplicate/dataset/raw'
    question_file = os.path.join(data_dir, 'questions_kway_pos.tsv')
    train_file = os.path.join(data_dir, 'train_15way_cos.tsv')
    dev_file = os.path.join(data_dir, 'dev_15way_cos.tsv')
    test_file = os.path.join(data_dir, 'test_15way_cos.tsv')
    data = DataKWay(question_file, train_file, dev_file, test_file, 5, batch_size=5, limit = 100)
    generator = data.train_generator()
    for _ in range(3):
        batch = next(generator)
        print("q")
        print(batch[0])
        print("choices")
        print(batch[1])
        print("target")
        print(batch[2])
        print("q_pos")
        print(batch[3])
        print("choices_pos")
        print(batch[4])
        for i in range(batch[0].shape[0]):
            print(data.point_to_words((batch[0][i], batch[1][i], batch[2][i])))

    print("self.id_to_pos")
    print(data.pos_id_to_tag)

    """
    train_file = os.path.join(data_dir, 'train.tsv')
    dev_file = os.path.join(data_dir, 'dev.tsv')
    test_file = os.path.join(data_dir, 'test.tsv')
    data = Data(train_file, dev_file, test_file, limit = 1000, batch_size=64)
    generator = data.dev_generator()
    print(next(generator))
    print(next(generator))
    """
