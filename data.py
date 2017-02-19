import csv
import contextlib
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter

def glove2dict(src_filename):
    with open(src_filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)
        return {line[0]: np.array(list(map(float, line[1: ])),
                                  dtype='float32') for line in reader}

class Data:
    def __init__(self, filename, vocabulary_size=100000):
        self.filename = filename
        self.glove = glove2dict('dataset/glove.6B.50d.txt')
        self.vocabulary = Counter()
        self.questions = {} # Map from id -> text
        for q1_id, q2_id, q1_text, q2_text, duplicates in self.quora_dataset(filename):
            print (q1_id)

    def quora_dataset(self, filename):
        with open(filename, 'r') as question_tsv:
            question_tsv = csv.reader(question_tsv, delimiter='\t')
            next(question_tsv)
            for row in question_tsv:
                yield (row[1], row[2], row[3], row[4], row[5])

    def glove_tokenize(self, sentence):
        return [np.asarray(self.glove[token] for token in word_tokenize(sentence))]

    def generator(self):
        with open(self.filename, 'r') as question_tsv:
            question_tsv = csv.reader(question_tsv, delimiter='\t')
            # move past headers
            next(question_tsv)
            for row in question_tsv:
                q1_id, q2_id = (row[1], row[2])
                q1_text, q2_text = (self.glove_tokenize(row[3]), self.glove_tokenize(row[4]))
                duplicate = row[5]
                questions[q1_id] = q1_text
                questions[q2_id] = q2_text
                yield q1_text, q2_text, duplicate

if __name__ == '__main__':
    data = Data('dataset/raw/quora_duplicate_questions.tsv')
