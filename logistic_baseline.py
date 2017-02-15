#!usr/bin/env
from featurizer import Featurizer
from data_generator import DataGenerator
from stanza.nlp import CoreNLP_pb2

"""
Just hold feature extraction functions
"""
class Features:
    """
    Intersection of q1's set of words and q2's set of words
    """
    def unigram_intersect(datapoint):
        pass

    """
    Intersection of q1 and q2 bigrams
    """
    def bigram_intersect(datapoint):
        pass

    """
    Cross unigram
    """
    def cross_unigram(datapoint):
        pass

    """
    Cross bigram
    """
    def cross_bigram(datapoint):
        pass

    """
    All Q2 Unigrams
    """
    def q1_unigram(datapoint):
        pass

    """
    All Q2 Bigrams 
    """
    def q1_bigram(datapoint):
        pass

    """
    All Unigrams
    """
    def unigrams(datapoint):
        pass

    """
    All Bigrams
    """
    def unigrams(datapoint):
        pass

    """
    BLEU score of q2 w/respect to q1, w/ bleu 1 through 4
    """
    def bleu(datapoint):
        pass

    """
    Difference in length
    """
    def length_difference(datapoint):
        pass

    """
    Overlap
    """
    def overlap_count(datapoint):
        pass

    def overlap_entities(datapoint):
        pass

    def overlap_nouns(datapoint):
        pass

    def overlap_adj(datapoint):
        pass

    def overlap_verbs(datapoint):
        pass

def main():
    pass


def get_features():
    directory = "dataset/processed"

