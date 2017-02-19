#!usr/bin/env
from featurizer import Featurizer
from data_generator import AnnotatedData
from stanza.nlp import CoreNLP_pb2
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
from sklearn.linear_model import SGDClassifier
import features
import gzip
import os
import numpy as np

def compute_features():
    DIRECTORY = "dataset/featurized"
    # Add all the features we want
    featurizer = Featurizer()
    featurizer.add_feature(features.unigram_intersect)
    featurizer.add_feature(features.bigram_intersect)
    featurizer.add_feature(features.cross_unigram)
    featurizer.add_feature(features.cross_bigram)
    featurizer.add_feature(features.unigrams)
    featurizer.add_feature(features.bigrams)
    featurizer.add_feature(features.bleu)
    featurizer.add_feature(features.length_difference)
    featurizer.add_feature(features.overlap_count)
    featurizer.add_feature(features.overlap_entities)
    featurizer.add_feature(features.overlap_nouns)
    featurizer.add_feature(features.overlap_adj)
    featurizer.add_feature(features.overlap_verbs)

    filename = "featurized.data.gz"
    if os.path.exists(os.path.join(DIRECTORY, filename)):
        return

    # now featurize if we don't have the features already 
    ad = AnnotatedData("dataset/processed/quora_annotated.data.gz", "dataset/raw/quora_duplicate_questions.tsv", 350000) 
    featurized_X = []
    y = []
    count = 0
    for data_point in ad:
        if data_point.q1_annotation.sentence and data_point.q2_annotation.sentence:
            if count % 100 == 0:
                print(count)
            featurized_X.append(featurizer.featurize(data_point))
            y.append(1 if data_point.is_duplicate else 0)
            count += 1

    # count number of features
    # feat_counter = Counter()
    # for feat_vector in featurized_X:
    #     feat_counter.update(feat_vector.keys())

    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(featurized_X)
    return (X, np.asarray(y), vectorizer)

def run_logistic_regression():
    X, y, vectorizer = compute_features();
    test_size = int(y.shape[0] / 5) * -1
    train_X = X[:test_size]
    train_y = y[:test_size]
    test_X = X[test_size:]
    test_y = y[test_size:]
    clf = SGDClassifier(loss = 'log', penalty = 'l2')
    clf.fit(train_X, train_y)
    print(clf.score(train_X, train_y))
    print(clf.score(test_X, test_y))

def rank_weights(clf, features):
    weights = clf.coef_

run_logistic_regression()