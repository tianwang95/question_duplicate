#!usr/bin/env
from featurizer import Featurizer
from data_generator import AnnotatedData
from stanza.nlp import CoreNLP_pb2
from sklearn.feature_extraction import DictVectorizer
import features
import gzip
import os

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

    filename = "{}.data.gz".format(featurizer.get_hash())
    if os.path.exists(os.path.join(DIRECTORY, filename)):
        return

    # now featurize if we don't have the features already 
    ad = AnnotatedData("dataset/processed/quora_annotated.data.gz", "dataset/raw/quora_duplicate_questions.tsv") 
    featurized_X = []
    y = []
    count = 0
    for data_point in ad:
        count += 1
        if (count >= 100):
            break;
        featurized_X.append(featurizer.featurize(data_point))
        y = 1.0 if data_point else 0.0

    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(featurized_X)

compute_features()

