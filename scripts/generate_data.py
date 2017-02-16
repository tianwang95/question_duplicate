#!/usr/bin/env
from stanza.nlp.corenlp import CoreNLPClient
from stanza.nlp import CoreNLP_pb2
import csv
import os
import gzip

"""
Python script for generating annotations for the Quora questions dataset
format is 'pair id, q1 id, q2 id, q1 text, q2 text, is duplicate'
"""
DIRECTORY = "dataset/processed"
FILENAME = "quora_annotated.data.gz"

def main():
    # clear contents of directory
    if os.path.exists(DIRECTORY):
        raise Exception("{} already exists.".format(DIRECTORY))
    os.makedirs(DIRECTORY)

    client = CoreNLPClient(server='http://localhost:9000') 
    with gzip.open(os.path.join(DIRECTORY, FILENAME), 'wb') as f:
        with open('dataset/raw/quora_duplicate_questions.tsv', 'rb') as question_tsv:
            question_tsv = csv.reader(question_tsv, delimiter='\t')
            # move past first line
            next(question_tsv)
            for row in question_tsv:
                q1_text = row[3]
                q2_text = row[4]
                annotated1 = client.annotate_proto(q1_text)
                annotated2 = client.annotate_proto(q2_text)
                f.write(get_packed_msg(annotated1.SerializeToString()))
                f.write(get_packed_msg(annotated2.SerializeToString()))

def get_packed_msg(proto):
    size_str = struct.pack("i", len(proto))
    assert(len(size_str) == 4)
    return size_str + proto

main()