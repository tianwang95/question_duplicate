#!/usr/bin/env
from stanza.nlp.corenlp import CoreNLPClient
from stanza.text.dataset import Dataset
from stanza.nlp import CoreNLP_pb2
import csv
import os
import shutil
import pickle

"""
Python script for generating annotations for the Quora questions dataset
format is 'pair id, q1 id, q2 id, q1 text, q2 text, is duplicate'
"""
DIRECTORY = "dataset/processed"
QUESTIONS_PER_FOLDER = 800

class DataPoint(object):
    def __init__(self, sample_id, q1_id, q2_id, q1_text, q2_text, q1_proto, q2_proto, is_duplicate):
        self.sample_id = sample_id
        self.q1_id = q1_id
        self.q2_id = q2_id
        self.q1_text = q1_text
        self.q2_text = q2_text
        self.q1_proto = q1_proto
        self.q2_proto = q2_proto
        self.is_duplicate = is_duplicate

def save_protos(proto_list, file_name):
    with open(file_name, 'w+') as dat_file:
        pickle.dump(proto_list, dat_file)

def main():
    client = CoreNLPClient(server='http://localhost:9000') 

    # clear contents of directory
    if os.path.exists(DIRECTORY):
        shutil.rmtree(DIRECTORY)
    os.makedirs(DIRECTORY)

    with open('dataset/raw/quora_duplicate_questions.tsv', 'rb') as question_tsv:
        question_tsv = csv.reader(question_tsv, delimiter='\t')
        # move past first line
        next(question_tsv)
        count = 0
        start = 163201 #1;
        proto_list = []
        for row in question_tsv:
            if (count < 163200):
                count += 1
                continue;
            sample_id = int(row[0])
            q1_id = int(row[1])
            q2_id = int(row[2])
            q1_text = row[3]
            q2_text = row[4]
            is_duplicate = True if int(row[5]) == 1 else False
            annotated1 = client.annotate_proto(q1_text)
            annotated2 = client.annotate_proto(q2_text)
            data_point = DataPoint(sample_id, q1_id, q2_id, q1_text, q2_text, annotated1.SerializeToString(), annotated2.SerializeToString(), is_duplicate)
            proto_list.append(data_point)
            count += 1

            if count % QUESTIONS_PER_FOLDER == 0:
                print count
                save_protos(proto_list, '{}/{}-{}.dat'.format(DIRECTORY, start, count))
                start = count + 1
                proto_list = []

        if count % QUESTIONS_PER_FOLDER > 0:
            save_protos(proto_list, '{}/{}-{}.dat'.format(DIRECTORY, start, count)) 

def test():
    files = os.listdir(DIRECTORY)
    with open(os.path.join(DIRECTORY, files[0]), 'rb') as dat_file:
        protos = pickle.load(dat_file)
        question_proto = CoreNLP_pb2.Document()
        question_proto.ParseFromString(protos[0].q1_proto)

main()
