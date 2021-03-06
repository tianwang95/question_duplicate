import random
from stanza.nlp import CoreNLP_pb2
from data_point import DataPoint
import gzip
import struct
import csv

class AnnotatedData(object):
    def __init__(self, annotated_file, raw_dataset_file, limit = None):
        self.annotated_file = gzip.open(annotated_file, 'rb')
        self.raw_dataset_file = open(raw_dataset_file, 'r')
        self.question_tsv = csv.reader(self.raw_dataset_file, delimiter='\t')
        self.limit = limit
        self.count = 0
        next(self.question_tsv)

    def __enter__(self):
        return self

    def __exit__(self):
        self.annotated_file.close()
        self.raw_dataset_file.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def get_next_proto(self):
        size_bytes = self.annotated_file.read(4)
        if size_bytes:
            size = struct.unpack('i', size_bytes)[0]
            return self.annotated_file.read(size)
        else:
            return ''

    def next(self):
        if self.limit and self.count >= self.limit:
            raise StopIteration
        q1_proto = self.get_next_proto()
        q2_proto = self.get_next_proto()
        if q1_proto and q2_proto:
            q1_annotation = CoreNLP_pb2.Document()
            q1_annotation.ParseFromString(q1_proto)
            q2_annotation = CoreNLP_pb2.Document()
            q2_annotation.ParseFromString(q2_proto)
            row = next(self.question_tsv)
            self.count += 1
            return DataPoint(q1_annotation, q2_annotation, bool(int(row[5])))
        else:
            raise StopIteration

def test():
    a = AnnotatedData("dataset/processed/quora_annotated.data.gz", "dataset/raw/quora_duplicate_questions.tsv")
    quora_tsv = "dataset/raw/quora_duplicate_questions.tsv"
    for x in a:
        print(x.q1_annotation.sentence[0].token)
        break

if __name__ == "__main__":
    test()
