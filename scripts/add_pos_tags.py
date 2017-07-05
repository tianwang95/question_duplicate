from stanza.nlp.corenlp import CoreNLPClient
from stanza.nlp.corenlp import AnnotatedDocument
from google.protobuf.internal.decoder import _DecodeVarint
from stanza.nlp.corenlp import CoreNLP_pb2
import csv


def question_dataset(filename):
    with open(filename, 'r') as f:
        question_tsv = csv.reader(f, delimiter = '\t')
        for row in question_tsv:
            yield row[0], row[1]

def annotate_text(client, text):
    properties = {
        'annotators': ','.join(client.default_annotators),
        'outputFormat': 'serialized',
        'tokenize.whitespace': 'true',
        'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
    }
    r = client._request(text, properties)
    buffer = r.content  # bytes
    size, pos = _DecodeVarint(buffer, 0)
    buffer = buffer[pos:(pos + size)]
    doc = CoreNLP_pb2.Document()
    doc.ParseFromString(buffer)
    return AnnotatedDocument.from_pb(doc)

def main():
    data_file = "../dataset/raw/questions_kway.tsv"
    output_file = "../dataset/raw/questions_kway_pos.tsv"
    client = CoreNLPClient(server='http://localhost:9000', default_annotators=['ssplit', 'pos'])
    count = 0
    with open(output_file, 'w+') as f:
        for q_id, text in question_dataset(data_file):
            count += 1
            if count % 1000 == 0:
                print(count)
            an = annotate_text(client, text)
            pos_tags = []
            for sentence in an.sentences:
                for token in sentence.tokens:
                    pos_tags.append(token.pos)
            pos_str = " ".join(pos_tags)
            """
            assert len(pos_tags) == len(text.split()), "{} != {}\t{}\t{}".format(len(pos_tags), len(text.split()), pos_str, text)
            """
            f.write("\t".join([q_id, text, pos_str]) + "\n")

if __name__ == '__main__':
    main()
