from stanza.nlp import CoreNLP_pb2

class DataPoint(object):
    def __init__(self, sample_id, q1_id, q2_id, q1_text, q2_text, q1_proto, q2_proto, is_duplicate):
        self.sample_id = sample_id
        self.q1_id = q1_id
        self.q2_id = q2_id
        self.q1_text = q1_text
        self.q2_text = q2_text
        self.q1_proto = CoreNLP_pb2.Document()
        self.q1_proto.ParseFromString(q1_proto)
        self.q2_proto = CoreNLP_pb2.Document()
        self.q2_proto.ParseFromString(q2_proto)
        self.is_duplicate = is_duplicate
