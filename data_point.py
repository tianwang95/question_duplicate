class DataPoint(object):
    def __init__(self, q1_annotation, q2_annotation, is_duplicate):
        self.q1_annotation = q1_annotation
        self.q2_annotation = q2_annotation
        self.is_duplicate = is_duplicate