import hashlib
class Featurizer(object):
    def __init__(self):
        features = {}

    def add_feature(self, function):
        if function.__name__ in features:
            raise ValueError("Already contains feature with name: " + function.__name__)
        else:
            feat_hash.update(function.__name__)
            features[function.__name__] = function

    def featurize(self, data_point):
        feature_vec = {};
        for _, func in features.iteritems():
            feature_vec.update(func(data_point))
        return feature_vec

    def get_hash(self):
        return feat_hash.hexdigest()

