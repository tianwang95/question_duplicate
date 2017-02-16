from collections import Counter
class Featurizer(object):
    def __init__(self):
        features = {}

    def add_feature(name, function):
        if name in features:
            raise ValueError("Already contains feature with name: " + name)
        else:
            features[name] = function

    def featurize(data_point):
        feature_vec = {};
        for _, func in features.iteritems():
            feature_vec.update(func(data_point))
        return feature_vec
