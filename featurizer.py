from collections import Counter
class Featurizer:
    def __init__(self):
        features = {}

    def add_feature(name, function):
        if name in features:
            raise ValueError("Already contains feature with name: " + name)
        else:
            features[name] = function

    def featurize(data_point):
        feature_vec = Counter();
        for _, func in features.iteritems():
            feature_vec += func(data_point)
        return feature_vec
