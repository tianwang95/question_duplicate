import hashlib
class Featurizer(object):
    def __init__(self):
        self.features = {}

    def add_feature(self, function):
        if function.__name__ in self.features:
            raise ValueError("Already contains feature with name: " + function.__name__)
        else:
            self.features[function.__name__] = function

    def featurize(self, data_point):
        feature_vec = {};
        for _, func in self.features.items():
            feature_vec.update(func(data_point))
        return feature_vec