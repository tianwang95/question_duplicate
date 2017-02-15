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
		feature_vec = Counter();
		for name, func in features.iteritems():
			func(feature_vec)
		return feature_vec
