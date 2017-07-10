from stanza.nlp import CoreNLP_pb2
from nltk.translate.bleu_score import sentence_bleu

"""
Intersection of q1's set of words and q2's set of words
"""
def unigram_intersect(datapoint):
    name = "unigram_intersect"
    fvec = {}
    q1_sentence = datapoint.q1_annotation.sentence[0].token
    q2_sentence = datapoint.q2_annotation.sentence[0].token
    for token1 in q1_sentence:
        for token2 in q2_sentence:
            if token1.lemma == token2.lemma:
                fvec[name + ":" + token1.lemma] = 1.0
    return fvec

"""
Intersection of q1 and q2 bigrams
"""
def bigram_intersect(datapoint):
    name = "bigram_intersect"
    fvec = {}
    q1_sentence = datapoint.q1_annotation.sentence[0].token
    q2_sentence = datapoint.q2_annotation.sentence[0].token
    for i in xrange(1, len(q1_sentence)):
        for j in xrange(1, len(q2_sentence)):
            if (q1_sentence[i-1].lemma == q2_sentence[j-1].lemma
            and q1_sentence[i].lemma == q2_sentence[j].lemma):
                fvec["{}:{},{}".format(name, q1_sentence[i-1].lemma, q1_sentence[i].lemma)] = 1.0
    return fvec

"""
Cross unigram: Feature over similar 
"""
def cross_unigram(datapoint):
    name = "cross_unigram"
    fvec = {}
    q1_sentence = datapoint.q1_annotation.sentence[0].token
    q2_sentence = datapoint.q2_annotation.sentence[0].token
    for token1 in q1_sentence:
        for token2 in q2_sentence:
            if token1.pos == token2.pos:
                a, b = _sort_pair(token1.lemma, token2.lemma)
                fvec["{}:{}:{}".format(name, a, b)] = 1.0
    return fvec

"""
Cross bigram
"""
def cross_bigram(datapoint):
    name = "cross_bigram"
    fvec = {}
    q1_sentence = datapoint.q1_annotation.sentence[0].token
    q2_sentence = datapoint.q2_annotation.sentence[0].token
    for i in xrange(1, len(q1_sentence)):
        for j in xrange(1, len(q2_sentence)):
            if q1_sentence[i].pos == q2_sentence[j].pos:
                a, b = _sort_pair(q1_sentence[i-1].lemma + "," + q1_sentence[i].lemma,
                                  q2_sentence[j-1].lemma + "," + q2_sentence[j].lemma)
                fvec["{}:{}:{}".format(name, a, b)] = 1.0
    return fvec

"""
All Unigrams
"""
def unigrams(datapoint):
    name = "unigrams"
    fvec = {}
    q1_sentence = datapoint.q1_annotation.sentence[0].token
    q2_sentence = datapoint.q2_annotation.sentence[0].token
    for token in q1_sentence:
        fvec["{}:{}".format(name, token.lemma)] = 1.0
    for token in q2_sentence:
        fvec["{}:{}".format(name, token.lemma)] = 1.0
    return fvec

"""
All Bigrams
"""
def bigrams(datapoint):
    name = "bigrams"
    fvec = {}
    q1_sentence = datapoint.q1_annotation.sentence[0].token
    q2_sentence = datapoint.q2_annotation.sentence[0].token
    for i in xrange(1, len(q1_sentence)):
        fvec["{}:{},{}".format(name, q1_sentence[i-1].lemma, q1_sentence[i].lemma)] = 1.0
    for i in xrange(1, len(q2_sentence)):
        fvec["{}:{},{}".format(name, q2_sentence[i-1].lemma, q2_sentence[i].lemma)] = 1.0
    return fvec

"""
BLEU score of q2 w/respect to q1, w/ bleu 1 through 4
"""
def bleu(datapoint):
    name = "bleu"
    fvec = {}
    for i in xrange(1, 5):
        q1 = [token.lemma for token in datapoint.q1_annotation.sentence[0].token]
        q2 = [token.lemma for token in datapoint.q2_annotation.sentence[0].token]
        score = sentence_bleu([q1], q2, tuple([1.0 / i if x < i else 0.0 for x in xrange(4)]))
        fvec["{}:{}".format(name, i)] = score
    return fvec

"""
Difference in length
"""
def length_difference(datapoint):
    name = "length_difference"
    fvec = {}
    q1_len = len(datapoint.q1_annotation.sentence[0].token)
    q2_len = len(datapoint.q2_annotation.sentence[0].token)
    fvec[name] = abs(q1_len - q2_len)
    return fvec

"""
Overlap
"""
def overlap_count(datapoint):
    name = "overlap_count"
    fvec = {}
    count = 0
    q1_sentence = set([token.lemma for token in datapoint.q1_annotation.sentence[0].token])
    q2_sentence = set([token.lemma for token in datapoint.q2_annotation.sentence[0].token])
    count = len(q1_sentence & q2_sentence)
    total = len(q1_sentence | q2_sentence)
    fvec[name] = count
    fvec[name + "_fraction"] = float(count) / total if total != 0 else 0.0
    return fvec

def overlap_entities(datapoint):
    name = "overlap_entities"
    fvec = {}
    q1_sentence = set([token.lemma for token in datapoint.q1_annotation.sentence[0].token if token.ner != "O"])
    q2_sentence = set([token.lemma for token in datapoint.q2_annotation.sentence[0].token if token.ner != "O"])
    count = len(q1_sentence & q2_sentence)
    total = len(q1_sentence | q2_sentence)
    fvec[name] = count
    fvec[name + "_fraction"] = float(count) / total if total != 0 else 0.0
    return fvec

def overlap_nouns(datapoint):
    name = "overlap_nouns"
    fvec = {}
    q1_sentence = set([token.lemma for token in datapoint.q1_annotation.sentence[0].token if token.pos[:2] == "NN"])
    q2_sentence = set([token.lemma for token in datapoint.q2_annotation.sentence[0].token if token.pos[:2] == "NN"])
    count = len(q1_sentence & q2_sentence)
    total = len(q1_sentence | q2_sentence)
    fvec[name] = count
    fvec[name + "_fraction"] = float(count) / total if total != 0 else 0.0
    return fvec

def overlap_adj(datapoint):
    name = "overlap_adj"
    fvec = {}
    q1_sentence = set([token.lemma for token in datapoint.q1_annotation.sentence[0].token if token.pos[:2] == "JJ"])
    q2_sentence = set([token.lemma for token in datapoint.q2_annotation.sentence[0].token if token.pos[:2] == "JJ"])
    count = len(q1_sentence & q2_sentence)
    total = len(q1_sentence | q2_sentence)
    fvec[name] = count
    fvec[name + "_fraction"] = float(count) / total if total != 0 else 0.0
    return fvec

def overlap_verbs(datapoint):
    name = "overlap_verbs"
    fvec = {}
    q1_sentence = set([token.lemma for token in datapoint.q1_annotation.sentence[0].token if token.pos[:2] == "VB"])
    q2_sentence = set([token.lemma for token in datapoint.q2_annotation.sentence[0].token if token.pos[:2] == "VB"])
    count = len(q1_sentence & q2_sentence)
    total = len(q1_sentence | q2_sentence)
    fvec[name] = count
    fvec[name + "_fraction"] = float(count) / total if total != 0 else 0.0
    return fvec

"""
Helper
"""
def _sort_pair(a, b):
    if a < b:
        return (a, b)
    else:
        return (b, a)
