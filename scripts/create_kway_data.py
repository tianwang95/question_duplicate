import csv
import time
import os
import itertools
import heapq
import random
import numpy as np

def quora_dataset(filename):
    with open(filename, 'r') as f:
        question_tsv = csv.reader(f, delimiter = '\t')
        for row in question_tsv:
            yield int(row[0]), row[1], row[2], int(row[3])

def glove2dict(src_filename):
    with open(src_filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)
        return {line[0]: np.array(list(map(float, line[1: ])),
                                  dtype='float32') for line in reader}

def embed_sentence(sentence, glove_dict):
    vec = np.zeros(50)
    tokens = [token.lower() for token in sentence.split()]
    for word in tokens:
        if word in glove_dict:
            vec += glove_dict[word]

    norm = np.linalg.norm(vec)
    if norm >= 1e-10:
        return vec / np.linalg.norm(vec)
    else:
        print("All unks...\t" + sentence)
        return np.zeros(50)

"""
target is vector
sentences is n x d matrix where n is number of sentences and d is embedding dim
"""
def find_highest_cosine(target, ignore_ids, sentences, k):
    start_time = time.time()
    sentence_scores = []
    cos_distances = np.sum(sentences * target, axis=1)
    results = [result[0] for result in heapq.nlargest(k + len(ignore_ids), enumerate(cos_distances), key=lambda x: x[1]) if result[0] not in ignore_ids]
    results = results[:k]
    print("elapsed: {}".format(time.time() - start_time))
    return results

def main():
    random.seed(42)
    k = 15 # generate 1 gold and k-1 distract sentences
    dataset_dir = '/mnt/disks/main/question_duplicate/dataset/raw'
    train_file = os.path.join(dataset_dir, 'train.tsv')
    dev_file = os.path.join(dataset_dir, 'dev.tsv')
    test_file = os.path.join(dataset_dir, 'test.tsv')
    train_output = os.path.join(dataset_dir, 'train_{}way_cos.tsv'.format(k))
    dev_output = os.path.join(dataset_dir, 'dev_{}way_cos.tsv'.format(k))
    test_output = os.path.join(dataset_dir, 'test_{}way_cos.tsv'.format(k))
    questions_output = os.path.join(dataset_dir, 'questions_kway.tsv')
    dev_count = 5000
    test_count = 5000

    print("Reading Glove")
    glove_dict = glove2dict('/mnt/disks/main/question_duplicate/dataset/glove.6B.50d.txt')

    question_to_id = {}
    id_to_question = []

    def add_to_dict(text):
        if text in question_to_id:
            return question_to_id[text]
        else:
            new_id = len(id_to_question)
            id_to_question.append(text)
            question_to_id[text] = new_id
            return new_id

    pairs = []

    for is_duplicate, q1_text, q2_text, _ in itertools.chain.from_iterable([
        quora_dataset(train_file),
        quora_dataset(dev_file),
        quora_dataset(test_file)]):

        q1_id = add_to_dict(q1_text)
        q2_id = add_to_dict(q2_text)

        if is_duplicate:
            pairs.append((q1_id, q2_id))

    print("Retrieved {} datapoints".format(len(pairs)))
    
    # write all the questions
    with open(questions_output, 'w+') as f:
        for question_text, q_id in question_to_id.items():
            f.write("{}\t{}\n".format(q_id, question_text))

    # split the dataset
    dev_split = pairs[:dev_count]
    test_split = pairs[dev_count:dev_count + test_count]
    train_split = pairs[dev_count+test_count:]

    print("Train size: {}".format(len(train_split)))
    print("Dev size: {}".format(len(dev_split)))
    print("Test size: {}".format(len(test_split)))

    # reverse question dict
    print("Embedding sentences")
    question_embed = np.array([embed_sentence(text, glove_dict) for text in id_to_question])
    print(question_embed)

    def write_dataset(output_file, data):
        with open(output_file, 'w+') as f:
            for q1_id, q2_id in data:
                choices = find_highest_cosine(question_embed[q1_id], [q1_id, q2_id], question_embed, k - 1)
                row = [str(x) for x in [q1_id, q2_id] + choices]
                f.write("\t".join(row) + "\n")

    write_dataset(train_output, train_split)
    write_dataset(dev_output, dev_split)
    write_dataset(test_output, test_split)

if __name__ == '__main__':
    main()