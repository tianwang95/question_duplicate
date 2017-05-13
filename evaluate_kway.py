import tensorflow as tf
import numpy as np
import os
import argparse
from data import DataKWay
from data import indices_to_words
from tensorflow.core.framework import graph_pb2
import time
import math
import csv
import heapq

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', dest='model')
    parser.add_argument('--size', action='store', dest='size', default=5000, type=int)
    parser.add_argument('--test', action='store_true', dest='test')
    parser.add_argument('--embed-node', action='store', dest='embed_node', default='model/question_embedding:0')
    parser.add_argument('--input-node', action='store', dest='input_node', default='inputs/x_1:0')
    parser.add_argument('--choices-node', action='store', dest='choices_node', default='inputs/x_2:0')
    parser.add_argument('--pred-node', action='store', dest='pred_node', default='prediction/prediction_index:0')
    parser.add_argument('--dataset', action='store', dest='dataset', default='dataset/raw/dev_15way_cos.tsv')
    parser.add_argument('--limit', action='store', dest='limit', type=int)
    parser.add_argument('--gpu-id', action='store', dest='gpu_id', choices = [0, 1, 2, 3], type=int)
    parser.add_argument('--k-choices', action='store', dest='k_choices', default=15, type=int)
    parser.add_argument('--random-distractors', action='store', dest='random_distractors', default=10, type=int)
    parser.add_argument('--eval-recall', action='store_true', dest='eval_recall')

    options = parser.parse_args()

    ## Print used options
    for arg in vars(options):
        print("{}\t{}".format(arg, getattr(options, arg)))

    return options

def embed_sentence(sentence, glove_matrix):
    vec = np.zeros(glove_matrix.shape[1])
    for idx in sentence:
        if idx != 0:
            vec += glove_matrix[idx]
        else:
            break

    norm = np.linalg.norm(vec)
    if norm >= 1e-10:
        return vec / np.linalg.norm(vec)
    else:
        return np.zeros(glove_matrix.shape[1])


def find_highest_cosine(target, ignore_ids, sentences, k):
    sentence_scores = []
    cos_distances = np.sum(sentences * target, axis=1)
    results = [result[0] for result in heapq.nlargest(k + len(ignore_ids), enumerate(cos_distances), key=lambda x: x[1]) if result[0] not in ignore_ids]
    results = results[:k]
    return results

def all_way_eval(data, eval_file, options):
    ### load model
    question_embed = None
    if options.model != None:
        with tf.Session() as sess:
            question_matrix = np.vstack(data.id_to_question)
            saver = tf.train.import_meta_graph(options.model + '.meta')
            saver.restore(sess, options.model)
            print("Restored Model...")

            input_node = tf.get_default_graph().get_tensor_by_name(options.input_node)
            embed_node = tf.get_default_graph().get_tensor_by_name(options.embed_node)
            start_time = time.time()
            embed_list = []
            for i in range(math.ceil(question_matrix.shape[0] / float(10000))):
                embed_list.append(sess.run(embed_node, feed_dict={input_node: question_matrix[i*10000:(i+1)*10000, :]}))
                print(time.time() - start_time)
            print("Finish Embedding Questions. Time: {:.2f}".format(time.time() - start_time))
            question_embed = np.vstack(embed_list)
            print(question_embed.shape)
    else:
        print("Using Avg GloVe")
        start_time = time.time()
        embed_list = [embed_sentence(question_point, data.embedding_matrix) for question_point in data.id_to_question]
        print("Finish Embedding Questions. Time: {:.2f}".format(time.time() - start_time))
        question_embed = np.vstack(embed_list)

    ### read in pairs
    pairs = []
    with open(eval_file, 'r') as f:
        tsv = csv.reader(f, delimiter = '\t')
        for row in tsv:
            pairs.append((int(row[0]), int(row[1])))

    ### run eval
    k = options.k_choices
    total = 0
    indices = []
    counts = [0.0 for _ in range(int(k / 5) + 1)]
    start_time = time.time()
    for q1_id, q2_id in pairs:
        choices = find_highest_cosine(question_embed[q1_id], [q1_id], question_embed, k)
        total += 1
        try:
            idx = choices.index(q2_id)
        except:
            idx = -1

        indices.append(idx)

        if idx != -1:
            for i in range(1, int(k / 5) + 1):
                if idx < i * 5:
                    counts[i] += 1.0
            if idx == 0:
                counts[0] += 1.0

        print('{}\tElapsed:\t{}'.format(total, time.time() - start_time))
        print('Question:\t{}'.format(indices_to_words(data.vocabulary, data.id_to_question[q1_id])))
        print('Gold:\t{}'.format(indices_to_words(data.vocabulary, data.id_to_question[q2_id])))
        print('=' * 80)
        for choice_id in choices:
            print(indices_to_words(data.vocabulary, data.id_to_question[choice_id]))
        print('\t'.join(['Corr: {:.2f}'.format(counts[0] / total)] + ['Top {} Acc: {:.2f}'.format((i) * 5, counts[i] / total) for i in range(1, int(k / 5) + 1)]))

def predict_datapoint(data_point, glove_matrix):
    embed_premise = embed_sentence(data_point[0], glove_matrix)
    embed_choices = np.vstack([embed_sentence(data_point[1][i], glove_matrix) for i in range(data_point[1].shape[0])])
    cos_distances = np.sum(embed_choices * embed_premise, axis = 1)
    idx = np.argmax(cos_distances)
    return idx

def eval_dataset_avg_vec(data, options):
    correct = 0
    total = 0
    generator = data.dev_generator(loop = False)
    for batch in generator:
        for data_point in zip(batch[0], batch[1], batch[2]):
            total += 1
            y_pred = predict_datapoint(data_point, data.embedding_matrix)
            if y_pred == data_point[2]:
                correct += 1
    print('Accuracy:\t{}'.format(correct / total))

def eval_dataset_neural(data, options):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(options.model + '.meta')
        saver.restore(sess, options.model)
        print("Restored Model...")

        input_node = tf.get_default_graph().get_tensor_by_name(options.input_node)
        choices_node = tf.get_default_graph().get_tensor_by_name(options.choices_node)
        pred_node = tf.get_default_graph().get_tensor_by_name(options.pred_node)

        def feed_dict(batch):
            return{input_node: batch[0], choices_node: batch[1]}

        correct = 0
        total = 0
        generator = data.dev_generator(loop=False)
        for batch in generator:
            predictions = sess.run(pred_node, feed_dict = feed_dict(batch))
            print(predictions)
            correct += sum(np.equal(batch[2], predictions).tolist())
            total += batch[0].shape[0]

        print('Accuracy:\t{}'.format(correct / total))

def main():
    options = parse_arguments()

    ### Set GPU
    if options.gpu_id != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''

    ### load data
    data_dir = '/mnt/disks/main/question_duplicate/dataset/raw'
    question_file = os.path.join(data_dir, 'questions_kway.tsv')
    train_file = os.path.join(data_dir, 'train_15way_cos.tsv')
    dev_file = os.path.join(data_dir, 'dev_15way_cos.tsv')
    test_file = os.path.join(data_dir, 'test_15way_cos.tsv')
    
    if options.eval_recall:
        data = DataKWay(question_file, train_file, dev_file, test_file, 5, batch_size=5, embed_dim=100)
        all_way_eval(data, test_file if options.test else dev_file, options)
    else:
        data = DataKWay(question_file, train_file, dev_file, test_file, options.k_choices, random_distractors = options.random_distractors, batch_size=64, embed_dim=100)
        if options.model != None:
            eval_dataset_neural(data, options)
        else:
            eval_dataset_avg_vec(data, options)

if __name__ == '__main__':
    main()
