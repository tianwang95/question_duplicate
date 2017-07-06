import tensorflow as tf
from data import Data, DataKWay
from models.concat_lstm import ConcatLSTM
from models.lstm_cosine import LSTMCosine
from models.bilstm_cosine import BiLstmCosine
from models.stacked_lstm_cosine import StackedLstmCosine
from base_model import BaseModel
import os
import sys
import argparse
import tempfile

def parse_arguments(avail_models):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', dest='model', choices=avail_models, required=True)
    parser.add_argument('--name', action='store', dest='name')
    parser.add_argument('--rnn-dim', action='store', default=128, dest='rnn_dim', type=int)
    parser.add_argument('--batch-size', action='store', default=64, dest='batch_size', type=int)
    parser.add_argument('--word-dim', action='store', default=100, dest='word_dim', type=int, choices = [50, 100, 200, 300])
    parser.add_argument('--num-dense', action='store', dest='num_dense', default=0, type=int)
    parser.add_argument('--lstm-stack-size', action='store', default=1, dest='stack_size', type=int)
    # parser.add_argument('--epochs', action='store', default=10, dest='epochs', type=int)
    parser.add_argument('--limit', action='store', dest='limit', type=int)
    parser.add_argument('--no-save', action='store_false', dest='save_model')
    parser.add_argument('--test', action='store_true', dest='test')
    parser.add_argument('--gpu-id', action='store', dest='gpu_id', choices = [0, 1, 2, 3], type=int)
    parser.add_argument('--k-choices', action='store', dest = 'k_choices', default=15, type=int)
    parser.add_argument('--random-distractors', action='store', default=10, dest='random_distractors', type=int)
    parser.add_argument('--train-embed', action='store_true', dest='train_embed')
    parser.add_argument('--randomize', action='store_true', dest='randomize')
    parser.add_argument('--use-pos', action='store_true', dest='use_pos')

    ## TODO: write arguments for evaluation
    ## TODO: write arguments for loading saved model
    options = parser.parse_args()

    ## Print used options
    for arg in vars(options):
        print("{}\t{}".format(arg, getattr(options, arg)))

    return options

def main():
    avail_models = ['concat_lstm', 'lstm_cosine', 'bilstm_cosine', 'stacked_lstm_cosine']
    k_way_models = ['lstm_cosine', 'bilstm_cosine', 'stacked_lstm_cosine']
    options = parse_arguments(avail_models)
    ### Set GPU
    if options.gpu_id != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    ### Prep Data
    if options.model in k_way_models:
        data = DataKWay("dataset/raw/questions_kway_pos.tsv",
                    "dataset/raw/train_15way_cos.tsv",
                    "dataset/raw/dev_15way_cos.tsv",
                    "dataset/raw/test_15way_cos.tsv",
                    options.k_choices,
                    embed_dim = options.word_dim,
                    batch_size = options.batch_size,
                    limit = options.limit,
                    randomize = options.randomize,
                    random_distractors = options.random_distractors)

    else:
        data = Data("dataset/raw/train.tsv",
                    "dataset/raw/dev.tsv",
                    "dataset/raw/test.tsv",
                    embed_dim=options.word_dim,
                    batch_size=options.batch_size,
                    limit=options.limit,
                    delim_questions=(options.model in ['basic_attention', 'read_forward']),
                    randomize = options.randomize)

    ### Set up the directory 
    name = options.name if options.name else options.model
    log_dir = None
    save_dir = None
    temp_dir = None
    if options.save_model:
        save_dir = os.path.abspath('saved_models/{}'.format(name))
        count = 1;
        while os.path.exists(save_dir):
            save_dir = os.path.abspath('saved_models/{}-{}'.format(name, count))
            count += 1
        os.makedirs(save_dir)
        log_dir = os.path.join(save_dir, 'log')
        os.makedirs(log_dir)
        print('Log Directory:\t{}'.format(log_dir))
    else:
        temp_dir = tempfile.TemporaryDirectory()
        log_dir = temp_dir.name
        print('Temp Directory:\t{}'.format(log_dir))

    ### Get the model
    model = None
    if options.model == 'concat_lstm':
        model = ConcatLSTM(data.max_sentence_length, options.rnn_dim, data.embedding_matrix.shape[1], num_hidden=options.num_dense)
    if options.model == 'lstm_cosine':
        model = LSTMCosine(options.rnn_dim, options.k_choices, num_dense=options.num_dense)
    if options.model == 'bilstm_cosine':
        model = BiLstmCosine(options.rnn_dim, options.k_choices, num_dense=options.num_dense)
    if options.model == 'stacked_lstm_cosine':
        model = StackedLstmCosine(options.rnn_dim, options.k_choices, num_dense=options.num_dense, stack_size = options.stack_size)

    # Create full model and train
    full_model = BaseModel(data, data.embedding_matrix, model, log_dir = log_dir, save_dir = save_dir, train_embed = options.train_embed, k_choices = options.k_choices if options.model in k_way_models else None, use_pos = options.use_pos)
    full_model.train()

if __name__ == '__main__':
    main()
