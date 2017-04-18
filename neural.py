import tensorflow as tf
from data import Data
from models.concat_lstm import ConcatLSTM
from models.lstm_cosine import LSTMCosine
from base_model import BaseModel
import os
import sys
import argparse
import tempfile

def parse_arguments(avail_models):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', dest='model', choices=avail_models)
    parser.add_argument('--name', action='store', dest='name')
    parser.add_argument('--rnn-dim', action='store', default=128, dest='rnn_dim', type=int)
    parser.add_argument('--batch-size', action='store', default=64, dest='batch_size', type=int)
    parser.add_argument('--word-dim', action='store', default=100, dest='word_dim', type=int, choices = [50, 100, 200, 300])
    # parser.add_argument('--epochs', action='store', default=10, dest='epochs', type=int)
    parser.add_argument('--limit', action='store', dest='limit', type=int)
    parser.add_argument('--no-save', action='store_false', dest='save_model')
    parser.add_argument('--test', action='store_true', dest='test')
    parser.add_argument('--gpu-id', action='store', dest='gpu_id', choices = [0, 1, 2, 3], type=int)
    parser.add_argument('--num-hidden', action='store', dest='num_hidden', default=0, type=int)
    parser.add_argument('--train-embed', action='store_true', dest='train_embed')

    ## TODO: write arguments for evaluation
    ## TODO: write arguments for loading saved model
    options = parser.parse_args()

    ## Print used options
    for arg in vars(options):
        print("{}\t{}".format(arg, getattr(options, arg)))

    return options

def main():
    avail_models = ['concat_lstm', 'lstm_cosine']
    options = parse_arguments(avail_models)
    ### Set GPU
    if options.gpu_id != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    ### Prep Data
    data = Data("dataset/raw/train.tsv",
                "dataset/raw/dev.tsv",
                "dataset/raw/test.tsv",
                embed_dim=options.word_dim,
                batch_size=options.batch_size,
                limit=options.limit,
                delim_questions=(options.model in ['basic_attention', 'read_forward']))

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
        model = ConcatLSTM(data.max_sentence_length, options.rnn_dim, data.embedding_matrix.shape[1], num_hidden=options.num_hidden)
    if options.model == 'lstm_cosine':
        model = LSTMCosine(data.max_sentence_length, options.rnn_dim, data.embedding_matrix.shape[1])

    # Create full model and train
    full_model = BaseModel(data, data.embedding_matrix, model, log_dir = log_dir, save_dir = save_dir, train_embed = options.train_embed)
    full_model.train()

def _deprecated_train(model, data, epochs, name, run_test_set = False, save_model = False):
    callbacks = None
    if save_model:
        ### Setup model checkpoint callback
        
        callbacks = [ModelCheckpoint(os.path.join(directory, 'model.{epoch:02d}-{val_acc:.2f}.hdf5'))] if save_model else None
        ### Save model
        model_json = model.to_json()
        with open(os.path.join(directory, 'model.json'), 'w') as f:
            f.write(model_json)

    ## Train
    model.fit_generator(data.train_generator(),
                        samples_per_epoch = data.train_count,
                        nb_epoch = epochs,
                        callbacks = callbacks,
                        verbose = 2,
                        validation_data = data.dev_generator(),
                        nb_val_samples = data.dev_count)

def _deprecated_main():
        ### Prep model
    model = None
    if options.model == 'concat_gru':
        model = concat_gru.get_model(
                data,
                dim = options.rnn_dim,
                weights = data.embedding_matrix,
                dropout_W = 0.2,
                dropout_U = 0.2,
                num_hidden=options.num_hidden)
    elif options.model == 'read_forward':
        model = read_forward.get_model(
                data,
                dim = options.rnn_dim,
                weights = data.embedding_matrix,
                dropout_W = 0.2,
                dropout_U = 0.2)
    elif options.model == 'basic_attention':
        model = basic_attention.get_model(
                data,
                dim = options.rnn_dim,
                weights = data.embedding_matrix,
                dropout_W = 0.2,
                dropout_U = 0.2)
        ### load weights from old model
    ### Set session log for tensorflow
    sess = K.get_session()
    file_writer = tf.summary.FileWriter('log/tf', sess.graph)
    ### Run experiment
    name = options.name if options.name else options.model
    train(model,
          data,
          options.epochs,
          name,
          options.test,
          options.save_model)

if __name__ == '__main__':
    main()
