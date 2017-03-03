from data import Data
from models import concat_gru, basic_attention, read_forward
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import os
import sys
import argparse

def parse_arguments(avail_models):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', dest='model', choices=avail_models)
    parser.add_argument('--name', action='store', dest='name')
    parser.add_argument('--rnn-dim', action='store', default=128, dest='rnn_dim', type=int)
    parser.add_argument('--batch-size', action='store', default=64, dest='batch_size', type=int)
    parser.add_argument('--word-dim', action='store', default=100, dest='word_dim', type=int, choices = ['50', '100', '200', '300'])
    parser.add_argument('--epochs', action='store', default=10, dest='epochs', type=int)
    parser.add_argument('--limit', action='store', dest='limit', type=int)
    parser.add_argument('--no-save', action='store_false', dest='save_model')
    parser.add_argument('--test', action='store_true', dest='test')
    parser.add_argument('--gpu-id', action='store', dest='gpu_id', choices = [0, 1, 2, 3], type=int)
    parser.add_argument('--num-hidden', action='store', dest='num_hidden', default=0, type=int)
    parser.add_argument('--dev-mode', action='store_true', dest='dev_mode')

    ## TODO: write arguments for evaluation
    ## TODO: write arguments for loading saved model
    options = parser.parse_args()

    ## Print used options
    for arg in vars(options):
        print("{}\t{}".format(arg, getattr(options, arg)))

    return options

def train(model, data, epochs, name, run_test_set = False, save_model = False):
    callbacks = None
    if save_model:
        ### Setup model checkpoint callback
        directory = 'saved_models/{}'.format(name)
        count = 1;
        while os.path.exists(directory):
            directory = 'saved_models/{}-{}'.format(name, count)
            count += 1
        os.makedirs(directory)
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

def main():
    avail_models = ['concat_gru', 'basic_attention', 'read_forward']
    options = parse_arguments(avail_models)
    ### Set GPU
    if options.gpu_id != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    ### Prep Data
    data = Data("dataset/raw/quora_duplicate_questions.tsv",
                embed_dim=options.word_dim,
                batch_size=options.batch_size,
                limit=options.limit,
                delim_questions=(options.model in ['basic_attention', 'read_forward']),
                dev_mode=options.dev_mode)
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
