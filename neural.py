from data import Data
from models import concat_gru, basic_attention
from keras.callbacks import ModelCheckpoint
import keras.backend.tensorflow_backend as K
import os
import sys

def parse_arguments(avail_models):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', dest=='model', choices=avail_models)
    parser.add_argument('--name', action='store', dest='name')
    parser.add_argument('--rnn-dim', action='store', default=128, dest='rnn_dim', type=int)
    parser.add_argument('--batch-size', action='store', default=64, dest='batch_size', type=int)
    parser.add_argument('--word-dim', action='store', default=200, dest='word_dim', type=int)
    parser.add_argument('--limit', action='store', dest='limit', type=int)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int)
    parser.add_argument('--no-save', action='store_false', dest='save_model')
    parser.add_argument('--test', action='store_true', dest='test')
    parser.add_argument('--output-console', action='store_true', dest='output_console')
    parser.add_argument('--gpu-id', action='store_true', dest='output_console')

    ## TODO: write arguments for evaluation
    ## TODO: write arguments for loading saved model
    options = parser.parse_args()

    ## Print used options
    for arg in vars(args):
        print("{}\t{}".format(arg, getattr(args, arg)))

    return options

def train(model, data, epochs, name, run_test_set = False, save_model = False, output_console = False):
    directory = 'saved_models/{}-{}'.format(name, 0)
    count = 1;
    while os.path.exists(directory):
        directory = 'saved_models/{}-{}'.format(name, count)
    os.makedirs(directory)
    callbacks = [ModelCheckpoint(os.path.join(directory, 'model.{epoch:02d}-{val_acc:.2f}.hdf5'))]
    ## Redirect to std out
    og_stdout = sys.stdout
    if not output_console:
        sys.stdout = open(os.path.join(directory, 'log.txt'), 'w')
    ## Train
    model.fit_generator(data.train_generator(),
                        samples_per_epoch = data.train_count,
                        nb_epoch = epochs,
                        callbacks = callbacks if save_model else None,
                        verbose = 2,
                        validation_data = data.dev_generator(),
                        nb_val_samples = data.dev_count)
    ## Reset stdout
    if not output_console:
        sys.stdout = og_stdout

def concat_gru(data, name, epochs, dim, embed_dim, batch_size):
    model = concat_gru.get_model(
            data,
            dim = dim,
            weights = data.embedding_matrix,
            dropout_W = 0.2,
            dropout_U = 0.2)
    train(model, data, epochs, name, save_model = True)

def basic_attention(name, epochs, dim, embed_dim, batch_size):
    data = Data("dataset/raw/quora_duplicate_questions.tsv",
                embed_dim=embed_dim,
                batch_size=batch_size)
    model = concat_gru.get_model(
            data,
            dim = dim,
            weights = data.embedding_matrix,
            dropout_W = 0.2,
            dropout_U = 0.2)
    train(model, data, epochs, name, save_model = False)
    
if __name__ == '__main__':
    avail_models = ['concat_gru', 'basic_attention']
    options = parse_arguments(avail_models)
    ### Prep Data
    data = Data("dataset/raw/quora_duplicate_questions.tsv",
                embed_dim=options.embed_dim,
                batch_size=options.batch_size,
                limit=options.limit)
    ### Prep model
    # concat_gru
    kwargs = {}
