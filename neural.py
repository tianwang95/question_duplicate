from data import Data
from models import concat_gru
from keras.callbacks import ModelCheckpoint
import keras.backend.tensorflow_backend as K
import os

def experiment(model, data, epochs, name, run_test_set = False, save_model = False):
    directory = 'saved_models/{}-{}'.format(name, 0)
    count = 1;
    while os.path.exists(directory):
        directory = 'saved_models/{}-{}'.format(name, count)
    os.makedirs(directory)
    callbacks = [ModelCheckpoint(os.path.join(directory, 'model.{epoch:02d}-{val_acc:.2f}.hdf5'))]
    model.fit_generator(data.train_generator(),
                        samples_per_epoch = data.train_count,
                        nb_epoch = epochs,
                        callbacks = callbacks if save_model else None,
                        verbose = 2,
                        validation_data = data.dev_generator(),
                        nb_val_samples = data.dev_count)

def run_concat_gru():
    data = Data("dataset/raw/quora_duplicate_questions.tsv", embed_dim=200, batch_size=64)
    model = concat_gru.get_model(data, dim = 128, weights = data.embedding_matrix)
    experiment(model, data, 10, "concat_gru", save_model = True)
    
if __name__ == '__main__':
    run_concat_gru()

