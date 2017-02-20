import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, GRU, LSTM, Dense, merge
import sys
sys.path.append('../')
from data import Data

"""
dim: size of lstm hidden dimension
optimizer: optimization function to be used
"""
def get_model(
        data,
        dim,
        weights,
        optimizer='rmsprop',
        loss='binary_crossentropy'):

    word_dim = weights.shape[1]

    #### MODEL ####
    embed = Embedding(input_dim = weights.shape[0],
            output_dim = word_dim,
            input_length = data.max_sentence_length,
            mask_zero = True,
            weights = [weights])

    q1_input = Input(shape=(data.max_sentence_length,), dtype='int32', name = 'q1_input')
    x_1 = embed(q1_input)

    q2_input = Input(shape=(data.max_sentence_length,), dtype='int32', name = 'q2_input')
    x_2 = embed(q2_input)
    
    question_gru = GRU(dim, consume_less='gpu')
    gru_1 = question_gru(x_1)
    gru_2 = question_gru(x_2)
    merged = merge([gru_1, gru_2], mode='concat')

    result = Dense(1, init='normal', activation='sigmoid')(merged)

    model = Model(input=[q1_input, q2_input], output = result)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    print(model.summary())
    return model
