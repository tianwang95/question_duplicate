import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, GRU, LSTM, Dense, merge
from keras.layers.core import RepeatVector, Lambda
from keras import backend as K
import sys
sys.path.append('../')
from data import Data
from custom.custom_layers import InitStateGRU, Attention, GetLastIndex
from custom.utils import get_time_index
from custom.metrics import binary_precision, binary_recall, binary_f1

"""
dim: size of lstm hidden dimension
optimizer: optimization function to be used
"""
def get_model(
        data,
        dim,
        weights,
        optimizer='rmsprop',
        all_regularizer = None,
        W_regularizer = None,
        U_regularizer = None,
        dropout_W = 0.0,
        dropout_U = 0.0,
        num_hidden = 0):

    if all_regularizer:
        if not W_regularizer:
            W_regularizer = all_regularizer
        if not U_regularizer:
            U_regularizer = all_regularizer

    word_dim = weights.shape[1]

    #### MODEL ####
    embed = Embedding(input_dim = weights.shape[0],
            output_dim = word_dim,
            input_length = data.max_sentence_length,
            mask_zero = True,
            weights = [weights])

    # Embed the inputs
    q1_input = Input(shape=(data.max_sentence_length,), dtype='int32', name = 'q1_input')
    x_1 = embed(q1_input)

    q2_input = Input(shape=(data.max_sentence_length,), dtype='int32', name = 'q2_input')
    x_2 = embed(q2_input)

    # Run over 1st question 
    gru_1 = GRU(
            dim,
            consume_less='gpu',
            return_sequences = True,
            dropout_W = dropout_W,
            dropout_U = dropout_U,
            W_regularizer = W_regularizer,
            U_regularizer = U_regularizer)(x_1)

    ### Get Last Index
    gru_1_last = GetLastIndex()(gru_1)

    #Init state with last time step and run over 2nd question
    gru_2 = InitStateGRU(
            dim,
            consume_less='gpu',
            dropout_W = dropout_W,
            dropout_U = dropout_U,
            W_regularizer = W_regularizer,
            U_regularizer = U_regularizer)([x_2, gru_1_last])

    prev = gru_2

    for _ in range(num_hidden):
        prev = Dense(dim, activation='tanh')(prev)

    result = Dense(1, activation='sigmoid')(prev)

    model = Model(input=[q1_input, q2_input], output = result)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', binary_precision, binary_recall, binary_f1])

    print(model.summary())
    return model
