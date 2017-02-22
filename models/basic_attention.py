import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, GRU, LSTM, Dense, merge
from keras.layers.core import RepeatVector
from keras import backend as K
import sys
sys.path.append('../')
from data import Data
from custom.init_state_gru import InitStateGru
from custom.utils import get_time_index

"""
dim: size of lstm hidden dimension
optimizer: optimization function to be used
"""
def get_model(
        data,
        dim,
        weights,
        optimizer='rmsprop',
        loss='binary_crossentropy',
        W_regularizer = None,
        U_regularizer = None,
        dropout_W = 0.0,
        dropout_U = 0.0):

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
    
    # Grab the last time step
    gru_1_last = get_time_index(x, K.shape(gru_1)[1] - 1)

    #Init state with last time step and run over 2nd question
    gru_2 = InitStateGRU(gru_1,
            dim,
            consume_less='gpu',
            return_sequences = True,
            dropout_W = dropout_W,
            dropout_U = dropout_U,
            W_regularizer = W_regularizer,
            U_regularizer = U_regularizer)(x_2)

    # Compute M for attention (see Rocktaschel '16)
    result = Dense(1, init='normal', activation='sigmoid')(gru_2)

    model = Model(input=[q1_input, q2_input], output = result)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    print(model.summary())
    return model
