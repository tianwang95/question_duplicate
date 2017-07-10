from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, GRU, LSTM, Dense, merge
import sys
sys.path.append('../')

"""
dim: size of lstm hidden dimension
optimizer: optimization function to be used
"""
def get_model(
        data,
        dim,
        weights,
        optimizer='rmsprop',
        W_regularizer=None,
        U_regularizer=None,
        dropout_W=0.0,
        dropout_U=0.0,
        dropout_embedding=0.0,
        num_hidden=0):

    word_dim = weights.shape[1]

    # MODEL #
    embed = Embedding(input_dim = weights.shape[0],
            output_dim = word_dim,
            input_length = data.max_sentence_length,
            mask_zero = True,
            weights = [weights])

    q1_input = Input(shape=(data.max_sentence_length,), dtype='int32', name = 'q1_input')
    x_1 = embed(q1_input)

    q2_input = Input(shape=(data.max_sentence_length,), dtype='int32', name = 'q2_input')
    x_2 = embed(q2_input)
    
    question_gru = GRU(dim, consume_less='gpu', dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer)
    gru_1 = question_gru(x_1)
    gru_2 = question_gru(x_2)

    prev = merge([gru_1, gru_2], mode='concat')
    for _ in range(num_hidden):
        prev = Dense(dim * 2, activation='tanh')(prev)

    result = Dense(1, activation='sigmoid')(prev)

    model = Model(input=[q1_input, q2_input], output = result)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    return model
