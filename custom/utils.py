from keras import backend as K

"""
Expects x to be of shape (batch_size, time_steps, output_dim)
"""
def get_time_index(x, index):
    # get shape
    batch_size = K.shape(x)[0]
    time_steps = K.shape(x)[1]
    output_dim = K.shape(x)[2]
    # reshape x, to timesteps, output_dim
    x_reshape = K.reshape(x, (-1, output_dim))
    index = K.arange(0, batch_size) * time_steps + index
    return K.gather(x_reshape, index)

