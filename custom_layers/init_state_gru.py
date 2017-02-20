from keras import backend as K
from keras.layers import GRU
import numpy as np

class InitStateGRU(GRU):
    def __init__(self, init_state, *args, **kwargs):
        self.init_state = init_state
        super(InitStateGRU, self).__init__(*args, **kwargs)

    def get_initial_states():
        return [self.init_state]
