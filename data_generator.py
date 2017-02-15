import os
import pickle
import random

class DataGenerator:
    def __init__(self, directory, batch_size = 1, randomize = False):
        self.directory = directory
        self.file_list = os.listdir(self.directory)
        self.batch_size = batch_size
        self.cur_samples = []
        self.idx_order = []
        self.sample_idx = 0
        self.file_idx = 0
        self.is_random = randomize

        # check for randomization
        if self.is_random:
            random.shuffle(self.file_list)

        print self.file_list
        print self.cur_samples
        print self.file_idx

    def __iter__(self):
        return self

    def reset(self):
        if self.is_random:
            random.shuffle(self.file_list)

    def __next__(self):
        return self.next()

    def init_next_file(self):
        if self.file_idx < len(self.file_list):
            with open(os.path.join(self.directory, self.file_list[self.file_idx]), 'rb') as data_file:
                self.cur_samples = pickle.load(data_file)
                self.idx_order = range(len(self.cur_samples))
                if self.is_random:
                    random.shuffle(self.idx_order)
            self.file_idx += 1
            self.sample_idx = 0
        else:
            self.sample_idx = 0
            self.cur_samples = []

    def next(self):
        samples = [];
        needed_samples = self.batch_size
        while needed_samples > 0:
            if self.sample_idx >= len(self.cur_samples):
                self.init_next_file()
            if self.sample_idx < len(self.cur_samples):
                samples.append(self.cur_samples[self.idx_order[self.sample_idx]])
                self.sample_idx += 1
                needed_samples -= 1
            else:
                raise StopIteration()
        return samples