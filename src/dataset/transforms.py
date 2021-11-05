import numpy as np


class RandomRoll(object):
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, sample):
        trace = sample[0][:1]

        r = np.random.randint(self.amount)
        np.roll(trace, 1, axis=1)
