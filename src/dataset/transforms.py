import numpy as np


class RandomRoll(object):
    def __init__(self, amount, steps):
        self.amount = amount
        self.steps = steps

    def __call__(self, sample):
        trace = sample[0][:1]

        r = np.random.randint(self.amount) * self.steps
        sample[0][:1] = np.roll(trace, r, axis=1)

        return sample
