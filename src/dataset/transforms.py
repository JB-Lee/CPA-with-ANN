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


class RandomPadding(object):
    def __init__(self, amount, steps):
        self.amount = amount
        self.steps = steps

    def __call__(self, sample):
        r = np.random.randint(self.amount) * self.steps
        tr = np.pad(sample, (r, 0), constant_values=0)
        return tr


class RandomMasking(object):
    def __init__(self, mask_rate):
        self.mask_rate = mask_rate

    def __call__(self, sample):
        mask = np.random.choice([True, False], size=sample.size, p=[self.mask_rate, 1.0 - self.mask_rate])
        sample[mask] = 0
        return sample
