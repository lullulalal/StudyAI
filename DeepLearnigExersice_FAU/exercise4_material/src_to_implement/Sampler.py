import random
import numpy as np

import random
from torch.utils.data.sampler import Sampler


class CustomSampler(Sampler):
    def __init__(self, qs, weights, mode = "mode1"):
        self.idx_list = [0, 1, 2, 3]
        self.indices = range(1600)
        self.mode = mode
        self.qs = qs
        self.weights = weights

        q = np.vstack((self.qs[0], self.qs[1]))
        q = np.vstack((q, self.qs[2]))
        q = np.vstack((q, self.qs[3]))
        np.random.shuffle(q)
        self.qt = q

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1

        idx = 0

        if self.mode == 'mode1':
            randIdx = random.choices(self.idx_list, weights=self.weights, k=1)[0]
            q = self.qs[randIdx]

            w = q[:, 0]
            di = q[:, 1]
            w = np.reciprocal(w.astype(float))
            p = w / np.sum(w)
            d = np.random.choice(di, 1, p=p)
            qi = np.where(di == d[0])
            q[qi,0] += 1
            idx = d[0]

        else:
            q = self.qt

            w = q[:, 0]
            di = q[:, 1]
            w = np.reciprocal(w.astype(float))
            p = w / np.sum(w)
            d = np.random.choice(di, 1, p=p)
            qi = np.where(di == d[0])
            q[qi, 0] += 1
            idx = d[0]

        return idx

    def __len__(self):
        return len(self.indices)