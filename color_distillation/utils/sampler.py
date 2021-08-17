import random
import numpy as np
from torch.utils.data.sampler import Sampler


class RandomSeqSampler(Sampler):
    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        random.seed(seed)
        np.random.seed(seed)
        self.index_list = np.random.permutation(len(data_source))

    def __iter__(self):
        return iter(self.index_list)

    def __len__(self) -> int:
        return len(self.data_source)
