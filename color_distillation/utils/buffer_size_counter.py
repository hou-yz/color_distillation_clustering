import numpy as np
import torch.multiprocessing as mp


class BufferSizeCounter(object):
    # please specify num_workers=0 when calling since
    # multiprocessing within dataloader can make the buffer size result unreliable
    def __init__(self):
        shared_array_base = mp.Array('i', 1)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array[0] = 0
        self.size = shared_array

    def reset(self):
        shared_array_base = mp.Array('i', 1)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array[0] = 0
        self.size = shared_array

    def update(self, new_buffer_size):
        self.size[0] += new_buffer_size
