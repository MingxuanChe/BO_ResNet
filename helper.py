
import numpy as np
import torch

class Options(object):
    '''
    A class for easy access to hyperparameters.
    '''
    def __init__(self, **kwargs):
        super(Options, self).__init__()
        self.__dict__.update(kwargs)
        

def set_seed(seed, device=torch.device("cpu")):
    # set the random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed) 
        