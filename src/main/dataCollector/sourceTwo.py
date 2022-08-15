import os
import argparse
from __future__ import print_function
import torch
import torchrec
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepL
import numpy as np

class SourceTwo():
    
    """
    
    SourceTwo.py: Public clothing datasets to recognize individual articles of clothing.

    Loads dataset for later consumption into model.
    -------------------------------------------------------------------------------------------

    Functions:

    """
    def __init__(self, ):
        
        '''
        Load the datasets for later consumption into model
        '''

        pass 
