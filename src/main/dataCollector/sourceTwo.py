import os 
import pandas as pd
from PIL import Image 

import torch 
from torch.utils.data import Dataset

class SourceTwo(Dataset):
    
    #root_dir = r'src/main/dataCollector/data'


    """
    
    SourceTwo.py: Fashion MNIST

    -------------------------------------------------------------------------------------------

    Functions:

    """
    def __init__(self, root_dir, annotation_file, transform = None):
        self.root_dir = root_dir
