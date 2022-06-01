from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
import matplotlib.pyplot as plt 

class Utils:
    
    """
    
    The Utilities module: 
    
    Base module where utility functions may be stored for use in any part of application.

    -------------------------------------------------------------------------------------------

    Functions:
    
    
    """
    def __init__(self, root_):
        self.root_ = "~/Capstone/data/toyData/mnist"
    
    def trainMNIST(self):
        train_data = datasets.FashionMNIST(
            root = self.root_
            , train = True
            , download = True 
            , transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        )
        return train_data
    
    def testMNIST(self):
        test_data = datasets.FashionMNIST(
            root = self.root_
            , train = False
            , download = True 
            , transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        )
        return test_data
