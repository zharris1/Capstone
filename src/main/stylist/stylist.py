import torch.nn as nn 
import torchvision.models as models

class Stylist(nn.Module):

    """
    
    The Stylist module:

        - Convolution: 
            - Purpose: 
                - Identify articles of clothing 
            - Training/Testing: 
                - Input: 
                    - Data from Sources 2 and 3 
            - Validation: 
                - Input: 
                    - Data from Source 1
    Returns: 
        - 

    -------------------------------------------------------------------------------------------

    Functions: 
    
    """

    def __init__(self, train_CNN = False, num_classes =1): #might need to change the num_classes to the number of classes found in Data Source 2 and 3
        super(Stylist, self).__init__()
        self.inception = self.models.inception_v3(pretrained = True, aux_logits = False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # might need to change this dropout rate
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        features = self.inception(images)
        return self.sigmoid(self.dropout(self.relu(features))).squeeze(1) # might need to change this squeeze rate 
        