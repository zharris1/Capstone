import torch
import torch.nn as nn 
import torch.nn.functional as f 

from sklearn.model_selection import train_test_split

class RecommenderMatrixFactorization(nn.Module):

    """
    
    The Recommender module: 

        Algorithm: 
        - SVD (Singular Value Decomposition)
            - https://en.wikipedia.org/wiki/Singular_value_decomposition
            - https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm
        - Matrix Factorization
        - Input: 
            - The Stylist's returned results, Source 4 data.


        Returns: 
            - complete outfit recommendation to the user.

    
        Notes on what to finish here: 
            - Save trained model in file format for later use

    -------------------------------------------------------------------------------------------

    Functions: 

    
    """

    def __init__(self, num_users, num_items, emb_size = 100):
        super(RecommenderMatrixFactorization, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.user_emb.weight.data.uniform_(0, 0.5)
        self.item_emb.weight.data.uniform(0, 0.5)

    
    def forward(self, u, v):
        u = self.user_emb(u)
        v = self.item_emb(v)
        return (u*v).sum(1)
    
    def train_and_test(self, model, epochs = 10, lr = 0.01, wd = 0.0):
        
        '''
        The data parameter in train_test_split is going to be the vectors from the Stylist
        '''

        train, test = train_test_split(data, test_size = 0.2)
        
        '''train'''
        optimizer_train = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
        model.train()
        for i in range(epochs):
            usernames_train = torch.LongTensor(train[0]) # This index needs to be the users from Data Source 4
            articles_of_clothing_train = torch.LongTensor(train[1]) # This index needs to be the articles of clothing from Data Source 1
            ratings_train = torch.FloatTensor(train[2]) # This index needs to be what the user prefers from Data Source 4
            y_hat_train = model(usernames_train, articles_of_clothing_train)
            loss_train = f.mse_loss(y_hat_train, ratings_train)
            optimizer_train.zero_grad()
            loss_train.backward()
            optimizer_train.step()
            print(loss_train.item())

        '''test'''
        model.eval()
        usernames_test = torch.LongTensor(test[0]) # Similar to usernames_train, this index needs to be the users from Data Source 4
        articles_of_clothing_test = torch.LongTensor(test[1]) # Similar to articles_of_clothing_train, this index needs to be the articles of clothing from Data Source 1
        ratings_test = torch.LongTensor(test[2]) # Similar to ratings_train, this index needs to be what the user prefers from Data Source 4
        y_hat_test = model(usernames_test, articles_of_clothing_test)
        loss_test = f.mse_loss(y_hat_test, ratings_test)
        print("Test Loss is: %.4f" % loss_test.item())
        trained_model = torch.save() # STUCK ON HOW TO SAVE THE MODEL FOR PREDICT FUNCTION
        return trained_model 
         
'''
This is the driver function. We can pass in a userID (from data source 1) an their preferences here 
'''
data = [] # Instantiated as empty list for now, this will need to be moved; will do this weekend
userID = 1

user = torch.tensor([userID])
clothes = torch.tensor(set(data[1]))
model = RecommenderMatrixFactorization.train_and_test(user, clothes)
predictions = [i/max(model) * 10 for i in model] # saves normalized predictions in variable
sortedItems = predictions.argsort()
recommendations = set(data[3]).sort() # Arbitrary, index should be equal to the title of the article of clothing from Data Sources 2 and 3
print(recommendations[:3]) # prints the top three clothing recommendations


        