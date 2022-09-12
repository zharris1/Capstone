import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
from main.stylist import Stylist 

from tqdm import tqdm

data = [] # import data class here; train_loader, validation_loader will be going here

'''
Notes: 

- criterion function is not working for some reason, did I miss something?
'''

'''Hyper Parameters'''
num_epochs = 10
learning_rate = 0.00001
train_CNN = False
batch_size = 32
shuffle = True
pin_memory = True
num_workers = 1

device = ("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((356, 356)), 
    transforms.RandomCrop((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

model = Stylist().to(device)
criterioin = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
for name, param in model.inception.named_parameters():
    if "fc.weight" in name or "fc.bias" in name: 
        param.requires_grad = True
    else: 
        param.requires_grad = train_CNN

'''Metric Checker'''
def metric_checker(loader, model): # using accuracy for now, would probably be wise to use recall instead?
    if loader == train_loader:
        print("----------Checking loss metric----------")
    else: 
        print("----------Checking validation loss metric----------")
    
    num_correct = 0 
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader: 
            x = x.to(device = device)
            y = y.to(device)
    scores = model(x)
    predictions = torch.tensor([1.0 if i >= 0.5 else 0.0 for i in scores]).to(device)
    num_correct += (predictions == y).sum()
    num_samples += predictions.size(0)
    print(f"Received {num_correct} / {num_samples} with accuracy of: {float(num_correct)/float(num_samples) * 100:.2f}")
    model.train()
    return f"{float(num_correct) / float(num_samples) * 100:.2f}"
    
'''Training Loop'''
def train():
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, total = len(train_loader), leave = True)
    if epoch % 2 == 0:
        loop.set_postfix(val_acc = metric_checker(validation_loader, model))
    for imgs, labels in loop:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss = loss.item())

if __name__ == '__main__':
    train()
    print('----------Eyes up Chief; we are training now----------')
    