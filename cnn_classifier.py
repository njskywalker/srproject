#%%
# Imports
from enum import unique
import os
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torchvision import models
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#%%
# Define datasets and functions

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__()

        # Read all BW images - transforms done in __getitem__ method
        self.annotations = pd.read_csv(csv_file)
        self.annotations = self.annotations[['id', 'category']] # Drop index
        print(set(self.annotations['category']))
        #self.annotations.set_index(['id'], inplace=True)  # Index on id so we can iloc
        self.root_dir = root_dir
        # Init transformer
        self.transform = transform

        # for image_file in os.listdir(self.root_dir):
        #     with Image.open(self.root_dir + image_file) as img:
        #         # Transform to vector and put into data dict, key = filename w/o extension
        #         self.annotations.iloc[image_file.split('.')[0], 'tensor'] = self.transform(img)

        # print(self.annotations.head())

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        # Load up image (features, basically)
        img_dir = os.path.join(self.root_dir, str(self.annotations.iloc[index, 0]) + '.jpg')
        image = Image.open(img_dir)

        # Get label - here, which category it's in
        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        # Transform features if we've said yes
        if self.transform:
            image = self.transform(image)

        return (image, label)


#%%
# Neural network
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, 7),
            nn.ReLU(),
            nn.Conv2d(8, 16, 7),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4000000, 13),
            nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)


# Define training function
def train(model, train_loader, optimiser, lf, epochs=10, writer=None):
    # Summary variables we will reference later
    epochs = range(epochs)
    total_steps = len(train_loader) # Just the number of batches
    # Set up TensorBoard variables
    running_loss = 0.0
    running_correct = 0

    # Actual training loop!
    for epoch in epochs:
        print(f'Training epoch {epoch} of {epochs}')
        for i, (features, labels) in enumerate(train_loader):
            print('Step ', i)
            
            # We've already separated features, labels from each batch
            predictions = model(features) # Return prediction
            loss = lf(predictions, labels) # Calculate loss
            print(loss)
            # Backpropagate to update model weights
            optimiser.zero_grad()
            loss.backward() 
            optimiser.step()

            # For writer!
            running_loss += loss.item() # Add our current loss
            # _, predicted = torch.max(predictions.data, 1)
            # And total running correct predictions
            # running_correct += (predicted == predictions).sum().item()

            if writer:
                if (i+1) % 100 == 0: # Every 100th step (batch) update TensorBoard
                    writer.add_scalar('Training loss', running_loss/ 100,  epoch * total_steps * i)
                    # writer.add_scalar('Accuracy', running_correct/ 100,  epoch * total_steps * i)
                    running_loss = 0.0
                    # running_correct = 0

        # Save model weights after every epoch
        model_dir = 'model_' + datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        filename = f'parameters_epoch_{epoch}.pt'

        path = os.path.join('model_evaluation', model_dir, 'weights', filename)
        torch.save(model.state_dict(), path)

        print(f'Finished training epoch {epoch}')

#%%
# Load datasets

root_dir = './data/bw_images/'
# This loads the dataset, and adds a transform object (which will convert to tensor only)
dataset = ImageDataset(csv_file='./data/Images_with_Labels.csv', root_dir=root_dir,
                       transform=transforms.ToTensor())

# Split dataset!
train_set, test_set = torch.utils.data.random_split(dataset, [8823, 3781])

# Instantiate dataloader
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=True)

#%%
# Train neural net

# Set up TensorBoard
writer = SummaryWriter('runs/image_dataset')

# Settings
lr = 1e-3

# Allow CUDA if it's available (GPU-based training)
device = torch.device('cpu') # 'cuda' if torch.cuda.is_availabe() else 'cpu')
model = ConvNet().to(device)

#%%
# Fine-tuning ResNET50

tuned_model = models.resnet50(pretrained=True)

# Get number of features (just before final layer) so
# we can recreate the final layer properly, but with
# our own categories
num_features = tuned_model.fc.in_features

# Recreate final layer with same number of features
tuned_model.fc = nn.Linear(num_features, 13) # we want to classify into 13
tuned_model.to(device)

model = tuned_model # reassign model so we can train it!

#%%
# Model training cell

# Set up loss function and optimiser method
lf = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=lr)

# Train model
train(model, train_loader, optimiser, lf, epochs=10, writer=writer)