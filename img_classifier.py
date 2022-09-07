# %% 
# Imports
import os
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor


# %%
# Functions

def transform(image):
    img_tensor = ToTensor()(image)
    img_vector = torch.flatten(img_tensor)
    return img_vector.numpy()


# %%
# Load datasets (image and text)
path = 'data/bw_images/'
csv_path = 'data/Products.csv'

# Get paths to all B&W images, as a directory
bw_paths = os.listdir(path)

# Text dataset
img_df = pd.read_csv(csv_path)
# Drop everything except id and category columns
img_df = img_df[['id', 'category']]
# Bin all categories except the highest-level one
# E.g. "Home / Bedrooms / Beds" turns into "Home"
img_df['category'] = img_df['category'].map(lambda x: x.split('/')[0])
labels = set(img_df['category'])  # Get all possible labels
img_df.set_index('id')  # Index by image/product UUID
img_df['img_vec'] = np.nan  # Give every product a NaN image vector

# Vectorise
# Enumerate (loop) through all images, and transform them
# Adding to DF with all its stuff
for n, image in enumerate(bw_paths):
    img_array = transform(image)

    # Combine image vector with image id in DataFrame
    img_id = image.split('/')[-1].split('.')[0]
    img_df.loc[img_id, 'img_vec'] = img_array

# Split datasets for training/testing
X = img_df['img_vec']
y = img_df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# %%
# Model

log_reg = LogisticRegression()

# TODO: Set up CV
params = {}

# Train the model
log_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)
