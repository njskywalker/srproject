import os
import pandas as pd

#%%
# Dataset creation
csv_path = './data/Products.csv'

# ID-Image dataset
img_df = pd.read_csv(csv_path, lineterminator='\n')
# Drop everything except id and category columns
img_df = img_df[['id', 'category']]
# Bin all categories except the highest-level one
# E.g. "Home / Bedrooms / Beds" turns into "Home"
img_df['category'] = img_df['category'].map(lambda x: x.split('/')[0])
# Now just remove leading/trailing whitespaces
img_df['category'] = img_df['category'].map(lambda x: x.strip())
# We have to rename 'id' to 'product_id' for img_df
# otherwise join returns nothing
img_df.rename(columns={'id': 'product_id'}, inplace=True)
# Since img_df's id != product_id (the latter of which is in the image filename) we must
# combine these data before getting labels

# Product ID-ID dataset
pid_df = pd.read_csv('./data/Images.csv', lineterminator='\n')
pid_df = pid_df[['id', 'product_id']]

merge_df = pd.merge(img_df, pid_df, how='inner', on='product_id')
# Combine to form Product ID-Image dataset, which is what we want
merge_df = merge_df[['id', 'category']]  # Truncate product_id
# Convert each category to (first Categorical, then) int
merge_df.category = pd.Categorical(merge_df.category)
categories = dict(enumerate(merge_df.category.cat.categories))
merge_df['category'] = merge_df.category.cat.codes
# Save!
merge_df.to_csv('./data/Images_with_Labels.csv')

# Do note some image files do not have a category!
# This is because of missing data. Truncate? Cannot impute!