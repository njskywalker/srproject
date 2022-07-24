# %%

# Imports
# Data
from datetime import datetime
import numpy as np
import pandas as pd

# Visualisation
import missingno as msno

# %%

# Clean tabular data

# Load and description of DF
tab_df = pd.read_csv('data/Products.csv', index_col=0, lineterminator='\n')
print(tab_df.info())
print(tab_df.isnull().sum())

# Typecast price str -> float (£25.00 to 25)
tab_df['price'] = tab_df['price'].str.replace('£', '')
tab_df['price'] = tab_df['price'].str.replace(',', '')
tab_df['price'].astype(float)

# TODO: Convert pageid, created to relevant dtypes
# tab_df['create_time'].astype(datetime)

# Visualise missing data
msno.matrix(tab_df)

# %%

