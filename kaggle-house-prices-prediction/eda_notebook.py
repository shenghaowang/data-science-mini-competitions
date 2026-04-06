# %%
from pathlib import Path

import pandas as pd
import plotly.express as px

# %%
script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "data" / "kaggle-house-prices-prediction"

train_df = pd.read_csv(data_dir / "train.csv")
print(train_df.shape)

train_df.head()

# %%
train_df.info()

# %%
# Compute correlation matrix (numeric columns only)
num_cols = train_df.select_dtypes(include='number').columns
num_cols = num_cols.drop('Id')  # Exclude 'Id' from correlation
corr = train_df[num_cols].corr()

fig = px.imshow(
    corr,
    text_auto='.2f',        # show correlation values
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1,
    title='Feature Correlation Matrix',
    width=1600, height=1200
)
fig.show()

# %%
