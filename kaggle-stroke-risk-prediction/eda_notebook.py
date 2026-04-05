# %%
from pathlib import Path

import pandas as pd
import plotly.express as px

# %%
script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "data" / "kaggle-stroke-risk-prediction"

train_df = pd.read_csv(data_dir / "train.csv")
print(train_df.shape)

train_df.head()

# %%
train_df.info()

# %%
# Gender
train_df['gender'].value_counts()
# %%
train_df.groupby('gender')['stroke'].mean()

# %%
# Age distribution by stroke
px.histogram(train_df, x='age', color='stroke', nbins=100, title='Age Distribution by Stroke')

# %%
train_df['hypertension'].value_counts()

# %%
train_df.groupby('hypertension')['stroke'].mean()

# %%
train_df['heart_disease'].value_counts()

# %%
train_df.groupby('heart_disease')['stroke'].mean()

# %%
train_df['ever_married'].value_counts()

# %%
train_df.groupby('ever_married')['stroke'].mean()

# %%
train_df['work_type'].value_counts()

# %%
train_df.groupby('work_type')['stroke'].mean()

# %%
train_df['Residence_type'].value_counts()

# %%
train_df.groupby('Residence_type')['stroke'].mean()

# %%
px.histogram(train_df, x='avg_glucose_level', color='stroke', nbins=100,
             title='Average Glucose Level Distribution by Stroke')

# %%
px.histogram(train_df, x='bmi', color='stroke', nbins=100,
             title='BMI Distribution by Stroke')

# %%
train_df['smoking_status'].value_counts()

# %%
train_df.groupby('smoking_status')['stroke'].mean()

# %%
# Plot heatmap of feature correlations
train_df['gender'] = train_df['gender'].map({'Male': 0, 'Female': 1})
train_df['ever_married'] = train_df['ever_married'].map({'No': 0, 'Yes': 1})
train_df['Residence_type'] = train_df['Residence_type'].map({'Rural': 0, 'Urban': 1})

# Compute correlation matrix (numeric columns only)
num_cols = train_df.select_dtypes(include='number').columns
num_cols = num_cols.drop('id')  # Exclude 'id' from correlation
corr = train_df[num_cols].corr()

# Heatmap with plotly express
fig = px.imshow(
    corr,
    text_auto='.2f',        # show correlation values
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1,
    title='Feature Correlation Matrix'
)
fig.show()

# %%
smoking_order = {
    'never smoked': 0,
    'Unknown': 1,
    'formerly smoked': 2,
    'smokes': 3,
}

train_df['smoking_status'] = train_df['smoking_status'].map(smoking_order)

# %%
# Create interaction features
interactions = {
    'age_x_hypertension': train_df['age'] * train_df['hypertension'],
    'age_x_heart_disease': train_df['age'] * train_df['heart_disease'],
    'age_x_bmi': train_df['age'] * train_df['bmi'],
    'glucose_x_bmi': train_df['avg_glucose_level'] * train_df['bmi'],
    'smoking_x_age': train_df['smoking_status'] * train_df['age'],
}

interact_df = pd.DataFrame(interactions)
interact_df['stroke'] = train_df['stroke']

# Rank by absolute correlation with target
corr_scores = interact_df.corr()['stroke'].drop('stroke').abs().sort_values(ascending=False)
print(corr_scores)

# %%
