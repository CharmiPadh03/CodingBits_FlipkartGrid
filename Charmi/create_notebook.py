import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')"""))

# Cell 2: Load Data
cells.append(nbf.v4.new_markdown_cell("## 1. Load Data"))
cells.append(nbf.v4.new_code_cell("""train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")"""))

# Cell 3: EDA
cells.append(nbf.v4.new_markdown_cell("## 2. Exploratory Data Analysis (EDA)"))
cells.append(nbf.v4.new_code_cell("""display(train_df.head())
print("\\n--- Missing Values in Train ---")
print(train_df.isnull().sum())
print("\\n--- Missing Values in Test ---")
print(test_df.isnull().sum())"""))

# Cell 4: Preprocessing
cells.append(nbf.v4.new_markdown_cell("## 3. Data Preprocessing & Feature Engineering"))
cells.append(nbf.v4.new_code_cell("""def preprocess_data(df):
    df = df.copy()
    
    # 1. Parse timestamp (e.g., "0:15" -> hour=0, minute=15)
    if 'timestamp' in df.columns:
        df[['hour', 'minute']] = df['timestamp'].str.split(':', expand=True).astype(int)
        df.drop('timestamp', axis=1, inplace=True)
        
    # 2. Fill Missing Values
    # We will use HistGradientBoostingRegressor which handles missing values natively,
    # but let's do a basic fill for categorical so we can encode them safely.
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    df['Weather'] = df['Weather'].fillna('Unknown')
    # Temperature can remain NaN, tree models can handle it.
    
    # 3. Label Encoding for Categorical columns
    cat_cols = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category')
            
    return df

X_train_full = preprocess_data(train_df.drop('demand', axis=1))
y_train_full = train_df['demand']
X_test = preprocess_data(test_df)
"""))

# Cell 5: Model Training
cells.append(nbf.v4.new_markdown_cell("## 4. Model Training & Validation"))
cells.append(nbf.v4.new_code_cell("""# Split the training data into train and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# HistGradientBoostingRegressor natively supports categorical features if we specify them
cat_features = [X_train_full.columns.get_loc(c) for c in X_train_full.select_dtypes(include=['category']).columns]

model = HistGradientBoostingRegressor(
    categorical_features=cat_features,
    random_state=42,
    max_iter=200
)

print("Training model...")
model.fit(X_tr, y_tr)

# Predict on validation set
val_preds = model.predict(X_val)

# Calculate R2 Score (Metric for evaluation)
score = max(0, 100 * r2_score(y_val, val_preds))
print(f"Validation R2 Score: {score:.2f}")
"""))

# Cell 6: Train on full data and Predict
cells.append(nbf.v4.new_markdown_cell("## 5. Train on Full Data & Make Predictions"))
cells.append(nbf.v4.new_code_cell("""# Retrain on full dataset for best performance
print("Retraining on full training data...")
model.fit(X_train_full, y_train_full)

print("Predicting on test data...")
test_preds = model.predict(X_test)
test_preds = np.clip(test_preds, 0, 1) # Demand should ideally be between 0 and 1

# Create submission dataframe
submission = pd.DataFrame({
    'Index': test_df['Index'],
    'demand': test_preds
})

submission.to_csv('submission.csv', index=False)
print("Submission saved to 'submission.csv'")
display(submission.head())
"""))

nb['cells'] = cells

with open('solution.ipynb', 'w') as f:
    nbf.write(nb, f)
