import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

featured_csv = "data/processed/featured_craigslist_van_cta_all.csv"

df = pd.read_csv(featured_csv)

df = df.loc[
    (df['age'] <= 25) & (df['price'].between(1000, 80000))
]

numeric_features = ['age', 'odometer', 'cylinders', 'Combined (L/100 km)']
categorical_features = ['make', 'market', 'title status', 'type', 'fuel', 'drive', 'transmission', 'is_dealer']

X = df[numeric_features + categorical_features]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Raw X_train shape:", X_train.shape)
print("Raw X_test shape:", X_test.shape)


# 1. Pipeline for numerical columns: fill blanks with column median, then set mean = 0 for each column and scale values as # std deviations from mean
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 2. Pipeline for text columns: fill blanks with 'missing', then turn each unique value in each column into its own column (ex. red -> is_red) and set as True (1) or False (0)
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 3. Sends numeric columns to num_pipeline, and text columns to cat_pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, categorical_features)
])

# 4. Apply rules using fit_transform for training data and transform for testing data
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

print("Preprocessed X_train shape:", X_train_prep.shape)
print("Preprocessed X_test shape:", X_test_prep.shape)
