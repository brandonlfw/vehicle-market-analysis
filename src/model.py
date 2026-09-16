import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score



featured_csv = "data/processed/featured_craigslist_van_cta_all.csv"

df = pd.read_csv(featured_csv)

df = df.loc[
    (df['age'] <= 25) & (df['price'].between(1000, 80000))
]

numeric_features = ['age', 'odometer', 'cylinders', 'Combined (L/100 km)']
categorical_features = ['make', 'market', 'title status', 'type', 'fuel', 'drive', 'transmission', 'is_dealer']

X = df[numeric_features + categorical_features]
y = df['price']


# X_train: specs (columns) for 80% of the cars
# y_train: real prices for those 80% of cars
# X_test: specs for the remaining 20% of the cars NOT used in training
# y_test: real prices for the remaining 20% of the cars trying to predict from X_test
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

# 4. Apply rules using fit_transform for the training data and transform for testing data
X_train_prep = preprocessor.fit_transform(X_train) # the 86 features
X_test_prep = preprocessor.transform(X_test)

print("Preprocessed X_train shape:", X_train_prep.shape)
print("Preprocessed X_test shape:", X_test_prep.shape)



def calc_error(y_true, y_pred):
    results_df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'error': np.abs(y_true - y_pred)
    })

    bins = [0, 10000, 25000, 45000, float('inf')]
    labels = ['Budget (<$10k)', 'Commuter ($10k-$25k)', 'Late Model ($25k-$45k)', 'Luxury/Truck ($45k+)']

    results_df['group'] = pd.cut(results_df['actual'], bins=bins, labels=labels, right=False)

    group_summary = results_df.groupby('group', observed=False)['error'].agg(
        cars_count = 'count',
        mae = 'mean',
        median_error = 'median'
    ).round(2)

    print(group_summary)



# MODEL 1: Linear Regression (Baseline)

# 1. Create and train model using .fit()
baseline_model = LinearRegression()
baseline_model.fit(X_train_prep, y_train)

# 2. Predict unseen test cars
y_pred_baseline = baseline_model.predict(X_test_prep)

# 3. Calculate Mean Absolute Error (MAE) and R-squared values
mae_baseline = mean_absolute_error(y_test, y_pred_baseline) # (true values, predicted values)
r2_baseline = r2_score(y_test, y_pred_baseline)

print("\nBaseline Linear Regression Results:")
print(f"R-squared Score: {r2_baseline:.3f}")
print(f"Average Error (MAE): ${mae_baseline:,.2f}\n")



if __name__ == "__main__":
    calc_error(y_test, y_pred_baseline)