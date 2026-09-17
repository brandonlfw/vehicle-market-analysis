import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
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

# print("Raw X_train shape:", X_train.shape)
# print("Raw X_test shape:", X_test.shape)


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

# print("Preprocessed X_train shape:", X_train_prep.shape)
# print("Preprocessed X_test shape:", X_test_prep.shape)



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

    print(group_summary, '\n')



# MODEL 1: Linear Regression (Baseline)

# 1. Create and train model using .fit()
baseline_model = LinearRegression()
baseline_model.fit(X_train_prep, y_train)

# 2. Predict unseen test cars
y_pred_baseline = baseline_model.predict(X_test_prep)

# 3. Calculate Mean Absolute Error (MAE) and R-squared values
mae_baseline = mean_absolute_error(y_test, y_pred_baseline) # (true values, predicted values)
r2_baseline = r2_score(y_test, y_pred_baseline)

# print("\n--- Baseline Linear Regression Results ---")
# print(f"R-squared Score: {r2_baseline:.3f}")
# print(f"Average Error (MAE): ${mae_baseline:,.2f}\n")

# print("\nGrading the baseline linear regression model:")
# calc_error(y_test, y_pred_baseline)



# MODEL 2: Random Forest Generator

# Create and train 100 decision trees
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_prep, y_train)

y_pred_rf = rf_model.predict(X_test_prep)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n--- Random Forest Results ---")
print(f"R-squared Score: {r2_rf:.3f}")
print(f"Average Error (MAE): ${mae_rf:,.2f}")

print("Grading the random forest model:")
calc_error(y_test, y_pred_rf)



# MODEL 3: XGBoost Regressor
# Create and train the sequential boosted trees
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_prep, y_train)


y_pred_xgb = xgb_model.predict(X_test_prep)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# print("\n--- XGBoost Results ---")
# print(f"R-squared Score: {r2_xgb:.3f}")
# print(f"Average Error (MAE): ${mae_xgb:,.2f}")

# print("Grading the XGB Regressor model:")
# calc_error(y_test, y_pred_xgb)



# Feature Importance

feature_names = preprocessor.get_feature_names_out()
clean_feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]

importances = pd.Series(rf_model.feature_importances_, index=clean_feature_names)

top10_features = importances.sort_values(ascending=False).head(10)
# print("\n--- Top 10 Most Important Features Driving Vehicle Price ---")
# print((top10_features * 100).round(2).to_string())

# Plot as a clean horizontal bar chart scaled to 100%
plt.figure(figsize=(10, 6))
plot_data = (top10_features.sort_values() * 100)
ax = plot_data.plot(kind='barh', color='#3498db', edgecolor='#2980b9', alpha=0.85)

# Add exact percentage labels to the end of each bar for clarity
for i, v in enumerate(plot_data):
    ax.text(v + 1.2, i, f"{v:.1f}%", va='center', fontsize=9.5, fontweight='bold', color='#2c3e50')

plt.title("Top 10 Features Driving Used Vehicle Prices in Metro Vancouver", fontsize=13, fontweight='bold')
plt.xlabel("Relative Importance (% of Total Variance Explained)", fontsize=11, fontweight='bold')
plt.xlim(0, 100)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("reports/figures/05_feature_importance.png", dpi=300)
plt.close()
print("Saved Feature vs. % Importance bar chart to reports/figures/05_feature_importance.png\n")



def predict_car_price(year, make, odometer,
    car_type='sedan',
    cylinders=4,
    title_status='clean',
    transmission='automatic',
    drive='fwd',
    fuel='gas',
    fuel_consumption=9.0,
    is_dealer=0,
    asking_price=None
):
    '''
    'year', 'make', 'odometer' are required args, but the rest will default to the values above if they are not set.
    If the correct value for the args is known, setting it will yield much more accurate prices, especially for fuel_consumption.
    Transforms the data passed in using Pipeline above, calculates how underpriced/fair/overpriced it is if asking_price provided,
    and returns the predicted price of the car.
    '''

    age = max(2026 - year, 1)

    specs_df = pd.DataFrame([{
        'age': age,
        'odometer': odometer,
        'cylinders': cylinders,
        'Combined (L/100 km)': fuel_consumption,
        'make': make,
        'market': np.nan,
        'title status': title_status,
        'type': car_type,
        'fuel': fuel,
        'drive': drive,
        'transmission': transmission,
        'is_dealer': is_dealer
    }])

    car_preprocessed = preprocessor.transform(specs_df)

    predicted_price = rf_model.predict(car_preprocessed)[0] # returns price as a numpy array, ex. [24581.20], take 0th index to get just price


    # Vehicle Valuation Report

    print(f"--- VEHICLE VALUATION REPORT ---")
    print(f"Vehicle: {year} {make} ({odometer:,} km)")
    print(f"Estimated Fair Market Value: ${predicted_price:,.2f}")
    
    if asking_price is not None:
        diff = asking_price - predicted_price
        pct_diff = (diff / predicted_price) * 100
        print(f"Asking Price: ${asking_price:,.2f}")

        if pct_diff < -10:
            print(f"Price Verdict: [GREAT DEAL] - ${abs(diff):,.2f} below market value, {abs(pct_diff):.1f}% discount")
        elif pct_diff > 10:
            print(f"Price Verdict: [OVERPRICED] - ${diff:,.2f} above market value, +{pct_diff:.1f}% markup")
        else:
            print(f"Price Verdict: [FAIR MARKET PRICE] - Within +/- 10% of market value")

    return predicted_price



predict_car_price(2025, 'Toyota', 0, fuel='hybrid', drive='fwd', fuel_consumption=4.9)