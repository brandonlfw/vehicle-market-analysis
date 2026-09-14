import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create the folder where charts will be saved
os.makedirs("reports/figures", exist_ok=True)

df = pd.read_csv("data/processed/featured_craigslist_van_cta_all.csv")

# Filter out extreme outliers (collector cars > 25 yrs and ultra luxury > $80,000)
filtered_df = df[(df['age'] <= 25) & (df['price'].between(1000, 80000))].copy()


# CHART 1: Metro Vancouver Depreciation Curve (Price vs. Age)
plt.figure(figsize=(10, 6)) # 10" W x 6" H

# Plot
sns.scatterplot(data=filtered_df, x='age', y='price', alpha=0.3, color='#3498db') # alpha = opacity (0-1)

# Labels
plt.title("Metro Vancouver Used Vehicle Depreciation Curve (Price vs. Age)")
plt.xlabel("Vehicle Age (Years)")
plt.ylabel("Asking Price (CAD)")

# Save to reports/figures
plt.savefig("reports/figures/01_depreciation_curve.png")
plt.close()