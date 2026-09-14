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
def plot_depreciation_curve(df):
    plt.figure(figsize=(10, 6)) # 10" W x 6" H

    # Plot
    sns.scatterplot(data=df, x='age', y='price', alpha=0.3, color='#3498db') # alpha = opacity (0-1)

    # Labels
    plt.title("Metro Vancouver Used Vehicle Depreciation Curve (Price vs. Age)")
    plt.xlabel("Vehicle Age (Years)")
    plt.ylabel("Asking Price (CAD)")

    # Save to reports/figures
    plt.savefig("reports/figures/01_depreciation_curve.png")
    plt.close()



# CHART 2: Annual Residual Value Retention Rate by Geographical Market
def plot_residual_value(df):
    plt.figure(figsize=(10, 6))

    yr_geo_df = df.loc[df['age'] <= 15]

    yr_geo_df = yr_geo_df.groupby(['age', 'market'])['price'].median().unstack()

    residual_pct = (yr_geo_df / yr_geo_df.iloc[0] * 100).round(1)

    sns.lineplot(data=residual_pct, markers=True, dashes=False)
    plt.title("Vehicle Residual Value Retention Rate by Geographical Market")
    plt.xlabel("Vehicle Age (Years)")
    plt.ylabel("Residual Value (% of Year 1 Price)")

    plt.axhline(50, color='gray', linestyle='--', label='50% Half-Life')
    plt.legend(title="Geographical Market")
    plt.xticks(range(1, 16))

    plt.savefig("reports/figures/02_residual_value.png")
    plt.close()


if __name__ == "__main__":
    plot_depreciation_curve(filtered_df)
    plot_residual_value(filtered_df)