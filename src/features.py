import pandas as pd
import numpy as np
import os

# File paths
INPUT_CSV = "data/processed/cleaned_craigslist_van_cta_all.csv"
OUTPUT_CSV = "data/processed/featured_craigslist_van_cta_all.csv"



def calc_km_fuel_metrics(input_path=INPUT_CSV, output_path=OUTPUT_CSV, annual_km=15000, gas_price_per_litre=1.80):
    '''
    Enhances the cleaned Craigslist vehicle dataset with the following columns:
    - km_per_year: Annual driving distance (odometer / age)
    - usage_ratio: Continuous ratio relative to the 15,000 km/year benchmark
    - usage_category: Binned categories from Ultra Low -> Heavy Commute
    - annual_fuel_cost: Estimated annual fuel cost in CAD based on NRC L/100km
    '''

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}. Please run clean_data.py first.")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} vehicles from {input_path}")

    # 1. Annual Mileage (km/year)
    df['km_per_year'] = (df['odometer'] / df['age']).round(1)

    # 2. Usage Ratio (relative to 15,000 km/year market baseline)
    # Evaluates previous owner driving intensity: < 1.0 = below average, 1.0 = average, > 1.0 = above average
    df['usage_ratio'] = (df['km_per_year'] / 15000).round(2)

    # 3. Usage Category
    bins = [-np.inf, 8000, 12000, 18000, 25000, np.inf]
    labels = [
        "Ultra Low/Weekend",
        "Below Average",
        "Average",
        "Above Average",
        "Highway/Heavy Commute"
    ]
    df['usage_category'] = pd.cut(df['km_per_year'], bins=bins, labels=labels, right=False)

    # 4. Estimated Annual Fuel Cost (CAD)
    # Formula: (Combined L / 100 km) * annual_km * gas_price_per_litre
    df['annual_fuel_cost'] = (
        (df['Combined (L/100 km)'] / 100) * annual_km * gas_price_per_litre
    ).round(0)

    # Export to CSV
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Mileage and fuel metrics successfully saved to {output_path}")


    return df



if __name__ == "__main__":
    calc_km_fuel_metrics()