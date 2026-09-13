import pandas as pd
import numpy as np
import re
import os


# Raw scraped data CSV file path
# raw_csv_path = "data/raw/craigslist_van_cta_p1.csv"
raw_csv_path = "data/raw/craigslist_van_cta_all.csv"

# NRC Fuel Consumption Rating Files 1995-2026
fcr_95_14 = "data/raw/my1995-2014-fuel-consumption-ratings-5-cycle.csv"
fcr_15_24 = "data/raw/my2015-2024-fuel-consumption-ratings.csv"
fcr_25 = "data/raw/my2025-fuel-consumption-ratings.csv"
fcr_26 = "data/raw/my2026-fuel-consumption-ratings.csv"

# Combine all the CSVs together as one master FCR df
fcr_files = [fcr_95_14, fcr_15_24, fcr_25, fcr_26]
fcr_dfs = [pd.read_csv(file, encoding="latin1") for file in fcr_files]
fcr_master_df = pd.concat(fcr_dfs, ignore_index=True)


# Get a list of the unique Makes from the master dataset
all_makes = fcr_master_df['Make'].dropna().unique().tolist()
all_makes = sorted(all_makes, key=len, reverse=True)

make_aliases = {
    "vw": "Volkswagen",
    "v.w.": "Volkswagen",
    "chevy": "Chevrolet",
    "mercedes": "Mercedes-Benz",
    "benz": "Mercedes-Benz"
}



def clean_fcr_model(model):
    '''
    Strips text in brackets (ex. '(2-Door)') and trims/drive-types (ex. '4X4', 'AWD') from the 'Model' value in the masterdataset.
    Returns the model name without trailing suffixes, parenthesized text, or symbols.
    '''

    # Remove parentheses and text inside like '(2-Door)' and symbols '#' and '*'
    cleaned_model = re.sub(r'\s*\(.*?\)\s*|[#*]', '', str(model))

    # Repeatedly strip common trailing suffixes from the end of the model name
    while True:
        prev = cleaned_model
        cleaned_model = re.sub(
            r'\s+\b(4x4|4wd|awd|2wd|fwd|rwd|quattro|4matic|xdrive|turbo|supercharged|hybrid|ffv|diesel|sedan|coupe|wagon|convertible|cabriolet|hatchback|sportback|roadster|all-terrain)\b.*$',
            '',
            cleaned_model,
            flags=re.IGNORECASE
        ).strip()

        # If the model name did not change from previous iteration, return it
        if cleaned_model == prev:
            return cleaned_model.strip()



def calc_median_fcr(make, cleaned_model, year, cylinders):
    '''
    Return the median city, hwy, and combined mileage (L/100 km) for each listing in the fcr_master_df
    using hierarchical fallback (exact -> prefix with cyl -> model only -> prefix -> reverse prefix).
    '''

    # Skip processing if any required column is missing
    if pd.isna(make) or pd.isna(cleaned_model) or pd.isna(year):
        return pd.Series([np.nan, np.nan, np.nan])

    # Narrow down the FCR df by Make and Year (my) first
    my_df = fcr_master_df.loc[
        (fcr_master_df['Make'] == make) &
        (fcr_master_df['Model year'] == year)
    ]
    if my_df.empty:
        return pd.Series([np.nan, np.nan, np.nan])

    matched_df = pd.DataFrame()


    # 1) Exact model match + cylinders
    if pd.notna(cylinders):
        exact_cy_match = my_df.loc[
            (my_df['cleaned_model'] == cleaned_model) &
            (my_df['Cylinders'] == cylinders)
        ]
        if not exact_cy_match.empty:
            matched_df = exact_cy_match

    # 2) Prefix match + cylinders (ex. 4-cyl 'Santa Fe' -> 'Santa Fe Sport', 5-cyl 'S60' -> 'S60 2.5T')
    if matched_df.empty and pd.notna(cylinders):
        prefix_cy_match = my_df.loc[
            (my_df['cleaned_model'].str.startswith(cleaned_model, na=False)) &
            (my_df['Cylinders'] == cylinders)
        ]
        if not prefix_cy_match.empty:
            matched_df = prefix_cy_match

    # 3) Exact model match WITHOUT cylinders (fallback if cylinders did not match or were missing)
    if matched_df.empty:
        exact_match = my_df.loc[my_df['cleaned_model'] == cleaned_model]
        if not exact_match.empty:
            matched_df = exact_match

    # 4) Prefix match WITHOUT cylinders (fallback for underspecified models with typos or missing characters)
    if matched_df.empty:
        prefix_match = my_df.loc[my_df['cleaned_model'].str.startswith(cleaned_model, na=False)]
        if not prefix_match.empty:
            matched_df = prefix_match

    # 5) Reverse prefix match (fallback for overspecified listings like 'Accord EX', but NRC only has 'Accord')
    if matched_df.empty:
        for fcr_model in my_df['cleaned_model'].dropna().unique():
            # Check if any FCR cleaned_model name is inside the extracted listing model name and gets the subset of rows for that FCR cleaned_model
            if cleaned_model.startswith(fcr_model):
                rev_match = my_df.loc[my_df['cleaned_model'] == fcr_model]
                if not rev_match.empty:
                    matched_df = rev_match
                    break

    # If no matches across all fallbacks, return NaN
    if matched_df.empty:
        return pd.Series([np.nan, np.nan, np.nan])

    # Compute medians once
    return pd.Series([
        matched_df['City (L/100 km)'].median(),
        matched_df['Highway (L/100 km)'].median(),
        matched_df['Combined (L/100 km)'].median()
    ])
    


# Extract clean model name for each model on master dataset and create a dict of all unique models
fcr_master_df['cleaned_model'] = fcr_master_df['Model'].apply(clean_fcr_model)
models_by_make = fcr_master_df.groupby('Make')['cleaned_model'].unique().to_dict()

# Export mega FCR dataset to CSV
fcr_master_df.to_csv("data/processed/1995-2026-fuel-consumption-ratings.csv", index=False, encoding="utf-8-sig")
print(f"Cleaned model names for 1995-2026 FCR and saved to data/processed/1995-2026-fuel-consumption-ratings.csv")



def extract_make(name):
    '''
    Extracts the vehicle's make by checking if an alias or the actual make is in the 'name' entry of each listing in raw_df
    '''

    # If the title is missing or not a string, return None
    if not isinstance(name, str):
        return None

    name = name.lower()

    # Check for makes that are written as aliases in the 'name' value
    for alias, make in make_aliases.items():
        if alias in name.split():
            return make

    # Check if actual make is inside the 'name' value
    for make in all_makes:
        if make.lower() in name:
            return make

    return None



def normalize_name(text):
    '''
    Normalizes text for matching by handling ampersands and removing punctuation/spaces.
    '''

    text = text.lower().replace('&', ' and ')
    return "".join(c for c in text if c.isalnum())



def extract_model(name, make):
    '''
    Extract the vehicle model from the 'name' and 'make' values
    '''

    # Return none if the 'name' value is not a string or if the 'make' is empty
    if not isinstance(name, str) or pd.isna(make):
        return None

    # Get all the models (list) for the make parameter
    models = models_by_make.get(make)

    if models is None:
        return None

    sorted_models = sorted(models, key=len, reverse=True)

    # Pass 1: Exact substring match
    name_lower = name.lower()

    for model in sorted_models:
        if model.lower() in name_lower:
            return model


    # Pass 2: Normalize text (handles spacing/hyphen differences like CR-V -> crv, F-150 -> f150, E 350 -> e350)
    norm_name = normalize_name(name)

    for model in sorted_models:
        norm_model = normalize_name(model) # Normalize the FCR model names to compare with scraped 'name' value

        # Require 3+ chars, or 2 chars if containing a digit (ex. '93' for '9-3') to prevent false matches
        if (
            len(norm_model) >= 3 or 
            (len(norm_model) >= 2 and any(c.isdigit() for c in norm_model))
        ) and norm_model in norm_name:
            return model

    return None



def clean_listing_details(raw_csv_path, output_csv_path=None):
    '''
    Cleans data, adds 'year' and 'age' columns, and filters out rows that do not meet requirements for model
    '''
    if output_csv_path is None:
        filename = os.path.basename(raw_csv_path)
        output_csv_path = f"data/processed/cleaned_{filename}"

    raw_df = pd.read_csv(raw_csv_path)

    # Remove '$' and ',' from price and odometer columns
    raw_df['price'] = raw_df['price'].replace(r'[\$,]', '', regex=True)
    raw_df["price"] = pd.to_numeric(raw_df["price"], errors="coerce")

    raw_df['odometer'] = raw_df['odometer'].replace(r'[,]', '', regex=True)
    raw_df["odometer"] = pd.to_numeric(raw_df["odometer"], errors="coerce")

    # Extract the car year into a separate 'year' column
    raw_df['year'] = raw_df['name'].str.extract(r"\b(19\d\d|20[0-2]\d)\b")
    raw_df['year'] = pd.to_numeric(raw_df['year'], errors="coerce")

    # Calculate vehicle age
    raw_df['age'] = 2026 - raw_df['year']

    # Extract only the number in the cylinders column
    raw_df['cylinders'] = raw_df['cylinders'].str.extract(r'(\d+)')
    raw_df['cylinders'] = pd.to_numeric(raw_df['cylinders'], errors="coerce")

    # Filter out 'WANTED' listings
    raw_df = raw_df[~raw_df['name'].str.contains('WANTED|WTB|W.T.B.', case=False, na=False)]

    # Filter out underpriced listings
    raw_df = raw_df.loc[raw_df['price'] >= 1000]

    # Drop rows that are missing price, mileage, or year
    raw_df = raw_df.dropna(subset=['price', 'odometer', 'year'])

    # Extract the make from the 'name' column
    raw_df['make'] = raw_df['name'].apply(extract_make)

    # Extract the model from the 'name' column
    raw_df['model'] = raw_df.apply(
        lambda row: extract_model(row['name'], row['make']),
        axis=1
    )

    # Retrieve the City, Highway, and Combined L/100km mileage rating for each listing
    raw_df[['City (L/100 km)', 'Highway (L/100 km)', 'Combined (L/100 km)']] = raw_df.apply(
        lambda row: calc_median_fcr(
            row['make'], row['model'], row['year'], row['cylinders']
        ),
        axis=1,
        result_type='expand',
    )

    # Export the cleaned and updated df as a CSV to /data/processed/
    raw_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Cleaned scraped vehicles CSV saved to {output_csv_path}")



clean_listing_details(raw_csv_path=raw_csv_path)