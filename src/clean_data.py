import pandas as pd


# Raw scraped data CSV file path
raw_csv_path = "data/raw/craigslist_van_cta_p1.csv"

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



def clean_listing_details(raw_csv_path):
    '''
    Cleans data, adds 'year' and 'age' columns, and filters out rows that do not meet requirements for model
    '''

    raw_df = pd.read_csv(raw_csv_path)

    # Remove '$' and ',' from price and odometer columns
    raw_df['price'] = raw_df['price'].replace(r'[\$,]', '', regex=True)
    raw_df["price"] = pd.to_numeric(raw_df["price"], errors="coerce")

    raw_df['odometer'] = raw_df['odometer'].str.replace(',', '', regex=False)
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

    # Export the cleaned and updated df as a CSV to /data/processed/
    raw_df.to_csv("data/processed/cleaned_craigslist_van_cta_p1.csv", index=False, encoding="utf-8-sig")
    print(f"Cleaned CSV saved to data/processed/cleaned_craigslist_van_cta_p1.csv")



clean_listing_details(raw_csv_path=raw_csv_path)