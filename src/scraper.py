import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import os

# Headers to emulate a standard web browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Craigslist internal JSON search API endpoint
SAPI_SEARCH_URL = "https://sapi.craigslist.org/web/v8/postings/search/full"


def fetch_all_listing_summaries(area_id=16, include_owner=True, include_dealer=True):
    '''
    Fetches vehicle listing summaries across Vancouver (area 16) using the Craigslist JSON search API.
    Retrieves both owner and dealer listings, capturing title, price, odometer, is_dealer, and URL.
    '''
    purveyors = []
    if include_owner:
        purveyors.append(("owner", 0))
    if include_dealer:
        purveyors.append(("dealer", 1))

    all_listings = []
    seen_tokens = set()

    for purveyor_name, is_dealer_flag in purveyors:
        print(f"\n--- Fetching {purveyor_name.upper()} listings via JSON API ---")

        # Query subareas to ensure complete coverage across Metro Vancouver
        subareas = ["cta", "van/cta", "bnc/cta", "rds/cta", "rch/cta", "nvn/cta", "pmc/cta"]
        # van: Vancouver city
        # bnc: Burnaby/New Westminster
        # rds: Delta/Surrey/Langley
        # rch: Richmond
        # nvn: North Shore
        # pmc: Tri-Cities/Pitt Meadows/Maple Ridge

        for subpath in subareas:
            # Each batch returns up to 360 listings; step through pages
            for page in range(0, 10, 2):
                batch = f"{area_id}-0-360-{page}-0" # ex. 16-0-360-0-0 means: 16 (Vancouver), 0 (default filter), 360 (return up to 360 cars at once), {page} (page index), 0 (sort by date)
                params = {
                    "batch": batch,
                    "cc": "CA", # Country
                    "lang": "en", # Language
                    "searchPath": subpath, # Current subarea
                    "purveyor": purveyor_name # Seller (owner or dealer)
                }

                try:
                    response = requests.get(SAPI_SEARCH_URL, params=params, headers=headers, timeout=10)
                    if response.status_code != 200:
                        break

                    data = response.json().get("data", {})
                    decode_locs = data.get("decode", {}).get("locationDescriptions", []) # get location names
                    items = data.get("items", []) # get list of 360 cars
                    if not items:
                        break

                    new_in_batch = 0
                    for item in items:
                        if not isinstance(item, list) or len(item) < 8:
                            continue

                        # Extract slug (tag 6), token (tag 13), and odometer (tag 9)
                        slug = None # ex. [6, 'north-vancouver-2018-toyota']: Tag 6 is the URL title slug
                        token = None # ex. [13, 'nYERSDL4B3tfLju4FpNz8H']: Tag 13 is the listing's unique post token
                        odometer = None # ex. [9, 100514]: Tag 9 is car's odometer mileage

                        for el in item:
                            if isinstance(el, list) and len(el) > 1:
                                if el[0] == 6 and isinstance(el[1], str):
                                    slug = el[1]
                                elif el[0] == 13 and isinstance(el[1], str):
                                    token = el[1]
                                elif el[0] == 9:
                                    odometer = el[1]

                        if not token or token in seen_tokens: # if this listings unique token was added to seen_tokens previously, skip this listing
                            continue

                        seen_tokens.add(token) # otherwise, add this listing to seen_tokens
                        name = item[-1] if isinstance(item[-1], str) else None # item[-1] is car title (ex. '2018 Toyota Prius)
                        price = item[3] if len(item) > 3 and isinstance(item[3], (int, float)) else None # item[3] is asking price (ex. 20500)
                        link = f"https://www.craigslist.org/view/d/{slug}/{token}" if slug and token else None # the listing URL

                        # Extract location from item[4] (format: "subarea:location_index~lat~lon", ex. '1:2~49.2551~-123.0944')
                        location = None
                        if len(item) > 4 and isinstance(item[4], str) and ":" in item[4]:
                            try:
                                loc_parts = item[4].split("~")[0].split(":")
                                if len(loc_parts) > 1 and loc_parts[1].isdigit():
                                    loc_idx = int(loc_parts[1])
                                    if loc_idx < len(decode_locs) and isinstance(decode_locs[loc_idx], str):
                                        location = decode_locs[loc_idx].strip()
                            except Exception:
                                location = None

                        all_listings.append({
                            "name": name,
                            "price": price,
                            "odometer": odometer,
                            "location": location,
                            "is_dealer": is_dealer_flag,
                            "link": link
                        })
                        new_in_batch += 1

                    if new_in_batch == 0 and page > 0:
                        break

                    time.sleep(0.3)

                except Exception as e:
                    print(f"Error fetching batch {batch} ({subpath}): {e}")
                    break

        print(f"Total unique listings collected so far: {len(all_listings)}")

    return all_listings


def get_listing_details(cars_data, output_csv="data/raw/craigslist_van_cta_all.csv", checkpoint_interval=50, max_listings=None, delay=1.0):
    '''
    Deep-scrapes vehicle specs (cylinders, transmission, drive, fuel, title status) from each car's URL.
    Includes automatic checkpoint saving and resume support if interrupted.
    '''
    # Check if a partial scrape already exists to resume from
    scraped_links = set()
    scraped_records = []

    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            scraped_records = existing_df.to_dict(orient="records")
            scraped_links = set(r["link"] for r in scraped_records if "link" in r and pd.notna(r["link"]))
            print(f"Found existing progress in {output_csv}: {len(scraped_records)} vehicles already scraped.")
        except Exception as e:
            print(f"Could not read existing {output_csv}: {e}")

    # Determine which listings still need to be scraped
    remaining_cars = [c for c in cars_data if c.get("link") not in scraped_links]
    if max_listings is not None:
        remaining_cars = remaining_cars[:max_listings]

    print(f"\nStarting deep scrape: {len(remaining_cars)} vehicles to scrape ({len(scraped_records)} already completed)...")

    total_to_scrape = len(remaining_cars)

    for i, car in enumerate(remaining_cars):
        link = car.get("link")
        if not link:
            continue

        specs = {}
        try:
            response = requests.get(link, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                listing_details = soup.find_all("div", class_="attr")

                for div in listing_details:
                    label = div.find("span", class_="labl")
                    value = div.find("span", class_="valu")

                    if label and value:
                        clean_label = label.text.replace(":", "").strip()
                        clean_value = value.text.strip()
                        specs[clean_label] = clean_value


            else:
                print(f"[{i + 1}/{total_to_scrape}] Status {response.status_code} for {link}")

        except requests.exceptions.RequestException as e:
            print(f"[{i + 1}/{total_to_scrape}] Failed to fetch {link}: {e}")

        # Update car dictionary with scraped specs (preserving existing odometer if page does not have it)
        car.update(specs)
        scraped_records.append(car)

        title_display = car.get("name", "Unknown")
        print(f"[{i + 1}/{total_to_scrape}] (Total: {len(scraped_records)}) Scraped: {title_display}")

        # Checkpoint save every interval
        if (i + 1) % checkpoint_interval == 0:
            pd.DataFrame(scraped_records).to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"--- [CHECKPOINT] Saved {len(scraped_records)} listings to {output_csv} ---")

        time.sleep(delay)

    # Final save
    final_df = pd.DataFrame(scraped_records)
    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nDone! Successfully saved {len(final_df)} listings to {output_csv}")
    return final_df


if __name__ == "__main__":
    # 1. Fetch all listing summaries via JSON API (Owner + Dealer)
    cars_data = fetch_all_listing_summaries(area_id=16, include_owner=True, include_dealer=True)
    print(f"\nTotal unique listings ready for deep scrape: {len(cars_data)}")

    # 2. Deep scrape individual pages for specs with checkpointing
    # Set max_listings=500 for a quick test, or None for all
    output_path = "data/raw/craigslist_van_cta_all.csv"
    get_listing_details(cars_data, output_csv=output_path, checkpoint_interval=50, max_listings=None, delay=1.0)