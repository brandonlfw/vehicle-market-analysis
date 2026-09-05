import requests
from bs4 import BeautifulSoup

url = "https://vancouver.craigslist.org/search/cta" # cta = "cars & trucks - all" on Craigslist

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
}

params = {
    "purveyor": "owner"
}


# Fetch page 1
response = requests.get(url, headers=headers, params=params)
print("Status Code:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")
listings = soup.find_all("li", class_="cl-static-search-result")



# Iterate through all car listings and store each as dictionaries containing name, price, location, and link in cars_data list
cars_data = []

for car in listings:
    name_element = car.find("div", class_="title")
    name = name_element.text.strip() if name_element else None

    price_element = car.find("div", class_="price")
    price = price_element.text.strip() if price_element else None

    location_element = car.find("div", class_="location")
    location = location_element.text.strip() if location_element else None

    link_element = car.find("a")
    link = link_element["href"] if link_element else None

    car_info = {
        "name": name,
        "price": price,
        "location": location,
        "link": link
    }

    cars_data.append(car_info)


print(f"Car data on {len(cars_data)} listings found. Displaying first 10 listings found:")

for car in cars_data[:10]:
    print(car)