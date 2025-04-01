import requests
from bs4 import BeautifulSoup
from collections import Counter

def count_films_per_year():
    # URL of the Wikipedia page
    url = "https://en.wikipedia.org/wiki/List_of_films_with_post-credits_scenes"
    
    # Send GET request to the page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all tables with class 'wikitable'
    tables = soup.find_all('table', class_='wikitable')
    
    # Dictionary to store year counts
    year_counts = Counter()
    
    # Process each table
    for table in tables:
        # Find all tr elements with an id attribute
        year_rows = table.find_all('tr', id=True)
        
        for row in year_rows:
            year = row.get('id')  # Get the id attribute (the year)
            try:
                year = int(year)  # Convert to integer
                if 1900 <= year <= 2025:  # Validate year range
                    # Find the td with rowspan in this row
                    td = row.find('td', rowspan=True)
                    if td and td.get('rowspan'):
                        count = int(td.get('rowspan'))  # Get the rowspan value
                        year_counts[year] += count
                    else:
                        # If no rowspan, assume 1 entry
                        year_counts[year] += 1
            except (ValueError, TypeError):
                # Skip if year or rowspan can't be parsed
                continue
    
    # Print results in chronological order
    print("Films with post-credits scenes per year:")
    for year in sorted(year_counts.keys()):
        print(f"{year}: {year_counts[year]}")
    
    # Print total
    total = sum(year_counts.values())
    print(f"\nTotal films: {total}")

    #any year counts that are missing, set to 0
    for year in range(min(year_counts.keys()), 2026):
        if year not in year_counts:
            year_counts[year] = 0

    #plot the data
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.bar(year_counts.keys(), year_counts.values(), color='dodgerblue')
    plt.xlabel('Year')
    plt.ylabel('Number of Films')
    plt.title('Number of Films with Post-Credits Scenes per Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Install required packages if not already installed:
    # pip install requests beautifulsoup4
    try:
        count_films_per_year()
    except Exception as e:
        print(f"An error occurred: {e}")