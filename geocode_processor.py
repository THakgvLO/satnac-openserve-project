import csv
import json
import time
import requests
import sys
import os  # <-- added import for os

# --- CONFIGURATION ---
OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY")  # <-- use env var
if not OPENCAGE_API_KEY:
    print("[FATAL] OPENCAGE_API_KEY not set in environment.")
    sys.exit(1)

INPUT_CSV_FILE = "openserve_property_sites.csv"
OUTPUT_CSV_FILE = "Openserve_Sites_Geocoded.csv"
COUNTRY_CODE = "za"  # Targeting South Africa for better accuracy
RATE_LIMIT_DELAY = 1.1 # Delay in seconds (OpenCage limit is ~1 request per second)

# Required columns
REQUIRED_LAT_NAME = 'LAT Trimmed'
REQUIRED_LON_NAME = 'LON Trimmed'

# These variables will hold the EXACT column names found in the CSV, 
# which might include hidden spaces, to ensure DictReader works correctly.
ACTUAL_LAT_COLUMN = None
ACTUAL_LON_COLUMN = None

# --- OPENCAGE API CALL FUNCTION ---
def reverse_geocode(lat, lon):
    """
    Performs a reverse geocoding API call to OpenCage.
    """
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {
        'q': f"{lat},{lon}",
        'key': OPENCAGE_API_KEY,
        'pretty': 0,
        'limit': 1,
        'no_annotations': 1,
        'language': 'en',
        'countrycode': COUNTRY_CODE
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        
        if data and data.get('results'):
            result = data['results'][0]
            components = result.get('components', {})
            
            formatted_address = result.get('formatted', 'N/A')
            city = components.get('city', components.get('town', components.get('village', 'N/A')))
            suburb = components.get('suburb', 'N/A')
            postcode = components.get('postcode', 'N/A')
            
            return {
                "Formatted_Address": formatted_address,
                "City": city,
                "Suburb": suburb,
                "Postcode": postcode
            }
        
        return {
            "Formatted_Address": "No result found",
            "City": "No result found",
            "Suburb": "No result found",
            "Postcode": "No result found"
        }

    except requests.exceptions.HTTPError as e:
        print(f"\n[ERROR] HTTP Error for {lat},{lon}: {e}. Status code: {response.status_code}")
        if response.status_code == 429:
            print("Rate limit exceeded. Waiting 60 seconds before retrying.")
            time.sleep(60)
            return reverse_geocode(lat, lon) # Recursive call to retry
        return None
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Request failed for {lat},{lon}: {e}")
        return None
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred for {lat},{lon}: {e}")
        return None

# --- MAIN EXECUTION LOGIC ---
def process_sites():
    """
    Reads the input CSV, processes each row, calls the API, and writes the output.
    """
    global ACTUAL_LAT_COLUMN, ACTUAL_LON_COLUMN # We need to update these global variables
    
    print(f"Starting geocoding process. Reading from: {INPUT_CSV_FILE}")
    
    try:
        # 1. Read the input CSV data
        with open(INPUT_CSV_FILE, mode='r', encoding='utf-8') as infile:
            # Added delimiter=';'
            reader = csv.DictReader(infile, delimiter=';')
            
            original_fieldnames = list(reader.fieldnames or [])
            print(f"Headers found: {original_fieldnames}")
            
            # --- ROBUST HEADER DETECTION LOGIC ---
            
            # 1. Look for the required columns, stripping spaces from the file's headers
            for fieldname in original_fieldnames:
                stripped_name = fieldname.strip()
                
                if stripped_name == REQUIRED_LAT_NAME:
                    ACTUAL_LAT_COLUMN = fieldname # Store the name WITH any potential spaces
                elif stripped_name == REQUIRED_LON_NAME:
                    ACTUAL_LON_COLUMN = fieldname # Store the name WITH any potential spaces

            if ACTUAL_LAT_COLUMN is None or ACTUAL_LON_COLUMN is None:
                print(f"\n[FATAL ERROR] Required columns '{REQUIRED_LAT_NAME}' and '{REQUIRED_LON_NAME}' not found.")
                print("Please check the file name and verify the spelling of the column headers.")
                sys.exit(1)
            
            print(f"Success! Using actual column names: LAT='{ACTUAL_LAT_COLUMN}', LON='{ACTUAL_LON_COLUMN}'")

            data_rows = list(reader) # Load data after checking fieldnames
            
    except FileNotFoundError:
        print(f"\n[FATAL ERROR] Input file '{INPUT_CSV_FILE}' not found. Please ensure it is in the same directory.")
        sys.exit(1)

    total_rows = len(data_rows)
    print(f"Total sites found in file: {total_rows}")
    
    
    # Define the new field names for the output
    fieldnames = original_fieldnames + [
        "Formatted_Address", 
        "City", 
        "Suburb", 
        "Postcode"
    ]
    
    output_rows = []
    
    # 2. Process each row
    for i, row in enumerate(data_rows):
        print(f"Processing row {i+1}/{total_rows}...", end='\r')
        
        try:
            # We now use the stored actual column names, which is guaranteed to match the DictReader
            lat = row[ACTUAL_LAT_COLUMN]
            lon = row[ACTUAL_LON_COLUMN]
            
            # Skip if latitude or longitude is missing or non-numeric
            if not lat or not lon:
                 new_row = row.copy()
                 new_row.update({
                    "Formatted_Address": "Error: Missing LAT/LON",
                    "City": "Error: Missing LAT/LON",
                    "Suburb": "Error: Missing LAT/LON",
                    "Postcode": "Error: Missing LAT/LON"
                })
                 output_rows.append(new_row)
                 continue
            
            geocode_data = reverse_geocode(lat, lon)
            
            if geocode_data is None:
                geocode_data = {
                    "Formatted_Address": "API Request Failed",
                    "City": "API Request Failed",
                    "Suburb": "API Request Failed",
                    "Postcode": "API Request Failed"
                }
            
            new_row = row.copy()
            new_row.update(geocode_data)
            output_rows.append(new_row)
            
            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            print(f"\n[ERROR] Failed to process row {i+1}: {e}")
            new_row = row.copy()
            new_row.update({
                "Formatted_Address": "Processing Error",
                "City": "Processing Error",
                "Suburb": "Processing Error",
                "Postcode": "Processing Error"
            })
            output_rows.append(new_row)
    
    print(f"\nGeocoding complete. Writing to: {OUTPUT_CSV_FILE}")

    # 3. Write the output CSV data
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print("Success! The geocoded data has been saved.")

    except Exception as e:
        print(f"\n[FATAL ERROR] Could not write output file: {e}")

if __name__ == "__main__":
    process_sites()