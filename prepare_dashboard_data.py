import csv
import json
import sys

# --- CONFIGURATION ---
INPUT_CSV_FILE = "Openserve_Sites_Geocoded.csv"
OUTPUT_JSON_FILE = "dashboard_data.json"

# Thresholds for colour-coding (These are placeholders that need manual assignment)
LOADSHEDDING_CATEGORIES = {
    'GREEN': 'Max 4 hours/month',
    'AMBER': 'Max 8 hours/month',
    'RED': '24 hours or more/month'
}

def filter_and_export():
    """
    Loads geocoded data, filters for Openserve Property sites, and exports
    a clean JSON object with a placeholder loadshedding category for manual update.
    """
    
    print(f"Starting data preparation. Reading from: {INPUT_CSV_FILE}")
    
    try:
        # 1. Load the CSV data
        with open(INPUT_CSV_FILE, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            data_rows = list(reader)
            
    except FileNotFoundError:
        print(f"\n[FATAL ERROR] Input file '{INPUT_CSV_FILE}' not found. Ensure the previous step completed successfully.")
        sys.exit(1)
    
    # 2. Filter for 'Openserve Property' and format data
    final_data = []
    
    for row in data_rows:
        # Check if the site is viable for charging infrastructure
        if row.get('Site Type', '').strip() == 'Openserve Property':
            
            # --- EV Demand & Range Simulation (Placeholder/Estimate) ---
            # Viable Range: Simulated 40km circle radius for map visualization
            # EV Demand: Placeholder value (0 to 100) based on City/Suburb size
            
            city = row.get('City', 'Unknown').strip()
            suburb = row.get('Suburb', 'Unknown').strip()

            # Simple heuristic for simulated EV Demand (you will replace this)
            if city in ['Johannesburg', 'Cape Town', 'Durban']:
                demand_index = 85
            elif city in ['Pretoria', 'Gqeberha', 'Bloemfontein']:
                demand_index = 60
            else:
                demand_index = 30
            
            
            final_data.append({
                "id": row.get('Site Identifier', 'N/A').strip(),
                "site_type": row.get('Site Type', '').strip(),
                "lat": float(row.get('LAT Trimmed', 0)),
                "lon": float(row.get('LON Trimmed', 0)),
                "city": city,
                "suburb": suburb,
                "address": row.get('Formatted_Address', 'N/A'),
                
                # *** PLACEHOLDER FOR MANUAL UPDATE ***
                # After running this script, you MUST manually assign the loadshedding
                # category (GREEN, AMBER, or RED) for each row in the output file.
                "loadshedding_category": "UNCATEGORIZED", # <-- REPLACE THIS!
                
                # Simulated Metrics for Dashboard
                "demand_index": demand_index,
                "charging_radius_km": 40
            })
            
    # 3. Export to JSON
    print(f"Found {len(final_data)} viable 'Openserve Property' sites.")
    
    try:
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as outfile:
            # We use indent=4 to make the JSON readable, which helps with manual edits
            json.dump(final_data, outfile, indent=4)
        
        print("\n--- NEXT STEP INSTRUCTIONS ---")
        print(f"1. A filtered list of sites is saved to: {OUTPUT_JSON_FILE}")
        print("2. You MUST manually edit this JSON file, replacing 'UNCATEGORIZED' with 'GREEN', 'AMBER', or 'RED' for each site based on your loadshedding risk assessment.")
        print("3. Once updated, open 'dashboard.html' and replace its internal 'sitesData' array with the contents of your updated dashboard_data.json.")

    except Exception as e:
        print(f"\n[FATAL ERROR] Could not write output JSON file: {e}")

if __name__ == "__main__":
    filter_and_export()