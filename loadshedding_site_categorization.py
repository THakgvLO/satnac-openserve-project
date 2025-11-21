import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import json
import math

# --- Configuration and Setup ---
SITE_DATA_PATH = 'Openserve_Sites_Geocoded.csv'
JSON_DATA_PATH = 'dashboard_data.json'
LOADSHEDDING_DATA_PATH = 'loadshedding_data.csv'
N_CLUSTERS = 3 # Green, Amber, Red categories
R_EARTH = 6371 # Radius of Earth in km
DEFAULT_NATIONAL_STRESS_FACTOR = 1.0 # Baseline if loadshedding data is unavailable

print(f"--- Starting Site Categorization Model ({N_CLUSTERS} Clusters) ---")

# --- Metro Definitions (Contextual Factor only) ---
METRO_CENTERS = {
    'Pretoria': (-25.7479, 28.2293),
    'Johannesburg': (-26.2041, 28.0473),
    'Cape Town': (-33.9249, 18.4241),
    'Durban': (-29.8587, 31.0218),
    'Port Elizabeth': (-33.9615, 25.6148),
    'Bloemfontein': (-29.0852, 26.2163)
}

# --- Scoring Weights ---
WEIGHTS = {
    'loadshedding': 0.25, # Grid Stability (Derived from National Data + Local Proxy)
    'security': 0.20,
    'demand': 0.20,
    'permitting': 0.10,
    'interop': 0.08,
    'physical': 0.07,
    'opCost': 0.10
}
total_weight = sum(WEIGHTS.values())
if not math.isclose(total_weight, 1.0):
    WEIGHTS = {k: v / total_weight for k, v in WEIGHTS.items()}


# --- Core Logic Functions (Loadshedding Data Processing) ---

def calculate_national_grid_stress():
    """
    Loads the Kaggle loadshedding data and calculates a National Grid Stress Index (NGSI).
    This factor scales the site-specific simulated risk based on historical severity.
    A factor > 1.0 means higher historical stress, increasing risk across all sites.
    """
    try:
        print(f"Processing national loadshedding data from: {LOADSHEDDING_DATA_PATH}...")
        df_loadshed = pd.read_csv(LOADSHEDDING_DATA_PATH)
        
        # Use ILS Usage and Manual Load Reduction (MLR) as indicators of system stress/loadshedding events
        stress_cols = ['ILS Usage', 'Manual Load_Reduction(MLR)']

        for col in stress_cols:
            if col not in df_loadshed.columns:
                print(f"WARNING: Required column '{col}' not found in loadshedding data.")
                return DEFAULT_NATIONAL_STRESS_FACTOR
            df_loadshed[col] = pd.to_numeric(df_loadshed[col], errors='coerce').fillna(0)

        # Calculate the total average stress metric across the historical period
        total_stress_average = df_loadshed[stress_cols].sum(axis=1).mean()
        
        # Scale the average stress to create a factor (1.0 = baseline, > 1.0 = high stress)
        # Normalizing against a historical average (e.g., 200) to get a scaling factor
        baseline_stress = 200
        stress_factor = max(0.8, min(1.5, total_stress_average / baseline_stress)) # Clamp between 0.8 and 1.5
        
        print(f"National Grid Stress Index (NGSI) calculated: {stress_factor:.2f}")
        return stress_factor

    except FileNotFoundError:
        print(f"WARNING: Loadshedding data file '{LOADSHEDDING_DATA_PATH}' not found. Using default stress factor.")
        return DEFAULT_NATIONAL_STRESS_FACTOR
    except Exception as e:
        print(f"ERROR processing loadshedding data: {e}. Using default stress factor.")
        return DEFAULT_NATIONAL_STRESS_FACTOR

# --- Helper Functions for Geolocation ---

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth (in km)."""
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = R_EARTH * c
    return distance_km

def calculate_metro_proximity(lat, lon):
    """Finds the distance to the closest metro center and returns a proximity score."""
    min_distance = float('inf')
    closest_metro = None
    for metro, (mlat, mlon) in METRO_CENTERS.items():
        dist = haversine(lat, lon, mlat, mlon)
        if dist < min_distance:
            min_distance = dist
            closest_metro = metro
    
    # Proximity Score: Higher for closer sites.
    proximity_score = 100 / (min_distance + 1)
    return min_distance, closest_metro, proximity_score


# --- Scoring Functions ---

def get_loadshedding_score(loadshedding_risk):
    """
    Converts the Site_Loadshedding_Risk (0=Low Risk, 100=High Risk)
    into a Score (10=Best, 0=Worst).
    """
    # Linear inverse scaling: Score = 10 - (Risk / 10)
    score = 10 - (loadshedding_risk / 10)
    return max(0, min(10, score)) # Ensure score stays between 0 and 10

def get_demand_score(demand_index):
    demand_index = demand_index if not pd.isna(demand_index) else 50
    return (demand_index / 100) * 10 # Scale demand index (0-100) to score (0-10)

def get_security_score(category):
    # Base risk category drives security assumption
    if category == 'Green': return 8.5
    if category == 'Amber': return 5.5
    if category == 'Red': return 3.0
    return 0.0

def get_permitting_score(category):
    # Base risk category drives permitting assumption
    if category == 'Green': return 8.0
    if category == 'Amber': return 6.0
    if category == 'Red': return 4.0
    return 0.0

def get_interop_score():
    # Fixed average score for interoperability
    return 6.0

def get_physical_score(category):
    # Base risk category drives physical assumption
    if category == 'Green': return 8.5
    if category == 'Amber': return 6.0
    if category == 'Red': return 4.0
    return 0.0

def get_op_cost_score(category):
    # Base risk category drives op cost assumption
    if category == 'Green': return 9.0
    if category == 'Amber': return 5.5
    if category == 'Red': return 3.0
    return 0.0

def get_proximity_hindsight_score(proximity_score):
    """
    NEW: Converts the Metro_Proximity_Score into a low-influence score (0-5).
    This acts as the 'hindsight' or contextual factor as requested.
    """
    # Log scaling and scaling down to ensure low influence
    return np.log1p(proximity_score) / np.log1p(10) * 5

def classify_final_score(score):
    """Returns the final classification label based on the weighted score."""
    if score >= 7.5:
        return 'Ready' # Corresponds to the GREEN deployment strategy
    elif 5.5 <= score < 7.5:
        return 'Amber' # Corresponds to the AMBER deployment strategy
    else:
        return 'Risky' # Corresponds to the RED deployment strategy

def calculate_site_suitability_score(row):
    """
    Calculates the single weighted suitability score and returns individual factor scores
    for dashboard visualization.
    """
    # The cluster label (Base_Risk_Category) informs non-grid related risks (security, permitting, etc.)
    base_risk_cat = row['Base_Risk_Category']
    demand_index = row['Demand_Index']
    site_loadshedding_risk = row['Site_Loadshedding_Risk']
    proximity_score = row['Metro_Proximity_Score']
    
    # 1. Calculate Factor Scores (0-10)
    factor_scores = {
        # Loadshedding score derived directly from the combined risk metric
        'loadshedding': get_loadshedding_score(site_loadshedding_risk),
        
        # The 'interop' factor is now used for the Metro Proximity 'hindsight'
        'interop': get_proximity_hindsight_score(proximity_score), # Re-purposed for hindsight
        
        'security': get_security_score(base_risk_cat),
        'demand': get_demand_score(demand_index),
        'permitting': get_permitting_score(base_risk_cat),
        'physical': get_physical_score(base_risk_cat),
        'opCost': get_op_cost_score(base_risk_cat)
    }
    
    # 2. Calculate Total Weighted Score
    
    # Recalculate weights based on the actual factors used in the score calculation
    # The original weights apply, but we explicitly use the 'interop' slot for the hindsight score
    total_weighted = sum(score * WEIGHTS[key] for key, score in factor_scores.items())
    
    # 3. Classify
    final_classification = classify_final_score(total_weighted)
    
    return pd.Series([total_weighted, final_classification, factor_scores])


# 1. Load Data & Calculate National Stress
try:
    # Calculate the national factor first
    national_stress_factor = calculate_national_grid_stress()
    
    print(f"Reading site geocodes from: {SITE_DATA_PATH}")
    df_csv = pd.read_csv(SITE_DATA_PATH, usecols=['Site Identifier', 'LAT Trimmed', 'LON Trimmed'])
    
    print(f"Reading categorized data from: {JSON_DATA_PATH}")
    df_json = pd.read_json(JSON_DATA_PATH)
    
    df_json.rename(columns={'id': 'Site Identifier', 'demand_index': 'Demand_Index'}, inplace=True)
    
    df_json = df_json[['Site Identifier', 'Demand_Index', 'city', 'suburb', 'address']]

    df = pd.merge(df_csv, df_json, on='Site Identifier', how='inner')
    
    df.rename(columns={'LAT Trimmed': 'Latitude', 'LON Trimmed': 'Longitude'}, inplace=True)
    
    df['Demand_Index'] = df['Demand_Index'].fillna(50).astype(int)

    if len(df) == 0:
        raise ValueError("Merge resulted in 0 sites.")

    print(f"Successfully loaded and merged data. Total sites: {len(df)}")

except FileNotFoundError as fnfe:
    print(f"ERROR: Could not find required file: {fnfe.filename}.")
    exit()
except Exception as e:
    print(f"ERROR during data loading/merging: {e}")
    exit()

# 2. Data Preparation: Creating the PRIMARY operational risk factor
df['Latitude'] = df['Latitude'].astype(float)
df['Longitude'] = df['Longitude'].astype(float)

# --- Site Loadshedding Risk ---
# 1. Local Risk Proxy: Inverse of Demand Index (Lower demand index -> Higher local risk)
#    (100 - Demand_Index) * 0.7 gives a base risk of 0 to 70.
base_local_risk = (100 - df['Demand_Index']) * 0.7

# 2. Apply National Stress Factor: Scale the local risk by the historical national stress
df['Site_Loadshedding_Risk'] = base_local_risk * national_stress_factor + (np.random.rand(len(df)) * 20)

# Clamp the risk between a floor (20) and ceiling (100)
df['Site_Loadshedding_Risk'] = df['Site_Loadshedding_Risk'].clip(20, 100).astype(int)
print(f"Site Loadshedding Risk (Local Proxy + National Stress) created.")

# 3. Geo-Feature Calculation (Contextual Factor)
print("Calculating Geo-Features (Distance to Metro & Proximity Score)...")
df[['Distance_to_Metro_KM', 'Closest_Metro', 'Metro_Proximity_Score']] = df.apply(
    lambda row: pd.Series(calculate_metro_proximity(row['Latitude'], row['Longitude'])), axis=1
)

# 4. Features for Clustering (Focus on operational risk)
# Clustering uses the Site_Loadshedding_Risk (the primary operational risk) and Demand.
features = df[['Site_Loadshedding_Risk', 'Demand_Index']].copy()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print("Features scaled (Loadshedding Risk and Demand Index).")

# 5. Clustering
print(f"Applying K-Means with {N_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(features_scaled)
print("Clustering complete.")

# 6. Labels (Order by lowest Loadshedding Risk)
cluster_score_mapping = df.groupby('Cluster')[['Site_Loadshedding_Risk']].mean()
cluster_order = cluster_score_mapping.sort_values(by='Site_Loadshedding_Risk', ascending=True).index

# Map: Lowest Risk -> Green, Medium -> Amber, Highest -> Red
risk_labels = ['Green', 'Amber', 'Red']
cluster_to_label = {cluster: label for cluster, label in zip(cluster_order, risk_labels)}
df['Base_Risk_Category'] = df['Cluster'].map(cluster_to_label)
print(f"Clusters mapped to Base Risk (based on Loadshedding/Demand): {cluster_to_label}")

# 7. Final Categorization for non-loadshedding factors
df['Adjusted_Risk_Category'] = df['Base_Risk_Category']
print("Base Risk Category finalized.")


# 8. Weighted scoring (Loadshedding score uses the derived risk data directly)
print("Applying rigorous weighted scoring...")
df[['Total_Weighted_Score', 'Final_Classification', 'Factor_Scores']] = df.apply(
    lambda row: calculate_site_suitability_score(row),
    axis=1
)
print("Weighted scoring complete.")

# Output
output_columns = [
    'Site Identifier', 'Latitude', 'Longitude', 'city', 'suburb', 'address',
    'Demand_Index', 'Site_Loadshedding_Risk', # Show the calculated risk
    'Distance_to_Metro_KM', 'Closest_Metro', 'Metro_Proximity_Score',
    'Base_Risk_Category', 'Adjusted_Risk_Category',
    'Total_Weighted_Score', 'Final_Classification', 'Factor_Scores'
]
results_df = df[df.columns.intersection(output_columns)].copy()

# Serialize Factor_Scores dictionary to JSON string for easy transfer/storage
results_df['Factor_Scores'] = results_df['Factor_Scores'].apply(json.dumps)

print("\n--- Categorization Summary ---")
print(results_df.groupby('Final_Classification').size().to_frame(name='Count').to_markdown())

output_file = 'categorized_sites_output.csv'
results_df.to_csv(output_file, index=False)
print(f"\nModel run finished. Output file saved: {output_file}")