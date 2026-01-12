#!/usr/bin/env python3
"""Load the latest processed ARGO data into the database."""

import json
import sqlite3
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_latest_data():
    """Load the latest processed data into the database."""
    
    # Database path
    db_path = "data/processed/argo_data_full.db"
    
    # Load metadata
    metadata_path = "processed_data_all_latest/metadata_all_latest.json"
    logger.info(f"Loading metadata from {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Found {len(metadata)} profiles in metadata")
    
    # Load numpy arrays
    temp_data = np.load("processed_data_all_latest/temperature_profiles_all_latest.npy")
    sal_data = np.load("processed_data_all_latest/salinity_profiles_all_latest.npy")
    pres_data = np.load("processed_data_all_latest/pressure_profiles_all_latest.npy")
    
    logger.info(f"Loaded arrays - Temp: {temp_data.shape}, Sal: {sal_data.shape}, Pres: {pres_data.shape}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear existing data to avoid duplicates
    logger.info("Clearing existing data...")
    cursor.execute("DELETE FROM argo_profiles")
    conn.commit()
    
    # Insert new data
    logger.info("Inserting new data...")
    
    for i, meta in enumerate(metadata):
        if i % 1000 == 0:
            logger.info(f"Processing profile {i}/{len(metadata)}")
        
        # Get the corresponding data arrays
        temp_profile = temp_data[i].tolist()
        sal_profile = sal_data[i].tolist()
        pres_profile = pres_data[i].tolist()
        
        # Convert to JSON strings
        temp_json = json.dumps(temp_profile)
        sal_json = json.dumps(sal_profile)
        pres_json = json.dumps(pres_profile)
        
        # Calculate max depth from pressure data
        max_depth = max(pres_profile) if pres_profile else 0
        
        # Create date string
        date_str = f"{meta['year']}-{meta['month']:02d}-01"
        
        # Insert into database
        cursor.execute("""
            INSERT INTO argo_profiles 
            (platform_number, cycle_number, latitude, longitude, 
             date, pressure_data, temperature_data, salinity_data, 
             depth_range, data_quality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            meta['platform_number'],
            meta['cycle_number'],
            meta['latitude'],
            meta['longitude'],
            date_str,
            pres_json,
            temp_json,
            sal_json,
            f"0-{max_depth}",  # Depth range format
            'A'  # Default quality
        ))
    
    # Commit and close
    conn.commit()
    conn.close()
    
    logger.info("✅ Successfully loaded all data into database!")
    
    # Verify the data
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM argo_profiles")
    count = cursor.fetchone()[0]
    conn.close()
    
    logger.info(f"✅ Database now contains {count} profiles")

if __name__ == "__main__":
    load_latest_data()
