#!/usr/bin/env python3
"""Fix coordinates by extracting real data from NetCDF files."""

import os
import sqlite3
import json
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import netCDF4 as nc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_coordinates_from_netcdf(file_path):
    """Extract coordinates from a single NetCDF file."""
    try:
        with nc.Dataset(file_path, 'r') as dataset:
            # Extract coordinates
            lat = float(dataset.variables['latitude'][0]) if 'latitude' in dataset.variables else None
            lon = float(dataset.variables['longitude'][0]) if 'longitude' in dataset.variables else None
            
            # Extract platform number
            platform = str(dataset.variables['platform_number'][0]).strip() if 'platform_number' in dataset.variables else None
            
            # Extract cycle number
            cycle = int(dataset.variables['cycle_number'][0]) if 'cycle_number' in dataset.variables else None
            
            # Extract Julian day
            juld = float(dataset.variables['juld'][0]) if 'juld' in dataset.variables else None
            
            # Convert Julian day to date
            if juld and juld > 0:
                try:
                    from datetime import datetime, timedelta
                    base_date = datetime(1950, 1, 1)
                    profile_date = base_date + timedelta(days=juld)
                    date_str = profile_date.strftime('%Y-%m-%d')
                except:
                    # Fallback: extract year from file path
                    year = int(file_path.split('/')[-3])
                    date_str = f"{year}-01-01"
            else:
                # Fallback: extract year from file path
                year = int(file_path.split('/')[-3])
                date_str = f"{year}-01-01"
            
            return {
                'platform_number': platform,
                'cycle_number': cycle,
                'latitude': lat,
                'longitude': lon,
                'date': date_str
            }
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

def fix_coordinates():
    """Fix coordinates by matching NetCDF files to database records."""
    
    # Connect to database
    db_path = "data/processed/argo_data_full.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all profiles that need coordinate fixes
    cursor.execute("SELECT id, platform_number, cycle_number, date FROM argo_profiles WHERE latitude = 0 OR longitude = 0")
    profiles_to_fix = cursor.fetchall()
    
    logger.info(f"Found {len(profiles_to_fix)} profiles that need coordinate fixes")
    
    # Find all NetCDF files
    raw_data_dir = Path("raw_data/argo/indian")
    netcdf_files = list(raw_data_dir.rglob("*.nc"))
    
    logger.info(f"Found {len(netcdf_files)} NetCDF files")
    
    # Create a mapping from platform+cycle to file path
    file_mapping = {}
    for file_path in netcdf_files:
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                platform = str(dataset.variables['platform_number'][0]).strip() if 'platform_number' in dataset.variables else None
                cycle = int(dataset.variables['cycle_number'][0]) if 'cycle_number' in dataset.variables else None
                
                if platform and cycle is not None:
                    key = f"{platform}_{cycle}"
                    file_mapping[key] = file_path
        except:
            continue
    
    logger.info(f"Created mapping for {len(file_mapping)} files")
    
    # Fix coordinates
    fixed_count = 0
    for profile_id, platform, cycle, date in profiles_to_fix:
        key = f"{platform}_{cycle}"
        
        if key in file_mapping:
            file_path = file_mapping[key]
            coords = extract_coordinates_from_netcdf(file_path)
            
            if coords and coords['latitude'] and coords['longitude']:
                cursor.execute("""
                    UPDATE argo_profiles 
                    SET latitude = ?, longitude = ?, date = ?
                    WHERE id = ?
                """, (coords['latitude'], coords['longitude'], coords['date'], profile_id))
                fixed_count += 1
                
                if fixed_count % 100 == 0:
                    logger.info(f"Fixed {fixed_count} profiles...")
    
    # Commit changes
    conn.commit()
    conn.close()
    
    logger.info(f"✅ Fixed coordinates for {fixed_count} profiles")
    
    # Verify results
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM argo_profiles WHERE latitude != 0 AND longitude != 0")
    valid_coords = cursor.fetchone()[0]
    
    cursor.execute("SELECT MIN(latitude), MAX(latitude), MIN(longitude), MAX(longitude) FROM argo_profiles WHERE latitude != 0 AND longitude != 0")
    lat_min, lat_max, lon_min, lon_max = cursor.fetchone()
    
    cursor.execute("SELECT SUBSTR(date, 1, 4) as year, COUNT(*) FROM argo_profiles WHERE latitude != 0 AND longitude != 0 GROUP BY year ORDER BY year")
    year_dist = cursor.fetchall()
    
    conn.close()
    
    logger.info(f"📊 Final Results:")
    logger.info(f"   Valid coordinates: {valid_coords}")
    if valid_coords > 0:
        logger.info(f"   Latitude range: {lat_min:.2f} to {lat_max:.2f}")
        logger.info(f"   Longitude range: {lon_min:.2f} to {lon_max:.2f}")
    logger.info(f"   Year distribution:")
    for year, count in year_dist:
        logger.info(f"     {year}: {count} profiles")

if __name__ == "__main__":
    fix_coordinates()


