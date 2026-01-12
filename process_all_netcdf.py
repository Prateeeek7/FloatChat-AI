#!/usr/bin/env python3
"""Process all NetCDF files to extract real coordinates and update database."""

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

def process_netcdf_file(file_path):
    """Extract data from a single NetCDF file."""
    try:
        with nc.Dataset(file_path, 'r') as dataset:
            # Extract coordinates
            lat = float(dataset.variables['latitude'][0]) if 'latitude' in dataset.variables else 0.0
            lon = float(dataset.variables['longitude'][0]) if 'longitude' in dataset.variables else 0.0
            
            # Extract date
            juld = float(dataset.variables['juld'][0]) if 'juld' in dataset.variables else 0.0
            juld_qc = int(dataset.variables['juld_qc'][0]) if 'juld_qc' in dataset.variables else 0
            
            # Extract platform number
            platform = str(dataset.variables['platform_number'][0]).strip() if 'platform_number' in dataset.variables else 'unknown'
            
            # Extract cycle number
            cycle = int(dataset.variables['cycle_number'][0]) if 'cycle_number' in dataset.variables else 0
            
            # Extract pressure, temperature, salinity
            pres = dataset.variables['pres'][:] if 'pres' in dataset.variables else []
            temp = dataset.variables['temp'][:] if 'temp' in dataset.variables else []
            psal = dataset.variables['psal'][:] if 'psal' in dataset.variables else []
            
            # Convert to lists and handle masked arrays
            if hasattr(pres, 'filled'):
                pres_list = pres.filled(0).tolist()
            elif hasattr(pres, 'tolist'):
                pres_list = pres.tolist()
            else:
                pres_list = list(pres) if pres is not None else []
                
            if hasattr(temp, 'filled'):
                temp_list = temp.filled(0).tolist()
            elif hasattr(temp, 'tolist'):
                temp_list = temp.tolist()
            else:
                temp_list = list(temp) if temp is not None else []
                
            if hasattr(psal, 'filled'):
                psal_list = psal.filled(0).tolist()
            elif hasattr(psal, 'tolist'):
                psal_list = psal.tolist()
            else:
                psal_list = list(psal) if psal is not None else []
            
            # Filter out invalid values
            pres_list = [p for p in pres_list if p > 0 and p < 2000]
            temp_list = [t for t in temp_list if t > -5 and t < 40]
            psal_list = [s for s in psal_list if s > 0 and s < 50]
            
            # Pad to 100 levels
            while len(pres_list) < 100:
                pres_list.append(0)
            while len(temp_list) < 100:
                temp_list.append(0)
            while len(psal_list) < 100:
                psal_list.append(0)
            
            # Truncate to 100 levels
            pres_list = pres_list[:100]
            temp_list = temp_list[:100]
            psal_list = psal_list[:100]
            
            # Convert Julian day to date
            try:
                from datetime import datetime, timedelta
                base_date = datetime(1950, 1, 1)
                profile_date = base_date + timedelta(days=juld)
                date_str = profile_date.strftime('%Y-%m-%d')
            except:
                # Fallback: extract year from file path
                year = int(file_path.split('/')[-3])
                date_str = f"{year}-01-01"
            
            return {
                'platform_number': platform,
                'cycle_number': cycle,
                'latitude': lat,
                'longitude': lon,
                'date': date_str,
                'pressure_data': json.dumps(pres_list),
                'temperature_data': json.dumps(temp_list),
                'salinity_data': json.dumps(psal_list),
                'max_depth': max(pres_list) if pres_list else 0,
                'data_quality': 'A' if juld_qc == 1 else 'B'
            }
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

def process_all_netcdf_files():
    """Process all NetCDF files and update database."""
    
    # Find all NetCDF files
    raw_data_dir = Path("raw_data/argo/indian")
    netcdf_files = list(raw_data_dir.rglob("*.nc"))
    
    logger.info(f"Found {len(netcdf_files)} NetCDF files to process")
    
    # Database setup
    db_path = "data/processed/argo_data_full.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear existing data
    logger.info("Clearing existing data...")
    cursor.execute("DELETE FROM argo_profiles")
    conn.commit()
    
    # Process files
    processed_count = 0
    error_count = 0
    
    for i, file_path in enumerate(netcdf_files):
        if i % 100 == 0:
            logger.info(f"Processing file {i}/{len(netcdf_files)}: {file_path}")
        
        profile_data = process_netcdf_file(file_path)
        
        if profile_data:
            try:
                cursor.execute("""
                    INSERT INTO argo_profiles 
                    (platform_number, cycle_number, latitude, longitude, 
                     date, pressure_data, temperature_data, salinity_data, 
                     depth_range, data_quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile_data['platform_number'],
                    profile_data['cycle_number'],
                    profile_data['latitude'],
                    profile_data['longitude'],
                    profile_data['date'],
                    profile_data['pressure_data'],
                    profile_data['temperature_data'],
                    profile_data['salinity_data'],
                    f"0-{profile_data['max_depth']}",
                    profile_data['data_quality']
                ))
                processed_count += 1
            except Exception as e:
                logger.error(f"Database error for {file_path}: {str(e)}")
                error_count += 1
        else:
            error_count += 1
    
    # Commit and close
    conn.commit()
    conn.close()
    
    logger.info(f"✅ Processing complete!")
    logger.info(f"✅ Successfully processed: {processed_count} files")
    logger.info(f"❌ Errors: {error_count} files")
    
    # Verify results
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get statistics
    cursor.execute("SELECT COUNT(*) FROM argo_profiles")
    total_profiles = cursor.fetchone()[0]
    
    cursor.execute("SELECT MIN(latitude), MAX(latitude), MIN(longitude), MAX(longitude) FROM argo_profiles WHERE latitude != 0 AND longitude != 0")
    coord_result = cursor.fetchone()
    lat_min, lat_max, lon_min, lon_max = coord_result if coord_result[0] is not None else (0, 0, 0, 0)
    
    cursor.execute("SELECT COUNT(*) FROM argo_profiles WHERE latitude != 0 AND longitude != 0")
    valid_coords = cursor.fetchone()[0]
    
    cursor.execute("SELECT SUBSTR(date, 1, 4) as year, COUNT(*) FROM argo_profiles GROUP BY year ORDER BY year")
    year_dist = cursor.fetchall()
    
    conn.close()
    
    logger.info(f"📊 Final Statistics:")
    logger.info(f"   Total profiles: {total_profiles}")
    logger.info(f"   Valid coordinates: {valid_coords}")
    if valid_coords > 0:
        logger.info(f"   Latitude range: {lat_min:.2f} to {lat_max:.2f}")
        logger.info(f"   Longitude range: {lon_min:.2f} to {lon_max:.2f}")
    else:
        logger.info(f"   No valid coordinates found")
    logger.info(f"   Year distribution:")
    for year, count in year_dist:
        logger.info(f"     {year}: {count} profiles")

if __name__ == "__main__":
    process_all_netcdf_files()