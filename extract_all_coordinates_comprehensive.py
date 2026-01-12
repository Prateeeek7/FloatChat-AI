#!/usr/bin/env python3
"""
Comprehensive coordinate extraction from ALL NetCDF files
Process every NetCDF file and update database with valid coordinates
"""

import os
import sqlite3
import json
import logging
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_coordinates_and_metadata(nc_file):
    """Extract coordinates and metadata from NetCDF file with multiple fallback methods"""
    try:
        with Dataset(nc_file, 'r') as ds:
            # Method 1: Try standard variable names
            lat = None
            lon = None
            platform = None
            cycle = None
            date_str = None
            
            # Extract latitude
            for lat_var in ['latitude', 'LATITUDE', 'lat', 'LAT']:
                if lat_var in ds.variables:
                    lat_val = ds.variables[lat_var][0]
                    if hasattr(lat_val, 'data'):
                        lat = float(lat_val.data)
                    else:
                        lat = float(lat_val)
                    break
            
            # Extract longitude  
            for lon_var in ['longitude', 'LONGITUDE', 'lon', 'LON']:
                if lon_var in ds.variables:
                    lon_val = ds.variables[lon_var][0]
                    if hasattr(lon_val, 'data'):
                        lon = float(lon_val.data)
                    else:
                        lon = float(lon_val)
                    break
            
            # Extract platform number
            for platform_var in ['platform_number', 'PLATFORM_NUMBER', 'platform', 'PLATFORM']:
                if platform_var in ds.variables:
                    platform_val = ds.variables[platform_var][0]
                    if hasattr(platform_val, 'data'):
                        platform = str(platform_val.data)
                    else:
                        platform = str(platform_val)
                    break
            
            # Extract cycle number
            for cycle_var in ['cycle_number', 'CYCLE_NUMBER', 'cycle', 'CYCLE']:
                if cycle_var in ds.variables:
                    cycle_val = ds.variables[cycle_var][0]
                    if hasattr(cycle_val, 'data'):
                        cycle = int(cycle_val.data)
                    else:
                        cycle = int(cycle_val)
                    break
            
            # Extract date - try multiple methods
            date_str = None
            
            # Method 1: Julian day
            for juld_var in ['juld', 'JULD', 'julian_day', 'JULIAN_DAY']:
                if juld_var in ds.variables:
                    try:
                        juld_val = ds.variables[juld_var][0]
                        if hasattr(juld_val, 'data'):
                            juld = float(juld_val.data)
                        else:
                            juld = float(juld_val)
                        
                        if juld > 0 and juld < 100000:  # Reasonable range
                            base_date = datetime(1950, 1, 1)
                            date = base_date + timedelta(days=juld)
                            date_str = date.strftime('%Y-%m-%d')
                            break
                    except:
                        continue
            
            # Method 2: Extract from filename if Julian day failed
            if not date_str:
                try:
                    year = nc_file.parent.parent.name
                    month = nc_file.parent.name
                    date_str = f"{year}-{month.zfill(2)}-01"
                except:
                    date_str = "1999-01-01"
            
            # Set defaults if still None
            if lat is None:
                lat = 0.0
            if lon is None:
                lon = 0.0
            if platform is None:
                platform = "unknown"
            if cycle is None:
                cycle = 0
            
            return lat, lon, platform, cycle, date_str
            
    except Exception as e:
        logger.error(f"Error processing {nc_file}: {e}")
        return 0.0, 0.0, "unknown", 0, "1999-01-01"

def extract_ocean_data_improved(nc_file):
    """Extract temperature, salinity, and pressure data with improved handling"""
    try:
        with Dataset(nc_file, 'r') as ds:
            # Extract pressure data - try multiple variable names
            pres_data = []
            for pres_var in ['pres', 'PRES', 'pressure', 'PRESSURE', 'depth', 'DEPTH']:
                if pres_var in ds.variables:
                    pres = ds.variables[pres_var][:]
                    if hasattr(pres, 'filled'):
                        pres_data = pres.filled(0).tolist()
                    elif hasattr(pres, 'tolist'):
                        pres_data = pres.tolist()
                    else:
                        pres_data = list(pres)
                    break
            
            # Extract temperature data - try multiple variable names
            temp_data = []
            for temp_var in ['temp', 'TEMP', 'temperature', 'TEMPERATURE']:
                if temp_var in ds.variables:
                    temp = ds.variables[temp_var][:]
                    if hasattr(temp, 'filled'):
                        temp_data = temp.filled(0).tolist()
                    elif hasattr(temp, 'tolist'):
                        temp_data = temp.tolist()
                    else:
                        temp_data = list(temp)
                    break
            
            # Extract salinity data - try multiple variable names
            sal_data = []
            for sal_var in ['psal', 'PSAL', 'salinity', 'SALINITY', 'sal', 'SAL']:
                if sal_var in ds.variables:
                    psal = ds.variables[sal_var][:]
                    if hasattr(psal, 'filled'):
                        sal_data = psal.filled(0).tolist()
                    elif hasattr(psal, 'tolist'):
                        sal_data = psal.tolist()
                    else:
                        sal_data = list(psal)
                    break
            
            # Flatten nested arrays and filter valid values
            def flatten_and_filter(data, min_val, max_val):
                result = []
                for item in data:
                    if isinstance(item, (int, float)) and item is not None and min_val < item < max_val:
                        result.append(item)
                    elif isinstance(item, list):
                        for sub_item in item:
                            if isinstance(sub_item, (int, float)) and sub_item is not None and min_val < sub_item < max_val:
                                result.append(sub_item)
                return result
            
            valid_pres = flatten_and_filter(pres_data, 0, 10000)
            valid_temp = flatten_and_filter(temp_data, -5, 40)
            valid_sal = flatten_and_filter(sal_data, 0, 50)
            
            return valid_pres, valid_temp, valid_sal
            
    except Exception as e:
        logger.error(f"Error extracting ocean data from {nc_file}: {e}")
        return [], [], []

def process_all_netcdf_files():
    """Process ALL NetCDF files and update database with comprehensive data"""
    
    db_path = "data/processed/argo_data_full.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear existing data to start fresh
    cursor.execute("DELETE FROM argo_profiles")
    conn.commit()
    logger.info("Cleared existing database to start fresh")
    
    # Process all NetCDF files from all years
    years = ['1999', '2000', '2001', '2002', '2003', '2004', '2005']
    total_processed = 0
    valid_coords = 0
    profiles_added = 0
    
    for year in years:
        year_path = Path(f"raw_data/argo/indian/{year}")
        if not year_path.exists():
            logger.warning(f"Year {year} directory not found")
            continue
            
        logger.info(f"Processing year {year}...")
        year_processed = 0
        year_valid = 0
        
        # Get all NetCDF files for this year
        nc_files = list(year_path.rglob("*.nc"))
        logger.info(f"Found {len(nc_files)} files in {year}")
        
        for nc_file in nc_files:
            try:
                # Extract coordinates and metadata
                lat, lon, platform, cycle, date_str = extract_coordinates_and_metadata(nc_file)
                
                # Extract ocean data
                pres_data, temp_data, sal_data = extract_ocean_data_improved(nc_file)
                
                # Only process if we have some data
                if len(pres_data) > 0 or len(temp_data) > 0 or len(sal_data) > 0:
                    # Convert to JSON strings
                    pres_json = json.dumps(pres_data)
                    temp_json = json.dumps(temp_data)
                    sal_json = json.dumps(sal_data)
                    
                    # Calculate depth range
                    max_depth = max(pres_data) if pres_data else 0
                    depth_range = f"0-{max_depth:.0f}"
                    
                    # Insert into database
                    cursor.execute("""
                        INSERT INTO argo_profiles 
                        (platform_number, cycle_number, latitude, longitude, 
                         date, pressure_data, temperature_data, salinity_data, 
                         depth_range, data_quality)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        platform, cycle, lat, lon, date_str,
                        pres_json, temp_json, sal_json, depth_range, 'A'
                    ))
                    
                    profiles_added += 1
                    if lat != 0.0 and lon != 0.0:
                        year_valid += 1
                        valid_coords += 1
                
                year_processed += 1
                total_processed += 1
                
                if total_processed % 500 == 0:
                    logger.info(f"Processed {total_processed} files, added {profiles_added} profiles...")
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Error processing {nc_file}: {e}")
                continue
        
        logger.info(f"Year {year}: {year_processed} files processed, {year_valid} profiles with valid coordinates")
        conn.commit()
    
    # Final statistics
    cursor.execute("SELECT COUNT(*) FROM argo_profiles")
    total_profiles = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM argo_profiles WHERE latitude != 0 AND longitude != 0")
    total_valid = cursor.fetchone()[0]
    
    logger.info(f"Processing complete!")
    logger.info(f"Total files processed: {total_processed}")
    logger.info(f"Total profiles in database: {total_profiles}")
    logger.info(f"Profiles with valid coordinates: {total_valid}")
    logger.info(f"Profiles added: {profiles_added}")
    
    conn.close()
    return total_profiles, total_valid

if __name__ == "__main__":
    logger.info("Starting comprehensive processing of ALL NetCDF files...")
    total_profiles, valid_coords = process_all_netcdf_files()
    logger.info(f"Processing complete! Total profiles: {total_profiles}, Valid coordinates: {valid_coords}")


