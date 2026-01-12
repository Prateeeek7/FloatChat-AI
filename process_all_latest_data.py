#!/usr/bin/env python3
"""
Process ALL Latest ARGO Data - Improved processing for maximum data extraction
Handles various NetCDF formats and provides detailed diagnostics
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
import gc
from datetime import datetime
import json
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('argo_all_latest_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImprovedLatestDataProcessor:
    """Improved processor for maximum data extraction from ARGO NetCDF files."""
    
    def __init__(self, raw_data_dir: str = "raw_data/argo/indian", 
                 output_dir: str = "processed_data_all_latest"):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = 100
        self.temperature_profiles = []
        self.salinity_profiles = []
        self.pressure_profiles = []
        self.metadata = []
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'no_temp_sal': 0,
            'insufficient_data': 0,
            'corrupted_files': 0,
            'variable_issues': 0
        }
        
        # Variable name mappings for different NetCDF formats
        self.temp_vars = ['temp', 'TEMP', 'temp_adjusted', 'TEMP_ADJUSTED', 'TEMP_CORRECTED', 'TEMP_QC', 'T']
        self.sal_vars = ['psal', 'PSAL', 'psal_adjusted', 'PSAL_ADJUSTED', 'PSAL_CORRECTED', 'PSAL_QC', 'S']
        self.pres_vars = ['pres', 'PRES', 'pres_adjusted', 'PRES_ADJUSTED', 'PRES_CORRECTED', 'PRES_QC', 'P', 'PRESSURE']
        
    def find_variables(self, ds: xr.Dataset) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Find temperature, salinity, and pressure variables in the dataset."""
        temp_var = None
        sal_var = None
        pres_var = None
        
        # Find temperature variable
        for var in self.temp_vars:
            if var in ds.variables:
                temp_var = var
                break
        
        # Find salinity variable
        for var in self.sal_vars:
            if var in ds.variables:
                sal_var = var
                break
        
        # Find pressure variable
        for var in self.pres_vars:
            if var in ds.variables:
                pres_var = var
                break
        
        return temp_var, sal_var, pres_var
    
    def process_netcdf_file(self, file_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
        """Process a single NetCDF file with improved error handling."""
        try:
            with xr.open_dataset(file_path) as ds:
                # Find variables
                temp_var, sal_var, pres_var = self.find_variables(ds)
                
                if temp_var is None or sal_var is None:
                    self.stats['no_temp_sal'] += 1
                    return None
                
                # Extract data
                temp_data = ds[temp_var].values
                sal_data = ds[sal_var].values
                
                # Get pressure data
                if pres_var is not None:
                    pres_data = ds[pres_var].values
                else:
                    # Create pressure from depth (assuming 1m = 0.1 dbar)
                    pres_data = np.arange(len(temp_data)) * 0.1
                
                # Handle different data shapes
                if temp_data.ndim > 1:
                    # Take the first profile if multiple profiles
                    temp_data = temp_data[0] if temp_data.shape[0] > 0 else temp_data.flatten()
                    sal_data = sal_data[0] if sal_data.shape[0] > 0 else sal_data.flatten()
                    pres_data = pres_data[0] if pres_data.shape[0] > 0 else pres_data.flatten()
                
                # Ensure all arrays have the same length
                min_length = min(len(temp_data), len(sal_data), len(pres_data))
                if min_length < 10:  # At least 10 valid points
                    self.stats['insufficient_data'] += 1
                    return None
                
                temp_data = temp_data[:min_length]
                sal_data = sal_data[:min_length]
                pres_data = pres_data[:min_length]
                
                # Remove NaN values
                valid_mask = ~(np.isnan(temp_data) | np.isnan(sal_data))
                if np.sum(valid_mask) < 10:  # At least 10 valid points
                    self.stats['insufficient_data'] += 1
                    return None
                
                temp_clean = temp_data[valid_mask]
                sal_clean = sal_data[valid_mask]
                pres_clean = pres_data[valid_mask]
                
                # Interpolate to fixed length
                if len(temp_clean) > self.sequence_length:
                    # Downsample
                    indices = np.linspace(0, len(temp_clean)-1, self.sequence_length, dtype=int)
                    temp_profile = temp_clean[indices]
                    sal_profile = sal_clean[indices]
                    pres_profile = pres_clean[indices]
                else:
                    # Upsample
                    temp_profile = np.interp(
                        np.linspace(0, len(temp_clean)-1, self.sequence_length),
                        np.arange(len(temp_clean)),
                        temp_clean
                    )
                    sal_profile = np.interp(
                        np.linspace(0, len(sal_clean)-1, self.sequence_length),
                        np.arange(len(sal_clean)),
                        sal_clean
                    )
                    pres_profile = np.interp(
                        np.linspace(0, len(pres_clean)-1, self.sequence_length),
                        np.arange(len(pres_clean)),
                        pres_clean
                    )
                
                # Extract metadata safely
                metadata = {
                    'file_path': str(file_path),
                    'platform_number': self._safe_extract(ds, 'PLATFORM_NUMBER', 'unknown'),
                    'cycle_number': self._safe_extract(ds, 'CYCLE_NUMBER', 0),
                    'latitude': self._safe_extract(ds, 'LATITUDE', 0.0),
                    'longitude': self._safe_extract(ds, 'LONGITUDE', 0.0),
                    'julian_day': self._safe_extract(ds, 'JULD', 0.0),
                    'data_mode': self._safe_extract(ds, 'DATA_MODE', 'unknown'),
                    'year': int(file_path.parts[-3]) if len(file_path.parts) >= 3 else 0,
                    'month': int(file_path.parts[-2]) if len(file_path.parts) >= 2 else 0,
                    'temp_var': temp_var,
                    'sal_var': sal_var,
                    'pres_var': pres_var,
                    'original_length': len(temp_clean)
                }
                
                return temp_profile, sal_profile, pres_profile, metadata
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing {file_path.name}: {e}")
            self.stats['failed_files'] += 1
            return None
    
    def _safe_extract(self, ds: xr.Dataset, var_name: str, default_value):
        """Safely extract a variable from the dataset."""
        try:
            if var_name in ds.variables:
                var = ds[var_name]
                if hasattr(var, 'values'):
                    if var.values.size == 1:
                        val = var.values.item()
                        # Convert bytes to string if needed
                        if isinstance(val, bytes):
                            return val.decode('utf-8')
                        return val
                    else:
                        return str(var.values)
                else:
                    return str(var)
            return default_value
        except:
            return default_value
    
    def process_all_latest_data(self) -> bool:
        """Process all data from 2003-2005 with improved diagnostics."""
        logger.info("📂 Processing ALL latest ARGO data (2003-2005) with improved diagnostics...")
        
        # Process years 2003-2005
        years_to_process = [2003, 2004, 2005]
        
        for year in years_to_process:
            year_dir = self.raw_data_dir / str(year)
            if not year_dir.exists():
                logger.warning(f"⚠️ Year {year} directory not found")
                continue
                
            logger.info(f"📅 Processing year {year}...")
            
            # Process all months in the year
            for month_dir in sorted(year_dir.iterdir()):
                if month_dir.is_dir() and month_dir.name.isdigit():
                    month = int(month_dir.name)
                    
                    # Skip March 2003 and earlier
                    if year == 2003 and month <= 3:
                        continue
                    
                    logger.info(f"📁 Processing {year}/{month:02d}...")
                    
                    # Process all NetCDF files in the month
                    nc_files = list(month_dir.glob("*.nc"))
                    self.stats['total_files'] += len(nc_files)
                    
                    month_processed = 0
                    for nc_file in nc_files:
                        result = self.process_netcdf_file(nc_file)
                        
                        if result is not None:
                            temp_profile, sal_profile, pres_profile, metadata = result
                            self.temperature_profiles.append(temp_profile)
                            self.salinity_profiles.append(sal_profile)
                            self.pressure_profiles.append(pres_profile)
                            self.metadata.append(metadata)
                            self.stats['processed_files'] += 1
                            month_processed += 1
                        
                        # Memory management
                        if self.stats['processed_files'] % 500 == 0:
                            logger.info(f"📊 Processed {self.stats['processed_files']} profiles...")
                            gc.collect()
                    
                    logger.info(f"✅ {year}/{month:02d}: {month_processed}/{len(nc_files)} files processed")
        
        # Convert to numpy arrays
        if self.stats['processed_files'] > 0:
            self.temperature_profiles = np.array(self.temperature_profiles)
            self.salinity_profiles = np.array(self.salinity_profiles)
            self.pressure_profiles = np.array(self.pressure_profiles)
            
            logger.info(f"✅ Processed {self.stats['processed_files']} profiles from {self.stats['total_files']} files")
            logger.info(f"📊 Data shape: {self.temperature_profiles.shape}")
            logger.info(f"📊 Temperature range: {self.temperature_profiles.min():.2f} to {self.temperature_profiles.max():.2f}")
            logger.info(f"📊 Salinity range: {self.salinity_profiles.min():.2f} to {self.salinity_profiles.max():.2f}")
            logger.info(f"📊 Pressure range: {self.pressure_profiles.min():.2f} to {self.pressure_profiles.max():.2f}")
            
            # Print detailed statistics
            logger.info(f"📊 Processing Statistics:")
            logger.info(f"   Total files: {self.stats['total_files']}")
            logger.info(f"   Successfully processed: {self.stats['processed_files']}")
            logger.info(f"   Failed files: {self.stats['failed_files']}")
            logger.info(f"   No temp/sal variables: {self.stats['no_temp_sal']}")
            logger.info(f"   Insufficient data: {self.stats['insufficient_data']}")
            
            return True
        else:
            logger.error("❌ No valid profiles found!")
            return False
    
    def save_processed_data(self) -> bool:
        """Save processed data to files."""
        logger.info("💾 Saving processed data...")
        
        try:
            # Save numpy arrays
            np.save(self.output_dir / "temperature_profiles_all_latest.npy", self.temperature_profiles)
            np.save(self.output_dir / "salinity_profiles_all_latest.npy", self.salinity_profiles)
            np.save(self.output_dir / "pressure_profiles_all_latest.npy", self.pressure_profiles)
            
            # Save metadata as JSON
            with open(self.output_dir / "metadata_all_latest.json", 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            
            # Save detailed statistics
            stats_summary = {
                'total_profiles': len(self.temperature_profiles),
                'sequence_length': self.sequence_length,
                'processing_stats': self.stats,
                'temperature_stats': {
                    'min': float(self.temperature_profiles.min()),
                    'max': float(self.temperature_profiles.max()),
                    'mean': float(self.temperature_profiles.mean()),
                    'std': float(self.temperature_profiles.std())
                },
                'salinity_stats': {
                    'min': float(self.salinity_profiles.min()),
                    'max': float(self.salinity_profiles.max()),
                    'mean': float(self.salinity_profiles.mean()),
                    'std': float(self.salinity_profiles.std())
                },
                'pressure_stats': {
                    'min': float(self.pressure_profiles.min()),
                    'max': float(self.pressure_profiles.max()),
                    'mean': float(self.pressure_profiles.mean()),
                    'std': float(self.pressure_profiles.std())
                },
                'years_processed': list(set([m['year'] for m in self.metadata])),
                'platforms': list(set([m['platform_number'] for m in self.metadata])),
                'processed_at': datetime.now().isoformat()
            }
            
            with open(self.output_dir / "stats_all_latest.json", 'w') as f:
                json.dump(stats_summary, f, indent=2)
            
            logger.info(f"✅ Data saved to {self.output_dir}")
            logger.info(f"📊 Summary: {stats_summary['total_profiles']} profiles from years {stats_summary['years_processed']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
            return False

def main():
    """Main processing function."""
    logger.info("🚀 Starting IMPROVED Latest ARGO Data Processing...")
    logger.info("📊 Processing ALL data from after March 2003 (2003-2005) with maximum extraction!")
    
    # Initialize processor
    processor = ImprovedLatestDataProcessor()
    
    # Process all latest data
    if not processor.process_all_latest_data():
        logger.error("❌ Failed to process latest data!")
        return False
    
    # Save processed data
    if not processor.save_processed_data():
        logger.error("❌ Failed to save processed data!")
        return False
    
    logger.info("🎉 ALL latest data processing completed successfully!")
    logger.info("🏆 Maximum data extraction achieved!")
    return True

if __name__ == "__main__":
    main()
