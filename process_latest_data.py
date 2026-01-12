#!/usr/bin/env python3
"""
Process Latest ARGO Data - Process raw NetCDF files from 2003-2005
Creates training data for the latest models
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
import gc
from datetime import datetime
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('argo_latest_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LatestDataProcessor:
    """Process latest ARGO data from 2003-2005 for training."""
    
    def __init__(self, raw_data_dir: str = "raw_data/argo/indian", 
                 output_dir: str = "processed_data_latest"):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = 100
        self.temperature_profiles = []
        self.salinity_profiles = []
        self.pressure_profiles = []
        self.metadata = []
        
    def process_netcdf_file(self, file_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
        """Process a single NetCDF file and extract profiles."""
        try:
            with xr.open_dataset(file_path) as ds:
                # Extract temperature and salinity data
                if 'TEMP' in ds.variables and 'PSAL' in ds.variables:
                    temp = ds['TEMP'].values
                    sal = ds['PSAL'].values
                    
                    # Get pressure if available, otherwise create depth-based pressure
                    if 'PRES' in ds.variables:
                        pres = ds['PRES'].values
                    else:
                        # Create pressure from depth (assuming 1m = 0.1 dbar)
                        pres = np.arange(len(temp)) * 0.1
                    
                    # Remove NaN values and ensure same length
                    valid_mask = ~(np.isnan(temp) | np.isnan(sal))
                    if np.sum(valid_mask) > 10:  # At least 10 valid points
                        temp_clean = temp[valid_mask]
                        sal_clean = sal[valid_mask]
                        pres_clean = pres[valid_mask]
                        
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
                        
                        # Extract metadata
                        metadata = {
                            'file_path': str(file_path),
                            'platform_number': str(ds.get('PLATFORM_NUMBER', 'unknown').values) if hasattr(ds.get('PLATFORM_NUMBER', 'unknown'), 'values') else str(ds.get('PLATFORM_NUMBER', 'unknown')),
                            'cycle_number': int(ds.get('CYCLE_NUMBER', 0).values) if hasattr(ds.get('CYCLE_NUMBER', 0), 'values') else int(ds.get('CYCLE_NUMBER', 0)),
                            'latitude': float(ds.get('LATITUDE', 0).values) if hasattr(ds.get('LATITUDE', 0), 'values') else float(ds.get('LATITUDE', 0)),
                            'longitude': float(ds.get('LONGITUDE', 0).values) if hasattr(ds.get('LONGITUDE', 0), 'values') else float(ds.get('LONGITUDE', 0)),
                            'julian_day': float(ds.get('JULD', 0).values) if hasattr(ds.get('JULD', 0), 'values') else float(ds.get('JULD', 0)),
                            'data_mode': str(ds.get('DATA_MODE', 'unknown').values) if hasattr(ds.get('DATA_MODE', 'unknown'), 'values') else str(ds.get('DATA_MODE', 'unknown')),
                            'year': int(file_path.parts[-3]) if len(file_path.parts) >= 3 else 0,
                            'month': int(file_path.parts[-2]) if len(file_path.parts) >= 2 else 0
                        }
                        
                        return temp_profile, sal_profile, pres_profile, metadata
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing {file_path.name}: {e}")
            return None
    
    def process_latest_data(self) -> bool:
        """Process all data from 2003-2005."""
        logger.info("📂 Processing latest ARGO data (2003-2005)...")
        
        # Process years 2003-2005
        years_to_process = [2003, 2004, 2005]
        total_files = 0
        processed_files = 0
        
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
                    total_files += len(nc_files)
                    
                    for nc_file in nc_files:
                        result = self.process_netcdf_file(nc_file)
                        
                        if result is not None:
                            temp_profile, sal_profile, pres_profile, metadata = result
                            self.temperature_profiles.append(temp_profile)
                            self.salinity_profiles.append(sal_profile)
                            self.pressure_profiles.append(pres_profile)
                            self.metadata.append(metadata)
                            processed_files += 1
                        
                        # Memory management
                        if processed_files % 1000 == 0:
                            logger.info(f"📊 Processed {processed_files} profiles...")
                            gc.collect()
        
        # Convert to numpy arrays
        if processed_files > 0:
            self.temperature_profiles = np.array(self.temperature_profiles)
            self.salinity_profiles = np.array(self.salinity_profiles)
            self.pressure_profiles = np.array(self.pressure_profiles)
            
            logger.info(f"✅ Processed {processed_files} profiles from {total_files} files")
            logger.info(f"📊 Data shape: {self.temperature_profiles.shape}")
            logger.info(f"📊 Temperature range: {self.temperature_profiles.min():.2f} to {self.temperature_profiles.max():.2f}")
            logger.info(f"📊 Salinity range: {self.salinity_profiles.min():.2f} to {self.salinity_profiles.max():.2f}")
            logger.info(f"📊 Pressure range: {self.pressure_profiles.min():.2f} to {self.pressure_profiles.max():.2f}")
            
            return True
        else:
            logger.error("❌ No valid profiles found!")
            return False
    
    def save_processed_data(self) -> bool:
        """Save processed data to files."""
        logger.info("💾 Saving processed data...")
        
        try:
            # Save numpy arrays
            np.save(self.output_dir / "temperature_profiles_latest.npy", self.temperature_profiles)
            np.save(self.output_dir / "salinity_profiles_latest.npy", self.salinity_profiles)
            np.save(self.output_dir / "pressure_profiles_latest.npy", self.pressure_profiles)
            
            # Save metadata as JSON
            import json
            with open(self.output_dir / "metadata_latest.json", 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            
            # Save summary statistics
            summary = {
                'total_profiles': len(self.temperature_profiles),
                'sequence_length': self.sequence_length,
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
            
            with open(self.output_dir / "summary_latest.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"✅ Data saved to {self.output_dir}")
            logger.info(f"📊 Summary: {summary['total_profiles']} profiles from years {summary['years_processed']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
            return False

def main():
    """Main processing function."""
    logger.info("🚀 Starting Latest ARGO Data Processing...")
    logger.info("📊 Processing data from after March 2003 (2003-2005)!")
    
    # Initialize processor
    processor = LatestDataProcessor()
    
    # Process latest data
    if not processor.process_latest_data():
        logger.error("❌ Failed to process latest data!")
        return False
    
    # Save processed data
    if not processor.save_processed_data():
        logger.error("❌ Failed to save processed data!")
        return False
    
    logger.info("🎉 Latest data processing completed successfully!")
    logger.info("🏆 Ready for training with latest data!")
    return True

if __name__ == "__main__":
    main()
