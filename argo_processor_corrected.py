#!/usr/bin/env python3
"""
Process ALL ARGO NetCDF files with CORRECT variable names (lowercase)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('argo_corrected_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CorrectedARGOProcessor:
    """Process ALL ARGO files with CORRECT variable names."""
    
    def __init__(self, data_dir: str = "raw_data/argo/indian", output_dir: str = "processed_data_all"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.temperature_profiles = []
        self.salinity_profiles = []
        self.pressure_profiles = []
        self.latitude_data = []
        self.longitude_data = []
        self.date_data = []
        self.platform_data = []
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'temp_only': 0,
            'sal_only': 0,
            'both_params': 0,
            'no_params': 0,
            'corrupted': 0,
            'temp_profiles': 0,
            'sal_profiles': 0,
            'pres_profiles': 0
        }
        
        # Fixed sequence length
        self.sequence_length = 100
        
    def process_all_files(self) -> bool:
        """Process all NetCDF files with correct variable names."""
        logger.info("🚀 Starting CORRECTED ARGO data processing...")
        
        nc_files = list(self.data_dir.rglob("*.nc"))
        self.stats['total_files'] = len(nc_files)
        
        if not nc_files:
            logger.error("❌ No NetCDF files found!")
            return False
        
        logger.info(f"📊 Found {len(nc_files)} NetCDF files to process")
        
        for i, nc_file in enumerate(nc_files):
            try:
                if i % 500 == 0:  # Log every 500 files
                    logger.info(f"📖 Processing {nc_file.name} ({i+1}/{len(nc_files)})")
                
                # Process each file
                self._process_single_file(nc_file)
                
            except Exception as e:
                self.stats['corrupted'] += 1
                if i % 500 == 0:
                    logger.warning(f"⚠️ Error processing {nc_file.name}: {e}")
                continue
        
        # Print comprehensive statistics
        self._print_statistics()
        
        # Save processed data
        self._save_processed_data()
        
        return True
    
    def _process_single_file(self, nc_file: Path):
        """Process a single NetCDF file with correct variable names."""
        try:
            # Load NetCDF file
            ds = xr.open_dataset(nc_file)
            
            # Extract all available parameters with CORRECT names (lowercase)
            temp_data = ds.get('temp', None)  # CORRECTED: lowercase
            sal_data = ds.get('psal', None)   # CORRECTED: lowercase
            pres_data = ds.get('pres', None)  # CORRECTED: lowercase
            lat_data = ds.get('latitude', None)
            lon_data = ds.get('longitude', None)
            juld_data = ds.get('juld', None)
            platform_data = ds.get('platform_number', None)
            
            # Check what data is available
            has_temp = temp_data is not None and not temp_data.isnull().all()
            has_sal = sal_data is not None and not sal_data.isnull().all()
            has_pres = pres_data is not None and not pres_data.isnull().all()
            
            # Extract metadata
            lat_val = float(lat_data.values) if lat_data is not None else None
            lon_val = float(lon_data.values) if lon_data is not None else None
            juld_val = float(juld_data.values) if juld_data is not None else None
            platform_val = str(platform_data.values) if platform_data is not None else "unknown"
            
            # Process temperature data
            if has_temp:
                temp_values = self._extract_clean_values(temp_data)
                if len(temp_values) >= 5:  # Minimum 5 points
                    temp_profile = self._normalize_and_pad(temp_values)
                    self.temperature_profiles.append(temp_profile)
                    self.latitude_data.append(lat_val)
                    self.longitude_data.append(lon_val)
                    self.date_data.append(juld_val)
                    self.platform_data.append(platform_val)
                    self.stats['temp_profiles'] += 1
            
            # Process salinity data
            if has_sal:
                sal_values = self._extract_clean_values(sal_data)
                if len(sal_values) >= 5:  # Minimum 5 points
                    sal_profile = self._normalize_and_pad(sal_values)
                    self.salinity_profiles.append(sal_profile)
                    self.stats['sal_profiles'] += 1
            
            # Process pressure data
            if has_pres:
                pres_values = self._extract_clean_values(pres_data)
                if len(pres_values) >= 5:  # Minimum 5 points
                    pres_profile = self._normalize_and_pad(pres_values)
                    self.pressure_profiles.append(pres_profile)
                    self.stats['pres_profiles'] += 1
            
            # Update statistics
            self.stats['processed_files'] += 1
            if has_temp and has_sal:
                self.stats['both_params'] += 1
            elif has_temp:
                self.stats['temp_only'] += 1
            elif has_sal:
                self.stats['sal_only'] += 1
            else:
                self.stats['no_params'] += 1
                
        except Exception as e:
            self.stats['corrupted'] += 1
            raise e
    
    def _extract_clean_values(self, data_array):
        """Extract clean values from xarray data."""
        values = data_array.values
        if values.ndim > 1:
            values = values.flatten()
        
        # Remove NaN values
        clean_values = values[~np.isnan(values)]
        
        # Remove extreme outliers (beyond 4 standard deviations)
        if len(clean_values) > 10:
            mean_val = np.mean(clean_values)
            std_val = np.std(clean_values)
            clean_values = clean_values[np.abs(clean_values - mean_val) < 4 * std_val]
        
        return clean_values
    
    def _normalize_and_pad(self, values):
        """Normalize values and pad/truncate to fixed length."""
        if len(values) == 0:
            return np.zeros(self.sequence_length)
        
        # Normalize
        normalized = (values - np.mean(values)) / (np.std(values) + 1e-8)
        
        # Pad or truncate
        if len(normalized) >= self.sequence_length:
            return normalized[:self.sequence_length]
        else:
            return np.pad(normalized, (0, self.sequence_length - len(normalized)), 
                         mode='constant', constant_values=0)
    
    def _print_statistics(self):
        """Print comprehensive processing statistics."""
        logger.info("\n" + "="*70)
        logger.info("📊 CORRECTED PROCESSING STATISTICS")
        logger.info("="*70)
        logger.info(f"📁 Total files processed: {self.stats['processed_files']}/{self.stats['total_files']}")
        logger.info(f"🌡️  Temperature profiles: {self.stats['temp_profiles']}")
        logger.info(f"🧂 Salinity profiles: {self.stats['sal_profiles']}")
        logger.info(f"📏 Pressure profiles: {self.stats['pres_profiles']}")
        logger.info(f"📍 Location data points: {len(self.latitude_data)}")
        logger.info("")
        logger.info("📈 Data Quality Breakdown:")
        logger.info(f"   • Both temp & salinity: {self.stats['both_params']} files")
        logger.info(f"   • Temperature only: {self.stats['temp_only']} files")
        logger.info(f"   • Salinity only: {self.stats['sal_only']} files")
        logger.info(f"   • No parameters: {self.stats['no_params']} files")
        logger.info(f"   • Corrupted files: {self.stats['corrupted']} files")
        logger.info("")
        logger.info(f"📊 Success rate: {(self.stats['processed_files']/self.stats['total_files']*100):.1f}%")
        logger.info(f"🌡️  Temperature success: {(self.stats['temp_profiles']/self.stats['total_files']*100):.1f}%")
        logger.info(f"🧂 Salinity success: {(self.stats['sal_profiles']/self.stats['total_files']*100):.1f}%")
        logger.info(f"📏 Pressure success: {(self.stats['pres_profiles']/self.stats['total_files']*100):.1f}%")
        logger.info("="*70)
    
    def _save_processed_data(self):
        """Save all processed data to files."""
        logger.info("💾 Saving processed data...")
        
        # Save temperature data
        if self.temperature_profiles:
            temp_array = np.array(self.temperature_profiles)
            np.save(self.output_dir / "temperature_profiles_all.npy", temp_array)
            logger.info(f"💾 Saved {len(self.temperature_profiles)} temperature profiles")
        
        # Save salinity data
        if self.salinity_profiles:
            sal_array = np.array(self.salinity_profiles)
            np.save(self.output_dir / "salinity_profiles_all.npy", sal_array)
            logger.info(f"💾 Saved {len(self.salinity_profiles)} salinity profiles")
        
        # Save pressure data
        if self.pressure_profiles:
            pres_array = np.array(self.pressure_profiles)
            np.save(self.output_dir / "pressure_profiles_all.npy", pres_array)
            logger.info(f"💾 Saved {len(self.pressure_profiles)} pressure profiles")
        
        # Save metadata
        metadata = {
            'latitude': self.latitude_data,
            'longitude': self.longitude_data,
            'date': self.date_data,
            'platform': self.platform_data,
            'stats': self.stats
        }
        
        import json
        with open(self.output_dir / "metadata_all.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✅ All data saved successfully!")

def main():
    """Main processing function."""
    logger.info("🚀 Starting CORRECTED ARGO Data Processing...")
    
    # Initialize processor
    processor = CorrectedARGOProcessor()
    
    # Process all files
    if processor.process_all_files():
        logger.info("🎉 All data processing completed successfully!")
        logger.info(f"📁 Processed data saved to: {processor.output_dir}")
    else:
        logger.error("❌ Data processing failed!")
        return False
    
    return True

if __name__ == "__main__":
    main()




