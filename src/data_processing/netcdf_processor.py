"""Memory-efficient NetCDF data processing for ARGO float data - optimized for 16GB RAM."""

import os
import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from datetime import datetime, timedelta
import dask.array as da
import zarr
import json
from memory_profiler import profile

logger = logging.getLogger(__name__)


class ARGONetCDFProcessor:
    """Memory-efficient ARGO NetCDF processor optimized for 16GB RAM."""
    
    def __init__(self, data_dir: str = "data/raw", max_memory_mb: int = 12000):
        """Initialize the processor with memory constraints."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_mb = max_memory_mb  # Leave 4GB for system
        
        # ARGO variable mappings
        self.variable_mappings = {
            'TEMP': 'temperature',
            'PSAL': 'salinity', 
            'PRES': 'pressure',
            'LATITUDE': 'latitude',
            'LONGITUDE': 'longitude',
            'JULD': 'julian_day',
            'CYCLE_NUMBER': 'cycle_number',
            'PLATFORM_NUMBER': 'platform_number',
            'DIRECTION': 'direction',
            'DATA_MODE': 'data_mode'
        }
        
        # Memory tracking
        self.current_memory_usage = 0
    
    def _check_memory_usage(self) -> bool:
        """Check if we're within memory limits."""
        import psutil
        memory_info = psutil.virtual_memory()
        used_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
        return used_mb < self.max_memory_mb
    
    def _cleanup_memory(self):
        """Force garbage collection to free memory."""
        gc.collect()
        logger.info("Memory cleanup performed")
    
    def _process_in_chunks(self, data_array, chunk_size: int = 1000):
        """Process large arrays in chunks to manage memory."""
        if len(data_array) <= chunk_size:
            return data_array
        
        chunks = []
        for i in range(0, len(data_array), chunk_size):
            chunk = data_array[i:i + chunk_size]
            chunks.append(chunk)
            if not self._check_memory_usage():
                self._cleanup_memory()
        
        return chunks
    
    def process_netcdf_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single NetCDF file and extract relevant data.
        
        Args:
            file_path: Path to the NetCDF file
            
        Returns:
            Dictionary containing processed data and metadata
        """
        try:
            # Open NetCDF file
            with xr.open_dataset(file_path) as ds:
                logger.info(f"Processing file: {file_path}")
                
                # Extract basic metadata
                metadata = self._extract_metadata(ds)
                
                # Extract profile data
                profile_data = self._extract_profile_data(ds)
                
                # Extract trajectory data
                trajectory_data = self._extract_trajectory_data(ds)
                
                # Extract BGC data if available
                bgc_data = self._extract_bgc_data(ds)
                
                return {
                    'metadata': metadata,
                    'profile_data': profile_data,
                    'trajectory_data': trajectory_data,
                    'bgc_data': bgc_data,
                    'file_path': file_path,
                    'processed_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def _extract_metadata(self, ds: xr.Dataset) -> Dict[str, Any]:
        """Extract metadata from the dataset."""
        metadata = {}
        
        # Global attributes
        for attr in ds.attrs:
            metadata[attr.lower()] = ds.attrs[attr]
        
        # Platform information
        if 'PLATFORM_NUMBER' in ds.variables:
            metadata['platform_number'] = str(ds.PLATFORM_NUMBER.values)
        
        # Data quality and mode
        if 'DATA_MODE' in ds.variables:
            metadata['data_mode'] = ds.DATA_MODE.values.tolist()
        
        # Time information
        if 'JULD' in ds.variables:
            julian_days = ds.JULD.values
            if len(julian_days) > 0:
                # Convert Julian day to datetime
                base_date = datetime(1950, 1, 1)
                dates = [base_date + timedelta(days=float(jd)) for jd in julian_days]
                metadata['date_range'] = {
                    'start': dates[0].isoformat(),
                    'end': dates[-1].isoformat()
                }
        
        return metadata
    
    def _extract_profile_data(self, ds: xr.Dataset) -> pd.DataFrame:
        """Extract profile data (depth-based measurements)."""
        profile_data = []
        
        # Get pressure levels
        if 'PRES' in ds.variables:
            pressure = ds.PRES.values
        else:
            return pd.DataFrame()
        
        # Extract measurements for each profile
        n_profiles = pressure.shape[0] if len(pressure.shape) > 1 else 1
        
        for i in range(n_profiles):
            profile_dict = {}
            
            # Pressure levels
            if len(pressure.shape) > 1:
                profile_dict['pressure'] = pressure[i, :]
            else:
                profile_dict['pressure'] = pressure
            
            # Temperature
            if 'TEMP' in ds.variables:
                temp = ds.TEMP.values
                if len(temp.shape) > 1:
                    profile_dict['temperature'] = temp[i, :]
                else:
                    profile_dict['temperature'] = temp
            
            # Salinity
            if 'PSAL' in ds.variables:
                sal = ds.PSAL.values
                if len(sal.shape) > 1:
                    profile_dict['salinity'] = sal[i, :]
                else:
                    profile_dict['salinity'] = sal
            
            # Time
            if 'JULD' in ds.variables:
                julian_days = ds.JULD.values
                if len(julian_days.shape) > 0:
                    profile_dict['julian_day'] = julian_days[i] if i < len(julian_days) else julian_days[0]
                else:
                    profile_dict['julian_day'] = julian_days
            
            # Cycle number
            if 'CYCLE_NUMBER' in ds.variables:
                cycles = ds.CYCLE_NUMBER.values
                if len(cycles.shape) > 0:
                    profile_dict['cycle_number'] = cycles[i] if i < len(cycles) else cycles[0]
                else:
                    profile_dict['cycle_number'] = cycles
            
            # Location
            if 'LATITUDE' in ds.variables:
                lats = ds.LATITUDE.values
                if len(lats.shape) > 0:
                    profile_dict['latitude'] = lats[i] if i < len(lats) else lats[0]
                else:
                    profile_dict['latitude'] = lats
            
            if 'LONGITUDE' in ds.variables:
                lons = ds.LONGITUDE.values
                if len(lons.shape) > 0:
                    profile_dict['longitude'] = lons[i] if i < len(lons) else lons[0]
                else:
                    profile_dict['longitude'] = lons
            
            profile_data.append(profile_dict)
        
        return pd.DataFrame(profile_data)
    
    def _extract_trajectory_data(self, ds: xr.Dataset) -> pd.DataFrame:
        """Extract trajectory data (surface positions)."""
        trajectory_data = []
        
        if 'LATITUDE' in ds.variables and 'LONGITUDE' in ds.variables:
            lats = ds.LATITUDE.values
            lons = ds.LONGITUDE.values
            
            # Handle different data shapes
            if len(lats.shape) == 0:
                lats = [lats]
                lons = [lons]
            
            for i, (lat, lon) in enumerate(zip(lats, lons)):
                if not np.isnan(lat) and not np.isnan(lon):
                    trajectory_dict = {
                        'latitude': float(lat),
                        'longitude': float(lon),
                        'point_index': i
                    }
                    
                    # Add time if available
                    if 'JULD' in ds.variables:
                        julian_days = ds.JULD.values
                        if len(julian_days.shape) > 0 and i < len(julian_days):
                            trajectory_dict['julian_day'] = float(julian_days[i])
                        else:
                            trajectory_dict['julian_day'] = float(julian_days)
                    
                    trajectory_data.append(trajectory_dict)
        
        return pd.DataFrame(trajectory_data)
    
    def _extract_bgc_data(self, ds: xr.Dataset) -> pd.DataFrame:
        """Extract Bio-Geo-Chemical data if available."""
        bgc_data = []
        
        # Common BGC variables
        bgc_variables = ['DOXY', 'NITRATE', 'CHLA', 'BBP', 'CDOM', 'DOWN_IRRADIANCE']
        
        for var in bgc_variables:
            if var in ds.variables:
                values = ds[var].values
                if len(values.shape) > 1:
                    # Profile data
                    for i in range(values.shape[0]):
                        bgc_dict = {
                            'variable': var,
                            'values': values[i, :].tolist(),
                            'profile_index': i
                        }
                        bgc_data.append(bgc_dict)
                else:
                    # Single value
                    bgc_dict = {
                        'variable': var,
                        'values': values.tolist(),
                        'profile_index': 0
                    }
                    bgc_data.append(bgc_dict)
        
        return pd.DataFrame(bgc_data)
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all NetCDF files in a directory.
        
        Args:
            directory_path: Path to directory containing NetCDF files
            
        Returns:
            List of processed data dictionaries
        """
        directory = Path(directory_path)
        processed_files = []
        
        # Find all NetCDF files
        netcdf_files = list(directory.glob("*.nc")) + list(directory.glob("**/*.nc"))
        
        logger.info(f"Found {len(netcdf_files)} NetCDF files to process")
        
        for file_path in netcdf_files:
            result = self.process_netcdf_file(str(file_path))
            if result:
                processed_files.append(result)
        
        return processed_files
    
    def save_processed_data(self, processed_data: List[Dict[str, Any]], 
                          output_dir: str = "data/processed") -> None:
        """Save processed data to various formats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet files
        for i, data in enumerate(processed_data):
            if data['profile_data'] is not None and not data['profile_data'].empty:
                profile_file = output_path / f"profile_data_{i}.parquet"
                data['profile_data'].to_parquet(profile_file)
            
            if data['trajectory_data'] is not None and not data['trajectory_data'].empty:
                trajectory_file = output_path / f"trajectory_data_{i}.parquet"
                data['trajectory_data'].to_parquet(trajectory_file)
            
            if data['bgc_data'] is not None and not data['bgc_data'].empty:
                bgc_file = output_path / f"bgc_data_{i}.parquet"
                data['bgc_data'].to_parquet(bgc_file)
        
        logger.info(f"Processed data saved to {output_path}")


def main():
    """Example usage of the ARGO NetCDF processor."""
    processor = ARGONetCDFProcessor()
    
    # Process a single file (replace with actual file path)
    # result = processor.process_netcdf_file("path/to/your/argo_file.nc")
    
    # Process all files in a directory
    # results = processor.process_directory("data/raw")
    # processor.save_processed_data(results)
    
    print("ARGO NetCDF Processor initialized. Ready to process files.")


if __name__ == "__main__":
    main()
