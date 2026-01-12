"""Memory-efficient data ingestion pipeline for ARGO data."""

import os
import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import zarr
from tqdm import tqdm

from .netcdf_processor import ARGONetCDFProcessor
from ..database.models import (
    DatabaseManager, ARGOFloat, ARGOProfile, ARGOTrajectory, 
    ARGOBGCData, compress_array_data, decompress_array_data, batch_insert
)

logger = logging.getLogger(__name__)


class ARGODataIngestionPipeline:
    """Memory-efficient pipeline for ingesting ARGO data into database."""
    
    def __init__(self, db_path: str = "data/floatchat.db", 
                 raw_data_dir: str = "data/raw",
                 processed_data_dir: str = "data/processed"):
        """Initialize the ingestion pipeline."""
        self.db_manager = DatabaseManager(db_path)
        self.processor = ARGONetCDFProcessor(raw_data_dir)
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing statistics
        self.stats = {
            'files_processed': 0,
            'profiles_ingested': 0,
            'trajectories_ingested': 0,
            'bgc_records_ingested': 0,
            'errors': 0
        }
    
    def ingest_netcdf_files(self, file_patterns: List[str] = None, 
                          batch_size: int = 10) -> Dict[str, Any]:
        """
        Ingest NetCDF files in batches to manage memory usage.
        
        Args:
            file_patterns: List of file patterns to process
            batch_size: Number of files to process in each batch
            
        Returns:
            Processing statistics
        """
        if file_patterns is None:
            # Find all NetCDF files
            netcdf_files = list(self.raw_data_dir.glob("*.nc")) + \
                          list(self.raw_data_dir.glob("**/*.nc"))
        else:
            netcdf_files = []
            for pattern in file_patterns:
                netcdf_files.extend(self.raw_data_dir.glob(pattern))
        
        logger.info(f"Found {len(netcdf_files)} NetCDF files to process")
        
        # Process files in batches
        for i in range(0, len(netcdf_files), batch_size):
            batch_files = netcdf_files[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_files)} files")
            
            try:
                self._process_batch(batch_files)
                self._cleanup_memory()
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                self.stats['errors'] += 1
        
        return self.stats
    
    def _process_batch(self, file_paths: List[Path]) -> None:
        """Process a batch of NetCDF files."""
        session = self.db_manager.get_session()
        
        try:
            for file_path in tqdm(file_paths, desc="Processing files"):
                try:
                    # Process NetCDF file
                    processed_data = self.processor.process_netcdf_file(str(file_path))
                    
                    if processed_data:
                        # Ingest into database
                        self._ingest_processed_data(session, processed_data)
                        self.stats['files_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    self.stats['errors'] += 1
                    continue
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Database error in batch processing: {str(e)}")
            raise
        finally:
            session.close()
    
    def _ingest_processed_data(self, session, processed_data: Dict[str, Any]) -> None:
        """Ingest processed data into database tables."""
        metadata = processed_data.get('metadata', {})
        profile_data = processed_data.get('profile_data')
        trajectory_data = processed_data.get('trajectory_data')
        bgc_data = processed_data.get('bgc_data')
        
        # Extract platform number
        platform_number = metadata.get('platform_number', 'unknown')
        
        # 1. Create or update ARGO float record
        self._upsert_argo_float(session, platform_number, metadata)
        
        # 2. Ingest profile data
        if profile_data is not None and not profile_data.empty:
            self._ingest_profile_data(session, platform_number, profile_data)
        
        # 3. Ingest trajectory data
        if trajectory_data is not None and not trajectory_data.empty:
            self._ingest_trajectory_data(session, platform_number, trajectory_data)
        
        # 4. Ingest BGC data
        if bgc_data is not None and not bgc_data.empty:
            self._ingest_bgc_data(session, platform_number, bgc_data)
    
    def _upsert_argo_float(self, session, platform_number: str, metadata: Dict[str, Any]) -> None:
        """Create or update ARGO float record."""
        existing_float = session.query(ARGOFloat).filter(
            ARGOFloat.platform_number == platform_number
        ).first()
        
        if existing_float:
            # Update existing record
            existing_float.data_mode = metadata.get('data_mode', existing_float.data_mode)
            existing_float.total_profiles += 1
        else:
            # Create new record
            new_float = ARGOFloat(
                platform_number=platform_number,
                wmo_number=metadata.get('wmo_number'),
                data_mode=metadata.get('data_mode'),
                region=self._determine_region(metadata),
                status='active'
            )
            session.add(new_float)
    
    def _ingest_profile_data(self, session, platform_number: str, profile_data: pd.DataFrame) -> None:
        """Ingest profile data with memory optimization."""
        profile_records = []
        
        for _, row in profile_data.iterrows():
            # Convert Julian day to datetime
            profile_date = None
            if 'julian_day' in row and not pd.isna(row['julian_day']):
                base_date = datetime(1950, 1, 1)
                profile_date = base_date + timedelta(days=float(row['julian_day']))
            
            # Compress array data
            pressure_data = compress_array_data(row.get('pressure', []), precision=2)
            temperature_data = compress_array_data(row.get('temperature', []), precision=3)
            salinity_data = compress_array_data(row.get('salinity', []), precision=3)
            
            profile_record = {
                'platform_number': platform_number,
                'cycle_number': int(row.get('cycle_number', 0)),
                'julian_day': float(row.get('julian_day', 0)),
                'latitude': float(row.get('latitude', 0)),
                'longitude': float(row.get('longitude', 0)),
                'profile_date': profile_date,
                'pressure_data': pressure_data,
                'temperature_data': temperature_data,
                'salinity_data': salinity_data,
                'max_depth': float(np.nanmax(row.get('pressure', [0]))),
                'data_quality': 'good'  # Could be determined from data quality flags
            }
            
            profile_records.append(profile_record)
            
            # Batch insert to manage memory
            if len(profile_records) >= 100:
                batch_insert(session, ARGOProfile, profile_records)
                profile_records = []
                self.stats['profiles_ingested'] += len(profile_records)
        
        # Insert remaining records
        if profile_records:
            batch_insert(session, ARGOProfile, profile_records)
            self.stats['profiles_ingested'] += len(profile_records)
    
    def _ingest_trajectory_data(self, session, platform_number: str, trajectory_data: pd.DataFrame) -> None:
        """Ingest trajectory data."""
        trajectory_records = []
        
        for _, row in trajectory_data.iterrows():
            # Convert Julian day to datetime
            position_date = None
            if 'julian_day' in row and not pd.isna(row['julian_day']):
                base_date = datetime(1950, 1, 1)
                position_date = base_date + timedelta(days=float(row['julian_day']))
            
            trajectory_record = {
                'platform_number': platform_number,
                'julian_day': float(row.get('julian_day', 0)),
                'latitude': float(row.get('latitude', 0)),
                'longitude': float(row.get('longitude', 0)),
                'position_date': position_date,
                'direction': row.get('direction', 'unknown'),
                'speed': None,  # Could be calculated from consecutive positions
                'distance_from_previous': None
            }
            
            trajectory_records.append(trajectory_record)
        
        if trajectory_records:
            batch_insert(session, ARGOTrajectory, trajectory_records)
            self.stats['trajectories_ingested'] += len(trajectory_records)
    
    def _ingest_bgc_data(self, session, platform_number: str, bgc_data: pd.DataFrame) -> None:
        """Ingest BGC data."""
        bgc_records = []
        
        for _, row in bgc_data.iterrows():
            bgc_record = {
                'platform_number': platform_number,
                'cycle_number': int(row.get('profile_index', 0)),
                'variable_name': row.get('variable', 'unknown'),
                'julian_day': 0,  # Would need to be extracted from profile data
                'latitude': 0,    # Would need to be extracted from profile data
                'longitude': 0,   # Would need to be extracted from profile data
                'values_data': compress_array_data(row.get('values', []), precision=3),
                'pressure_levels': compress_array_data([], precision=2),
                'data_quality': 'good'
            }
            
            bgc_records.append(bgc_record)
        
        if bgc_records:
            batch_insert(session, ARGOBGCData, bgc_records)
            self.stats['bgc_records_ingested'] += len(bgc_records)
    
    def _determine_region(self, metadata: Dict[str, Any]) -> str:
        """Determine ocean region based on metadata or coordinates."""
        # Simple region determination - could be enhanced
        lat = metadata.get('latitude', 0)
        lon = metadata.get('longitude', 0)
        
        if 20 <= lat <= 40 and 60 <= lon <= 100:
            return 'Indian Ocean'
        elif -20 <= lat <= 20 and 60 <= lon <= 100:
            return 'Indian Ocean'
        else:
            return 'Global'
    
    def _cleanup_memory(self):
        """Clean up memory after processing."""
        gc.collect()
        logger.info("Memory cleanup performed")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.stats.copy()
    
    def optimize_database(self):
        """Optimize database after ingestion."""
        self.db_manager.optimize_database()
        logger.info("Database optimization completed")


def main():
    """Example usage of the ingestion pipeline."""
    pipeline = ARGODataIngestionPipeline()
    
    # Ingest all NetCDF files
    stats = pipeline.ingest_netcdf_files()
    
    print("Ingestion completed!")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Profiles ingested: {stats['profiles_ingested']}")
    print(f"Trajectories ingested: {stats['trajectories_ingested']}")
    print(f"BGC records ingested: {stats['bgc_records_ingested']}")
    print(f"Errors: {stats['errors']}")
    
    # Optimize database
    pipeline.optimize_database()


if __name__ == "__main__":
    main()





