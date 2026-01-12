"""Database models for ARGO float data - optimized for 16GB RAM."""

from sqlalchemy import Column, Integer, Float, String, DateTime, Text, Index, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from pathlib import Path

Base = declarative_base()


class ARGOFloat(Base):
    """ARGO float metadata - lightweight table for basic info."""
    __tablename__ = 'argo_floats'
    
    id = Column(Integer, primary_key=True)
    platform_number = Column(String(20), unique=True, nullable=False, index=True)
    wmo_number = Column(String(20), index=True)
    data_mode = Column(String(10))
    first_deployment_date = Column(DateTime)
    last_profile_date = Column(DateTime)
    total_profiles = Column(Integer, default=0)
    region = Column(String(50), index=True)  # e.g., 'Indian Ocean'
    status = Column(String(20), default='active')
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_platform_region', 'platform_number', 'region'),
        Index('idx_region_status', 'region', 'status'),
    )


class ARGOProfile(Base):
    """ARGO profile data - chunked for memory efficiency."""
    __tablename__ = 'argo_profiles'
    
    id = Column(Integer, primary_key=True)
    platform_number = Column(String(20), nullable=False, index=True)
    cycle_number = Column(Integer, nullable=False)
    julian_day = Column(Float, nullable=False, index=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    profile_date = Column(DateTime, index=True)
    
    # Compressed data storage (JSON strings for efficiency)
    pressure_data = Column(Text)  # JSON array of pressure values
    temperature_data = Column(Text)  # JSON array of temperature values
    salinity_data = Column(Text)  # JSON array of salinity values
    
    # Metadata
    max_depth = Column(Float)
    data_quality = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes for spatial and temporal queries
    __table_args__ = (
        Index('idx_platform_cycle', 'platform_number', 'cycle_number'),
        Index('idx_lat_lon', 'latitude', 'longitude'),
        Index('idx_date_platform', 'profile_date', 'platform_number'),
    )


class ARGOTrajectory(Base):
    """ARGO trajectory data - surface positions."""
    __tablename__ = 'argo_trajectories'
    
    id = Column(Integer, primary_key=True)
    platform_number = Column(String(20), nullable=False, index=True)
    julian_day = Column(Float, nullable=False, index=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    position_date = Column(DateTime, index=True)
    
    # Additional trajectory info
    direction = Column(String(10))
    speed = Column(Float)  # km/day if calculable
    distance_from_previous = Column(Float)  # km
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Spatial index for location queries
    __table_args__ = (
        Index('idx_platform_date', 'platform_number', 'position_date'),
        Index('idx_spatial', 'latitude', 'longitude'),
    )


class ARGOBGCData(Base):
    """ARGO Bio-Geo-Chemical data - if available."""
    __tablename__ = 'argo_bgc_data'
    
    id = Column(Integer, primary_key=True)
    platform_number = Column(String(20), nullable=False, index=True)
    cycle_number = Column(Integer, nullable=False)
    variable_name = Column(String(20), nullable=False, index=True)
    julian_day = Column(Float, nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    # BGC measurements (compressed JSON)
    values_data = Column(Text)  # JSON array of values
    pressure_levels = Column(Text)  # JSON array of pressure levels
    
    # Quality flags
    data_quality = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_platform_variable', 'platform_number', 'variable_name'),
        Index('idx_variable_date', 'variable_name', 'julian_day'),
    )


class VectorEmbedding(Base):
    """Vector embeddings for RAG - stored as base64 for efficiency."""
    __tablename__ = 'vector_embeddings'
    
    id = Column(Integer, primary_key=True)
    content_type = Column(String(20), nullable=False)  # 'profile', 'trajectory', 'metadata'
    content_id = Column(Integer, nullable=False, index=True)
    platform_number = Column(String(20), index=True)
    
    # Embedding data (compressed)
    embedding_vector = Column(Text)  # Base64 encoded vector
    embedding_model = Column(String(50))
    content_summary = Column(Text)  # Text summary for retrieval
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_content_type_id', 'content_type', 'content_id'),
        Index('idx_platform_type', 'platform_number', 'content_type'),
    )


class DatabaseManager:
    """Memory-efficient database manager for 16GB RAM constraint."""
    
    def __init__(self, db_path: str = "data/floatchat.db"):
        """Initialize database with memory optimizations."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # SQLite with memory optimizations
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            # Memory optimizations
            pool_pre_ping=True,
            pool_recycle=3600,
            # SQLite specific optimizations
            connect_args={
                "check_same_thread": False,
                "timeout": 30
            }
        )
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Session factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.SessionLocal = SessionLocal
    
    def get_session(self):
        """Get database session with memory management."""
        return self.SessionLocal()
    
    def optimize_database(self):
        """Optimize database for memory usage."""
        with self.engine.connect() as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            
            # Optimize for memory usage
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA temp_store=memory")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            
            # Analyze tables for query optimization
            conn.execute("ANALYZE")
    
    def get_memory_usage(self) -> dict:
        """Get current database memory usage."""
        with self.engine.connect() as conn:
            result = conn.execute("PRAGMA page_count").fetchone()
            page_count = result[0] if result else 0
            
            result = conn.execute("PRAGMA page_size").fetchone()
            page_size = result[0] if result else 4096
            
            total_size = page_count * page_size
            
            return {
                "page_count": page_count,
                "page_size": page_size,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "cache_size_mb": 64  # Our configured cache size
            }
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old data to manage storage."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        with self.get_session() as session:
            # Clean up old vector embeddings
            old_embeddings = session.query(VectorEmbedding).filter(
                VectorEmbedding.created_at < cutoff_date
            ).delete()
            
            session.commit()
            return {"cleaned_embeddings": old_embeddings}


# Utility functions for memory-efficient data operations
def compress_array_data(data_array, precision=2):
    """Compress numpy array to JSON string with specified precision."""
    import json
    import numpy as np
    
    if isinstance(data_array, np.ndarray):
        # Round to specified precision to reduce size
        compressed = np.round(data_array, precision)
        # Remove NaN values and convert to list
        compressed = compressed[~np.isnan(compressed)].tolist()
    else:
        compressed = data_array
    
    return json.dumps(compressed)


def decompress_array_data(json_string):
    """Decompress JSON string back to numpy array."""
    import json
    import numpy as np
    
    data = json.loads(json_string)
    return np.array(data)


def batch_insert(session, model_class, data_list, batch_size=1000):
    """Insert data in batches to manage memory usage."""
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        session.bulk_insert_mappings(model_class, batch)
        session.commit()
        session.flush()  # Clear session memory


if __name__ == "__main__":
    # Initialize database
    db_manager = DatabaseManager()
    db_manager.optimize_database()
    
    # Check memory usage
    usage = db_manager.get_memory_usage()
    print(f"Database initialized. Memory usage: {usage['total_size_mb']:.2f} MB")





