"""Configuration settings for FloatChat - optimized for 16GB M4 MacBook."""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Database settings (optimized for 16GB RAM)
DATABASE_CONFIG = {
    "db_path": str(DATA_DIR / "floatchat.db"),
    "cache_size_mb": 64,  # 4% of 16GB RAM
    "mmap_size_mb": 256,  # 16% of 16GB RAM
    "temp_store": "memory",
    "journal_mode": "WAL"
}

# Memory management settings
MEMORY_CONFIG = {
    "max_memory_mb": 12000,  # Leave 4GB for system
    "chunk_size": 1000,  # Process data in chunks
    "batch_size": 100,  # Database batch size
    "cleanup_frequency": 10  # Cleanup every N operations
}

# Vector database settings
VECTOR_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 384,  # Dimension for the model above
    "faiss_index_type": "IndexFlatIP",  # Inner product similarity
    "top_k": 5  # Number of similar documents to retrieve
}

# LLM settings
LLM_CONFIG = {
    "openai_model": "gpt-3.5-turbo",
    "max_tokens": 500,
    "temperature": 0.1,
    "timeout": 30
}

# Data processing settings
DATA_PROCESSING_CONFIG = {
    "precision": 2,  # Decimal places for compressed data
    "max_file_size_mb": 100,  # Skip files larger than this
    "supported_formats": [".nc", ".netcdf"],
    "compression": "zlib"  # Compression for arrays
}

# API settings
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "workers": 1  # Single worker for 16GB RAM
}

# Dashboard settings
DASHBOARD_CONFIG = {
    "host": "0.0.0.0",
    "port": 8501,
    "theme": "light",
    "layout": "wide"
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(DATA_DIR / "logs" / "floatchat.log")
}

# ARGO data specific settings
ARGO_CONFIG = {
    "regions": {
        "Indian Ocean": {"lat_range": (20, 40), "lon_range": (60, 100)},
        "Atlantic Ocean": {"lat_range": (-60, 60), "lon_range": (-80, 20)},
        "Pacific Ocean": {"lat_range": (-60, 60), "lon_range": (100, -80)},
        "Global": {"lat_range": (-90, 90), "lon_range": (-180, 180)}
    },
    "variables": {
        "TEMP": "temperature",
        "PSAL": "salinity",
        "PRES": "pressure",
        "DOXY": "dissolved_oxygen",
        "NITRATE": "nitrate",
        "CHLA": "chlorophyll_a"
    },
    "quality_flags": {
        "good": 1,
        "probably_good": 2,
        "probably_bad": 3,
        "bad": 4
    }
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    "enable_profiling": True,
    "profile_memory": True,
    "log_performance": True,
    "max_query_time": 30  # seconds
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "raw_data_dir": RAW_DATA_DIR,
        "processed_data_dir": PROCESSED_DATA_DIR,
        "vector_db_dir": VECTOR_DB_DIR,
        "database": DATABASE_CONFIG,
        "memory": MEMORY_CONFIG,
        "vector": VECTOR_CONFIG,
        "llm": LLM_CONFIG,
        "data_processing": DATA_PROCESSING_CONFIG,
        "api": API_CONFIG,
        "dashboard": DASHBOARD_CONFIG,
        "logging": LOGGING_CONFIG,
        "argo": ARGO_CONFIG,
        "performance": PERFORMANCE_CONFIG
    }

def create_directories():
    """Create necessary directories."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        VECTOR_DB_DIR,
        DATA_DIR / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def validate_config() -> bool:
    """Validate configuration settings."""
    try:
        # Check memory settings
        if MEMORY_CONFIG["max_memory_mb"] > 15000:
            print("Warning: max_memory_mb is set higher than recommended for 16GB RAM")
        
        # Check database settings
        if DATABASE_CONFIG["cache_size_mb"] > 1000:
            print("Warning: database cache size is very large")
        
        # Check if directories exist
        create_directories()
        
        return True
        
    except Exception as e:
        print(f"Configuration validation error: {str(e)}")
        return False

if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("FloatChat Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\nConfiguration valid: {validate_config()}")





