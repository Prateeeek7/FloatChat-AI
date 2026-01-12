#!/usr/bin/env python3
"""Main entrypoint for ARGO data download and training pipeline."""

import os
import sys
import logging
import argparse
from pathlib import Path
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from argo_downloader import ARGODownloader
from argo_trainer import ARGOTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('argo_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'requests', 'beautifulsoup4', 'xarray', 'torch', 'numpy', 
        'pandas', 'sklearn', 'netCDF4'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'beautifulsoup4':
                import bs4
            elif package == 'netCDF4':
                import netCDF4
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        logger.error("Please install them with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All dependencies are installed")
    return True

def run_download_phase():
    """Run the download phase."""
    logger.info("🚀 Starting ARGO data download phase...")
    
    downloader = ARGODownloader()
    success = downloader.download_all()
    
    if success:
        logger.info("✅ Download phase completed successfully!")
        return True
    else:
        logger.error("❌ Download phase failed!")
        return False

def run_training_phase():
    """Run the training phase."""
    logger.info("🤖 Starting ARGO model training phase...")
    
    # Check if data exists
    data_dir = Path("raw_data/argo/indian")
    if not data_dir.exists() or not any(data_dir.rglob("*.nc")):
        logger.error("❌ No ARGO data found! Please run download phase first.")
        return False
    
    trainer = ARGOTrainer()
    
    # Load data
    if not trainer.load_nc_files():
        logger.error("❌ Failed to load ARGO data")
        return False
    
    # Train models
    logger.info("🧠 Training LSTM model...")
    if not trainer.train_model("lstm", epochs=50):
        logger.error("❌ LSTM training failed")
        return False
    
    logger.info("🧠 Training CNN model...")
    if not trainer.train_model("cnn", epochs=50):
        logger.error("❌ CNN training failed")
        return False
    
    logger.info("✅ Training phase completed successfully!")
    return True

def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="ARGO Data Download and Training Pipeline")
    parser.add_argument("--download-only", action="store_true", help="Only run download phase")
    parser.add_argument("--train-only", action="store_true", help="Only run training phase")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency check")
    
    args = parser.parse_args()
    
    logger.info("🌊 ARGO Data Pipeline Starting...")
    logger.info("=" * 50)
    
    # Check dependencies
    if not args.skip_deps:
        if not check_dependencies():
            sys.exit(1)
    
    # Determine what to run
    run_download = not args.train_only
    run_training = not args.download_only
    
    success = True
    
    # Download phase
    if run_download:
        logger.info("\n📥 PHASE 1: DATA DOWNLOAD")
        logger.info("-" * 30)
        
        start_time = time.time()
        download_success = run_download_phase()
        download_time = time.time() - start_time
        
        if download_success:
            logger.info(f"⏱️ Download completed in {download_time:.1f} seconds")
        else:
            logger.error("❌ Download phase failed!")
            success = False
        
        # If download failed and we're not training only, exit
        if not download_success and not args.train_only:
            sys.exit(1)
    
    # Training phase
    if run_training and success:
        logger.info("\n🧠 PHASE 2: MODEL TRAINING")
        logger.info("-" * 30)
        
        start_time = time.time()
        training_success = run_training_phase()
        training_time = time.time() - start_time
        
        if training_success:
            logger.info(f"⏱️ Training completed in {training_time:.1f} seconds")
        else:
            logger.error("❌ Training phase failed!")
            success = False
    
    # Final summary
    logger.info("\n" + "=" * 50)
    if success:
        logger.info("🎉 ARGO Pipeline completed successfully!")
        
        # Show results
        data_dir = Path("raw_data/argo/indian")
        model_dir = Path("models")
        
        if data_dir.exists():
            nc_files = list(data_dir.rglob("*.nc"))
            total_size = sum(f.stat().st_size for f in nc_files)
            logger.info(f"📊 Downloaded {len(nc_files)} NetCDF files ({total_size / 1024 / 1024 / 1024:.2f} GB)")
        
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pt"))
            logger.info(f"🤖 Trained {len(model_files)} models")
            for model_file in model_files:
                logger.info(f"   - {model_file.name}")
        
    else:
        logger.error("❌ ARGO Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
