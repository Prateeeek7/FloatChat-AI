#!/usr/bin/env python3
"""ARGO NetCDF Data Downloader with Progress Tracking and Permission Checks."""

import os
import requests
from bs4 import BeautifulSoup
import time
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse
import hashlib
from typing import List, Tuple
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('argo_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ARGODownloader:
    """Downloads ARGO NetCDF files with progress tracking and permission checks."""
    
    def __init__(self, base_url: str = "https://www.ncei.noaa.gov/data/oceans/argo/gadr/data/indian/"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.downloaded_size = 0
        self.downloaded_files = 0
        self.skipped_files = 0
        
    def get_folder_structure(self) -> List[Tuple[str, str, str]]:
        """Get all year/month combinations from the base URL."""
        logger.info("🔍 Discovering ARGO data folder structure...")
        
        try:
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            years = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('/') and href[:-1].isdigit() and len(href[:-1]) == 4:
                    year = href[:-1]
                    years.append(year)
            
            logger.info(f"📅 Found years: {sorted(years)}")
            
            # Get all year/month combinations
            year_month_combos = []
            for year in sorted(years):
                year_url = urljoin(self.base_url, f"{year}/")
                try:
                    year_response = self.session.get(year_url, timeout=30)
                    year_response.raise_for_status()
                    year_soup = BeautifulSoup(year_response.content, 'html.parser')
                    
                    for link in year_soup.find_all('a', href=True):
                        href = link['href']
                        if href.endswith('/') and href[:-1].isdigit() and 1 <= int(href[:-1]) <= 12:
                            month = href[:-1]
                            year_month_combos.append((year, month, urljoin(year_url, f"{month}/")))
                            
                except Exception as e:
                    logger.warning(f"⚠️ Could not access year {year}: {e}")
                    continue
            
            logger.info(f"📊 Found {len(year_month_combos)} year/month combinations")
            return year_month_combos
            
        except Exception as e:
            logger.error(f"❌ Error discovering folder structure: {e}")
            return []
    
    def get_nc_files(self, url: str) -> List[str]:
        """Get list of .nc files from a specific URL."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            nc_files = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.nc'):
                    nc_files.append(urljoin(url, href))
            
            return nc_files
            
        except Exception as e:
            logger.warning(f"⚠️ Could not access {url}: {e}")
            return []
    
    def get_file_size(self, url: str) -> int:
        """Get file size without downloading."""
        try:
            response = self.session.head(url, timeout=30)
            return int(response.headers.get('content-length', 0))
        except:
            return 0
    
    def should_skip_file(self, file_path: Path, expected_size: int) -> bool:
        """Check if file should be skipped (already exists and correct size)."""
        if not file_path.exists():
            return False
        
        try:
            actual_size = file_path.stat().st_size
            return actual_size == expected_size and actual_size > 0
        except:
            return False
    
    def download_file(self, url: str, file_path: Path) -> bool:
        """Download a single file with progress tracking."""
        try:
            # Get file size
            file_size = self.get_file_size(url)
            if file_size == 0:
                logger.warning(f"⚠️ Could not determine size for {url}")
                return False
            
            # Check if file should be skipped
            if self.should_skip_file(file_path, file_size):
                logger.info(f"⏭️ Skipping {file_path.name} (already exists)")
                self.skipped_files += 1
                return True
            
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            logger.info(f"⬇️ Downloading {file_path.name} ({file_size / 1024 / 1024:.1f} MB)")
            
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        self.downloaded_size += len(chunk)
            
            self.downloaded_files += 1
            logger.info(f"✅ Downloaded {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {url}: {e}")
            return False
    
    def check_permission(self, downloaded_gb: float) -> bool:
        """Ask user permission to continue after 1GB download."""
        print(f"\n📊 Download Progress:")
        print(f"   Files downloaded: {self.downloaded_files}")
        print(f"   Files skipped: {self.skipped_files}")
        print(f"   Total size: {downloaded_gb:.2f} GB")
        print(f"\n🤔 You've downloaded {downloaded_gb:.2f} GB. Continue downloading?")
        
        while True:
            response = input("Continue? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    def download_all(self) -> bool:
        """Download all ARGO NetCDF files with permission checks."""
        logger.info("🚀 Starting ARGO data download...")
        
        # Get folder structure
        year_month_combos = self.get_folder_structure()
        if not year_month_combos:
            logger.error("❌ No data folders found")
            return False
        
        # Create base directory
        base_dir = Path("raw_data/argo/indian")
        base_dir.mkdir(parents=True, exist_ok=True)
        
        total_files = 0
        for year, month, url in year_month_combos:
            nc_files = self.get_nc_files(url)
            total_files += len(nc_files)
        
        logger.info(f"📈 Total files to process: {total_files}")
        
        # Download files
        for year, month, url in year_month_combos:
            logger.info(f"📁 Processing {year}/{month}")
            
            nc_files = self.get_nc_files(url)
            if not nc_files:
                logger.warning(f"⚠️ No .nc files found in {year}/{month}")
                continue
            
            for nc_url in nc_files:
                # Create file path
                filename = os.path.basename(urlparse(nc_url).path)
                file_path = base_dir / year / month / filename
                
                # Download file
                success = self.download_file(nc_url, file_path)
                if not success:
                    continue
                
                # Check permission every 1GB
                downloaded_gb = self.downloaded_size / (1024 * 1024 * 1024)
                if downloaded_gb >= 1.0 and int(downloaded_gb) % 1 == 0:
                    if not self.check_permission(downloaded_gb):
                        logger.info("🛑 Download stopped by user")
                        return False
                
                # Small delay to be respectful
                time.sleep(0.1)
        
        # Final summary
        final_gb = self.downloaded_size / (1024 * 1024 * 1024)
        logger.info(f"\n🎉 Download completed!")
        logger.info(f"   Files downloaded: {self.downloaded_files}")
        logger.info(f"   Files skipped: {self.skipped_files}")
        logger.info(f"   Total size: {final_gb:.2f} GB")
        
        return True

def main():
    """Main function to run the downloader."""
    downloader = ARGODownloader()
    success = downloader.download_all()
    
    if success:
        logger.info("✅ ARGO data download completed successfully!")
        return True
    else:
        logger.error("❌ ARGO data download failed!")
        return False

if __name__ == "__main__":
    main()





