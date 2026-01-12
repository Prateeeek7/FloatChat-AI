#!/usr/bin/env python3
"""ARGO Trainer - Main trainer class for ARGO data model training."""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple
import warnings
import gc
import xarray as xr
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ARGOProfileDataset(Dataset):
    """Memory-efficient dataset for ARGO profile data."""
    
    def __init__(self, temperature_data, salinity_data):
        self.temperature_data = torch.FloatTensor(temperature_data)
        self.salinity_data = torch.FloatTensor(salinity_data)
    
    def __len__(self):
        return len(self.temperature_data)
    
    def __getitem__(self, idx):
        return self.temperature_data[idx], self.salinity_data[idx]

class ARGOLSTM(nn.Module):
    """LSTM model for ARGO data."""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
        super(ARGOLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = torch.relu(self.fc1(lstm_out))
        output = self.fc2(output)
        return output

class ARGOCNN(nn.Module):
    """CNN model for ARGO data."""
    
    def __init__(self, sequence_length=100):
        super(ARGOCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        
        conv_output_size = (sequence_length // 4) * 128
        
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, sequence_length)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ARGOTrainer:
    """Main trainer class for ARGO data."""
    
    def __init__(self, data_dir: str = "raw_data/argo/indian", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = 100
        self.batch_size = 64
        self.temperature_profiles = []
        self.salinity_profiles = []
        
    def process_netcdf_file(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single NetCDF file and extract temperature/salinity profiles."""
        try:
            ds = xr.open_dataset(file_path)
            
            if 'TEMP' in ds.variables and 'PSAL' in ds.variables:
                temp = ds['TEMP'].values
                sal = ds['PSAL'].values
                
                valid_mask = ~(np.isnan(temp) | np.isnan(sal))
                if np.sum(valid_mask) > 10:
                    temp_clean = temp[valid_mask]
                    sal_clean = sal[valid_mask]
                    
                    if len(temp_clean) > self.sequence_length:
                        indices = np.linspace(0, len(temp_clean)-1, self.sequence_length, dtype=int)
                        temp_profile = temp_clean[indices]
                        sal_profile = sal_clean[indices]
                    else:
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
                    
                    return temp_profile, sal_profile
            
            ds.close()
            return None, None
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing {file_path.name}: {e}")
            return None, None
    
    def load_nc_files(self) -> bool:
        """Load NetCDF files and extract profiles."""
        logger.info("📂 Loading NetCDF files...")
        
        try:
            nc_files = list(self.data_dir.rglob("*.nc"))
            if not nc_files:
                logger.error("❌ No NetCDF files found!")
                return False
            
            logger.info(f"📊 Found {len(nc_files)} NetCDF files")
            
            temp_profiles = []
            sal_profiles = []
            
            for i, nc_file in enumerate(nc_files):
                if (i + 1) % 100 == 0:
                    logger.info(f"📊 Processing file {i+1}/{len(nc_files)}...")
                
                temp_profile, sal_profile = self.process_netcdf_file(nc_file)
                if temp_profile is not None and sal_profile is not None:
                    temp_profiles.append(temp_profile)
                    sal_profiles.append(sal_profile)
            
            if not temp_profiles:
                logger.error("❌ No valid profiles extracted!")
                return False
            
            self.temperature_profiles = np.array(temp_profiles)
            self.salinity_profiles = np.array(sal_profiles)
            
            logger.info(f"✅ Loaded {len(self.temperature_profiles)} profiles")
            logger.info(f"📊 Data shape: {self.temperature_profiles.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading NetCDF files: {e}")
            return False
    
    def create_data_loaders(self, model_type: str = "lstm") -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for training."""
        logger.info(f"🔄 Creating data loaders for {model_type.upper()} model...")
        
        X = self.temperature_profiles.copy()
        y = self.salinity_profiles.copy()
        
        if model_type.lower() == "lstm":
            X = X.reshape(X.shape[0], X.shape[1], 1)
            y = y.reshape(y.shape[0], y.shape[1], 1)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        train_dataset = ARGOProfileDataset(X_train, y_train)
        val_dataset = ARGOProfileDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        logger.info(f"📊 Training samples: {len(train_dataset)}")
        logger.info(f"📊 Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_model(self, model_type: str = "lstm", epochs: int = 50) -> bool:
        """Train model."""
        logger.info(f"🚀 Starting training with {model_type.upper()} model...")
        
        train_loader, val_loader = self.create_data_loaders(model_type)
        
        if model_type.lower() == "lstm":
            model = ARGOLSTM(input_size=1, hidden_size=128, num_layers=2, output_size=1)
        elif model_type.lower() == "cnn":
            model = ARGOCNN(sequence_length=self.sequence_length)
        else:
            logger.error(f"❌ Unknown model type: {model_type}")
            return False
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"🖥️ Using device: {device}")
        
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            total_train_samples = 0
            
            for temp, sal in train_loader:
                temp, sal = temp.to(device), sal.to(device)
                
                optimizer.zero_grad()
                outputs = model(temp)
                loss = criterion(outputs, sal)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(temp)
                total_train_samples += len(temp)
            
            model.eval()
            val_loss = 0.0
            total_val_samples = 0
            
            with torch.no_grad():
                for temp, sal in val_loader:
                    temp, sal = temp.to(device), sal.to(device)
                    outputs = model(temp)
                    loss = criterion(outputs, sal)
                    val_loss += loss.item() * len(temp)
                    total_val_samples += len(temp)
            
            avg_train_loss = train_loss / total_train_samples
            avg_val_loss = val_loss / total_val_samples
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = self.model_dir / f"argo_{model_type}.pt"
                torch.save(model.state_dict(), model_path)
        
        final_model_path = self.model_dir / f"argo_{model_type}_final.pt"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"✅ Training completed! Final model saved to {final_model_path}")
        
        return True
