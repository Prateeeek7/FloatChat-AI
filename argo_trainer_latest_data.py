#!/usr/bin/env python3
"""
Latest Data ARGO Trainer - Train with data from after March 2003
Uses the newly downloaded data from 2003-2005 for training
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, List
import warnings
import gc
import xarray as xr
from datetime import datetime
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('argo_latest_training.log'),
        logging.StreamHandler()
    ]
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

class LatestLSTM(nn.Module):
    """LSTM model optimized for latest data (2003-2005)."""
    
    def __init__(self, input_size=1, hidden_size=256, num_layers=4, output_size=1):
        super(LatestLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Enhanced LSTM for latest data
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply dropout and fully connected layers
        output = self.dropout(lstm_out)
        output = torch.relu(self.fc1(output))
        output = self.dropout(output)
        output = torch.relu(self.fc2(output))
        output = self.dropout(output)
        output = self.fc3(output)
        
        return output

class LatestCNN(nn.Module):
    """CNN model optimized for latest data (2003-2005)."""
    
    def __init__(self, sequence_length=100):
        super(LatestCNN, self).__init__()
        
        # Enhanced convolutional layers
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = (sequence_length // 16) * 1024
        
        # Enhanced fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, sequence_length)
        
    def forward(self, x):
        # Input shape: (batch, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.dropout(x)
        
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x

class LatestDataTrainer:
    """Trainer for latest ARGO data (2003-2005)."""
    
    def __init__(self, data_dir: str = "raw_data/argo/indian", model_dir: str = "models_latest"):
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
            # Open NetCDF file
            ds = xr.open_dataset(file_path)
            
            # Extract temperature and salinity data
            if 'TEMP' in ds.variables and 'PSAL' in ds.variables:
                temp = ds['TEMP'].values
                sal = ds['PSAL'].values
                
                # Remove NaN values and ensure same length
                valid_mask = ~(np.isnan(temp) | np.isnan(sal))
                if np.sum(valid_mask) > 10:  # At least 10 valid points
                    temp_clean = temp[valid_mask]
                    sal_clean = sal[valid_mask]
                    
                    # Interpolate to fixed length
                    if len(temp_clean) > self.sequence_length:
                        # Downsample
                        indices = np.linspace(0, len(temp_clean)-1, self.sequence_length, dtype=int)
                        temp_profile = temp_clean[indices]
                        sal_profile = sal_clean[indices]
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
                    
                    return temp_profile, sal_profile
            
            ds.close()
            return None, None
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing {file_path.name}: {e}")
            return None, None
    
    def load_latest_data(self) -> bool:
        """Load processed data from after March 2003 (2003-2005)."""
        logger.info("📂 Loading processed latest ARGO data (2003-2005)...")
        
        try:
            # Load processed data
            processed_dir = Path("processed_data_all_latest")
            temp_file = processed_dir / "temperature_profiles_all_latest.npy"
            sal_file = processed_dir / "salinity_profiles_all_latest.npy"
            
            if not temp_file.exists() or not sal_file.exists():
                logger.error("❌ Processed data files not found! Please run process_latest_data.py first.")
                return False
            
            # Load data
            self.temperature_profiles = np.load(temp_file)
            self.salinity_profiles = np.load(sal_file)
            
            logger.info(f"✅ Loaded {len(self.temperature_profiles)} profiles")
            logger.info(f"📊 Data shape: {self.temperature_profiles.shape}")
            logger.info(f"📊 Temperature range: {self.temperature_profiles.min():.2f} to {self.temperature_profiles.max():.2f}")
            logger.info(f"📊 Salinity range: {self.salinity_profiles.min():.2f} to {self.salinity_profiles.max():.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading processed data: {e}")
            return False
    
    def create_data_loaders(self, model_type: str = "lstm") -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for training."""
        logger.info(f"🔄 Creating data loaders for {model_type.upper()} model...")
        
        # Prepare data
        X = self.temperature_profiles.copy()
        y = self.salinity_profiles.copy()
        
        if model_type.lower() == "lstm":
            # Reshape for LSTM: (samples, sequence_length, features)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            y = y.reshape(y.shape[0], y.shape[1], 1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = ARGOProfileDataset(X_train, y_train)
        val_dataset = ARGOProfileDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        logger.info(f"📊 Training samples: {len(train_dataset)}")
        logger.info(f"📊 Validation samples: {len(val_dataset)}")
        logger.info(f"📊 Batch size: {self.batch_size}")
        
        return train_loader, val_loader
    
    def train_model(self, model_type: str = "lstm", epochs: int = 100) -> bool:
        """Train model with latest data."""
        logger.info(f"🚀 Starting training with {model_type.upper()} model...")
        logger.info(f"📊 Using latest data from 2003-2005!")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(model_type)
        
        # Initialize model
        if model_type.lower() == "lstm":
            model = LatestLSTM(input_size=1, hidden_size=256, num_layers=4, output_size=1)
        elif model_type.lower() == "cnn":
            model = LatestCNN(sequence_length=self.sequence_length)
        else:
            logger.error(f"❌ Unknown model type: {model_type}")
            return False
        
        # Setup training
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"🖥️ Using device: {device}")
        
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
        
        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training
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
            
            # Validation
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
            
            # Calculate average losses
            avg_train_loss = train_loss / total_train_samples
            avg_val_loss = val_loss / total_val_samples
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Log progress every 10 epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                logger.info(f"📊 Processed {total_train_samples} training samples, {total_val_samples} validation samples")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                model_path = self.model_dir / f"argo_latest_{model_type}.pt"
                torch.save(model.state_dict(), model_path)
                if epoch % 10 == 0:
                    logger.info(f"💾 Saved best model to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                    break
            
            # Memory cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        # Save final model
        final_model_path = self.model_dir / f"argo_latest_{model_type}_final.pt"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"✅ Training completed! Final model saved to {final_model_path}")
        logger.info(f"📊 Best validation loss: {best_val_loss:.6f}")
        logger.info(f"📊 Processed {total_train_samples} training samples successfully!")
        
        return True

def main():
    """Main training function."""
    logger.info("🚀 Starting Latest Data ARGO Training...")
    logger.info("📊 Using data from after March 2003 (2003-2005)!")
    
    # Initialize trainer
    trainer = LatestDataTrainer()
    
    # Load latest data
    if not trainer.load_latest_data():
        logger.error("❌ Failed to load latest data!")
        return False
    
    # Train LSTM model
    logger.info("🤖 Training LSTM model with latest data...")
    if not trainer.train_model("lstm", epochs=100):
        logger.error("❌ LSTM training failed!")
        return False
    
    # Train CNN model
    logger.info("🤖 Training CNN model with latest data...")
    if not trainer.train_model("cnn", epochs=100):
        logger.error("❌ CNN training failed!")
        return False
    
    logger.info("🎉 Latest data training finished successfully!")
    logger.info("🏆 Models trained with data from 2003-2005!")
    logger.info("🌡️ Your MacBook processed the latest data like a champion!")
    return True

if __name__ == "__main__":
    main()
