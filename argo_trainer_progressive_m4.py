#!/usr/bin/env python3
"""
Progressive M4 MacBook ARGO Trainer - Gradually increase data usage
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
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('argo_progressive_training.log'),
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

class ProgressiveLSTM(nn.Module):
    """Progressive LSTM that can handle varying data sizes."""
    
    def __init__(self, input_size=1, hidden_size=96, num_layers=2, output_size=1):
        super(ProgressiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Slightly larger than lightweight but still M4-friendly
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.25)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply dropout and fully connected layer
        output = self.dropout(lstm_out)
        output = self.fc(output)
        
        return output

class ProgressiveCNN(nn.Module):
    """Progressive CNN that can handle varying data sizes."""
    
    def __init__(self, sequence_length=100):
        super(ProgressiveCNN, self).__init__()
        
        # Balanced convolutional layers
        self.conv1 = nn.Conv1d(1, 48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(48, 96, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(96, 192, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = (sequence_length // 8) * 192
        
        # Balanced fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, sequence_length)
        
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
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class ProgressiveM4Trainer:
    """Progressive M4 trainer that gradually increases data usage."""
    
    def __init__(self, data_dir: str = "processed_data_all", model_dir: str = "models_progressive"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = 100
        self.data_sizes = [1000, 2000, 3000, 4000, 5000]  # Progressive data sizes
        self.current_size = 0
        
    def load_processed_data_progressive(self, size: int) -> bool:
        """Load processed data with specified size."""
        logger.info(f"📂 Loading {size} profiles for progressive training...")
        
        try:
            # Load temperature profiles
            temp_file = self.data_dir / "temperature_profiles_all.npy"
            sal_file = self.data_dir / "salinity_profiles_all.npy"
            
            if not temp_file.exists() or not sal_file.exists():
                logger.error("❌ Processed data files not found!")
                return False
            
            # Load full data
            full_temp = np.load(temp_file)
            full_sal = np.load(sal_file)
            
            logger.info(f"📊 Full dataset: {len(full_temp)} profiles")
            
            # Take a representative subset
            if len(full_temp) > size:
                # Use stratified sampling to get diverse data
                indices = np.random.choice(len(full_temp), size, replace=False)
                self.temperature_profiles = full_temp[indices]
                self.salinity_profiles = full_sal[indices]
                logger.info(f"📊 Using subset: {size} profiles")
            else:
                self.temperature_profiles = full_temp
                self.salinity_profiles = full_sal
                logger.info(f"📊 Using full dataset: {len(full_temp)} profiles")
            
            # Clear memory
            del full_temp, full_sal
            gc.collect()
            
            logger.info(f"📊 Data shape: {self.temperature_profiles.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def prepare_data(self, model_type: str = "lstm") -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        logger.info(f"🔄 Preparing data for {model_type.upper()} model...")
        
        # Use current data
        X = self.temperature_profiles
        y = self.salinity_profiles
        
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
        
        # Adjust batch size based on data size
        batch_size = min(32, max(8, len(train_dataset) // 50))
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"📊 Training samples: {len(train_dataset)}")
        logger.info(f"📊 Validation samples: {len(val_dataset)}")
        logger.info(f"📊 Batch size: {batch_size}")
        
        return train_loader, val_loader
    
    def train_model(self, model_type: str = "lstm", epochs: int = 30) -> bool:
        """Train the specified model type."""
        logger.info(f"🚀 Starting progressive training with {model_type.upper()} model...")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(model_type)
        
        # Initialize model
        if model_type.lower() == "lstm":
            model = ProgressiveLSTM(input_size=1, hidden_size=96, num_layers=2, output_size=1)
        elif model_type.lower() == "cnn":
            model = ProgressiveCNN(sequence_length=self.sequence_length)
        else:
            logger.error(f"❌ Unknown model type: {model_type}")
            return False
        
        # Setup training with M4 optimizations
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"🖥️ Using device: {device}")
        
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
        
        best_val_loss = float('inf')
        patience = 8
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_idx, (temp, sal) in enumerate(train_loader):
                temp, sal = temp.to(device), sal.to(device)
                
                optimizer.zero_grad()
                outputs = model(temp)
                loss = criterion(outputs, sal)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Memory cleanup every 20 batches
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for temp, sal in val_loader:
                    temp, sal = temp.to(device), sal.to(device)
                    outputs = model(temp)
                    loss = criterion(outputs, sal)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Log progress every 5 epochs
            if epoch % 5 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                model_path = self.model_dir / f"argo_progressive_{model_type}_{self.current_size}.pt"
                torch.save(model.state_dict(), model_path)
                if epoch % 5 == 0:
                    logger.info(f"💾 Saved best model to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                    break
            
            # Memory cleanup after each epoch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        # Save final model
        final_model_path = self.model_dir / f"argo_progressive_{model_type}_{self.current_size}_final.pt"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"✅ Training completed! Final model saved to {final_model_path}")
        logger.info(f"📊 Best validation loss: {best_val_loss:.6f}")
        
        return True
    
    def progressive_training(self):
        """Run progressive training with increasing data sizes."""
        logger.info("🚀 Starting Progressive M4 ARGO Training...")
        
        for i, size in enumerate(self.data_sizes):
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 PROGRESSIVE TRAINING PHASE {i+1}/{len(self.data_sizes)}")
            logger.info(f"📊 Data size: {size} profiles")
            logger.info(f"{'='*60}")
            
            self.current_size = size
            
            # Load data for this phase
            if not self.load_processed_data_progressive(size):
                logger.error(f"❌ Failed to load data for size {size}")
                continue
            
            # Train LSTM model
            logger.info(f"🤖 Training LSTM model with {size} profiles...")
            if not self.train_model("lstm", epochs=30):
                logger.error(f"❌ LSTM training failed for size {size}")
                continue
            
            # Train CNN model
            logger.info(f"🤖 Training CNN model with {size} profiles...")
            if not self.train_model("cnn", epochs=30):
                logger.error(f"❌ CNN training failed for size {size}")
                continue
            
            logger.info(f"✅ Phase {i+1} completed successfully!")
            
            # Memory cleanup between phases
            del self.temperature_profiles, self.salinity_profiles
            gc.collect()
        
        logger.info("🎉 All progressive training phases completed!")
        logger.info("🌡️ Your MacBook stayed cool throughout the process!")

def main():
    """Main training function."""
    # Initialize trainer
    trainer = ProgressiveM4Trainer()
    
    # Run progressive training
    trainer.progressive_training()
    
    return True

if __name__ == "__main__":
    main()





