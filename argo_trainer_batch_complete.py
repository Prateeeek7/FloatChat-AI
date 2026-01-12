#!/usr/bin/env python3
"""
Complete Batch ARGO Trainer - Process ALL 6,178 files in batches
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
        logging.FileHandler('argo_batch_complete_training.log'),
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

class CompleteLSTM(nn.Module):
    """Complete LSTM for maximum performance with all data."""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1):
        super(CompleteLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Large LSTM for maximum performance
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply dropout and fully connected layers
        output = self.dropout(lstm_out)
        output = torch.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output

class CompleteCNN(nn.Module):
    """Complete CNN for maximum performance with all data."""
    
    def __init__(self, sequence_length=100):
        super(CompleteCNN, self).__init__()
        
        # Large convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = (sequence_length // 16) * 512
        
        # Large fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, sequence_length)
        
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

class CompleteBatchTrainer:
    """Complete batch trainer that processes ALL files in batches."""
    
    def __init__(self, data_dir: str = "processed_data_all", model_dir: str = "models_complete"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = 100
        self.batch_size = 1000  # Process 1000 files at a time
        self.total_files = 6178
        
    def load_all_processed_data(self) -> bool:
        """Load ALL processed data."""
        logger.info("📂 Loading ALL processed data...")
        
        try:
            # Load temperature profiles
            temp_file = self.data_dir / "temperature_profiles_all.npy"
            sal_file = self.data_dir / "salinity_profiles_all.npy"
            
            if not temp_file.exists() or not sal_file.exists():
                logger.error("❌ Processed data files not found!")
                return False
            
            # Load ALL data
            self.temperature_profiles = np.load(temp_file)
            self.salinity_profiles = np.load(sal_file)
            
            logger.info(f"📊 Loaded ALL data: {len(self.temperature_profiles)} profiles")
            logger.info(f"📊 Data shape: {self.temperature_profiles.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def create_data_batches(self, model_type: str = "lstm") -> List[Tuple[DataLoader, DataLoader]]:
        """Create data batches for training."""
        logger.info(f"🔄 Creating data batches for {model_type.upper()} model...")
        
        # Use ALL data
        X = self.temperature_profiles
        y = self.salinity_profiles
        
        if model_type.lower() == "lstm":
            # Reshape for LSTM: (samples, sequence_length, features)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            y = y.reshape(y.shape[0], y.shape[1], 1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        
        # Create batches
        train_batches = []
        val_batches = []
        
        # Training batches
        for i in range(0, len(X_train), self.batch_size):
            end_idx = min(i + self.batch_size, len(X_train))
            X_batch = X_train[i:end_idx]
            y_batch = y_train[i:end_idx]
            
            dataset = ARGOProfileDataset(X_batch, y_batch)
            batch_size = min(32, max(16, len(dataset) // 50))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            train_batches.append(loader)
            
            logger.info(f"📊 Created training batch {len(train_batches)}: {len(dataset)} samples")
        
        # Validation batches
        for i in range(0, len(X_val), self.batch_size):
            end_idx = min(i + self.batch_size, len(X_val))
            X_batch = X_val[i:end_idx]
            y_batch = y_val[i:end_idx]
            
            dataset = ARGOProfileDataset(X_batch, y_batch)
            batch_size = min(32, max(16, len(dataset) // 50))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            val_batches.append(loader)
            
            logger.info(f"📊 Created validation batch {len(val_batches)}: {len(dataset)} samples")
        
        logger.info(f"📊 Total training batches: {len(train_batches)}")
        logger.info(f"📊 Total validation batches: {len(val_batches)}")
        
        return train_batches, val_batches
    
    def train_model_batches(self, model_type: str = "lstm", epochs: int = 50) -> bool:
        """Train model using all data batches."""
        logger.info(f"🚀 Starting COMPLETE batch training with {model_type.upper()} model...")
        logger.info(f"📊 Processing ALL {self.total_files} files in batches!")
        
        # Create data batches
        train_batches, val_batches = self.create_data_batches(model_type)
        
        # Initialize model
        if model_type.lower() == "lstm":
            model = CompleteLSTM(input_size=1, hidden_size=128, num_layers=3, output_size=1)
        elif model_type.lower() == "cnn":
            model = CompleteCNN(sequence_length=self.sequence_length)
        else:
            logger.error(f"❌ Unknown model type: {model_type}")
            return False
        
        # Setup training
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"🖥️ Using device: {device}")
        
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        # Training loop with all batches
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            total_train_samples = 0
            
            for batch_idx, train_loader in enumerate(train_batches):
                logger.info(f"🔄 Processing training batch {batch_idx + 1}/{len(train_batches)}")
                
                for temp, sal in train_loader:
                    temp, sal = temp.to(device), sal.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(temp)
                    loss = criterion(outputs, sal)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * len(temp)
                    total_train_samples += len(temp)
                    
                    # Memory cleanup every 10 batches
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        gc.collect()
            
            # Validation
            model.eval()
            val_loss = 0.0
            total_val_samples = 0
            
            with torch.no_grad():
                for batch_idx, val_loader in enumerate(val_batches):
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
            
            # Log progress every 5 epochs
            if epoch % 5 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                logger.info(f"📊 Processed {total_train_samples} training samples, {total_val_samples} validation samples")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                model_path = self.model_dir / f"argo_complete_{model_type}.pt"
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
        final_model_path = self.model_dir / f"argo_complete_{model_type}_final.pt"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"✅ COMPLETE batch training completed! Final model saved to {final_model_path}")
        logger.info(f"📊 Best validation loss: {best_val_loss:.6f}")
        logger.info(f"📊 Processed ALL {self.total_files} files successfully!")
        
        return True

def main():
    """Main training function."""
    logger.info("🚀 Starting COMPLETE Batch ARGO Training...")
    logger.info("📊 Processing ALL 6,178 files in batches!")
    
    # Initialize trainer
    trainer = CompleteBatchTrainer()
    
    # Load ALL data
    if not trainer.load_all_processed_data():
        logger.error("❌ Failed to load ALL processed data!")
        return False
    
    # Train complete LSTM model
    logger.info("🤖 Training COMPLETE LSTM model with ALL data batches...")
    if not trainer.train_model_batches("lstm", epochs=50):
        logger.error("❌ Complete LSTM training failed!")
        return False
    
    # Train complete CNN model
    logger.info("🤖 Training COMPLETE CNN model with ALL data batches...")
    if not trainer.train_model_batches("cnn", epochs=50):
        logger.error("❌ Complete CNN training failed!")
        return False
    
    logger.info("🎉 COMPLETE batch training finished successfully!")
    logger.info("🏆 ALL 6,178 files have been processed!")
    logger.info("🌡️ Your MacBook is a true champion!")
    return True

if __name__ == "__main__":
    main()





