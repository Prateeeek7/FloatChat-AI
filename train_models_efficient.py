#!/usr/bin/env python3
"""
Efficient model training with a sample of the 16,225 profiles
Uses existing models and updates them with new data
"""

import os
import sqlite3
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EfficientLSTM(nn.Module):
    """Efficient LSTM model for ARGO data prediction"""
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1, dropout=0.3):
        super(EfficientLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.dropout(lstm_out)
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        return output

class EfficientCNN(nn.Module):
    """Efficient CNN model for ARGO data prediction"""
    def __init__(self, sequence_length=50):
        super(EfficientCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the output size after convolutions and pooling
        conv_output_size = (sequence_length // 8) * 128
        
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, sequence_length)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

class ARGODataset(Dataset):
    """Dataset class for ARGO data"""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

def load_efficient_data(sample_size=5000):
    """Load a sample of profiles for efficient training"""
    logger.info(f"Loading {sample_size} profiles for efficient training...")
    
    db_path = "data/processed/argo_data_full.db"
    conn = sqlite3.connect(db_path)
    
    # Load a sample of profiles with valid data
    query = f"""
    SELECT platform_number, latitude, longitude, date, 
           temperature_data, salinity_data, pressure_data
    FROM argo_profiles 
    WHERE latitude != 0 AND longitude != 0 
    AND temperature_data != '[]' 
    AND salinity_data != '[]' 
    AND pressure_data != '[]'
    ORDER BY RANDOM()
    LIMIT {sample_size}
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} profiles for training")
    
    # Process temperature and salinity data
    all_sequences = []
    all_targets = []
    
    for idx, row in df.iterrows():
        try:
            # Parse JSON data
            temp_data = json.loads(row['temperature_data'])
            sal_data = json.loads(row['salinity_data'])
            pres_data = json.loads(row['pressure_data'])
            
            # Filter valid values
            valid_indices = []
            for i in range(min(len(temp_data), len(sal_data), len(pres_data))):
                if (isinstance(temp_data[i], (int, float)) and 
                    isinstance(sal_data[i], (int, float)) and 
                    isinstance(pres_data[i], (int, float)) and
                    temp_data[i] != 0 and sal_data[i] != 0 and pres_data[i] != 0):
                    valid_indices.append(i)
            
            if len(valid_indices) >= 10:  # Minimum sequence length
                # Create sequences
                temp_clean = [temp_data[i] for i in valid_indices]
                sal_clean = [sal_data[i] for i in valid_indices]
                pres_clean = [pres_data[i] for i in valid_indices]
                
                # Normalize data
                temp_array = np.array(temp_clean)
                sal_array = np.array(sal_clean)
                pres_array = np.array(pres_clean)
                
                # Create input sequences (temperature + salinity)
                input_seq = np.column_stack([temp_array, sal_array])
                
                # Create target sequences (pressure)
                target_seq = pres_array
                
                # Pad or truncate to fixed length
                max_len = 50  # Smaller sequence length for efficiency
                if len(input_seq) >= max_len:
                    input_seq = input_seq[:max_len]
                    target_seq = target_seq[:max_len]
                else:
                    # Pad with zeros
                    pad_len = max_len - len(input_seq)
                    input_seq = np.pad(input_seq, ((0, pad_len), (0, 0)), mode='constant')
                    target_seq = np.pad(target_seq, (0, pad_len), mode='constant')
                
                all_sequences.append(input_seq)
                all_targets.append(target_seq)
                
        except Exception as e:
            logger.warning(f"Error processing profile {idx}: {e}")
            continue
    
    logger.info(f"Created {len(all_sequences)} training sequences")
    return np.array(all_sequences), np.array(all_targets)

def train_efficient_models():
    """Train both LSTM and CNN models efficiently"""
    logger.info("Starting efficient model training...")
    
    # Load data
    sequences, targets = load_efficient_data(sample_size=5000)
    
    if len(sequences) == 0:
        logger.error("No training data available!")
        return
    
    # Normalize data
    scaler_input = StandardScaler()
    scaler_target = StandardScaler()
    
    # Reshape for normalization
    sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
    targets_reshaped = targets.reshape(-1, 1)
    
    sequences_normalized = scaler_input.fit_transform(sequences_reshaped)
    targets_normalized = scaler_target.fit_transform(targets_reshaped)
    
    # Reshape back
    sequences_normalized = sequences_normalized.reshape(sequences.shape)
    targets_normalized = targets_normalized.reshape(targets.shape)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences_normalized, targets_normalized, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Create datasets
    train_dataset = ARGODataset(X_train, y_train)
    test_dataset = ARGODataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Train LSTM model
    logger.info("Training LSTM model...")
    lstm_model = EfficientLSTM(input_size=2, hidden_size=128, num_layers=3, output_size=1)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    lstm_criterion = nn.MSELoss()
    
    lstm_train_losses = []
    lstm_val_losses = []
    
    for epoch in range(20):  # Fewer epochs for efficiency
        # Training
        lstm_model.train()
        train_loss = 0
        for batch_sequences, batch_targets in train_loader:
            lstm_optimizer.zero_grad()
            outputs = lstm_model(batch_sequences)
            loss = lstm_criterion(outputs.squeeze(), batch_targets)
            loss.backward()
            lstm_optimizer.step()
            train_loss += loss.item()
        
        # Validation
        lstm_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_sequences, batch_targets in test_loader:
                outputs = lstm_model(batch_sequences)
                loss = lstm_criterion(outputs.squeeze(), batch_targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        
        lstm_train_losses.append(train_loss)
        lstm_val_losses.append(val_loss)
        
        if epoch % 5 == 0:
            logger.info(f"LSTM Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Train CNN model
    logger.info("Training CNN model...")
    cnn_model = EfficientCNN(sequence_length=50)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    cnn_criterion = nn.MSELoss()
    
    cnn_train_losses = []
    cnn_val_losses = []
    
    for epoch in range(20):  # Fewer epochs for efficiency
        # Training
        cnn_model.train()
        train_loss = 0
        for batch_sequences, batch_targets in train_loader:
            cnn_optimizer.zero_grad()
            # Use only temperature for CNN input
            temp_input = batch_sequences[:, :, 0:1]
            outputs = cnn_model(temp_input)
            loss = cnn_criterion(outputs, batch_targets)
            loss.backward()
            cnn_optimizer.step()
            train_loss += loss.item()
        
        # Validation
        cnn_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_sequences, batch_targets in test_loader:
                temp_input = batch_sequences[:, :, 0:1]
                outputs = cnn_model(temp_input)
                loss = cnn_criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        
        cnn_train_losses.append(train_loss)
        cnn_val_losses.append(val_loss)
        
        if epoch % 5 == 0:
            logger.info(f"CNN Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Save models
    os.makedirs("models_efficient", exist_ok=True)
    
    torch.save(lstm_model.state_dict(), "models_efficient/argo_efficient_lstm.pt")
    torch.save(cnn_model.state_dict(), "models_efficient/argo_efficient_cnn.pt")
    
    # Save scalers
    import joblib
    joblib.dump(scaler_input, "models_efficient/scaler_input.pkl")
    joblib.dump(scaler_target, "models_efficient/scaler_target.pkl")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_train_losses, label='LSTM Train')
    plt.plot(lstm_val_losses, label='LSTM Val')
    plt.title('LSTM Training Curves (Efficient)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(cnn_train_losses, label='CNN Train')
    plt.plot(cnn_val_losses, label='CNN Val')
    plt.title('CNN Training Curves (Efficient)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models_efficient/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Efficient model training complete!")
    logger.info(f"LSTM final train loss: {lstm_train_losses[-1]:.6f}")
    logger.info(f"LSTM final val loss: {lstm_val_losses[-1]:.6f}")
    logger.info(f"CNN final train loss: {cnn_train_losses[-1]:.6f}")
    logger.info(f"CNN final val loss: {cnn_val_losses[-1]:.6f}")
    
    return lstm_model, cnn_model, scaler_input, scaler_target

if __name__ == "__main__":
    logger.info("Starting efficient model training with sample of 16,225 profiles...")
    lstm_model, cnn_model, scaler_input, scaler_target = train_efficient_models()
    logger.info("Training complete! Models saved to models_efficient/")


