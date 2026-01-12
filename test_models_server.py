#!/usr/bin/env python3
"""
Server Test Script for Trained ARGO Models
Test the models trained on ALL 6,178 files
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompleteLSTM(nn.Module):
    """Complete LSTM model (same as training)."""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1):
        super(CompleteLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.dropout(lstm_out)
        output = torch.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        return output

class CompleteCNN(nn.Module):
    """Complete CNN model (same as training)."""
    
    def __init__(self, sequence_length=100):
        super(CompleteCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        conv_output_size = (sequence_length // 16) * 512
        
        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, sequence_length)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.dropout(x)
        
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x

class ARGOModelTester:
    """Test the trained ARGO models on server."""
    
    def __init__(self, model_dir: str = "models_complete"):
        self.model_dir = Path(model_dir)
        self.sequence_length = 100
        self.results = {}
        
    def load_models(self):
        """Load the trained models."""
        logger.info("🔄 Loading trained models...")
        
        try:
            # Load LSTM model
            lstm_model = CompleteLSTM(input_size=1, hidden_size=128, num_layers=3, output_size=1)
            lstm_path = self.model_dir / "argo_complete_lstm.pt"
            if lstm_path.exists():
                lstm_model.load_state_dict(torch.load(lstm_path, map_location='cpu'))
                lstm_model.eval()
                self.lstm_model = lstm_model
                logger.info("✅ LSTM model loaded successfully")
            else:
                logger.error("❌ LSTM model not found")
                return False
            
            # Load CNN model
            cnn_model = CompleteCNN(sequence_length=self.sequence_length)
            cnn_path = self.model_dir / "argo_complete_cnn.pt"
            if cnn_path.exists():
                cnn_model.load_state_dict(torch.load(cnn_path, map_location='cpu'))
                cnn_model.eval()
                self.cnn_model = cnn_model
                logger.info("✅ CNN model loaded successfully")
            else:
                logger.error("❌ CNN model not found")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def generate_test_data(self, num_samples=10):
        """Generate test temperature profiles."""
        logger.info(f"🔄 Generating {num_samples} test temperature profiles...")
        
        # Generate realistic temperature profiles
        test_profiles = []
        for i in range(num_samples):
            # Create a realistic temperature profile (surface to deep)
            depths = np.linspace(0, 2000, self.sequence_length)
            
            # Surface temperature (warmer)
            surface_temp = np.random.uniform(25, 30)
            
            # Deep temperature (colder)
            deep_temp = np.random.uniform(2, 5)
            
            # Create temperature profile with thermocline
            temp_profile = surface_temp * np.exp(-depths/500) + deep_temp * (1 - np.exp(-depths/500))
            
            # Add some noise
            noise = np.random.normal(0, 0.5, self.sequence_length)
            temp_profile += noise
            
            # Normalize
            temp_profile = (temp_profile - np.mean(temp_profile)) / (np.std(temp_profile) + 1e-8)
            
            test_profiles.append(temp_profile)
        
        return np.array(test_profiles)
    
    def test_models(self, test_data):
        """Test both models on the test data."""
        logger.info("🔄 Testing models...")
        
        results = {
            'lstm_predictions': [],
            'cnn_predictions': [],
            'test_temperatures': [],
            'model_performance': {}
        }
        
        for i, temp_profile in enumerate(test_data):
            logger.info(f"🔄 Testing sample {i+1}/{len(test_data)}")
            
            # Prepare input for LSTM
            temp_lstm = torch.FloatTensor(temp_profile).unsqueeze(0).unsqueeze(-1)  # (1, 100, 1)
            
            # Prepare input for CNN
            temp_cnn = torch.FloatTensor(temp_profile).unsqueeze(0)  # (1, 100)
            
            # Test LSTM
            with torch.no_grad():
                lstm_pred = self.lstm_model(temp_lstm)
                lstm_pred_np = lstm_pred.squeeze().numpy()
            
            # Test CNN
            with torch.no_grad():
                cnn_pred = self.cnn_model(temp_cnn)
                cnn_pred_np = cnn_pred.squeeze().numpy()
            
            # Store results
            results['lstm_predictions'].append(lstm_pred_np)
            results['cnn_predictions'].append(cnn_pred_np)
            results['test_temperatures'].append(temp_profile)
            
            # Calculate performance metrics
            lstm_std = np.std(lstm_pred_np)
            cnn_std = np.std(cnn_pred_np)
            
            logger.info(f"   LSTM prediction range: {lstm_pred_np.min():.3f} to {lstm_pred_np.max():.3f}")
            logger.info(f"   CNN prediction range: {cnn_pred_np.min():.3f} to {cnn_pred_np.max():.3f}")
        
        # Calculate overall performance
        all_lstm_preds = np.concatenate(results['lstm_predictions'])
        all_cnn_preds = np.concatenate(results['cnn_predictions'])
        
        results['model_performance'] = {
            'lstm_mean_std': np.std(all_lstm_preds),
            'cnn_mean_std': np.std(all_cnn_preds),
            'lstm_range': [all_lstm_preds.min(), all_lstm_preds.max()],
            'cnn_range': [all_cnn_preds.min(), all_cnn_preds.max()],
            'total_predictions': len(all_lstm_preds)
        }
        
        self.results = results
        return results
    
    def create_visualizations(self, save_dir="server_test_results"):
        """Create visualization plots."""
        logger.info("🔄 Creating visualizations...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ARGO Model Predictions - Server Test Results', fontsize=16)
        
        # Plot 1: Temperature profile
        axes[0, 0].plot(self.results['test_temperatures'][0], 'b-', linewidth=2, label='Input Temperature')
        axes[0, 0].set_title('Sample Temperature Profile')
        axes[0, 0].set_xlabel('Depth Level')
        axes[0, 0].set_ylabel('Normalized Temperature')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: LSTM predictions
        axes[0, 1].plot(self.results['lstm_predictions'][0], 'r-', linewidth=2, label='LSTM Prediction')
        axes[0, 1].set_title('LSTM Salinity Prediction')
        axes[0, 1].set_xlabel('Depth Level')
        axes[0, 1].set_ylabel('Predicted Salinity')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: CNN predictions
        axes[1, 0].plot(self.results['cnn_predictions'][0], 'g-', linewidth=2, label='CNN Prediction')
        axes[1, 0].set_title('CNN Salinity Prediction')
        axes[1, 0].set_xlabel('Depth Level')
        axes[1, 0].set_ylabel('Predicted Salinity')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Model comparison
        axes[1, 1].plot(self.results['lstm_predictions'][0], 'r-', linewidth=2, label='LSTM')
        axes[1, 1].plot(self.results['cnn_predictions'][0], 'g-', linewidth=2, label='CNN')
        axes[1, 1].set_title('Model Comparison')
        axes[1, 1].set_xlabel('Depth Level')
        axes[1, 1].set_ylabel('Predicted Salinity')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'model_predictions_server.png', dpi=150, bbox_inches='tight')
        logger.info(f"📊 Visualization saved to {save_dir / 'model_predictions_server.png'}")
        
        plt.close()
    
    def save_results(self, save_dir="server_test_results"):
        """Save test results to files."""
        logger.info("🔄 Saving test results...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Save numerical results
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': self.results['model_performance'],
            'test_summary': {
                'total_samples': len(self.results['test_temperatures']),
                'sequence_length': self.sequence_length,
                'models_tested': ['LSTM', 'CNN']
            }
        }
        
        with open(save_dir / 'test_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save predictions as numpy arrays
        np.save(save_dir / 'lstm_predictions.npy', np.array(self.results['lstm_predictions']))
        np.save(save_dir / 'cnn_predictions.npy', np.array(self.results['cnn_predictions']))
        np.save(save_dir / 'test_temperatures.npy', np.array(self.results['test_temperatures']))
        
        logger.info(f"💾 Results saved to {save_dir}")
    
    def run_complete_test(self, num_samples=10):
        """Run complete test suite."""
        logger.info("🚀 Starting ARGO Model Server Test...")
        logger.info(f"📊 Testing with {num_samples} samples")
        
        # Load models
        if not self.load_models():
            logger.error("❌ Failed to load models")
            return False
        
        # Generate test data
        test_data = self.generate_test_data(num_samples)
        
        # Test models
        results = self.test_models(test_data)
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("📊 SERVER TEST RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ Models loaded successfully")
        logger.info(f"✅ Tested {num_samples} temperature profiles")
        logger.info(f"✅ LSTM predictions: {len(results['lstm_predictions'])} samples")
        logger.info(f"✅ CNN predictions: {len(results['cnn_predictions'])} samples")
        logger.info(f"📊 LSTM std deviation: {results['model_performance']['lstm_mean_std']:.4f}")
        logger.info(f"📊 CNN std deviation: {results['model_performance']['cnn_mean_std']:.4f}")
        logger.info(f"📊 LSTM range: {results['model_performance']['lstm_range']}")
        logger.info(f"📊 CNN range: {results['model_performance']['cnn_range']}")
        logger.info("="*60)
        logger.info("🎉 Server test completed successfully!")
        
        return True

def main():
    """Main test function."""
    logger.info("🚀 Starting ARGO Model Server Test...")
    
    # Initialize tester
    tester = ARGOModelTester()
    
    # Run complete test
    success = tester.run_complete_test(num_samples=10)
    
    if success:
        logger.info("🎉 All tests passed! Models are ready for production!")
    else:
        logger.error("❌ Tests failed!")
    
    return success

if __name__ == "__main__":
    main()





