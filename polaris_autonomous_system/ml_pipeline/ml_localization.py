#!/usr/bin/env python3

"""
Machine Learning-based Localization for Autonomous Vehicles

This module implements neural network-based localization that learns to replicate
the behavior of the Extended Kalman Filter using real sensor data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

class LocalizationDataset(Dataset):
    """
    PyTorch Dataset for localization training data.
    
    Input: Sensor data (IMU, GPS, odometry)
    Output: Position, velocity, attitude estimates
    """
    
    def __init__(self, sensor_data, target_data, sequence_length=10):
        """
        Initialize dataset.
        
        Args:
            sensor_data: Input sensor measurements
            target_data: Target localization outputs (from EKF)
            sequence_length: Number of timesteps for sequence modeling
        """
        self.sensor_data = torch.FloatTensor(sensor_data)
        self.target_data = torch.FloatTensor(target_data)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sensor_data) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Get sequence of sensor data
        sensor_seq = self.sensor_data[idx:idx + self.sequence_length]
        
        # Get target for the last timestep in sequence
        target = self.target_data[idx + self.sequence_length - 1]
        
        return sensor_seq, target

class LocalizationLSTM(nn.Module):
    """
    LSTM-based neural network for vehicle localization.
    
    Architecture:
    - Input: Sequence of sensor measurements
    - LSTM layers: Process temporal dependencies
    - Dense layers: Map to position/velocity/attitude
    - Output: 12-DOF state vector (same as EKF)
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=12):
        super(LocalizationLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers for temporal processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Dense layers for output mapping
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Dense layers
        output = self.fc_layers(last_output)
        
        return output

class MLLocalizationTrainer:
    """
    Trainer class for machine learning localization model.

    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler_input = StandardScaler()
        self.scaler_output = StandardScaler()
        
    def prepare_data(self, data_file, sequence_length=10, test_size=0.2):
        """
        Prepare training data from processed sensor data.
        
        Args:
            data_file: Path to processed CSV file
            sequence_length: Length of input sequences
            test_size: Fraction of data for testing
        """
        print("Preparing ML training data...")
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Prepare input features (sensor data)
        sensor_cols = [
            'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
            'lin_acc_x', 'lin_acc_y', 'lin_acc_z',
            'latitude', 'longitude', 'altitude',
            'speed'
        ]
        
        # Filter available columns
        available_sensor_cols = [col for col in sensor_cols if col in df.columns]
        sensor_data = df[available_sensor_cols].fillna(0).values
        
        # Prepare target features (from EKF or ground truth)
        target_cols = [
            'enu_x', 'enu_y', 'enu_z',  # Position
            'vel_x', 'vel_y', 'vel_z',  # Velocity
            'roll', 'pitch', 'yaw',     # Attitude
            'omega_roll', 'omega_pitch', 'omega_yaw'  # Angular velocity
        ]
        
        # Filter available target columns
        available_target_cols = [col for col in target_cols if col in df.columns]
        target_data = df[available_target_cols].fillna(0).values
        
        print(f"Input features: {len(available_sensor_cols)}")
        print(f"Target features: {len(available_target_cols)}")
        print(f"Data points: {len(sensor_data)}")
        
        # Normalize data
        sensor_data_scaled = self.scaler_input.fit_transform(sensor_data)
        target_data_scaled = self.scaler_output.fit_transform(target_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sensor_data_scaled, target_data_scaled, 
            test_size=test_size, random_state=42
        )
        
        # Create datasets
        train_dataset = LocalizationDataset(X_train, y_train, sequence_length)
        test_dataset = LocalizationDataset(X_test, y_test, sequence_length)
        
        return train_dataset, test_dataset, available_sensor_cols, available_target_cols
    
    def train(self, train_dataset, test_dataset, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train the localization model.
        """
        print("Training ML localization model...")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training history
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_sensor, batch_target in train_loader:
                batch_sensor = batch_sensor.to(self.device)
                batch_target = batch_target.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_sensor)
                loss = criterion(outputs, batch_target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for batch_sensor, batch_target in test_loader:
                    batch_sensor = batch_sensor.to(self.device)
                    batch_target = batch_target.to(self.device)
                    
                    outputs = self.model(batch_sensor)
                    loss = criterion(outputs, batch_target)
                    test_loss += loss.item()
            
            # Average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_test_loss)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, Test Loss = {avg_test_loss:.6f}")
        
        return train_losses, test_losses
    
    def evaluate(self, test_dataset):
        """
        Evaluate the trained model.
        """
        print("Evaluating ML localization model...")
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_sensor, batch_target in test_loader:
                batch_sensor = batch_sensor.to(self.device)
                batch_target = batch_target.to(self.device)
                
                predictions = self.model(batch_sensor)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_target.cpu().numpy())
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Inverse transform to original scale
        predictions_original = self.scaler_output.inverse_transform(predictions)
        targets_original = self.scaler_output.inverse_transform(targets)
        
        # Calculate metrics
        mse = mean_squared_error(targets_original, predictions_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets_original, predictions_original)
        
        print(f"Evaluation Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")
        
        return predictions_original, targets_original, rmse, r2
    
    def save_model(self, model_path, scaler_path):
        """
        Save the trained model and scalers.
        """
        torch.save(self.model.state_dict(), model_path)
        joblib.dump({
            'input_scaler': self.scaler_input,
            'output_scaler': self.scaler_output
        }, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scalers saved to {scaler_path}")
    
    def load_model(self, model_path, scaler_path):
        """
        Load a trained model and scalers.
        """
        self.model.load_state_dict(torch.load(model_path))
        scalers = joblib.load(scaler_path)
        self.scaler_input = scalers['input_scaler']
        self.scaler_output = scalers['output_scaler']
        print(f"Model loaded from {model_path}")

class MLLocalizationProcessor:
    """
    Real-time ML localization processor.
    """
    
    def __init__(self, model_path, scaler_path, device='cpu'):
        self.device = device
        self.model = LocalizationLSTM(input_size=10, hidden_size=128, num_layers=2)
        self.trainer = MLLocalizationTrainer(self.model, device)
        self.trainer.load_model(model_path, scaler_path)
        
        # Buffer for sequence processing
        self.sensor_buffer = []
        self.sequence_length = 10
        
    def process_measurement(self, sensor_data):
        """
        Process a single sensor measurement.
        
        Args:
            sensor_data: Dictionary with sensor measurements
        """
        # Convert to array format
        sensor_array = np.array([
            sensor_data.get('ang_vel_x', 0),
            sensor_data.get('ang_vel_y', 0),
            sensor_data.get('ang_vel_z', 0),
            sensor_data.get('lin_acc_x', 0),
            sensor_data.get('lin_acc_y', 0),
            sensor_data.get('lin_acc_z', 0),
            sensor_data.get('latitude', 0),
            sensor_data.get('longitude', 0),
            sensor_data.get('altitude', 0),
            sensor_data.get('speed', 0)
        ]).reshape(1, -1)
        
        # Normalize
        sensor_scaled = self.trainer.scaler_input.transform(sensor_array)
        
        # Add to buffer
        self.sensor_buffer.append(sensor_scaled[0])
        
        # Keep only recent measurements
        if len(self.sensor_buffer) > self.sequence_length:
            self.sensor_buffer.pop(0)
        
        # Process if we have enough data
        if len(self.sensor_buffer) == self.sequence_length:
            # Convert to tensor
            sequence = torch.FloatTensor(self.sensor_buffer).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                prediction = self.model(sequence)
                prediction_original = self.trainer.scaler_output.inverse_transform(
                    prediction.cpu().numpy()
                )[0]
            
            return {
                'position': prediction_original[0:3],
                'velocity': prediction_original[3:6],
                'attitude': prediction_original[6:9],
                'angular_velocity': prediction_original[9:12]
            }
        
        return None

def main():
    """
    Main function to train and evaluate ML localization model.
    """
    print("🚀 Starting ML Localization Training")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = LocalizationLSTM(input_size=10, hidden_size=128, num_layers=2, output_size=12)
    trainer = MLLocalizationTrainer(model, device)
    
    # Prepare data
    data_file = '/home/valid_monke/ros2_ws/polaris_localization/data/processed/localization_training_data.csv'
    train_dataset, test_dataset, input_cols, target_cols = trainer.prepare_data(data_file)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Train model
    train_losses, test_losses = trainer.train(train_dataset, test_dataset, epochs=50)
    
    # Evaluate model
    predictions, targets, rmse, r2 = trainer.evaluate(test_dataset)
    
    # Save model
    model_path = '/home/valid_monke/ros2_ws/polaris_localization/models/ml_localization_model.pth'
    scaler_path = '/home/valid_monke/ros2_ws/polaris_localization/models/ml_scalers.pkl'
    
    # Create models directory
    Path(model_path).parent.mkdir(exist_ok=True)
    
    trainer.save_model(model_path, scaler_path)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(targets[:, 0], predictions[:, 0], alpha=0.5)
    plt.plot([targets[:, 0].min(), targets[:, 0].max()], 
             [targets[:, 0].min(), targets[:, 0].max()], 'r--')
    plt.xlabel('True Position X')
    plt.ylabel('Predicted Position X')
    plt.title(f'Position Prediction (R² = {r2:.3f})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/valid_monke/ros2_ws/polaris_localization/results/ml_training_results.png', dpi=300)
    plt.show()
    
    print("\n🎉 ML Localization Training Complete!")
    print(f"Final RMSE: {rmse:.4f}")
    print(f"Final R² Score: {r2:.4f}")

if __name__ == '__main__':
    main()
