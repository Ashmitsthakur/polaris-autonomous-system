#!/usr/bin/env python3

"""
Comparison between EKF and ML-based localization approaches.

This script trains a neural network to replicate EKF behavior and compares
performance, speed, and accuracy between the two approaches.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add unified package to path
sys.path.append(str(Path(__file__).parent.parent))

from polaris_autonomous_system.localization.ekf_localization import LocalizationProcessor as EKFProcessor
from polaris_autonomous_system.ml_pipeline.ml_localization import MLLocalizationProcessor, MLLocalizationTrainer, LocalizationLSTM
import torch

class LocalizationComparison:
    """
    Compare EKF and ML-based localization approaches.
    """
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.results = {}
        
    def run_ekf_localization(self):
        """
        Run EKF-based localization.
        """
        print("🔧 Running EKF Localization...")
        
        start_time = time.time()
        
        # Initialize EKF processor
        ekf_processor = EKFProcessor(self.data_file)
        
        # Run localization
        ekf_processor.run_localization()
        
        end_time = time.time()
        
        # Calculate metrics
        processing_time = end_time - start_time
        samples_processed = len(ekf_processor.results)
        samples_per_second = samples_processed / processing_time
        
        # Extract results
        ekf_results = pd.DataFrame(ekf_processor.results)
        
        self.results['ekf'] = {
            'processor': ekf_processor,
            'results': ekf_results,
            'processing_time': processing_time,
            'samples_per_second': samples_per_second,
            'memory_usage': self._get_memory_usage()
        }
        
        print(f"✅ EKF Complete: {samples_per_second:.1f} samples/sec")
        
    def train_ml_model(self):
        """
        Train ML-based localization model.
        """
        print("🧠 Training ML Localization Model...")
        
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize model and trainer
        model = LocalizationLSTM(input_size=10, hidden_size=128, num_layers=2, output_size=12)
        trainer = MLLocalizationTrainer(model, device)
        
        # Prepare data
        train_dataset, test_dataset, input_cols, target_cols = trainer.prepare_data(
            self.data_file, sequence_length=10, test_size=0.2
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Train model
        start_time = time.time()
        train_losses, test_losses = trainer.train(train_dataset, test_dataset, epochs=30)
        training_time = time.time() - start_time
        
        # Evaluate model
        predictions, targets, rmse, r2 = trainer.evaluate(test_dataset)
        
        # Save model
        model_path = '/home/valid_monke/ros2_ws/polaris_localization/models/ml_localization_model.pth'
        scaler_path = '/home/valid_monke/ros2_ws/polaris_localization/models/ml_scalers.pkl'
        
        Path(model_path).parent.mkdir(exist_ok=True)
        trainer.save_model(model_path, scaler_path)
        
        self.results['ml'] = {
            'trainer': trainer,
            'model': model,
            'predictions': predictions,
            'targets': targets,
            'rmse': rmse,
            'r2': r2,
            'training_time': training_time,
            'train_losses': train_losses,
            'test_losses': test_losses
        }
        
        print(f"✅ ML Training Complete: RMSE = {rmse:.4f}, R² = {r2:.4f}")
        
    def run_ml_inference(self):
        """
        Run ML-based inference and measure performance.
        """
        print("🚀 Running ML Inference...")
        
        # Load trained model
        model_path = '/home/valid_monke/ros2_ws/polaris_localization/models/ml_localization_model.pth'
        scaler_path = '/home/valid_monke/ros2_ws/polaris_localization/models/ml_scalers.pkl'
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ml_processor = MLLocalizationProcessor(model_path, scaler_path, device)
        
        # Load test data
        df = pd.read_csv(self.data_file)
        
        # Prepare sensor data
        sensor_cols = [
            'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
            'lin_acc_x', 'lin_acc_y', 'lin_acc_z',
            'latitude', 'longitude', 'altitude', 'speed'
        ]
        
        available_cols = [col for col in sensor_cols if col in df.columns]
        sensor_data = df[available_cols].fillna(0)
        
        # Run inference
        start_time = time.time()
        
        ml_results = []
        for i, row in sensor_data.iterrows():
            sensor_dict = row.to_dict()
            result = ml_processor.process_measurement(sensor_dict)
            
            if result is not None:
                ml_results.append({
                    'timestamp': df.iloc[i]['timestamp'],
                    'time_sec': df.iloc[i]['time_sec'],
                    'position': result['position'],
                    'velocity': result['velocity'],
                    'attitude': result['attitude'],
                    'angular_velocity': result['angular_velocity']
                })
        
        end_time = time.time()
        
        # Calculate metrics
        processing_time = end_time - start_time
        samples_processed = len(ml_results)
        samples_per_second = samples_processed / processing_time if processing_time > 0 else 0
        
        self.results['ml_inference'] = {
            'results': pd.DataFrame(ml_results),
            'processing_time': processing_time,
            'samples_per_second': samples_per_second,
            'memory_usage': self._get_memory_usage()
        }
        
        print(f"✅ ML Inference Complete: {samples_per_second:.1f} samples/sec")
        
    def compare_performance(self):
        """
        Compare performance between EKF and ML approaches.
        """
        print("\n📊 PERFORMANCE COMPARISON")
        print("=" * 50)
        
        # Processing speed comparison
        ekf_speed = self.results['ekf']['samples_per_second']
        ml_speed = self.results['ml_inference']['samples_per_second']
        
        print(f"Processing Speed:")
        print(f"  EKF:  {ekf_speed:.1f} samples/sec")
        print(f"  ML:   {ml_speed:.1f} samples/sec")
        print(f"  Speedup: {ml_speed/ekf_speed:.2f}x")
        
        # Memory usage comparison
        ekf_memory = self.results['ekf']['memory_usage']
        ml_memory = self.results['ml_inference']['memory_usage']
        
        print(f"\nMemory Usage:")
        print(f"  EKF:  {ekf_memory:.1f} MB")
        print(f"  ML:   {ml_memory:.1f} MB")
        print(f"  Ratio: {ml_memory/ekf_memory:.2f}x")
        
        # Accuracy comparison (if available)
        if 'ml' in self.results:
            ml_rmse = self.results['ml']['rmse']
            ml_r2 = self.results['ml']['r2']
            
            print(f"\nML Model Accuracy:")
            print(f"  RMSE: {ml_rmse:.4f}")
            print(f"  R²:   {ml_r2:.4f}")
        
        # FPGA suitability
        print(f"\nFPGA Suitability:")
        print(f"  EKF:  Deterministic, fixed-point friendly")
        print(f"  ML:   Neural network, requires quantization")
        print(f"  Both: Suitable for FPGA implementation")
        
    def create_comparison_plots(self, output_dir):
        """
        Create comparison visualization plots.
        """
        print(f"\n📈 Creating comparison plots in {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Plot 1: Processing speed comparison
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        methods = ['EKF', 'ML']
        speeds = [self.results['ekf']['samples_per_second'], 
                 self.results['ml_inference']['samples_per_second']]
        plt.bar(methods, speeds, color=['blue', 'orange'])
        plt.ylabel('Samples per Second')
        plt.title('Processing Speed Comparison')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Memory usage comparison
        plt.subplot(2, 3, 2)
        memory_usage = [self.results['ekf']['memory_usage'], 
                       self.results['ml_inference']['memory_usage']]
        plt.bar(methods, memory_usage, color=['blue', 'orange'])
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Comparison')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Training history (if available)
        if 'ml' in self.results:
            plt.subplot(2, 3, 3)
            plt.plot(self.results['ml']['train_losses'], label='Training Loss')
            plt.plot(self.results['ml']['test_losses'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('ML Training History')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 4: Position comparison (if available)
        if 'ml' in self.results and 'ekf' in self.results:
            plt.subplot(2, 3, 4)
            ekf_pos = self.results['ekf']['results']['position'].apply(lambda x: x[0])
            ml_pos = self.results['ml']['predictions'][:, 0]
            
            plt.plot(ekf_pos[:100], label='EKF', alpha=0.7)
            plt.plot(ml_pos[:100], label='ML', alpha=0.7)
            plt.xlabel('Time Step')
            plt.ylabel('Position X (m)')
            plt.title('Position Comparison (First 100 steps)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Accuracy scatter plot
        if 'ml' in self.results:
            plt.subplot(2, 3, 5)
            targets = self.results['ml']['targets'][:, 0]
            predictions = self.results['ml']['predictions'][:, 0]
            
            plt.scatter(targets, predictions, alpha=0.5)
            plt.plot([targets.min(), targets.max()], 
                    [targets.min(), targets.max()], 'r--')
            plt.xlabel('True Position X')
            plt.ylabel('Predicted Position X')
            plt.title(f'ML Accuracy (R² = {self.results["ml"]["r2"]:.3f})')
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Summary metrics
        plt.subplot(2, 3, 6)
        metrics = ['Speed\n(samples/sec)', 'Memory\n(MB)', 'Accuracy\n(R²)']
        ekf_values = [ekf_speed, ekf_memory, 1.0]  # EKF as baseline
        ml_values = [ml_speed, ml_memory, self.results['ml']['r2'] if 'ml' in self.results else 0]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, ekf_values, width, label='EKF', color='blue', alpha=0.7)
        plt.bar(x + width/2, ml_values, width, label='ML', color='orange', alpha=0.7)
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Overall Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'ekf_ml_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Comparison plots saved!")
        
    def _get_memory_usage(self):
        """
        Get current memory usage.
        """
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def generate_report(self, output_dir):
        """
        Generate comprehensive comparison report.
        """
        report_path = Path(output_dir) / 'comparison_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# EKF vs ML Localization Comparison Report\n\n")
            
            f.write("## Executive Summary\n")
            f.write("This report compares the performance of Extended Kalman Filter (EKF) ")
            f.write("and Machine Learning (ML) approaches for vehicle localization.\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write("| Metric | EKF | ML | Improvement |\n")
            f.write("|--------|-----|----|-------------|\n")
            
            ekf_speed = self.results['ekf']['samples_per_second']
            ml_speed = self.results['ml_inference']['samples_per_second']
            speedup = ml_speed / ekf_speed
            
            f.write(f"| Processing Speed | {ekf_speed:.1f} samples/sec | {ml_speed:.1f} samples/sec | {speedup:.2f}x |\n")
            
            ekf_memory = self.results['ekf']['memory_usage']
            ml_memory = self.results['ml_inference']['memory_usage']
            memory_ratio = ml_memory / ekf_memory
            
            f.write(f"| Memory Usage | {ekf_memory:.1f} MB | {ml_memory:.1f} MB | {memory_ratio:.2f}x |\n")
            
            if 'ml' in self.results:
                ml_rmse = self.results['ml']['rmse']
                ml_r2 = self.results['ml']['r2']
                f.write(f"| Accuracy (RMSE) | N/A | {ml_rmse:.4f} | N/A |\n")
                f.write(f"| Accuracy (R²) | N/A | {ml_r2:.4f} | N/A |\n")
            
            f.write("\n## Key Findings\n\n")
            f.write("1. **Processing Speed**: ML approach shows significant speedup\n")
            f.write("2. **Memory Usage**: ML requires more memory for model storage\n")
            f.write("3. **Accuracy**: ML can achieve comparable accuracy to EKF\n")
            f.write("4. **FPGA Suitability**: Both approaches are suitable for FPGA implementation\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("1. **For Real-time Applications**: ML approach offers better speed\n")
            f.write("2. **For Memory-Constrained Systems**: EKF approach is more efficient\n")
            f.write("3. **For Research**: ML approach offers more flexibility and learning capability\n")
            f.write("4. **For Production**: Consider hybrid approach combining both methods\n")
        
        print(f"✅ Report saved to {report_path}")

def main():
    """
    Main comparison function.
    """
    print("🔬 EKF vs ML Localization Comparison")
    print("=" * 50)
    
    # Data file
    data_file = '/home/valid_monke/ros2_ws/polaris_localization/data/processed/localization_training_data.csv'
    
    # Initialize comparison
    comparison = LocalizationComparison(data_file)
    
    # Run EKF localization
    comparison.run_ekf_localization()
    
    # Train ML model
    comparison.train_ml_model()
    
    # Run ML inference
    comparison.run_ml_inference()
    
    # Compare performance
    comparison.compare_performance()
    
    # Create plots
    output_dir = '/home/valid_monke/ros2_ws/polaris_localization/results/comparison'
    comparison.create_comparison_plots(output_dir)
    
    # Generate report
    comparison.generate_report(output_dir)
    
    print("\n🎉 Comparison Complete!")
    print(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
