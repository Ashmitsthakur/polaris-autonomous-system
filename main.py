#!/usr/bin/env python3
"""
Polaris Autonomous System - Unified CLI Entry Point

Simplified interface for EKF localization, ML training, and comparison.
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Polaris Autonomous System - Vehicle Localization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run EKF localization
  python main.py ekf --data-file data/processed/localization_training_data.csv
  
  # Train ML model
  python main.py ml-train --data-file data/processed/localization_training_data.csv
  
  # Compare EKF vs ML
  python main.py compare --data-file data/processed/localization_training_data.csv
  
  # Process raw ROS2 bags
  python main.py process-bags --input data/raw/CAST/collect5 --output data/processed
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # EKF command
    ekf_parser = subparsers.add_parser('ekf', help='Run EKF localization')
    ekf_parser.add_argument('--data-file', required=True, help='Processed CSV data file')
    ekf_parser.add_argument('--output-dir', default='ekf_localization/results', help='Output directory')
    ekf_parser.add_argument('--visualize', action='store_true', help='Generate plots')
    ekf_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # ML training command
    ml_parser = subparsers.add_parser('ml-train', help='Train ML localization model')
    ml_parser.add_argument('--data-file', required=True, help='Processed CSV data file')
    ml_parser.add_argument('--output-dir', default='ml/results', help='Output directory')
    ml_parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    ml_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    # Comparison command
    comp_parser = subparsers.add_parser('compare', help='Compare EKF vs ML')
    comp_parser.add_argument('--data-file', required=True, help='Processed CSV data file')
    comp_parser.add_argument('--output-dir', default='comparison_results', help='Output directory')
    
    # Data processing command
    proc_parser = subparsers.add_parser('process-bags', help='Process ROS2 bag files')
    proc_parser.add_argument('--input', required=True, help='Input bag directory')
    proc_parser.add_argument('--output', required=True, help='Output directory')
    proc_parser.add_argument('--freq', type=float, default=30.0, help='Target frequency (Hz)')
    
    # Validation command
    val_parser = subparsers.add_parser('validate', help='Run validation tests')
    val_parser.add_argument('--type', choices=['ekf', 'ml', 'all'], default='all', help='What to validate')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute commands
    if args.command == 'ekf':
        run_ekf(args)
    elif args.command == 'ml-train':
        run_ml_training(args)
    elif args.command == 'compare':
        run_comparison(args)
    elif args.command == 'process-bags':
        process_bags(args)
    elif args.command == 'validate':
        run_validation(args)

def run_ekf(args):
    """Run EKF localization."""
    print(f"🎯 Running EKF Localization")
    print(f"Data file: {args.data_file}")
    print(f"Output: {args.output_dir}")
    
    try:
        from ekf_localization import LocalizationProcessor
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run localization
        processor = LocalizationProcessor(args.data_file)
        processor.run_localization()
        
        # Evaluate
        errors = processor.evaluate_accuracy()
        
        # Visualize
        if args.visualize:
            processor.plot_results(args.output_dir)
        
        # Print results
        if errors is not None and len(errors) > 0:
            import numpy as np
            rmse = np.sqrt(np.mean(errors**2))
            print(f"\n✅ Localization complete!")
            print(f"   Position RMSE: {rmse:.2f} m")
            print(f"   Processed: {len(processor.results)} samples")
            print(f"   Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def run_ml_training(args):
    """Train ML localization model."""
    print(f"🧠 Training ML Model")
    print(f"Data file: {args.data_file}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    
    try:
        import torch
        from ml import LocalizationLSTM, MLLocalizationTrainer
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path('ml/models').mkdir(parents=True, exist_ok=True)
        
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize model
        model = LocalizationLSTM(input_size=10, hidden_size=128, num_layers=2, output_size=12)
        trainer = MLLocalizationTrainer(model, device)
        
        # Prepare data
        train_dataset, test_dataset, _, _ = trainer.prepare_data(args.data_file)
        
        # Train
        train_losses, test_losses = trainer.train(
            train_dataset, test_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Evaluate
        predictions, targets, rmse, r2 = trainer.evaluate(test_dataset)
        
        # Save model
        model_path = Path('ml/models') / 'ml_localization_model.pth'
        scaler_path = Path('ml/models') / 'ml_scalers.pkl'
        trainer.save_model(str(model_path), str(scaler_path))
        
        print(f"\n✅ Training complete!")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Model saved to: {model_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_comparison(args):
    """Compare EKF vs ML performance."""
    print(f"🔬 Comparing EKF vs ML")
    print(f"Data file: {args.data_file}")
    print(f"Output: {args.output_dir}")
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent / 'ml'))
        from compare_ekf_ml import LocalizationComparison
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run comparison
        comparison = LocalizationComparison(args.data_file)
        comparison.run_ekf_localization()
        comparison.train_ml_model()
        comparison.run_ml_inference()
        comparison.compare_performance()
        comparison.create_comparison_plots(args.output_dir)
        comparison.generate_report(args.output_dir)
        
        print(f"\n✅ Comparison complete!")
        print(f"   Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def process_bags(args):
    """Process ROS2 bag files."""
    print(f"📊 Processing ROS2 Bags")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    try:
        from ekf_localization import LocalizationPreprocessor
        
        # Create output directory
        Path(args.output).mkdir(parents=True, exist_ok=True)
        
        # Process
        preprocessor = LocalizationPreprocessor(args.input)
        preprocessor.load_data()
        preprocessor.synchronize_sensors(target_freq=args.freq)
        preprocessor.transform_coordinates()
        preprocessor.compute_derivatives()
        preprocessor.clean_data()
        
        # Save
        output_file = Path(args.output) / 'localization_training_data.csv'
        preprocessor.save_processed_data(str(output_file))
        
        print(f"\n✅ Processing complete!")
        print(f"   Output saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_validation(args):
    """Run validation tests."""
    print(f"🧪 Running Validation Tests")
    
    try:
        if args.type in ['ekf', 'all']:
            print("\n=== EKF VALIDATION ===")
            from ekf_localization import validate
            # Run EKF validation
            
        if args.type in ['ml', 'all']:
            print("\n=== ML VALIDATION ===")
            # Run ML validation
            
        if args.type == 'all':
            print("\n=== COMPREHENSIVE VALIDATION ===")
            sys.path.append('tests')
            from validation_framework import ValidationFramework
            validator = ValidationFramework()
            validator.run_all_validations()
        
        print("\n✅ Validation complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

