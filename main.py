#!/usr/bin/env python3

"""
Main entry point for the Polaris Localization Algorithm.
This script provides a command-line interface for running the complete pipeline.
"""

import argparse
import sys
import os
from pathlib import Path

from polaris_autonomous_system.localization.localization_preprocessor import LocalizationPreprocessor
from polaris_autonomous_system.localization.ekf_localization import LocalizationProcessor

def main():
    """Main entry point for the localization pipeline."""
    parser = argparse.ArgumentParser(
        description="Polaris Localization Algorithm - Real-time vehicle localization using EKF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process bag data and run localization
  python main.py --bag-dir /path/to/bags --output-dir /path/to/output
  
  # Run localization on processed data
  python main.py --data-file processed_data.csv --output-dir /path/to/output
  
  # Run validation tests
  python main.py --validate
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--bag-dir", 
        type=str, 
        help="Directory containing ROS2 bag files to process"
    )
    input_group.add_argument(
        "--data-file", 
        type=str, 
        help="Path to processed CSV data file"
    )
    input_group.add_argument(
        "--validate", 
        action="store_true", 
        help="Run validation tests"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./results", 
        help="Output directory for results (default: ./results)"
    )
    
    # Processing options
    parser.add_argument(
        "--freq", 
        type=float, 
        default=30.0, 
        help="Target processing frequency in Hz (default: 30.0)"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true", 
        help="Generate visualization plots"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.validate:
        run_validation()
    elif args.bag_dir:
        process_bag_data(args.bag_dir, output_dir, args.freq, args.visualize, args.verbose)
    elif args.data_file:
        run_localization(args.data_file, output_dir, args.visualize, args.verbose)

def process_bag_data(bag_dir, output_dir, freq, visualize, verbose):
    """Process ROS2 bag data and run localization."""
    if verbose:
        print(f"Processing bag data from: {bag_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Target frequency: {freq} Hz")
    
    try:
        # Step 1: Preprocess data
        if verbose:
            print("\n=== STEP 1: DATA PREPROCESSING ===")
        
        preprocessor = LocalizationPreprocessor(bag_dir)
        preprocessor.load_data()
        preprocessor.synchronize_sensors(target_freq=freq)
        preprocessor.transform_coordinates()
        preprocessor.compute_derivatives()
        preprocessor.clean_data()
        
        # Save processed data
        processed_file = output_dir / "localization_training_data.csv"
        preprocessor.save_processed_data(str(processed_file))
        
        if visualize:
            viz_dir = output_dir / "visualizations"
            preprocessor.visualize_data(str(viz_dir))
        
        if verbose:
            print(f"✅ Data preprocessing complete. Saved to: {processed_file}")
        
        # Step 2: Run localization
        if verbose:
            print("\n=== STEP 2: LOCALIZATION ALGORITHM ===")
        
        processor = LocalizationProcessor(str(processed_file))
        processor.run_localization()
        
        # Evaluate accuracy
        errors = processor.evaluate_accuracy()
        
        if visualize:
            results_dir = output_dir / "localization_results"
            processor.plot_results(str(results_dir))
        
        if verbose:
            print(f"✅ Localization complete. Processed {len(processor.results)} samples")
            if len(errors) > 0:
                rmse = (sum(e**2 for e in errors) / len(errors))**0.5
                print(f"   Position RMSE: {rmse:.2f} m")
        
        print(f"\n🎉 Pipeline complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error processing bag data: {str(e)}")
        sys.exit(1)

def run_localization(data_file, output_dir, visualize, verbose):
    """Run localization on processed data."""
    if verbose:
        print(f"Running localization on: {data_file}")
        print(f"Output directory: {output_dir}")
    
    try:
        processor = LocalizationProcessor(data_file)
        processor.run_localization()
        
        # Evaluate accuracy
        errors = processor.evaluate_accuracy()
        
        if visualize:
            results_dir = output_dir / "localization_results"
            processor.plot_results(str(results_dir))
        
        if verbose:
            print(f"✅ Localization complete. Processed {len(processor.results)} samples")
            if len(errors) > 0:
                rmse = (sum(e**2 for e in errors) / len(errors))**0.5
                print(f"   Position RMSE: {rmse:.2f} m")
        
        print(f"\n🎉 Localization complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error running localization: {str(e)}")
        sys.exit(1)

def run_validation():
    """Run validation tests."""
    print("🧪 Running validation tests...")
    
    try:
        # Import and run unit tests
        sys.path.append(str(Path(__file__).parent / "tests"))
        from unit_tests import run_unit_tests
        from validation_framework import ValidationFramework
        
        # Run unit tests
        print("\n=== UNIT TESTS ===")
        unit_success = run_unit_tests()
        
        # Run comprehensive validation
        print("\n=== COMPREHENSIVE VALIDATION ===")
        validator = ValidationFramework()
        validator.run_all_validations()
        
        if unit_success:
            print("\n🎉 All validation tests completed!")
        else:
            print("\n⚠️ Some validation tests failed. Check output above.")
            
    except Exception as e:
        print(f"❌ Error running validation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
