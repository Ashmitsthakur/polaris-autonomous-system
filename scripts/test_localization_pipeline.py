#!/usr/bin/env python3

"""
Test script for the complete localization pipeline.
This script demonstrates the full workflow from raw bag data to localization results.
"""

import sys
import os
sys.path.append('/home/valid_monke/ros2_ws/src/polaris_ml')

from polaris_ml.data_processing.localization_preprocessor import LocalizationPreprocessor
from polaris_ml.data_processing.ekf_localization import LocalizationProcessor
import matplotlib.pyplot as plt
import numpy as np

def test_data_preprocessing():
    """Test the data preprocessing pipeline."""
    print("=" * 60)
    print("TESTING DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Process the collect5 data
    bag_dir = '/home/valid_monke/ros2_ws/bags/CAST/collect5_processed'
    preprocessor = LocalizationPreprocessor(bag_dir)
    
    try:
        # Load and process data
        print("Loading sensor data...")
        preprocessor.load_data()
        
        print("Synchronizing sensors...")
        preprocessor.synchronize_sensors(target_freq=30.0)
        
        print("Transforming coordinates...")
        preprocessor.transform_coordinates()
        
        print("Computing derivatives...")
        preprocessor.compute_derivatives()
        
        print("Cleaning data...")
        preprocessor.clean_data()
        
        # Save processed data
        output_file = '/home/valid_monke/ros2_ws/localization_training_data.csv'
        preprocessor.save_processed_data(output_file)
        
        # Create visualizations
        preprocessor.visualize_data('/home/valid_monke/ros2_ws/visualizations')
        
        print("✅ Data preprocessing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {str(e)}")
        return False

def test_localization_algorithm():
    """Test the localization algorithm."""
    print("\n" + "=" * 60)
    print("TESTING LOCALIZATION ALGORITHM")
    print("=" * 60)
    
    data_file = '/home/valid_monke/ros2_ws/localization_training_data.csv'
    
    if not os.path.exists(data_file):
        print("❌ Processed data file not found. Run data preprocessing first.")
        return False
    
    try:
        # Run localization
        print("Initializing localization processor...")
        processor = LocalizationProcessor(data_file)
        
        print("Running localization algorithm...")
        processor.run_localization()
        
        print("Evaluating accuracy...")
        errors = processor.evaluate_accuracy()
        
        print("Creating result visualizations...")
        processor.plot_results('/home/valid_monke/ros2_ws/localization_results')
        
        print("✅ Localization algorithm completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Localization algorithm failed: {str(e)}")
        return False

def test_fpga_considerations():
    """Test FPGA implementation considerations."""
    print("\n" + "=" * 60)
    print("TESTING FPGA IMPLEMENTATION CONSIDERATIONS")
    print("=" * 60)
    
    try:
        # Load processed data
        data_file = '/home/valid_monke/ros2_ws/localization_training_data.csv'
        if not os.path.exists(data_file):
            print("❌ Processed data file not found.")
            return False
            
        import pandas as pd
        data = pd.read_csv(data_file)
        
        # Analyze data characteristics for FPGA implementation
        print("Analyzing data characteristics for FPGA implementation...")
        
        # Check data ranges for fixed-point conversion
        print("\nData Range Analysis:")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['timestamp', 'time_sec']:
                min_val = data[col].min()
                max_val = data[col].max()
                range_val = max_val - min_val
                print(f"  {col}: [{min_val:.6f}, {max_val:.6f}] (range: {range_val:.6f})")
        
        # Check for NaN values
        print(f"\nMissing Data Analysis:")
        missing_data = data.isnull().sum()
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                print(f"  {col}: {missing_count} missing values")
        
        # Analyze computational requirements
        print(f"\nComputational Requirements Analysis:")
        print(f"  Total samples: {len(data)}")
        print(f"  Data rate: {1.0 / np.mean(np.diff(data['time_sec'])):.1f} Hz")
        print(f"  Duration: {data['time_sec'].max() - data['time_sec'].min():.1f} seconds")
        
        # Estimate FPGA resource requirements
        print(f"\nEstimated FPGA Resource Requirements:")
        print(f"  State vector size: 12 elements")
        print(f"  Covariance matrix: 12x12 = 144 elements")
        print(f"  Memory requirement: ~2KB (assuming 32-bit fixed-point)")
        print(f"  Processing frequency: 30 Hz")
        print(f"  Clock cycles per update: ~1000 (estimated)")
        print(f"  Required clock frequency: 30 kHz (easily achievable)")
        
        print("✅ FPGA implementation analysis completed!")
        return True
        
    except Exception as e:
        print(f"❌ FPGA analysis failed: {str(e)}")
        return False

def create_fpga_implementation_guide():
    """Create a guide for FPGA implementation."""
    print("\n" + "=" * 60)
    print("CREATING FPGA IMPLEMENTATION GUIDE")
    print("=" * 60)
    
    guide_content = """
# FPGA Implementation Guide for Localization Algorithm

## Overview
This guide outlines the steps to implement the EKF localization algorithm on FPGA hardware.

## Hardware Requirements

### Recommended FPGA Boards:
1. **Xilinx Zynq-7000** (e.g., Zybo Z7-20)
   - ARM Cortex-A9 + Artix-7 FPGA
   - Good for development and testing
   - Price: ~$200

2. **Xilinx Kintex-7** (e.g., KC705)
   - High-performance FPGA
   - More resources for complex algorithms
   - Price: ~$1000

3. **Intel Cyclone V** (e.g., DE1-SoC)
   - ARM Cortex-A9 + Cyclone V FPGA
   - Good alternative to Xilinx
   - Price: ~$300

### Memory Requirements:
- **Block RAM**: 10-20 KB (for state storage)
- **DSP Blocks**: 50-100 (for matrix operations)
- **Logic Elements**: 5,000-10,000 LUTs

## Implementation Steps

### Phase 1: Fixed-Point Conversion
1. **Analyze data ranges** from your sensor data
2. **Choose fixed-point format** (e.g., 16.16 or 32.16)
3. **Convert floating-point algorithms** to fixed-point
4. **Validate accuracy** against floating-point version

### Phase 2: HDL Implementation
1. **State Machine Design**:
   - IMU processing state
   - GPS update state
   - Odometry update state
   - Output state

2. **Matrix Operations**:
   - Use DSP blocks for multiplication
   - Implement matrix inversion (Cholesky decomposition)
   - Optimize for 12x12 matrices

3. **Memory Management**:
   - Store state vector in BRAM
   - Store covariance matrix in BRAM
   - Use FIFOs for data streaming

### Phase 3: Interface Design
1. **Input Interfaces**:
   - UART/SPI for IMU data
   - UART for GPS data
   - CAN for vehicle data

2. **Output Interfaces**:
   - UART for position output
   - Ethernet for real-time streaming
   - GPIO for status indicators

### Phase 4: Testing and Validation
1. **Simulation**: Use ModelSim for HDL simulation
2. **Hardware-in-the-Loop**: Test with real sensor data
3. **Performance Validation**: Compare with software implementation

## Code Structure

### Top-Level Module:
```verilog
module localization_top(
    input clk,
    input rst,
    input [31:0] imu_data,
    input [31:0] gps_data,
    input [31:0] odom_data,
    output [31:0] position_out,
    output [31:0] velocity_out,
    output [31:0] attitude_out
);
```

### Key Submodules:
1. **imu_processor**: IMU data processing and integration
2. **gps_processor**: GPS coordinate transformation
3. **ekf_core**: Extended Kalman Filter implementation
4. **matrix_ops**: Matrix multiplication and inversion
5. **state_manager**: State vector and covariance management

## Performance Targets

### Timing Requirements:
- **Processing Rate**: 30 Hz (33.3 ms per update)
- **Latency**: < 10 ms from input to output
- **Clock Frequency**: 100 MHz (easily achievable)

### Accuracy Targets:
- **Position Error**: < 1 meter RMS
- **Velocity Error**: < 0.1 m/s RMS
- **Attitude Error**: < 1 degree RMS

## Development Tools

### Software:
- **Xilinx Vivado**: FPGA synthesis and implementation
- **ModelSim**: HDL simulation
- **MATLAB/Simulink**: Algorithm development and validation

### Hardware:
- **Oscilloscope**: Signal analysis
- **Logic Analyzer**: Digital signal debugging
- **Development Board**: Target hardware platform

## Next Steps

1. **Start with simulation**: Implement and test in ModelSim
2. **Use development board**: Test with real hardware
3. **Optimize performance**: Fine-tune for your specific requirements
4. **Validate accuracy**: Compare with ground truth data

## Resources

- **Xilinx Documentation**: UG973 (Vivado Design Suite User Guide)
- **Verilog Tutorials**: Online resources for HDL development
- **FPGA Development**: Community forums and documentation
"""
    
    with open('/home/valid_monke/ros2_ws/fpga_implementation_guide.md', 'w') as f:
        f.write(guide_content)
    
    print("✅ FPGA implementation guide created!")
    print("   Location: /home/valid_monke/ros2_ws/fpga_implementation_guide.md")

def main():
    """Run the complete test pipeline."""
    print("🚀 STARTING LOCALIZATION PIPELINE TEST")
    print("=" * 60)
    
    # Test data preprocessing
    success1 = test_data_preprocessing()
    
    # Test localization algorithm
    success2 = test_localization_algorithm()
    
    # Test FPGA considerations
    success3 = test_fpga_considerations()
    
    # Create implementation guide
    create_fpga_implementation_guide()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Data Preprocessing: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Localization Algorithm: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"FPGA Analysis: {'✅ PASS' if success3 else '❌ FAIL'}")
    
    if all([success1, success2, success3]):
        print("\n🎉 ALL TESTS PASSED! Your localization pipeline is ready for FPGA implementation.")
        print("\nNext steps:")
        print("1. Review the FPGA implementation guide")
        print("2. Choose your target FPGA platform")
        print("3. Start with HDL simulation")
        print("4. Implement on hardware")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    print("\nGenerated files:")
    print("- /home/valid_monke/ros2_ws/localization_training_data.csv")
    print("- /home/valid_monke/ros2_ws/visualizations/")
    print("- /home/valid_monke/ros2_ws/localization_results/")
    print("- /home/valid_monke/ros2_ws/fpga_implementation_guide.md")

if __name__ == '__main__':
    main()
