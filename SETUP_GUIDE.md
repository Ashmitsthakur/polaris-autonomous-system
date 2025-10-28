# 🚀 Setup Guide - Polaris Autonomous System

This guide will help you get the Polaris Autonomous System up and running on your machine.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Verification](#verification)
- [Quick Start Examples](#quick-start-examples)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

## 🔧 Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows WSL2
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for software, additional space for data

### Required Software

1. **Python 3.8+**
   ```bash
   python3 --version  # Should show 3.8 or higher
   ```

2. **pip (Python package manager)**
   ```bash
   pip --version
   ```

3. **ROS2** (Optional - only needed for bag file processing)
   - [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html) or later
   - Not required if you already have processed CSV data

## 📦 Installation

### Step 1: Clone the Repository

```bash
# Navigate to your workspace
cd ~/your_workspace

# Clone the repository
git clone https://github.com/valid_monke/polaris-autonomous-system.git
cd polaris-autonomous-system
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install basic requirements
pip install -r requirements.txt

# Install in development mode (recommended for contributors)
pip install -e .

# Optional: Install development tools
pip install -e ".[dev]"
```

### Step 4: Verify Installation

```bash
# Check if package is installed
python -c "import polaris_autonomous_system; print('✅ Installation successful!')"
```

## ✅ Verification

### Run Basic Tests

```bash
# Run validation framework
python main.py --validate
```

**Expected output:**
- Unit tests should pass
- Validation tests should complete
- You should see performance metrics

### Test with Sample Data

If you have the sample data included:

```bash
# Run localization on sample data
python main.py --data-file data/localization_training_data.csv \
               --output-dir results_test \
               --visualize --verbose
```

**Expected output:**
```
✅ Localization complete. Processed 6480 samples
   Position RMSE: 0.62 m
```

## 🎯 Quick Start Examples

### Example 1: Validate EKF Implementation

```bash
python scripts/validate_ekf.py
```

This will:
- Test mathematical consistency
- Validate with synthetic data
- Check convergence properties
- Generate validation plots

### Example 2: Process Your Own Data

```bash
# If you have ROS2 bag files
python main.py --bag-dir /path/to/your/bags \
               --output-dir results \
               --visualize

# If you have processed CSV data
python main.py --data-file /path/to/your/data.csv \
               --output-dir results \
               --visualize
```

### Example 3: Use Python API

```python
from polaris_autonomous_system.localization.ekf_localization import LocalizationProcessor

# Load data
processor = LocalizationProcessor('data/localization_training_data.csv')

# Run localization
processor.run_localization()

# Evaluate
errors = processor.evaluate_accuracy()
print(f"RMSE: {errors.mean():.2f} m")

# Visualize
processor.plot_results('my_results')
```

## 🐛 Troubleshooting

### Issue: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'polaris_autonomous_system'`

**Solution**:
```bash
# Make sure you installed the package
pip install -e .

# Check if you're in the virtual environment
which python  # Should point to venv/bin/python
```

### Issue: Missing Dependencies

**Problem**: `ImportError: No module named 'numpy'` (or other packages)

**Solution**:
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### Issue: ROS2 Bag Processing Fails

**Problem**: Can't process ROS2 bag files

**Solution**:
```bash
# Install ROS2 dependencies
pip install rclpy rosbag2-py

# Or source your ROS2 installation
source /opt/ros/humble/setup.bash
```

### Issue: Memory Errors

**Problem**: `MemoryError` during processing

**Solution**:
- Process data in smaller batches
- Close other applications
- Increase system swap space
- Use a machine with more RAM

### Issue: Low Performance

**Problem**: Processing is slower than expected

**Solution**:
- Check CPU usage
- Ensure no other heavy processes are running
- Try processing a smaller dataset first
- Consider using a more powerful machine

## 🔍 Data Format

If you're using your own data, ensure your CSV has these columns:

**Required columns:**
- `timestamp`: Unix timestamp
- `ang_vel_x`, `ang_vel_y`, `ang_vel_z`: IMU angular velocity
- `lin_acc_x`, `lin_acc_y`, `lin_acc_z`: IMU linear acceleration
- `latitude`, `longitude`, `altitude`: GPS data
- `enu_x`, `enu_y`, `enu_z`: Local coordinates (ground truth)

**Optional columns:**
- `speed`: Vehicle speed
- `vel_x`, `vel_y`, `vel_z`: Velocity estimates
- Other sensor data

## 📚 Next Steps

### Learn More

1. **Read the Documentation**
   - [Algorithm Design](docs/localization_algorithm_design.md)
   - [Validation Guide](docs/EKF_VALIDATION_GUIDE.md)
   - [FPGA Implementation](docs/fpga_implementation_guide.md)

2. **Explore the Code**
   - Check out `polaris_autonomous_system/localization/ekf_localization.py`
   - Look at test examples in `tests/`

3. **Run More Tests**
   ```bash
   # Component tests
   python tests/test_ekf_components.py
   
   # Full pipeline test
   python scripts/test_localization_pipeline.py
   
   # Compare algorithms
   python scripts/compare_ekf_ml.py
   ```

### Contributing

Ready to contribute? Check out [CONTRIBUTING.md](CONTRIBUTING.md)!

### Get Help

- **Issues**: [GitHub Issues](https://github.com/valid_monke/polaris-autonomous-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/valid_monke/polaris-autonomous-system/discussions)
- **Email**: valid_monke@tamu.edu

## 🎉 You're All Set!

You should now have a working installation of Polaris Autonomous System. Happy coding! 🚗💨

