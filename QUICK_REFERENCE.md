# 📖 Quick Reference Guide

One-page reference for common tasks with Polaris Autonomous System.

## 🚀 Installation

```bash
git clone https://github.com/valid_monke/polaris-autonomous-system.git
cd polaris-autonomous-system
pip install -r requirements.txt
pip install -e .
```

## 🧪 Testing & Validation

```bash
# Run all validation tests
python main.py --validate

# Validate EKF implementation
python scripts/validate_ekf.py

# Run unit tests
python tests/unit_tests.py

# Test EKF components
python tests/test_ekf_components.py
```

## 🏃 Running Localization

```bash
# With processed CSV data
python main.py --data-file data/localization_training_data.csv \
               --output-dir results --visualize --verbose

# With ROS2 bag files
python main.py --bag-dir /path/to/bags \
               --output-dir results --visualize

# Custom frequency
python main.py --data-file data.csv --freq 50.0 --output-dir results
```

## 🐍 Python API

```python
# Quick localization
from polaris_autonomous_system.localization.ekf_localization import LocalizationProcessor

processor = LocalizationProcessor('data.csv')
processor.run_localization()
errors = processor.evaluate_accuracy()
processor.plot_results('output_dir')

# Data preprocessing
from polaris_autonomous_system.localization.localization_preprocessor import LocalizationPreprocessor

preprocessor = LocalizationPreprocessor('bag_dir')
preprocessor.load_data()
preprocessor.synchronize_sensors(target_freq=30.0)
preprocessor.transform_coordinates()
preprocessor.save_processed_data('output.csv')
```

## 📊 Key Performance Metrics

| Metric | Value |
|--------|-------|
| Position RMSE | 0.62 m |
| Processing Rate | 30 Hz |
| Real-time Factor | 124x |
| Memory Usage | 172 MB |
| FPGA Memory | 624 B |

## 📁 Important Directories

- `polaris_autonomous_system/` - Source code
- `tests/` - Test suites
- `scripts/` - Utility scripts
- `docs/` - Documentation
- `data/` - Data files
- `results/` - Outputs (generated)

## 🔧 Utilities

```bash
# Compare EKF vs ML
python scripts/compare_ekf_ml.py

# Diagnose EKF issues
python scripts/diagnose_ekf_issues.py

# Read ROS2 bag info
python scripts/read_ros2_bag.py /path/to/bag

# Test complete pipeline
python scripts/test_localization_pipeline.py
```

## 📝 File Formats

### Input CSV Format
Required columns:
- `timestamp`, `time_sec`
- `ang_vel_x/y/z`, `lin_acc_x/y/z` (IMU)
- `latitude`, `longitude`, `altitude` (GPS)
- `enu_x/y/z` (local coordinates)
- `speed` (odometry)

### Output Format
Results saved as:
- CSV files in `results/`
- Plots in `results/visualizations/`
- Validation reports in `config/`

## 🐛 Common Issues

**Import Error**: Run `pip install -e .` from project root

**Memory Error**: Process smaller datasets or increase RAM

**ROS2 Not Found**: Source ROS2 or install `rclpy rosbag2-py`

**Large RMSE**: Check coordinate frames (use ENU not lat/lon)

## 🔗 Quick Links

- **Full Documentation**: `docs/`
- **Setup Guide**: `SETUP_GUIDE.md`
- **Contributing**: `CONTRIBUTING.md`
- **GitHub Prep**: `GITHUB_PREP.md`
- **Changelog**: `CHANGELOG.md`

## 💡 Pro Tips

- Use `--verbose` flag to see detailed processing info
- Use `--visualize` to generate plots automatically
- Check `config/validation_report.yaml` for test results
- Results are cached - delete `results/` to regenerate
- Use virtual environment to avoid dependency conflicts

## 📧 Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: GitHub Issues (after publishing)
- **Email**: valid_monke@tamu.edu

---

**Version**: 1.0.0 | **License**: Apache 2.0 | **Python**: 3.8+

