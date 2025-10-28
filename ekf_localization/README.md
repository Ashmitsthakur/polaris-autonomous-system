# 🎯 EKF Localization Module

Extended Kalman Filter implementation for vehicle localization using multi-sensor fusion.

## 📁 Structure

```
ekf_localization/
├── ekf_core.py                # Main EKF implementation (12-DOF)
├── preprocessor.py            # Data preprocessing pipeline
├── validate.py                # EKF validation tests
├── diagnose.py                # Debugging utilities
├── config.yaml                # EKF parameters
├── results/                   # EKF outputs
│   ├── convergence_validation.png
│   ├── synthetic_validation.png
│   └── localization_results.png
├── tests_*.py                 # Test suites
├── VALIDATION_GUIDE.md        # How to validate EKF
├── FPGA_GUIDE.md             # Hardware implementation
├── ALGORITHM_DESIGN.md        # Technical details
├── VALIDATION_SUMMARY.md      # Test results
└── TEST_REPORT.md            # Comprehensive report
```

## 🚀 Quick Start

### Run EKF Localization

```python
from ekf_localization import LocalizationProcessor

# Load data and run
processor = LocalizationProcessor('data/processed/localization_training_data.csv')
processor.run_localization()

# Evaluate
errors = processor.evaluate_accuracy()
print(f"Position RMSE: {errors.mean():.2f} m")

# Visualize
processor.plot_results('ekf_localization/results')
```

### Command Line

```bash
# Run localization
python -m ekf_localization.ekf_core

# Run validation
python ekf_localization/validate.py

# Diagnose issues
python ekf_localization/diagnose.py
```

## 🎯 Algorithm Overview

### State Vector (12-DOF)
```
x = [x, y, z,              # Position (m)
     vx, vy, vz,           # Velocity (m/s)
     roll, pitch, yaw,     # Orientation (rad)
     ωx, ωy, ωz]           # Angular velocity (rad/s)
```

### Sensor Fusion
- **IMU** (30 Hz): Angular velocity, linear acceleration → Prediction step
- **GPS** (variable): Lat/lon/alt → Position update
- **Odometry** (30 Hz): Vehicle speed → Velocity update

### Key Features
- ✅ Multi-sensor fusion
- ✅ Robust error handling
- ✅ Numerical stability
- ✅ FPGA-ready (624 bytes memory)
- ✅ Real-time capable (30 Hz)

## 📊 Performance

| Metric | Value | Status |
|--------|-------|--------|
| Position RMSE | 0.62 m | ✅ |
| Processing Rate | 30 Hz | ✅ |
| Real-time Factor | 124x | ✅ |
| Memory (FPGA) | 624 B | ✅ |
| Data Coverage | 100% | ✅ |

## 🧪 Testing

```bash
# Run all tests
python ekf_localization/tests_unit.py

# Run EKF validation
python ekf_localization/validate.py

# Run comprehensive validation
python ekf_localization/tests_validation.py
```

## 📚 Documentation

- **[ALGORITHM_DESIGN.md](ALGORITHM_DESIGN.md)** - Technical specifications
- **[VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)** - How to validate
- **[FPGA_GUIDE.md](FPGA_GUIDE.md)** - Hardware implementation
- **[VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md)** - Test results
- **[TEST_REPORT.md](TEST_REPORT.md)** - Comprehensive report

## ⚙️ Configuration

Edit `config.yaml` to tune EKF parameters:

```yaml
ekf:
  dt: 0.033333           # Time step (30 Hz)
  process_noise: 0.01    # Process noise covariance
  imu_noise: 0.1         # IMU measurement noise
  gps_noise: 1.0         # GPS measurement noise
  odom_noise: 0.5        # Odometry noise
```

## 🔧 Key Components

### EKFLocalization Class
```python
from ekf_localization.ekf_core import EKFLocalization

ekf = EKFLocalization(dt=1.0/30.0)

# Prediction step (IMU)
ekf.predict({'ang_vel': [...], 'lin_acc': [...]})

# Update step (GPS)
ekf.update_gps({'position': [x, y, z]})

# Update step (odometry)
ekf.update_odometry({'velocity': [vx, vy, vz]})

# Get state
position = ekf.state[0:3]
uncertainty = ekf.get_position_uncertainty()
```

### LocalizationPreprocessor Class
```python
from ekf_localization.preprocessor import LocalizationPreprocessor

preprocessor = LocalizationPreprocessor('data/raw/CAST/collect5')
preprocessor.load_data()
preprocessor.synchronize_sensors(target_freq=30.0)
preprocessor.transform_coordinates()
preprocessor.compute_derivatives()
preprocessor.clean_data()
preprocessor.save_processed_data('data/processed/output.csv')
```

## 🚀 FPGA Implementation

See **[FPGA_GUIDE.md](FPGA_GUIDE.md)** for details on:
- Fixed-point conversion
- Memory requirements
- HDL implementation
- Hardware platform selection

**Memory Footprint**:
- State vector: 384 bits (48 bytes)
- Covariance matrix: 4,608 bits (576 bytes)
- **Total: 624 bytes** (fits in block RAM)

## 📈 Results

Position RMSE: **0.62m** (beats 1.0m target!)

![Localization Results](results/localization_results.png)

## 🤝 Integration with ML

The EKF output serves as ground truth for ML training:

```python
# Run EKF
ekf_processor = LocalizationProcessor('data.csv')
ekf_processor.run_localization()

# Use EKF results for ML training
# See ml/ folder for details
```

## ❓ Troubleshooting

### High Position Error
- Check coordinate frame (use ENU not lat/lon)
- Verify GPS data quality
- Tune noise parameters

### NaN Values
- Check IMU data for outliers
- Verify covariance matrix stays positive definite
- Increase process noise if needed

### Slow Processing
- Normal: ~4000 samples/sec on CPU
- Vectorize operations where possible
- Consider FPGA for ultra-low latency

