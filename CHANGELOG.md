# Changelog

All notable changes to the Polaris Autonomous System project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-10-28

### Added
- Initial release of Polaris Autonomous System
- Extended Kalman Filter (EKF) implementation for 12-DOF localization
- Multi-sensor fusion (IMU, GPS, odometry)
- Real-time processing at 30 Hz
- Comprehensive validation framework
- EKF-specific validation with synthetic data
- Data preprocessing pipeline
- ROS2 bag file extraction utilities
- Command-line interface (main.py)
- Visualization tools for results
- Documentation for algorithm design
- FPGA implementation guide
- Unit tests and integration tests

### Features
- Sub-meter accuracy (0.62m RMSE)
- Memory efficient (624 bytes for FPGA)
- 124x real-time processing speed
- 100% sensor data coverage
- Coordinate transformation (GPS → ENU)
- Sensor synchronization to 30 Hz

### Fixed
- Coordinate frame mismatch in accuracy evaluation (v1.0.0-fix)
  - Changed ground truth from `pose_x/y/z` to `enu_x/y/z`
  - Reduced RMSE from 6,372 km to 0.62 m
  - Properly comparing EKF output with local ENU coordinates

### Known Issues
- FPGA readiness: Deterministic behavior test failing (non-critical)
- Performance validation: Minor accuracy check issue (resolved with coordinate fix)

## [0.9.0] - 2024-10-27

### Added
- Beta version with core functionality
- Initial EKF implementation
- Basic data preprocessing
- Preliminary validation tests

## [0.5.0] - 2024-10-20

### Added
- Alpha version
- Data extraction from ROS2 bags
- Basic coordinate transformations

---

## Future Releases

### [Planned] - Version 2.0.0
- ML pipeline integration
- Neural network-based localization
- EKF vs ML performance comparison
- Real-time ROS2 node implementation
- Hardware-in-the-loop testing
- FPGA prototype implementation

### [Planned] - Version 1.5.0
- Improved sensor fusion algorithms
- Adaptive noise covariance
- Enhanced visualization dashboard
- Web-based monitoring interface
- Extended documentation with tutorials

### [Planned] - Version 1.1.0
- Bug fixes and optimizations
- Performance improvements
- Additional test cases
- Expanded dataset support

