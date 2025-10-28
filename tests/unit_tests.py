#!/usr/bin/env python3

"""
Unit tests for individual components of the localization pipeline.
These tests verify that each component works correctly in isolation.
"""

import sys
import os
import numpy as np
import pandas as pd
import unittest
from pathlib import Path

# Add the package to path
sys.path.append('/home/valid_monke/ros2_ws/src/polaris_ml')

from polaris_ml.data_processing.localization_preprocessor import LocalizationPreprocessor
from polaris_ml.data_processing.ekf_localization import EKFLocalization

class TestLocalizationPreprocessor(unittest.TestCase):
    """Test cases for LocalizationPreprocessor."""
    
    def setUp(self):
        """Set up test data."""
        self.bag_dir = '/home/valid_monke/ros2_ws/bags/CAST/collect5_processed'
        self.preprocessor = LocalizationPreprocessor(self.bag_dir)
    
    def test_initialization(self):
        """Test that preprocessor initializes correctly."""
        self.assertIsNotNone(self.preprocessor.bag_data_dir)
        self.assertEqual(self.preprocessor.bag_data_dir, Path(self.bag_dir))
    
    def test_load_data(self):
        """Test data loading functionality."""
        try:
            self.preprocessor.load_data()
            self.assertIsNotNone(self.preprocessor.nav_df)
            self.assertIsNotNone(self.preprocessor.control_df)
            self.assertIsNotNone(self.preprocessor.speed_df)
            print("✅ Data loading test passed")
        except Exception as e:
            self.fail(f"Data loading failed: {str(e)}")
    
    def test_synchronize_sensors(self):
        """Test sensor synchronization."""
        try:
            self.preprocessor.load_data()
            self.preprocessor.synchronize_sensors(target_freq=30.0)
            
            self.assertIsNotNone(self.preprocessor.synchronized_data)
            self.assertEqual(len(self.preprocessor.synchronized_data), 6480)
            print("✅ Sensor synchronization test passed")
        except Exception as e:
            self.fail(f"Sensor synchronization failed: {str(e)}")
    
    def test_coordinate_transformation(self):
        """Test GPS to ENU coordinate transformation."""
        try:
            self.preprocessor.load_data()
            self.preprocessor.synchronize_sensors(target_freq=30.0)
            self.preprocessor.transform_coordinates()
            
            # Check that ENU coordinates exist
            self.assertIn('enu_x', self.preprocessor.synchronized_data.columns)
            self.assertIn('enu_y', self.preprocessor.synchronized_data.columns)
            self.assertIn('enu_z', self.preprocessor.synchronized_data.columns)
            
            # Check that ENU coordinates are reasonable
            enu_x = self.preprocessor.synchronized_data['enu_x'].dropna()
            enu_y = self.preprocessor.synchronized_data['enu_y'].dropna()
            enu_z = self.preprocessor.synchronized_data['enu_z'].dropna()
            
            self.assertGreater(len(enu_x), 0)
            self.assertGreater(len(enu_y), 0)
            self.assertGreater(len(enu_z), 0)
            print("✅ Coordinate transformation test passed")
        except Exception as e:
            self.fail(f"Coordinate transformation failed: {str(e)}")

class TestEKFLocalization(unittest.TestCase):
    """Test cases for EKFLocalization."""
    
    def setUp(self):
        """Set up test data."""
        self.ekf = EKFLocalization(dt=1.0/30.0)
    
    def test_initialization(self):
        """Test EKF initialization."""
        self.assertEqual(len(self.ekf.state), 12)
        self.assertEqual(self.ekf.covariance.shape, (12, 12))
        self.assertEqual(self.ekf.dt, 1.0/30.0)
        print("✅ EKF initialization test passed")
    
    def test_prediction_step(self):
        """Test EKF prediction step."""
        # Test with synthetic IMU data
        imu_data = {
            'ang_vel': np.array([0.1, 0.05, 0.02]),
            'lin_acc': np.array([0.5, 0.3, 9.8])
        }
        
        initial_state = self.ekf.state.copy()
        self.ekf.predict(imu_data)
        final_state = self.ekf.state.copy()
        
        # State should change after prediction
        self.assertFalse(np.allclose(initial_state, final_state))
        print("✅ EKF prediction test passed")
    
    def test_gps_update(self):
        """Test GPS update step."""
        gps_data = {'position': np.array([1.0, 2.0, 3.0])}
        
        initial_pos = self.ekf.state[self.ekf.idx_pos].copy()
        self.ekf.update_gps(gps_data)
        final_pos = self.ekf.state[self.ekf.idx_pos]
        
        # Position should be updated
        self.assertFalse(np.allclose(initial_pos, final_pos))
        print("✅ GPS update test passed")
    
    def test_odometry_update(self):
        """Test odometry update step."""
        odom_data = {'velocity': np.array([1.0, 0.5, 0.0])}
        
        initial_vel = self.ekf.state[self.ekf.idx_vel].copy()
        self.ekf.update_odometry(odom_data)
        final_vel = self.ekf.state[self.ekf.idx_vel]
        
        # Velocity should be updated
        self.assertFalse(np.allclose(initial_vel, final_vel))
        print("✅ Odometry update test passed")
    
    def test_covariance_properties(self):
        """Test that covariance matrix remains valid."""
        # Run a few prediction and update steps
        imu_data = {
            'ang_vel': np.array([0.1, 0.05, 0.02]),
            'lin_acc': np.array([0.5, 0.3, 9.8])
        }
        gps_data = {'position': np.array([1.0, 2.0, 3.0])}
        
        for _ in range(10):
            self.ekf.predict(imu_data)
            self.ekf.update_gps(gps_data)
        
        # Covariance should be positive definite
        eigenvals = np.linalg.eigvals(self.ekf.covariance)
        self.assertTrue(np.all(eigenvals > 0))
        print("✅ Covariance properties test passed")

class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and consistency."""
    
    def test_processed_data_exists(self):
        """Test that processed data file exists and is readable."""
        data_file = '/home/valid_monke/ros2_ws/polaris_localization/data/processed/localization_training_data.csv'
        self.assertTrue(os.path.exists(data_file))
        
        data = pd.read_csv(data_file)
        self.assertGreater(len(data), 0)
        print("✅ Processed data file test passed")
    
    def test_data_consistency(self):
        """Test data consistency."""
        data_file = '/home/valid_monke/ros2_ws/polaris_localization/data/processed/localization_training_data.csv'
        data = pd.read_csv(data_file)
        
        # Check required columns exist
        required_cols = ['timestamp', 'time_sec', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']
        for col in required_cols:
            self.assertIn(col, data.columns)
        
        # Check time column is monotonic
        time_diffs = np.diff(data['time_sec'])
        self.assertTrue(np.all(time_diffs > 0))
        print("✅ Data consistency test passed")
    
    def test_data_ranges(self):
        """Test that data is in reasonable ranges."""
        data_file = '/home/valid_monke/ros2_ws/polaris_localization/data/processed/localization_training_data.csv'
        data = pd.read_csv(data_file)
        
        # Check IMU data ranges
        imu_cols = ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']
        for col in imu_cols:
            if col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    self.assertLess(np.abs(col_data).max(), 10.0)  # Less than 10 rad/s
        
        print("✅ Data ranges test passed")

def run_unit_tests():
    """Run all unit tests."""
    print("🧪 RUNNING UNIT TESTS")
    print("=" * 40)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestLocalizationPreprocessor))
    test_suite.addTest(unittest.makeSuite(TestEKFLocalization))
    test_suite.addTest(unittest.makeSuite(TestDataIntegrity))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 40)
    print("UNIT TEST SUMMARY")
    print("=" * 40)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("🎉 ALL UNIT TESTS PASSED!")
    else:
        print("⚠️  Some unit tests failed.")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    run_unit_tests()
