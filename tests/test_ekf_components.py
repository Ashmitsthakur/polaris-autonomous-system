#!/usr/bin/env python3

"""
Unit Tests for EKF Components

Individual tests for specific EKF components to help debug issues.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add unified package to path
sys.path.append(str(Path(__file__).parent.parent))

from polaris_autonomous_system.localization.ekf_localization import EKFLocalization

class TestEKFComponents(unittest.TestCase):
    """
    Unit tests for individual EKF components.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.ekf = EKFLocalization(dt=1.0/30.0)
        
    def test_initialization(self):
        """Test proper EKF initialization."""
        # Check state vector size
        self.assertEqual(len(self.ekf.state), 12)
        
        # Check covariance matrix properties
        self.assertTrue(np.allclose(self.ekf.covariance, self.ekf.covariance.T))  # Symmetric
        self.assertTrue(np.all(np.linalg.eigvals(self.ekf.covariance) > 0))  # Positive definite
        
        # Check noise matrices
        self.assertTrue(np.allclose(self.ekf.Q, self.ekf.Q.T))  # Process noise symmetric
        self.assertTrue(np.allclose(self.ekf.R_imu, self.ekf.R_imu.T))  # IMU noise symmetric
        self.assertTrue(np.allclose(self.ekf.R_gps, self.ekf.R_gps.T))  # GPS noise symmetric
        
    def test_state_indices(self):
        """Test state vector indexing."""
        # Position indices
        self.assertEqual(self.ekf.idx_pos.start, 0)
        self.assertEqual(self.ekf.idx_pos.stop, 3)
        
        # Velocity indices
        self.assertEqual(self.ekf.idx_vel.start, 3)
        self.assertEqual(self.ekf.idx_vel.stop, 6)
        
        # Attitude indices  
        self.assertEqual(self.ekf.idx_att.start, 6)
        self.assertEqual(self.ekf.idx_att.stop, 9)
        
        # Angular velocity indices
        self.assertEqual(self.ekf.idx_omega.start, 9)
        self.assertEqual(self.ekf.idx_omega.stop, 12)
        
    def test_predict_step(self):
        """Test prediction step."""
        initial_state = self.ekf.state.copy()
        initial_cov = self.ekf.covariance.copy()
        
        # Simple IMU data
        imu_data = {
            'ang_vel': [0.1, 0.0, 0.0],
            'lin_acc': [0.0, 0.0, 9.81]
        }
        
        # Run prediction
        self.ekf.predict(imu_data)
        
        # State should change
        self.assertFalse(np.allclose(initial_state, self.ekf.state))
        
        # Covariance should increase (uncertainty grows)
        self.assertGreater(np.trace(self.ekf.covariance), np.trace(initial_cov))
        
        # Covariance should remain symmetric and positive definite
        self.assertTrue(np.allclose(self.ekf.covariance, self.ekf.covariance.T))
        self.assertTrue(np.all(np.linalg.eigvals(self.ekf.covariance) > 0))
        
    def test_gps_update(self):
        """Test GPS measurement update."""
        initial_cov_trace = np.trace(self.ekf.covariance)
        
        # GPS measurement
        gps_data = {
            'position': [10.0, 5.0, 2.0]
        }
        
        # Run GPS update
        self.ekf.update_gps(gps_data)
        
        # Covariance should decrease (uncertainty reduces with measurement)
        final_cov_trace = np.trace(self.ekf.covariance)
        self.assertLess(final_cov_trace, initial_cov_trace)
        
        # Covariance should remain symmetric and positive definite
        self.assertTrue(np.allclose(self.ekf.covariance, self.ekf.covariance.T))
        self.assertTrue(np.all(np.linalg.eigvals(self.ekf.covariance) > 0))
        
    def test_imu_update(self):
        """Test IMU measurement update."""
        initial_cov_trace = np.trace(self.ekf.covariance)
        
        # IMU measurement (EKF uses predict for IMU data, not update_imu)
        imu_data = {
            'ang_vel': [0.1, 0.05, 0.02],
            'lin_acc': [0.5, 0.2, 9.81]
        }
        
        # Run IMU prediction (this is how IMU data is processed)
        self.ekf.predict(imu_data)
        
        # Prediction step should increase covariance (uncertainty grows)
        final_cov_trace = np.trace(self.ekf.covariance)
        self.assertGreater(final_cov_trace, initial_cov_trace)
        
        # Covariance properties should be maintained
        self.assertTrue(np.allclose(self.ekf.covariance, self.ekf.covariance.T))
        self.assertTrue(np.all(np.linalg.eigvals(self.ekf.covariance) > 0))
        
    def test_multiple_updates(self):
        """Test multiple prediction and update cycles."""
        for i in range(10):
            # Predict
            imu_data = {
                'ang_vel': [0.01 * i, 0.0, 0.0],
                'lin_acc': [0.1, 0.0, 9.81]
            }
            self.ekf.predict(imu_data)
            
            # Update with GPS every 3rd step
            if i % 3 == 0:
                gps_data = {
                    'position': [i * 0.1, 0.0, 0.0]
                }
                self.ekf.update_gps(gps_data)
            
            # Check no NaN or infinity values
            self.assertFalse(np.any(np.isnan(self.ekf.state)))
            self.assertFalse(np.any(np.isinf(self.ekf.state)))
            self.assertFalse(np.any(np.isnan(self.ekf.covariance)))
            self.assertFalse(np.any(np.isinf(self.ekf.covariance)))
            
            # Check covariance properties
            self.assertTrue(np.allclose(self.ekf.covariance, self.ekf.covariance.T))
            eigenvals = np.linalg.eigvals(self.ekf.covariance)
            self.assertTrue(np.all(eigenvals > 0))
            
    def test_angle_normalization(self):
        """Test angle normalization in attitude states."""
        # Set large angle values
        self.ekf.state[self.ekf.idx_att] = [4 * np.pi, 3 * np.pi, 5 * np.pi]
        
        # Run a prediction step (should normalize angles)
        imu_data = {
            'ang_vel': [0.0, 0.0, 0.0],
            'lin_acc': [0.0, 0.0, 9.81]
        }
        self.ekf.predict(imu_data)
        
        # Angles should be normalized to [-π, π]
        attitudes = self.ekf.state[self.ekf.idx_att]
        self.assertTrue(np.all(attitudes >= -np.pi))
        self.assertTrue(np.all(attitudes <= np.pi))
        
    def test_covariance_bounds(self):
        """Test that covariance doesn't grow unbounded."""
        initial_trace = np.trace(self.ekf.covariance)
        
        # Run many prediction steps without updates
        for i in range(100):
            imu_data = {
                'ang_vel': [0.1, 0.0, 0.0],
                'lin_acc': [0.0, 0.0, 9.81]
            }
            self.ekf.predict(imu_data)
        
        final_trace = np.trace(self.ekf.covariance)
        growth_factor = final_trace / initial_trace
        
        # Should grow but not excessively (less than 1000x)
        self.assertLess(growth_factor, 1000)
        
    def test_state_bounds(self):
        """Test that state values remain within reasonable bounds."""
        # Run simulation with reasonable inputs
        for i in range(50):
            imu_data = {
                'ang_vel': [0.1, 0.05, 0.02],
                'lin_acc': [1.0, 0.5, 9.81]
            }
            self.ekf.predict(imu_data)
            
            if i % 10 == 0:
                gps_data = {
                    'position': [i * 0.5, i * 0.2, 0.0]
                }
                self.ekf.update_gps(gps_data)
        
        # Check reasonable bounds
        position = self.ekf.state[self.ekf.idx_pos]
        velocity = self.ekf.state[self.ekf.idx_vel]
        attitude = self.ekf.state[self.ekf.idx_att]
        
        # Position should be reasonable (< 1km from origin)
        self.assertTrue(np.all(np.abs(position) < 1000))
        
        # Velocity should be reasonable (< 100 m/s)
        self.assertTrue(np.all(np.abs(velocity) < 100))
        
        # Attitude should be normalized
        self.assertTrue(np.all(attitude >= -np.pi))
        self.assertTrue(np.all(attitude <= np.pi))

class TestEKFErrorCases(unittest.TestCase):
    """
    Test error cases and edge conditions.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.ekf = EKFLocalization(dt=1.0/30.0)
        
    def test_zero_dt(self):
        """Test behavior with zero time step."""
        with self.assertRaises((ValueError, ZeroDivisionError)):
            ekf = EKFLocalization(dt=0.0)
            
    def test_negative_dt(self):
        """Test behavior with negative time step."""
        with self.assertRaises(ValueError):
            ekf = EKFLocalization(dt=-0.1)
            
    def test_missing_imu_data(self):
        """Test behavior with missing IMU data."""
        incomplete_data = {
            'ang_vel': [0.1, 0.0]  # Missing z component
        }
        
        with self.assertRaises((KeyError, IndexError, ValueError)):
            self.ekf.predict(incomplete_data)
            
    def test_missing_gps_data(self):
        """Test behavior with missing GPS data."""
        incomplete_data = {
            'position': [10.0, 5.0]  # Missing z component
        }
        
        with self.assertRaises((KeyError, IndexError, ValueError)):
            self.ekf.update_gps(incomplete_data)
            
    def test_large_time_step(self):
        """Test behavior with very large time step."""
        # Large time step should raise ValueError due to validation
        with self.assertRaises(ValueError):
            ekf = EKFLocalization(dt=10.0)  # 10 second time step

def run_component_tests():
    """
    Run all component tests with nice output.
    """
    print("🧪 Running EKF Component Tests")
    print("=" * 40)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestEKFComponents))
    suite.addTest(unittest.makeSuite(TestEKFErrorCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 40)
    if result.wasSuccessful():
        print("✅ All component tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"💥 {len(result.errors)} error(s) occurred")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    run_component_tests()
