#!/usr/bin/env python3

"""
Comprehensive EKF Validation Framework

This module provides multiple approaches to validate the correctness of the EKF implementation:
1. Mathematical consistency checks
2. Simulation with known ground truth
3. Sensor fusion validation
4. Performance benchmarking
5. Statistical analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import inv, cholesky
from pathlib import Path
import sys
import time

# Add the localization module to path
sys.path.append(str(Path(__file__).parent.parent))
from polaris_autonomous_system.localization.ekf_localization import EKFLocalization

class EKFValidator:
    """
    Comprehensive validation framework for EKF localization.
    """
    
    def __init__(self, dt=1.0/30.0):
        self.dt = dt
        self.validation_results = {}
        
    def validate_mathematical_consistency(self):
        """
        Test 1: Mathematical Consistency Checks
        Verify that the EKF maintains mathematical properties.
        """
        print("🧮 Testing Mathematical Consistency...")
        
        ekf = EKFLocalization(self.dt)
        results = {}
        
        # Test 1.1: Covariance Matrix Properties
        P = ekf.covariance
        
        # Check symmetry
        symmetric = np.allclose(P, P.T)
        results['covariance_symmetric'] = symmetric
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(P)
        positive_definite = np.all(eigenvals > 0)
        results['covariance_positive_definite'] = positive_definite
        
        # Test 1.2: Process Noise Properties
        Q = ekf.Q
        Q_symmetric = np.allclose(Q, Q.T)
        Q_eigenvals = np.linalg.eigvals(Q)
        Q_positive_definite = np.all(Q_eigenvals >= 0)
        
        results['process_noise_symmetric'] = Q_symmetric
        results['process_noise_positive_definite'] = Q_positive_definite
        
        # Test 1.3: Measurement Noise Properties
        for name, R_matrix in [('imu', ekf.R_imu), ('gps', ekf.R_gps), ('odom', ekf.R_odom)]:
            R_symmetric = np.allclose(R_matrix, R_matrix.T)
            R_eigenvals = np.linalg.eigvals(R_matrix)
            R_positive_definite = np.all(R_eigenvals > 0)
            
            results[f'{name}_noise_symmetric'] = R_symmetric
            results[f'{name}_noise_positive_definite'] = R_positive_definite
        
        # Test 1.4: State Vector Bounds
        # Check if state vector has reasonable bounds
        state_reasonable = True
        if np.any(np.abs(ekf.state[ekf.idx_pos]) > 10000):  # Position > 10km
            state_reasonable = False
        if np.any(np.abs(ekf.state[ekf.idx_vel]) > 100):    # Velocity > 100 m/s
            state_reasonable = False
        if np.any(np.abs(ekf.state[ekf.idx_att]) > 2*np.pi): # Attitude > 2π
            state_reasonable = False
            
        results['state_bounds_reasonable'] = state_reasonable
        
        self.validation_results['mathematical_consistency'] = results
        
        # Print results
        print("   Mathematical Consistency Results:")
        for test, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test}: {status}")
        
        return all(results.values())
    
    def validate_with_synthetic_data(self):
        """
        Test 2: Validation with Synthetic Data
        Generate synthetic trajectory with known ground truth.
        """
        print("🎯 Testing with Synthetic Data...")
        
        # Generate synthetic trajectory
        t_max = 30.0  # 30 seconds
        t = np.arange(0, t_max, self.dt)
        n_steps = len(t)
        
        # Create circular trajectory
        radius = 50.0  # 50m radius
        angular_vel = 0.2  # rad/s
        
        # Ground truth trajectory
        true_x = radius * np.cos(angular_vel * t)
        true_y = radius * np.sin(angular_vel * t)
        true_z = np.zeros_like(t)
        
        true_vx = -radius * angular_vel * np.sin(angular_vel * t)
        true_vy = radius * angular_vel * np.cos(angular_vel * t)
        true_vz = np.zeros_like(t)
        
        true_yaw = angular_vel * t + np.pi/2
        true_roll = np.zeros_like(t)
        true_pitch = np.zeros_like(t)
        
        # Generate synthetic sensor measurements with noise
        np.random.seed(42)  # Reproducible results
        
        # IMU measurements
        imu_noise_std = 0.1
        acc_x = -radius * angular_vel**2 * np.cos(angular_vel * t) + np.random.normal(0, imu_noise_std, n_steps)
        acc_y = -radius * angular_vel**2 * np.sin(angular_vel * t) + np.random.normal(0, imu_noise_std, n_steps)
        acc_z = np.random.normal(0, imu_noise_std, n_steps)
        
        gyro_x = np.random.normal(0, imu_noise_std, n_steps)
        gyro_y = np.random.normal(0, imu_noise_std, n_steps)
        gyro_z = angular_vel + np.random.normal(0, imu_noise_std, n_steps)
        
        # GPS measurements (less frequent, noisier)
        gps_noise_std = 2.0
        gps_frequency = 10  # Every 10th measurement
        
        # Initialize EKF
        ekf = EKFLocalization(self.dt)
        ekf.state[ekf.idx_pos] = [true_x[0], true_y[0], true_z[0]]  # Initialize with true position
        
        # Run EKF with synthetic data
        ekf_positions = []
        ekf_velocities = []
        ekf_attitudes = []
        
        for i in range(n_steps):
            # IMU update
            imu_data = {
                'ang_vel': [gyro_x[i], gyro_y[i], gyro_z[i]],
                'lin_acc': [acc_x[i], acc_y[i], acc_z[i]]
            }
            ekf.predict(imu_data)
            
            # GPS update (less frequent)
            if i % gps_frequency == 0:
                gps_x = true_x[i] + np.random.normal(0, gps_noise_std)
                gps_y = true_y[i] + np.random.normal(0, gps_noise_std)
                gps_z = true_z[i] + np.random.normal(0, gps_noise_std)
                
                gps_data = {
                    'position': [gps_x, gps_y, gps_z]
                }
                ekf.update_gps(gps_data)
            
            # Store results
            ekf_positions.append(ekf.state[ekf.idx_pos].copy())
            ekf_velocities.append(ekf.state[ekf.idx_vel].copy())
            ekf_attitudes.append(ekf.state[ekf.idx_att].copy())
        
        # Convert to numpy arrays
        ekf_positions = np.array(ekf_positions)
        ekf_velocities = np.array(ekf_velocities)
        ekf_attitudes = np.array(ekf_attitudes)
        
        # Calculate errors
        pos_errors = np.sqrt(np.sum((ekf_positions - np.column_stack([true_x, true_y, true_z]))**2, axis=1))
        vel_errors = np.sqrt(np.sum((ekf_velocities - np.column_stack([true_vx, true_vy, true_vz]))**2, axis=1))
        yaw_errors = np.abs(ekf_attitudes[:, 2] - true_yaw)
        
        # Handle angle wrapping
        yaw_errors = np.minimum(yaw_errors, 2*np.pi - yaw_errors)
        
        # Calculate metrics
        rmse_pos = np.sqrt(np.mean(pos_errors**2))
        rmse_vel = np.sqrt(np.mean(vel_errors**2))
        rmse_yaw = np.sqrt(np.mean(yaw_errors**2))
        
        max_pos_error = np.max(pos_errors)
        max_vel_error = np.max(vel_errors)
        max_yaw_error = np.max(yaw_errors)
        
        # Define acceptance criteria
        pos_threshold = 5.0  # 5m RMSE
        vel_threshold = 2.0  # 2 m/s RMSE
        yaw_threshold = 0.2  # 0.2 rad RMSE
        
        results = {
            'rmse_position': rmse_pos,
            'rmse_velocity': rmse_vel,
            'rmse_yaw': rmse_yaw,
            'max_position_error': max_pos_error,
            'max_velocity_error': max_vel_error,
            'max_yaw_error': max_yaw_error,
            'position_accuracy_ok': rmse_pos < pos_threshold,
            'velocity_accuracy_ok': rmse_vel < vel_threshold,
            'yaw_accuracy_ok': rmse_yaw < yaw_threshold,
            'true_trajectory': {'x': true_x, 'y': true_y, 'z': true_z, 'vx': true_vx, 'vy': true_vy, 'vz': true_vz, 'yaw': true_yaw},
            'ekf_trajectory': {'pos': ekf_positions, 'vel': ekf_velocities, 'att': ekf_attitudes},
            'errors': {'pos': pos_errors, 'vel': vel_errors, 'yaw': yaw_errors}
        }
        
        self.validation_results['synthetic_data'] = results
        
        # Print results
        print(f"   Position RMSE: {rmse_pos:.3f}m (threshold: {pos_threshold}m)")
        print(f"   Velocity RMSE: {rmse_vel:.3f}m/s (threshold: {vel_threshold}m/s)")
        print(f"   Yaw RMSE: {rmse_yaw:.3f}rad (threshold: {yaw_threshold}rad)")
        
        accuracy_ok = results['position_accuracy_ok'] and results['velocity_accuracy_ok'] and results['yaw_accuracy_ok']
        status = "✅ PASS" if accuracy_ok else "❌ FAIL"
        print(f"   Overall Accuracy: {status}")
        
        return accuracy_ok
    
    def validate_convergence_properties(self):
        """
        Test 3: Convergence Properties
        Test if EKF converges properly with sufficient measurements.
        """
        print("📈 Testing Convergence Properties...")
        
        ekf = EKFLocalization(self.dt)
        
        # Initial large uncertainty
        ekf.covariance *= 100
        initial_trace = np.trace(ekf.covariance)
        
        # Simulate measurements to see if uncertainty reduces
        traces = [initial_trace]
        
        for i in range(100):
            # Simulate IMU measurement
            imu_data = {
                'ang_vel': [0.1, 0.05, 0.0],
                'lin_acc': [0.0, 0.0, 9.81]
            }
            ekf.predict(imu_data)
            
            # Simulate GPS measurement every 10 steps
            if i % 10 == 0:
                gps_data = {
                    'position': [0.0, 0.0, 0.0]
                }
                ekf.update_gps(gps_data)
            
            traces.append(np.trace(ekf.covariance))
        
        # Check if uncertainty decreased
        final_trace = traces[-1]
        convergence_ratio = final_trace / initial_trace
        
        results = {
            'initial_uncertainty': initial_trace,
            'final_uncertainty': final_trace,
            'convergence_ratio': convergence_ratio,
            'converged': convergence_ratio < 0.1,  # Should reduce by at least 90%
            'trace_history': traces
        }
        
        self.validation_results['convergence'] = results
        
        print(f"   Initial uncertainty: {initial_trace:.3f}")
        print(f"   Final uncertainty: {final_trace:.3f}")
        print(f"   Reduction ratio: {convergence_ratio:.3f}")
        
        status = "✅ PASS" if results['converged'] else "❌ FAIL"
        print(f"   Convergence: {status}")
        
        return results['converged']
    
    def validate_sensor_fusion(self):
        """
        Test 4: Sensor Fusion Validation
        Test if different sensors contribute appropriately to the estimate.
        """
        print("🔗 Testing Sensor Fusion...")
        
        # Test with GPS-only
        ekf_gps = EKFLocalization(self.dt)
        
        # Test with IMU-only
        ekf_imu = EKFLocalization(self.dt)
        
        # Test with both sensors
        ekf_fused = EKFLocalization(self.dt)
        
        # Simulate 100 steps with different sensor configurations
        for i in range(100):
            # Common IMU data
            imu_data = {
                'ang_vel': [0.0, 0.0, 0.1],
                'lin_acc': [0.1, 0.0, 9.81]
            }
            
            # GPS data (every 10 steps)
            if i % 10 == 0:
                gps_data = {
                    'position': [i * 0.1, 0.0, 0.0]  # Moving forward
                }
            
            # GPS-only EKF
            if i % 10 == 0:
                ekf_gps.predict(imu_data)  # Still need to predict
                ekf_gps.update_gps(gps_data)
            else:
                ekf_gps.predict(imu_data)
            
            # IMU-only EKF
            ekf_imu.predict(imu_data)
            
            # Fused EKF
            ekf_fused.predict(imu_data)
            if i % 10 == 0:
                ekf_fused.update_gps(gps_data)
        
        # Compare final uncertainties
        gps_uncertainty = np.trace(ekf_gps.covariance)
        imu_uncertainty = np.trace(ekf_imu.covariance)
        fused_uncertainty = np.trace(ekf_fused.covariance)
        
        # Fused should have lower uncertainty than individual sensors
        fusion_better_than_gps = fused_uncertainty < gps_uncertainty
        fusion_better_than_imu = fused_uncertainty < imu_uncertainty
        
        results = {
            'gps_only_uncertainty': gps_uncertainty,
            'imu_only_uncertainty': imu_uncertainty,
            'fused_uncertainty': fused_uncertainty,
            'fusion_improves_gps': fusion_better_than_gps,
            'fusion_improves_imu': fusion_better_than_imu,
            'sensor_fusion_working': fusion_better_than_gps and fusion_better_than_imu
        }
        
        self.validation_results['sensor_fusion'] = results
        
        print(f"   GPS-only uncertainty: {gps_uncertainty:.3f}")
        print(f"   IMU-only uncertainty: {imu_uncertainty:.3f}")
        print(f"   Fused uncertainty: {fused_uncertainty:.3f}")
        
        status = "✅ PASS" if results['sensor_fusion_working'] else "❌ FAIL"
        print(f"   Sensor Fusion: {status}")
        
        return results['sensor_fusion_working']
    
    def validate_numerical_stability(self):
        """
        Test 5: Numerical Stability
        Test if EKF remains stable under various conditions.
        """
        print("⚖️ Testing Numerical Stability...")
        
        results = {}
        
        # Test 5.1: Large initial uncertainty
        ekf = EKFLocalization(self.dt)
        ekf.covariance *= 1e6  # Very large initial uncertainty
        
        stable_large_uncertainty = True
        try:
            for i in range(50):
                imu_data = {'ang_vel': [0.1, 0.0, 0.0], 'lin_acc': [0.0, 0.0, 9.81]}
                ekf.predict(imu_data)
                
                if i % 10 == 0:
                    gps_data = {'position': [0.0, 0.0, 0.0]}
                    ekf.update_gps(gps_data)
                
                # Check for NaN or infinity
                if np.any(np.isnan(ekf.state)) or np.any(np.isinf(ekf.state)):
                    stable_large_uncertainty = False
                    break
                    
                if np.any(np.isnan(ekf.covariance)) or np.any(np.isinf(ekf.covariance)):
                    stable_large_uncertainty = False
                    break
        except:
            stable_large_uncertainty = False
        
        results['stable_large_uncertainty'] = stable_large_uncertainty
        
        # Test 5.2: High-frequency measurements
        ekf = EKFLocalization(0.001)  # 1000 Hz
        
        stable_high_frequency = True
        try:
            for i in range(100):
                imu_data = {'ang_vel': [0.01, 0.0, 0.0], 'lin_acc': [0.0, 0.0, 9.81]}
                ekf.predict(imu_data)
                
                # Check for NaN or infinity
                if np.any(np.isnan(ekf.state)) or np.any(np.isinf(ekf.state)):
                    stable_high_frequency = False
                    break
        except:
            stable_high_frequency = False
        
        results['stable_high_frequency'] = stable_high_frequency
        
        # Test 5.3: Large measurement values
        ekf = EKFLocalization(self.dt)
        
        stable_large_measurements = True
        try:
            for i in range(20):
                # Large GPS coordinates (simulating different coordinate systems)
                gps_data = {'position': [1e6, 1e6, 1e3]}
                ekf.update_gps(gps_data)
                
                if np.any(np.isnan(ekf.state)) or np.any(np.isinf(ekf.state)):
                    stable_large_measurements = False
                    break
        except:
            stable_large_measurements = False
        
        results['stable_large_measurements'] = stable_large_measurements
        
        # Overall stability
        results['numerically_stable'] = all([
            stable_large_uncertainty,
            stable_high_frequency,
            stable_large_measurements
        ])
        
        self.validation_results['numerical_stability'] = results
        
        print(f"   Large uncertainty stability: {'✅ PASS' if stable_large_uncertainty else '❌ FAIL'}")
        print(f"   High frequency stability: {'✅ PASS' if stable_high_frequency else '❌ FAIL'}")
        print(f"   Large measurement stability: {'✅ PASS' if stable_large_measurements else '❌ FAIL'}")
        
        status = "✅ PASS" if results['numerically_stable'] else "❌ FAIL"
        print(f"   Overall Stability: {status}")
        
        return results['numerically_stable']
    
    def run_all_validations(self):
        """
        Run all validation tests and provide comprehensive results.
        """
        print("🔍 COMPREHENSIVE EKF VALIDATION")
        print("=" * 50)
        
        # Run all tests
        test_results = {}
        test_results['mathematical_consistency'] = self.validate_mathematical_consistency()
        test_results['synthetic_data'] = self.validate_with_synthetic_data()
        test_results['convergence'] = self.validate_convergence_properties()
        test_results['sensor_fusion'] = self.validate_sensor_fusion()
        test_results['numerical_stability'] = self.validate_numerical_stability()
        
        # Overall assessment
        all_passed = all(test_results.values())
        
        print("\n" + "=" * 50)
        print("📋 VALIDATION SUMMARY")
        print("=" * 50)
        
        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 ALL TESTS PASSED - EKF IMPLEMENTATION IS CORRECT!")
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW EKF IMPLEMENTATION")
        print("=" * 50)
        
        return all_passed, self.validation_results
    
    def create_validation_plots(self, output_dir):
        """
        Create visualization plots for validation results.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if 'synthetic_data' in self.validation_results:
            self._plot_synthetic_validation(output_path)
        
        if 'convergence' in self.validation_results:
            self._plot_convergence(output_path)
        
        print(f"📊 Validation plots saved to {output_path}")
    
    def _plot_synthetic_validation(self, output_path):
        """Plot synthetic data validation results."""
        data = self.validation_results['synthetic_data']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Position trajectory
        ax = axes[0, 0]
        true_traj = data['true_trajectory']
        ekf_traj = data['ekf_trajectory']
        
        ax.plot(true_traj['x'], true_traj['y'], 'b-', label='True Trajectory', linewidth=2)
        ax.plot(ekf_traj['pos'][:, 0], ekf_traj['pos'][:, 1], 'r--', label='EKF Estimate', linewidth=2)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Position Trajectory Comparison')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        # Position errors
        ax = axes[0, 1]
        t = np.arange(len(data['errors']['pos'])) * self.dt
        ax.plot(t, data['errors']['pos'], 'r-', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error (m)')
        ax.set_title(f'Position Error (RMSE: {data["rmse_position"]:.3f}m)')
        ax.grid(True)
        
        # Velocity comparison
        ax = axes[1, 0]
        ax.plot(t, true_traj['vx'], 'b-', label='True Vx', linewidth=2)
        ax.plot(t, ekf_traj['vel'][:, 0], 'r--', label='EKF Vx', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity X (m/s)')
        ax.set_title('Velocity Comparison')
        ax.legend()
        ax.grid(True)
        
        # Yaw comparison
        ax = axes[1, 1]
        ax.plot(t, true_traj['yaw'], 'b-', label='True Yaw', linewidth=2)
        ax.plot(t, ekf_traj['att'][:, 2], 'r--', label='EKF Yaw', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Yaw (rad)')
        ax.set_title('Yaw Comparison')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'synthetic_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_convergence(self, output_path):
        """Plot convergence validation results."""
        data = self.validation_results['convergence']
        
        plt.figure(figsize=(10, 6))
        steps = np.arange(len(data['trace_history']))
        plt.semilogy(steps, data['trace_history'], 'b-', linewidth=2)
        plt.xlabel('Update Steps')
        plt.ylabel('Trace of Covariance Matrix (log scale)')
        plt.title('EKF Convergence: Uncertainty Reduction Over Time')
        plt.grid(True)
        
        # Add annotations
        plt.annotate(f'Initial: {data["initial_uncertainty"]:.1f}', 
                    xy=(0, data['initial_uncertainty']), 
                    xytext=(10, data['initial_uncertainty']), 
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.annotate(f'Final: {data["final_uncertainty"]:.1f}', 
                    xy=(len(data['trace_history'])-1, data['final_uncertainty']), 
                    xytext=(len(data['trace_history'])-20, data['final_uncertainty']*10), 
                    arrowprops=dict(arrowstyle='->', color='green'))
        
        plt.savefig(output_path / 'convergence_validation.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Main function to run EKF validation.
    """
    print("🔍 EKF Localization Validation Framework")
    print("=" * 50)
    
    # Initialize validator
    validator = EKFValidator(dt=1.0/30.0)
    
    # Run all validations
    all_passed, results = validator.run_all_validations()
    
    # Create validation plots
    output_dir = Path(__file__).parent.parent / 'results' / 'ekf_validation'
    validator.create_validation_plots(output_dir)
    
    # Generate detailed report
    report_path = output_dir / 'validation_report.txt'
    with open(report_path, 'w') as f:
        f.write("EKF VALIDATION DETAILED REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for test_name, test_data in results.items():
            f.write(f"{test_name.upper().replace('_', ' ')}\n")
            f.write("-" * 30 + "\n")
            
            if isinstance(test_data, dict):
                for key, value in test_data.items():
                    if not isinstance(value, (list, np.ndarray)):
                        f.write(f"{key}: {value}\n")
            f.write("\n")
    
    print(f"\n📁 Detailed results saved to: {output_dir}")
    
    return all_passed

if __name__ == '__main__':
    main()
