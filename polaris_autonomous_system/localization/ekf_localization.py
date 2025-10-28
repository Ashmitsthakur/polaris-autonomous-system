#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.linalg import inv, cholesky
import matplotlib.pyplot as plt

class EKFLocalization:
    """
    Extended Kalman Filter for vehicle localization using IMU, GPS, and odometry.
    Optimized for FPGA implementation with fixed-point arithmetic considerations.
    """
    
    def __init__(self, dt=1.0/30.0):
        """
        Initialize EKF for localization.
        
        Args:
            dt: Time step in seconds (default 30 Hz)
        """
        if dt <= 0:
            raise ValueError("Time step dt must be positive")
        if dt > 1.0:
            raise ValueError("Time step dt should be <= 1.0 seconds for stability")
        
        self.dt = dt
        
        # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        self.n_states = 12
        self.state = np.zeros(self.n_states)
        self.covariance = np.eye(self.n_states) * 0.1
        
        # Process noise covariance
        self.Q = np.eye(self.n_states) * 0.01
        
        # Measurement noise covariances
        self.R_imu = np.eye(6) * 0.1  # IMU noise
        self.R_gps = np.eye(3) * 1.0  # GPS noise (higher uncertainty)
        self.R_odom = np.eye(3) * 0.5  # Odometry noise
        
        # State indices
        self.idx_pos = slice(0, 3)    # x, y, z
        self.idx_vel = slice(3, 6)    # vx, vy, vz
        self.idx_att = slice(6, 9)    # roll, pitch, yaw
        self.idx_omega = slice(9, 12) # wx, wy, wz
        
        # Measurement history for analysis
        self.measurement_history = []
        self.state_history = []
        self.covariance_history = []
        
    def predict(self, imu_data):
        """
        Prediction step using IMU data.
        
        Args:
            imu_data: dict with 'ang_vel' and 'lin_acc' arrays
        """
        # Extract current state
        pos = self.state[self.idx_pos]
        vel = self.state[self.idx_vel]
        att = self.state[self.idx_att]
        omega = self.state[self.idx_omega]
        
        # IMU measurements
        ang_vel = imu_data['ang_vel']  # [wx, wy, wz]
        lin_acc = imu_data['lin_acc']  # [ax, ay, az]
        
        # Update angular velocity
        omega_new = ang_vel
        
        # Update attitude using angular velocity
        # Simple Euler integration (can be improved with quaternions)
        att_new = att + omega * self.dt
        
        # Normalize angles to [-π, π]
        att_new = self._normalize_angles(att_new)
        
        # Update velocity using linear acceleration
        # Transform acceleration to world frame
        R_world = self._euler_to_rotation_matrix(att)
        acc_world = R_world @ lin_acc
        
        # Add gravity compensation (assuming z-up)
        acc_world[2] += 9.81
        
        vel_new = vel + acc_world * self.dt
        
        # Update position
        pos_new = pos + vel * self.dt + 0.5 * acc_world * self.dt**2
        
        # Update state
        self.state[self.idx_pos] = pos_new
        self.state[self.idx_vel] = vel_new
        self.state[self.idx_att] = att_new
        self.state[self.idx_omega] = omega_new
        
        # Compute Jacobian matrix F
        F = self._compute_jacobian_F(att, vel, lin_acc)
        
        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q
        
    def update_gps(self, gps_data):
        """
        Update step using GPS data.
        
        Args:
            gps_data: dict with 'position' array [x, y, z] in ENU frame
        """
        # Check for valid GPS data
        if np.any(np.isnan(gps_data['position'])) or np.any(np.isinf(gps_data['position'])):
            return  # Skip update if GPS data is invalid
        
        # Measurement model: h(x) = [x, y, z]
        H = np.zeros((3, self.n_states))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # z
        
        # Predicted measurement
        h_pred = self.state[self.idx_pos]
        
        # Measurement
        z = gps_data['position']
        
        # Innovation
        y = z - h_pred
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + self.R_gps
        
        # Check for numerical stability
        if np.any(np.diag(S) <= 0):
            return  # Skip update if covariance is not positive definite
        
        # Kalman gain
        try:
            K = self.covariance @ H.T @ inv(S)
        except np.linalg.LinAlgError:
            return  # Skip update if matrix inversion fails
        
        # Update state
        self.state += K @ y
        
        # Update covariance (Joseph form for numerical stability)
        I = np.eye(self.n_states)
        self.covariance = (I - K @ H) @ self.covariance
        
        # Store measurement
        self.measurement_history.append({
            'type': 'gps',
            'data': gps_data,
            'innovation': y,
            'innovation_cov': S
        })
        
    def update_odometry(self, odom_data):
        """
        Update step using odometry data.
        
        Args:
            odom_data: dict with 'velocity' array [vx, vy, vz]
        """
        # Check for valid odometry data
        if np.any(np.isnan(odom_data['velocity'])) or np.any(np.isinf(odom_data['velocity'])):
            return  # Skip update if odometry data is invalid
        
        # Measurement model: h(x) = [vx, vy, vz]
        H = np.zeros((3, self.n_states))
        H[0, 3] = 1  # vx
        H[1, 4] = 1  # vy
        H[2, 5] = 1  # vz
        
        # Predicted measurement
        h_pred = self.state[self.idx_vel]
        
        # Measurement
        z = odom_data['velocity']
        
        # Innovation
        y = z - h_pred
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + self.R_odom
        
        # Check for numerical stability
        if np.any(np.diag(S) <= 0):
            return  # Skip update if covariance is not positive definite
        
        # Kalman gain
        try:
            K = self.covariance @ H.T @ inv(S)
        except np.linalg.LinAlgError:
            return  # Skip update if matrix inversion fails
        
        # Update state
        self.state += K @ y
        
        # Update covariance
        I = np.eye(self.n_states)
        self.covariance = (I - K @ H) @ self.covariance
        
        # Store measurement
        self.measurement_history.append({
            'type': 'odom',
            'data': odom_data,
            'innovation': y,
            'innovation_cov': S
        })
        
    def _euler_to_rotation_matrix(self, euler):
        """Convert Euler angles to rotation matrix."""
        roll, pitch, yaw = euler
        
        # Rotation matrices
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return R_z @ R_y @ R_x
        
    def _compute_jacobian_F(self, att, vel, acc):
        """Compute Jacobian matrix F for the process model."""
        F = np.eye(self.n_states)
        
        # Position derivatives
        F[0:3, 3:6] = np.eye(3) * self.dt  # pos depends on vel
        
        # Velocity derivatives
        F[3:6, 6:9] = self._compute_velocity_jacobian(att, acc) * self.dt
        
        # Attitude derivatives
        F[6:9, 9:12] = np.eye(3) * self.dt  # att depends on omega
        
        return F
        
    def _compute_velocity_jacobian(self, att, acc):
        """Compute Jacobian of velocity w.r.t. attitude."""
        roll, pitch, yaw = att
        
        # Simplified Jacobian (can be made more accurate)
        jac = np.zeros((3, 3))
        
        # This is a simplified version - full Jacobian would be more complex
        jac[0, 1] = -acc[2] * np.sin(pitch)  # vx w.r.t. pitch
        jac[1, 0] = acc[2] * np.cos(pitch) * np.sin(roll)  # vy w.r.t. roll
        jac[2, 0] = acc[2] * np.cos(pitch) * np.cos(roll)  # vz w.r.t. roll
        jac[2, 1] = -acc[2] * np.sin(pitch)  # vz w.r.t. pitch
        
        return jac
        
    def process_measurement(self, measurement):
        """
        Process a single measurement.
        
        Args:
            measurement: dict with 'type' and sensor data
        """
        if measurement['type'] == 'imu':
            self.predict(measurement['data'])
        elif measurement['type'] == 'gps':
            self.update_gps(measurement['data'])
        elif measurement['type'] == 'odom':
            self.update_odometry(measurement['data'])
        else:
            raise ValueError(f"Unknown measurement type: {measurement['type']}")
            
        # Store state history
        self.state_history.append(self.state.copy())
        self.covariance_history.append(self.covariance.copy())
        
    def get_position_uncertainty(self):
        """Get position uncertainty (standard deviation)."""
        return np.sqrt(np.diag(self.covariance[self.idx_pos, self.idx_pos]))
        
    def get_velocity_uncertainty(self):
        """Get velocity uncertainty (standard deviation)."""
        return np.sqrt(np.diag(self.covariance[self.idx_vel, self.idx_vel]))
        
    def get_attitude_uncertainty(self):
        """Get attitude uncertainty (standard deviation)."""
        return np.sqrt(np.diag(self.covariance[self.idx_att, self.idx_att]))
        
    def reset(self):
        """Reset the filter to initial state."""
        self.state = np.zeros(self.n_states)
        self.covariance = np.eye(self.n_states) * 0.1
        self.measurement_history = []
        self.state_history = []
        self.covariance_history = []
    
    def _normalize_angles(self, angles):
        """
        Normalize angles to [-π, π] range.
        
        Args:
            angles: Array of angles in radians
            
        Returns:
            Normalized angles in [-π, π]
        """
        return np.arctan2(np.sin(angles), np.cos(angles))
    
    def _euler_to_rotation_matrix(self, euler_angles):
        """
        Convert Euler angles to rotation matrix.
        
        Args:
            euler_angles: [roll, pitch, yaw] in radians
            
        Returns:
            3x3 rotation matrix
        """
        roll, pitch, yaw = euler_angles
        
        # Rotation matrices for each axis
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        # Combined rotation matrix (Z-Y-X order)
        return R_z @ R_y @ R_x
    
    def _compute_jacobian_F(self, att, vel, lin_acc):
        """
        Compute the Jacobian matrix F for the prediction step.
        
        Args:
            att: Current attitude [roll, pitch, yaw]
            vel: Current velocity [vx, vy, vz]
            lin_acc: Linear acceleration [ax, ay, az]
            
        Returns:
            12x12 Jacobian matrix F
        """
        F = np.eye(self.n_states)
        dt = self.dt
        
        # Position derivatives
        F[0:3, 3:6] = np.eye(3) * dt  # dx/dv
        
        # Velocity derivatives (simplified)
        # This is a simplified version - full implementation would include
        # derivatives of rotation matrix w.r.t. attitude
        
        # Attitude derivatives
        F[6:9, 9:12] = np.eye(3) * dt  # datt/domega
        
        return F

class LocalizationProcessor:
    """
    High-level processor that handles data preprocessing and EKF execution.
    """
    
    def __init__(self, data_file):
        """
        Initialize with processed sensor data.
        
        Args:
            data_file: Path to processed CSV file
        """
        self.data = pd.read_csv(data_file)
        self.ekf = EKFLocalization(dt=1.0/30.0)
        self.results = []
        
    def run_localization(self):
        """Run the complete localization algorithm."""
        print("Running localization algorithm...")
        
        for i, row in self.data.iterrows():
            # Process IMU data (every timestep)
            if not pd.isna(row['ang_vel_x']):
                imu_measurement = {
                    'type': 'imu',
                    'data': {
                        'ang_vel': np.array([row['ang_vel_x'], row['ang_vel_y'], row['ang_vel_z']]),
                        'lin_acc': np.array([row['lin_acc_x'], row['lin_acc_y'], row['lin_acc_z']])
                    }
                }
                self.ekf.process_measurement(imu_measurement)
            
            # Process GPS data (when available)
            if not pd.isna(row['latitude']):
                gps_measurement = {
                    'type': 'gps',
                    'data': {
                        'position': np.array([row['enu_x'], row['enu_y'], row['enu_z']])
                    }
                }
                self.ekf.process_measurement(gps_measurement)
            
            # Process odometry data (when available)
            if not pd.isna(row['speed']):
                odom_measurement = {
                    'type': 'odom',
                    'data': {
                        'velocity': np.array([row['vel_x'], row['vel_y'], row['vel_z']])
                    }
                }
                self.ekf.process_measurement(odom_measurement)
            
            # Store results
            result = {
                'timestamp': row['timestamp'],
                'time_sec': row['time_sec'],
                'position': self.ekf.state[self.ekf.idx_pos].copy(),
                'velocity': self.ekf.state[self.ekf.idx_vel].copy(),
                'attitude': self.ekf.state[self.ekf.idx_att].copy(),
                'position_uncertainty': self.ekf.get_position_uncertainty(),
                'velocity_uncertainty': self.ekf.get_velocity_uncertainty(),
                'attitude_uncertainty': self.ekf.get_attitude_uncertainty()
            }
            self.results.append(result)
            
            if i % 1000 == 0:
                print(f"Processed {i}/{len(self.data)} samples")
        
        print("Localization complete!")
        
    def evaluate_accuracy(self, ground_truth_cols=['enu_x', 'enu_y', 'enu_z']):
        """Evaluate localization accuracy against ground truth."""
        print("Evaluating localization accuracy...")
        
        results_df = pd.DataFrame(self.results)
        
        # Compare with ground truth (using ENU coordinates)
        errors = []
        for i, result in enumerate(self.results):
            if i < len(self.data):
                gt_pos = self.data.iloc[i][ground_truth_cols].values
                if not pd.isna(gt_pos).any():
                    est_pos = result['position']
                    error = np.linalg.norm(est_pos - gt_pos)
                    errors.append(error)
        
        errors = np.array(errors)
        
        print(f"Position Error Statistics:")
        print(f"  Mean: {np.mean(errors):.3f} m")
        print(f"  Std:  {np.std(errors):.3f} m")
        print(f"  Max:  {np.max(errors):.3f} m")
        print(f"  RMSE: {np.sqrt(np.mean(errors**2)):.3f} m")
        
        return errors
        
    def plot_results(self, output_dir):
        """Create visualization plots of the localization results."""
        print(f"Creating result plots in {output_dir}")
        
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results_df = pd.DataFrame(self.results)
        
        # Plot trajectory
        plt.figure(figsize=(15, 10))
        
        # Position trajectory
        plt.subplot(2, 3, 1)
        plt.plot(results_df['position'].apply(lambda x: x[0]), 
                results_df['position'].apply(lambda x: x[1]))
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        plt.title('Estimated Trajectory')
        plt.grid(True)
        plt.axis('equal')
        
        # Position vs time
        plt.subplot(2, 3, 2)
        plt.plot(results_df['time_sec'], results_df['position'].apply(lambda x: x[0]), label='X')
        plt.plot(results_df['time_sec'], results_df['position'].apply(lambda x: x[1]), label='Y')
        plt.plot(results_df['time_sec'], results_df['position'].apply(lambda x: x[2]), label='Z')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Position vs Time')
        plt.legend()
        plt.grid(True)
        
        # Velocity vs time
        plt.subplot(2, 3, 3)
        plt.plot(results_df['time_sec'], results_df['velocity'].apply(lambda x: x[0]), label='Vx')
        plt.plot(results_df['time_sec'], results_df['velocity'].apply(lambda x: x[1]), label='Vy')
        plt.plot(results_df['time_sec'], results_df['velocity'].apply(lambda x: x[2]), label='Vz')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocity vs Time')
        plt.legend()
        plt.grid(True)
        
        # Attitude vs time
        plt.subplot(2, 3, 4)
        plt.plot(results_df['time_sec'], results_df['attitude'].apply(lambda x: x[0]), label='Roll')
        plt.plot(results_df['time_sec'], results_df['attitude'].apply(lambda x: x[1]), label='Pitch')
        plt.plot(results_df['time_sec'], results_df['attitude'].apply(lambda x: x[2]), label='Yaw')
        plt.xlabel('Time (s)')
        plt.ylabel('Attitude (rad)')
        plt.title('Attitude vs Time')
        plt.legend()
        plt.grid(True)
        
        # Position uncertainty
        plt.subplot(2, 3, 5)
        plt.plot(results_df['time_sec'], results_df['position_uncertainty'].apply(lambda x: x[0]), label='X')
        plt.plot(results_df['time_sec'], results_df['position_uncertainty'].apply(lambda x: x[1]), label='Y')
        plt.plot(results_df['time_sec'], results_df['position_uncertainty'].apply(lambda x: x[2]), label='Z')
        plt.xlabel('Time (s)')
        plt.ylabel('Position Uncertainty (m)')
        plt.title('Position Uncertainty')
        plt.legend()
        plt.grid(True)
        
        # Error analysis
        plt.subplot(2, 3, 6)
        errors = self.evaluate_accuracy()
        plt.hist(errors, bins=50, alpha=0.7)
        plt.xlabel('Position Error (m)')
        plt.ylabel('Frequency')
        plt.title('Position Error Distribution')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'localization_results.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Example usage of the localization system."""
    # Process data first
    from localization_preprocessor import LocalizationPreprocessor
    
    bag_dir = '/home/valid_monke/ros2_ws/bags/CAST/collect5_processed'
    preprocessor = LocalizationPreprocessor(bag_dir)
    preprocessor.load_data()
    preprocessor.synchronize_sensors(target_freq=30.0)
    preprocessor.transform_coordinates()
    preprocessor.compute_derivatives()
    preprocessor.clean_data()
    
    # Save processed data
    output_file = '/home/valid_monke/ros2_ws/localization_training_data.csv'
    preprocessor.save_processed_data(output_file)
    
    # Run localization
    processor = LocalizationProcessor(output_file)
    processor.run_localization()
    
    # Evaluate and plot results
    processor.evaluate_accuracy()
    processor.plot_results('/home/valid_monke/ros2_ws/localization_results')
    
    print("Localization algorithm development complete!")

if __name__ == '__main__':
    main()
