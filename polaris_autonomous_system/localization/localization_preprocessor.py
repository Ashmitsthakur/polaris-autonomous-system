#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

class LocalizationPreprocessor:
    """
    Preprocesses ROS2 bag data for localization algorithm development.
    Handles sensor synchronization, coordinate transformation, and data cleaning.
    """
    
    def __init__(self, bag_data_dir):
        self.bag_data_dir = Path(bag_data_dir)
        self.nav_df = None
        self.control_df = None
        self.speed_df = None
        self.synchronized_data = None
        
    def load_data(self):
        """Load all processed CSV files from bag extraction."""
        print("Loading sensor data...")
        
        # Load navigation data (IMU, GPS, Pose)
        nav_file = self.bag_data_dir / 'navigation.csv'
        if nav_file.exists():
            self.nav_df = pd.read_csv(nav_file)
            print(f"Loaded navigation data: {len(self.nav_df)} samples")
        else:
            raise FileNotFoundError(f"Navigation data not found: {nav_file}")
            
        # Load control data
        control_file = self.bag_data_dir / 'control.csv'
        if control_file.exists():
            self.control_df = pd.read_csv(control_file)
            print(f"Loaded control data: {len(self.control_df)} samples")
            
        # Load speed data
        speed_file = self.bag_data_dir / 'vehicle_speed.csv'
        if speed_file.exists():
            self.speed_df = pd.read_csv(speed_file)
            print(f"Loaded speed data: {len(self.speed_df)} samples")
            
    def synchronize_sensors(self, target_freq=30.0):
        """
        Synchronize all sensors to a common time base.
        
        Args:
            target_freq: Target frequency in Hz for synchronized data
        """
        print(f"Synchronizing sensors to {target_freq} Hz...")
        
        # Create time base
        start_time = min(
            self.nav_df['timestamp'].min(),
            self.control_df['timestamp'].min(),
            self.speed_df['timestamp'].min()
        )
        end_time = max(
            self.nav_df['timestamp'].max(),
            self.control_df['timestamp'].max(),
            self.speed_df['timestamp'].max()
        )
        
        # Convert to seconds and create uniform time grid
        start_sec = start_time / 1e9
        end_sec = end_time / 1e9
        dt = 1.0 / target_freq
        
        time_grid = np.arange(start_sec, end_sec, dt)
        time_ns = (time_grid * 1e9).astype(np.int64)
        
        # Create synchronized dataframe
        sync_data = pd.DataFrame({
            'timestamp': time_ns,
            'time_sec': time_grid
        })
        
        # Interpolate IMU data
        imu_cols = ['ang_vel_x', 'ang_vel_y', 'ang_vel_z', 
                   'lin_acc_x', 'lin_acc_y', 'lin_acc_z']
        for col in imu_cols:
            valid_data = self.nav_df.dropna(subset=[col])
            if len(valid_data) > 1:
                f = interpolate.interp1d(
                    valid_data['timestamp'] / 1e9, 
                    valid_data[col], 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value='extrapolate'
                )
                sync_data[col] = f(time_grid)
        
        # Interpolate GPS data
        gps_cols = ['latitude', 'longitude', 'altitude']
        for col in gps_cols:
            valid_data = self.nav_df.dropna(subset=[col])
            if len(valid_data) > 1:
                f = interpolate.interp1d(
                    valid_data['timestamp'] / 1e9, 
                    valid_data[col], 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value='extrapolate'
                )
                sync_data[col] = f(time_grid)
        
        # Interpolate pose data
        pose_cols = ['pose_x', 'pose_y', 'pose_z', 
                    'pose_qx', 'pose_qy', 'pose_qz', 'pose_qw']
        for col in pose_cols:
            valid_data = self.nav_df.dropna(subset=[col])
            if len(valid_data) > 1:
                f = interpolate.interp1d(
                    valid_data['timestamp'] / 1e9, 
                    valid_data[col], 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value='extrapolate'
                )
                sync_data[col] = f(time_grid)
        
        # Interpolate control data
        control_cols = ['accel', 'steering', 'brake']
        for col in control_cols:
            valid_data = self.control_df.dropna(subset=[col])
            if len(valid_data) > 1:
                f = interpolate.interp1d(
                    valid_data['timestamp'] / 1e9, 
                    valid_data[col], 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value='extrapolate'
                )
                sync_data[col] = f(time_grid)
        
        # Interpolate speed data
        speed_cols = ['speed']
        for col in speed_cols:
            valid_data = self.speed_df.dropna(subset=[col])
            if len(valid_data) > 1:
                f = interpolate.interp1d(
                    valid_data['timestamp'] / 1e9, 
                    valid_data[col], 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value='extrapolate'
                )
                sync_data[col] = f(time_grid)
        
        self.synchronized_data = sync_data
        print(f"Synchronized data: {len(sync_data)} samples at {target_freq} Hz")
        
    def transform_coordinates(self):
        """Transform GPS coordinates to local ENU frame."""
        print("Transforming coordinates to local ENU frame...")
        
        if self.synchronized_data is None:
            raise ValueError("Must synchronize data first")
            
        # Get reference point (first valid GPS reading)
        gps_data = self.synchronized_data.dropna(subset=['latitude', 'longitude'])
        if len(gps_data) == 0:
            print("Warning: No GPS data available for coordinate transformation")
            return
            
        ref_lat = gps_data['latitude'].iloc[0]
        ref_lon = gps_data['longitude'].iloc[0]
        ref_alt = gps_data['altitude'].iloc[0]
        
        print(f"Reference point: Lat={ref_lat:.6f}, Lon={ref_lon:.6f}, Alt={ref_alt:.2f}")
        
        # Convert GPS to ENU coordinates
        enu_x, enu_y, enu_z = self._gps_to_enu(
            self.synchronized_data['latitude'].values,
            self.synchronized_data['longitude'].values,
            self.synchronized_data['altitude'].values,
            ref_lat, ref_lon, ref_alt
        )
        
        self.synchronized_data['enu_x'] = enu_x
        self.synchronized_data['enu_y'] = enu_y
        self.synchronized_data['enu_z'] = enu_z
        
    def _gps_to_enu(self, lat, lon, alt, ref_lat, ref_lon, ref_alt):
        """Convert GPS coordinates to ENU (East-North-Up) frame."""
        # WGS84 ellipsoid parameters
        a = 6378137.0  # Semi-major axis
        f = 1/298.257223563  # Flattening
        e2 = 2*f - f*f  # First eccentricity squared
        
        # Convert to radians
        lat = np.radians(lat)
        lon = np.radians(lon)
        ref_lat = np.radians(ref_lat)
        ref_lon = np.radians(ref_lon)
        
        # Reference point calculations
        N_ref = a / np.sqrt(1 - e2 * np.sin(ref_lat)**2)
        x_ref = (N_ref + ref_alt) * np.cos(ref_lat) * np.cos(ref_lon)
        y_ref = (N_ref + ref_alt) * np.cos(ref_lat) * np.sin(ref_lon)
        z_ref = (N_ref * (1 - e2) + ref_alt) * np.sin(ref_lat)
        
        # Current point calculations
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        x = (N + alt) * np.cos(lat) * np.cos(lon)
        y = (N + alt) * np.cos(lat) * np.sin(lon)
        z = (N * (1 - e2) + alt) * np.sin(lat)
        
        # Convert to ENU
        dx = x - x_ref
        dy = y - y_ref
        dz = z - z_ref
        
        # Rotation matrix from ECEF to ENU
        sin_lat = np.sin(ref_lat)
        cos_lat = np.cos(ref_lat)
        sin_lon = np.sin(ref_lon)
        cos_lon = np.cos(ref_lon)
        
        enu_x = -sin_lon * dx + cos_lon * dy
        enu_y = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        enu_z = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
        
        return enu_x, enu_y, enu_z
        
    def compute_derivatives(self):
        """Compute velocity and acceleration from position data."""
        print("Computing derivatives...")
        
        if self.synchronized_data is None:
            raise ValueError("Must synchronize data first")
            
        # Use uniform time step for gradient computation
        dt = 1.0 / 30.0  # 30 Hz
        
        # Compute velocities from position
        if 'enu_x' in self.synchronized_data.columns:
            self.synchronized_data['vel_x'] = np.gradient(self.synchronized_data['enu_x'], dt)
            self.synchronized_data['vel_y'] = np.gradient(self.synchronized_data['enu_y'], dt)
            self.synchronized_data['vel_z'] = np.gradient(self.synchronized_data['enu_z'], dt)
        
        # Compute accelerations from velocity
        if 'vel_x' in self.synchronized_data.columns:
            self.synchronized_data['acc_x'] = np.gradient(self.synchronized_data['vel_x'], dt)
            self.synchronized_data['acc_y'] = np.gradient(self.synchronized_data['vel_y'], dt)
            self.synchronized_data['acc_z'] = np.gradient(self.synchronized_data['vel_z'], dt)
        
        # Compute angular velocities from quaternions
        if all(col in self.synchronized_data.columns for col in ['pose_qx', 'pose_qy', 'pose_qz', 'pose_qw']):
            quats = self.synchronized_data[['pose_qx', 'pose_qy', 'pose_qz', 'pose_qw']].values
            r = R.from_quat(quats)
            euler = r.as_euler('xyz', degrees=False)
            
            self.synchronized_data['roll'] = euler[:, 0]
            self.synchronized_data['pitch'] = euler[:, 1]
            self.synchronized_data['yaw'] = euler[:, 2]
            
            # Compute angular velocities
            self.synchronized_data['omega_roll'] = np.gradient(euler[:, 0], dt)
            self.synchronized_data['omega_pitch'] = np.gradient(euler[:, 1], dt)
            self.synchronized_data['omega_yaw'] = np.gradient(euler[:, 2], dt)
    
    def clean_data(self):
        """Remove outliers and apply smoothing filters."""
        print("Cleaning data...")
        
        if self.synchronized_data is None:
            raise ValueError("Must synchronize data first")
            
        # Remove extreme outliers in IMU data
        imu_cols = ['ang_vel_x', 'ang_vel_y', 'ang_vel_z', 
                   'lin_acc_x', 'lin_acc_y', 'lin_acc_z']
        for col in imu_cols:
            if col in self.synchronized_data.columns:
                # Remove values beyond 3 standard deviations
                mean_val = self.synchronized_data[col].mean()
                std_val = self.synchronized_data[col].std()
                mask = np.abs(self.synchronized_data[col] - mean_val) < 3 * std_val
                self.synchronized_data.loc[~mask, col] = np.nan
                
                # Interpolate missing values
                self.synchronized_data[col] = self.synchronized_data[col].interpolate()
        
        # Apply smoothing to noisy signals
        from scipy.signal import savgol_filter
        
        # Smooth angular velocities
        for col in ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']:
            if col in self.synchronized_data.columns:
                valid_data = self.synchronized_data[col].dropna()
                if len(valid_data) > 10:
                    window_length = min(11, len(valid_data) // 2 * 2 + 1)
                    smoothed = savgol_filter(valid_data, window_length, 3)
                    self.synchronized_data.loc[valid_data.index, col] = smoothed
    
    def save_processed_data(self, output_file):
        """Save processed data for algorithm development."""
        print(f"Saving processed data to {output_file}")
        
        if self.synchronized_data is None:
            raise ValueError("Must process data first")
            
        self.synchronized_data.to_csv(output_file, index=False)
        
        # Save metadata
        metadata = {
            'total_samples': len(self.synchronized_data),
            'duration_seconds': self.synchronized_data['time_sec'].max() - self.synchronized_data['time_sec'].min(),
            'frequency_hz': 1.0 / np.mean(np.diff(self.synchronized_data['time_sec'])),
            'sensors': {
                'imu': 'ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z',
                'gps': 'latitude,longitude,altitude',
                'pose': 'pose_x,pose_y,pose_z,pose_qx,pose_qy,pose_qz,pose_qw',
                'control': 'accel,steering,brake',
                'speed': 'speed'
            }
        }
        
        metadata_file = output_file.replace('.csv', '_metadata.yaml')
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
    
    def visualize_data(self, output_dir):
        """Create visualization plots of the processed data."""
        print(f"Creating visualizations in {output_dir}")
        
        if self.synchronized_data is None:
            raise ValueError("Must process data first")
            
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Plot trajectory
        if 'enu_x' in self.synchronized_data.columns:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(self.synchronized_data['enu_x'], self.synchronized_data['enu_y'])
            plt.xlabel('East (m)')
            plt.ylabel('North (m)')
            plt.title('Vehicle Trajectory (ENU)')
            plt.grid(True)
            plt.axis('equal')
        
        # Plot IMU data
        if 'ang_vel_x' in self.synchronized_data.columns:
            plt.subplot(2, 2, 2)
            plt.plot(self.synchronized_data['time_sec'], self.synchronized_data['ang_vel_x'], label='X')
            plt.plot(self.synchronized_data['time_sec'], self.synchronized_data['ang_vel_y'], label='Y')
            plt.plot(self.synchronized_data['time_sec'], self.synchronized_data['ang_vel_z'], label='Z')
            plt.xlabel('Time (s)')
            plt.ylabel('Angular Velocity (rad/s)')
            plt.title('IMU Angular Velocity')
            plt.legend()
            plt.grid(True)
        
        # Plot GPS data
        if 'latitude' in self.synchronized_data.columns:
            plt.subplot(2, 2, 3)
            plt.plot(self.synchronized_data['time_sec'], self.synchronized_data['latitude'])
            plt.xlabel('Time (s)')
            plt.ylabel('Latitude (deg)')
            plt.title('GPS Latitude')
            plt.grid(True)
        
        # Plot speed data
        if 'speed' in self.synchronized_data.columns:
            plt.subplot(2, 2, 4)
            plt.plot(self.synchronized_data['time_sec'], self.synchronized_data['speed'])
            plt.xlabel('Time (s)')
            plt.ylabel('Speed (m/s)')
            plt.title('Vehicle Speed')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'sensor_data_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Example usage of the LocalizationPreprocessor."""
    # Process the collect5 data
    bag_dir = '/home/valid_monke/ros2_ws/bags/CAST/collect5_processed'
    preprocessor = LocalizationPreprocessor(bag_dir)
    
    # Load and process data
    preprocessor.load_data()
    preprocessor.synchronize_sensors(target_freq=30.0)
    preprocessor.transform_coordinates()
    preprocessor.compute_derivatives()
    preprocessor.clean_data()
    
    # Save processed data
    output_file = '/home/valid_monke/ros2_ws/localization_training_data.csv'
    preprocessor.save_processed_data(output_file)
    
    # Create visualizations
    preprocessor.visualize_data('/home/valid_monke/ros2_ws/visualizations')
    
    print("Data preprocessing complete!")

if __name__ == '__main__':
    main()
