#!/usr/bin/env python3

import rclpy
from rclpy.serialization import deserialize_message
import sqlite3
from rclpy.logging import get_logger
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from rosidl_runtime_py.utilities import get_message
import importlib

class BagExtractor:
    def __init__(self, bag_path):
        self.bag_path = Path(bag_path)
        self.logger = get_logger('bag_extractor')
        
        # Topics we want to extract based on the actual bag contents
        self.topics_of_interest = {
            '/pacmod/accel_rpt': 'pacmod3_msgs/msg/SystemRptFloat',  # % Value
            '/pacmod/steering_rpt': 'pacmod3_msgs/msg/SystemRptFloat',  # radians value
            '/pacmod/brake_rpt': 'pacmod3_msgs/msg/SystemRptFloat',  # % Value
            '/pacmod/vehicle_speed_rpt': 'pacmod3_msgs/msg/VehicleSpeedRpt',  # m/s value
            '/vectornav/pose': 'geometry_msgs/msg/PoseWithCovarianceStamped',  # Position and orientation
            '/vectornav/imu': 'sensor_msgs/msg/Imu',  # orientation, angular_velocity, linear_acceleration
            '/vectornav/gnss': 'sensor_msgs/msg/NavSatFix'  # latitude, longitude
        }
        
        self.data = {topic: [] for topic in self.topics_of_interest.keys()}

    def connect_to_bag(self):
        db_path = next(self.bag_path.glob('*.db3'))
        self.logger.info(f"Attempting to connect to database: {db_path}")
        
        try:
            # First try to repair the database using sqlite3 recovery
            import subprocess
            import os
            
            # Create a backup of the original database
            backup_path = db_path.parent / f"backup_{db_path.name}"
            if not backup_path.exists():
                import shutil
                shutil.copy2(db_path, backup_path)
                self.logger.info(f"Created backup at: {backup_path}")
            
            # Try to repair the database
            self.logger.info("Attempting to repair database...")
            repair_cmd = f"echo '.recover' | sqlite3 {db_path}"
            try:
                os.system(repair_cmd)
                self.logger.info("Database repair attempt completed")
            except Exception as e:
                self.logger.warning(f"Repair attempt failed: {str(e)}")
            
            # Try to connect to the database
            self.conn = sqlite3.connect(str(db_path))
            self.cursor = self.conn.cursor()
            
            # Test if we can read the tables
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = self.cursor.fetchall()
            self.logger.info(f"Found tables: {[t[0] for t in tables]}")
            
            # Check for messages table
            self.cursor.execute("SELECT COUNT(*) FROM messages")
            msg_count = self.cursor.fetchone()[0]
            self.logger.info(f"Found {msg_count} messages in the database")
            
        except sqlite3.DatabaseError as e:
            self.logger.error(f"Database error: {str(e)}")
            self.logger.info("Attempting to repair database...")
            
            # Try to repair the database using sqlite3 recovery
            import subprocess
            output_path = db_path.parent / f"recovered_{db_path.name}"
            try:
                subprocess.run(['sqlite3', str(db_path), '.recover'], 
                             stdout=open(str(output_path), 'wb'),
                             stderr=subprocess.PIPE,
                             check=True)
                self.logger.info(f"Recovery attempted. New database created at: {output_path}")
                self.conn = sqlite3.connect(str(output_path))
                self.cursor = self.conn.cursor()
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Recovery failed: {e.stderr.decode()}")
                raise

    def get_msg_type(self, msg_type_str):
        """Get the message type from the string representation."""
        try:
            return get_message(msg_type_str)
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to import message type {msg_type_str}: {str(e)}")
            raise

    def extract_messages(self):
        for topic, msg_type_str in self.topics_of_interest.items():
            query = """
            SELECT m.timestamp, m.data 
            FROM messages m JOIN topics t ON m.topic_id = t.id 
            WHERE t.name = ? ORDER BY m.timestamp
            """
            self.cursor.execute(query, (topic,))
            rows = self.cursor.fetchall()
            
            msg_type = self.get_msg_type(msg_type_str)
            for timestamp, data in rows:
                msg_dict = {
                    'timestamp': timestamp,
                    'data': deserialize_message(data, msg_type)
                }
                self.data[topic].append(msg_dict)

    def process_vehicle_data(self):
        data = []
        for msg in self.data['/pacmod/vehicle_speed_rpt']:
            data.append({
                'timestamp': msg['timestamp'],
                'speed': msg['data'].vehicle_speed,  # m/s value
                'rolling_counter': msg['data'].rolling_counter,
                'fault_wrt_sensor': msg['data'].fault_wrt_sensor
            })
        return pd.DataFrame(data)

    def process_control_data(self):
        control_data = []
        for msg in self.data['/pacmod/accel_rpt']:
            control_data.append({
                'timestamp': msg['timestamp'],
                'accel': msg['data'].manual_input,  # % value
                'accel_cmd': msg['data'].command,
                'accel_enabled': msg['data'].enabled
            })
        
        # Match timestamps for steering and brake data
        steering_dict = {msg['timestamp']: {
            'steering': msg['data'].manual_input,
            'steering_cmd': msg['data'].command,
            'steering_enabled': msg['data'].enabled
        } for msg in self.data['/pacmod/steering_rpt']}
        
        brake_dict = {msg['timestamp']: {
            'brake': msg['data'].manual_input,
            'brake_cmd': msg['data'].command,
            'brake_enabled': msg['data'].enabled
        } for msg in self.data['/pacmod/brake_rpt']}
        
        for entry in control_data:
            steering_data = steering_dict.get(entry['timestamp'], {})
            brake_data = brake_dict.get(entry['timestamp'], {})
            entry.update(steering_data)
            entry.update(brake_data)
            
        return pd.DataFrame(control_data)

    def process_pose_data(self):
        pose_data = []
        for msg in self.data['/vectornav/pose']:
            pose_with_cov = msg['data']
            pose_data.append({
                'timestamp': msg['timestamp'],
                'x': pose_with_cov.pose.pose.position.x,
                'y': pose_with_cov.pose.pose.position.y,
                'z': pose_with_cov.pose.pose.position.z,
                'qx': pose_with_cov.pose.pose.orientation.x,
                'qy': pose_with_cov.pose.pose.orientation.y,
                'qz': pose_with_cov.pose.pose.orientation.z,
                'qw': pose_with_cov.pose.pose.orientation.w,
                'covariance': list(pose_with_cov.pose.covariance)  # Include covariance data
            })
        return pd.DataFrame(pose_data)

    def process_navigation_data(self):
        nav_data = []
        
        # Get pose and IMU data
        pose_dict = {msg['timestamp']: msg['data'] for msg in self.data['/vectornav/pose']}
        imu_dict = {msg['timestamp']: msg['data'] for msg in self.data['/vectornav/imu']}
        gnss_dict = {msg['timestamp']: msg['data'] for msg in self.data['/vectornav/gnss']}
        
        # Combine all timestamps
        all_timestamps = sorted(set(pose_dict.keys()) | set(imu_dict.keys()) | set(gnss_dict.keys()))
        
        for timestamp in all_timestamps:
            entry = {'timestamp': timestamp}
            
            if timestamp in pose_dict:
                pose = pose_dict[timestamp]
                entry.update({
                    'pose_x': pose.pose.position.x,
                    'pose_y': pose.pose.position.y,
                    'pose_z': pose.pose.position.z,
                    'pose_qx': pose.pose.orientation.x,
                    'pose_qy': pose.pose.orientation.y,
                    'pose_qz': pose.pose.orientation.z,
                    'pose_qw': pose.pose.orientation.w
                })
            
            if timestamp in imu_dict:
                imu = imu_dict[timestamp]
                entry.update({
                    'ang_vel_x': imu.angular_velocity.x,
                    'ang_vel_y': imu.angular_velocity.y,
                    'ang_vel_z': imu.angular_velocity.z,
                    'lin_acc_x': imu.linear_acceleration.x,
                    'lin_acc_y': imu.linear_acceleration.y,
                    'lin_acc_z': imu.linear_acceleration.z
                })
            
            if timestamp in gnss_dict:
                gnss = gnss_dict[timestamp]
                entry.update({
                    'latitude': gnss.latitude,
                    'longitude': gnss.longitude,
                    'altitude': gnss.altitude
                })
            
            nav_data.append(entry)
            
        return pd.DataFrame(nav_data)

    def save_to_csv(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Process and save vehicle dynamics data
        vehicle_df = self.process_vehicle_data()
        control_df = self.process_control_data()
        nav_df = self.process_navigation_data()
        
        vehicle_df.to_csv(output_path / 'vehicle_speed.csv', index=False)
        control_df.to_csv(output_path / 'control.csv', index=False)
        nav_df.to_csv(output_path / 'navigation.csv', index=False)

def list_available_topics(conn):
    """List all available topics in the bag file."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT name, type, COUNT(*) as msg_count
        FROM topics t
        JOIN messages m ON t.id = m.topic_id
        GROUP BY name, type
        ORDER BY name;
    """)
    return cursor.fetchall()

def main(args=None):
    rclpy.init(args=args)
    logger = get_logger('bag_extractor')
    
    # Process all bags
    bag_paths = [
        '/home/valid_monke/ros2_ws/bags/CAST/collect5',
        '/home/valid_monke/ros2_ws/bags/CAST/follow5.1',
        '/home/valid_monke/ros2_ws/bags/CAST/follow5.2'
    ]
    
    for bag_path in bag_paths:
        try:
            logger.info(f"Processing bag: {bag_path}")
            extractor = BagExtractor(bag_path)
            
            # Verify bag file exists and is readable
            db_files = list(Path(bag_path).glob('*.db3'))
            if not db_files:
                logger.error(f"No .db3 files found in {bag_path}")
                continue
                
            for db_file in db_files:
                try:
                    logger.info(f"Checking database file: {db_file}")
                    # Test database connection and list available topics
                    with sqlite3.connect(str(db_file)) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        tables = cursor.fetchall()
                        if not tables:
                            logger.error(f"No tables found in database: {db_file}")
                            continue
                            
                        # List available topics
                        topics = list_available_topics(conn)
                        logger.info("Available topics in the bag:")
                        for topic_name, msg_type, count in topics:
                            logger.info(f"  - {topic_name} ({msg_type}): {count} messages")
                            
                    extractor.connect_to_bag()
                    extractor.extract_messages()
                    output_dir = f'{bag_path}_processed'
                    extractor.save_to_csv(output_dir)
                    logger.info(f"Successfully processed bag: {bag_path}")
                    logger.info(f"Data saved to: {output_dir}")
                except sqlite3.DatabaseError as dbe:
                    logger.error(f"Database error in {db_file}: {str(dbe)}")
                except Exception as e:
                    logger.error(f"Error processing {db_file}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing bag {bag_path}: {str(e)}")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
