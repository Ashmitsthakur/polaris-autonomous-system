#!/usr/bin/env python3

import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'polaris_autonomous_system'

# Read requirements from file
def read_requirements(filename):
    """Read requirements from a file and return as list."""
    try:
        with open(filename, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

# Read README for long description
def read_readme():
    """Read README file for long description."""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Polaris Autonomous System - Real-time localization and ML pipeline for autonomous vehicles"

setup(
    name=package_name,
    version='1.0.0',
    author='valid_monke',
    author_email='valid_monke@tamu.edu',
    description='Unified autonomous driving system with localization and ML pipeline',
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/valid_monke/polaris-autonomous-system",
    
    # Package discovery
    packages=find_packages(),
    
    # Data files for ROS2
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.[pxy][yma]*')),
    ],
    
    # Dependencies
    install_requires=[
        'setuptools',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'pyyaml>=5.4.0',
        'psutil>=5.8.0',
        'rclpy>=3.0.0',
        'rosbag2_py>=0.15.0',
        'geometry_msgs',
        'sensor_msgs',
    ],
    
    # Optional extras
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.12.0',
            'black>=21.0.0',
            'flake8>=3.9.0',
        ],
        'ml': [
            'torch>=1.9.0',
            'scikit-learn>=1.0.0',
            'joblib>=1.0.0',
        ],
        'fpga': [
            'cocotb>=1.6.0',  # For FPGA simulation
        ],
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    python_requires=">=3.8",
    zip_safe=True,
    
    # Console scripts
    entry_points={
        'console_scripts': [
            'polaris-extract-bags=polaris_autonomous_system.data_processing.bag_extractor:main',
            'polaris-preprocess=polaris_autonomous_system.localization.localization_preprocessor:main',
            'polaris-localize=polaris_autonomous_system.localization.ekf_localization:main',
            'polaris-system=main:main',
        ],
    },
    
    # Include package data
    include_package_data=True,
    package_data={
        package_name: ['*.yaml', '*.yml', '*.json'],
    },
)
