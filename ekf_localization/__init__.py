"""
EKF Localization Module

Extended Kalman Filter implementation for vehicle localization
using multi-sensor fusion (IMU, GPS, odometry).
"""

from .ekf_core import EKFLocalization, LocalizationProcessor
from .preprocessor import LocalizationPreprocessor

__all__ = ['EKFLocalization', 'LocalizationProcessor', 'LocalizationPreprocessor']

