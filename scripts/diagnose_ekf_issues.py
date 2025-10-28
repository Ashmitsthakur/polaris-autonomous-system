#!/usr/bin/env python3

"""
EKF Issue Diagnosis Script

Analyzes the specific issues found in EKF validation and provides recommendations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add unified package to path
sys.path.append(str(Path(__file__).parent.parent))

from polaris_autonomous_system.localization.ekf_localization import EKFLocalization

def diagnose_process_noise():
    """Diagnose if process noise is appropriate."""
    print("🔍 Diagnosing Process Noise...")
    
    ekf = EKFLocalization(dt=1.0/30.0)
    
    # Check process noise magnitude
    Q_trace = np.trace(ekf.Q)
    print(f"   Process noise trace: {Q_trace:.6f}")
    
    # Run prediction steps and see how fast uncertainty grows
    initial_trace = np.trace(ekf.covariance)
    
    for i in range(30):  # 1 second at 30 Hz
        imu_data = {'ang_vel': [0.0, 0.0, 0.0], 'lin_acc': [0.0, 0.0, 9.81]}
        ekf.predict(imu_data)
    
    final_trace = np.trace(ekf.covariance)
    growth_rate = (final_trace / initial_trace - 1) * 100
    
    print(f"   Uncertainty growth in 1s: {growth_rate:.1f}%")
    
    if growth_rate > 1000:
        print("   ❌ Process noise too high - uncertainty grows too fast")
        recommendation = "Reduce Q matrix values by factor of 10-100"
    elif growth_rate < 10:
        print("   ❌ Process noise too low - EKF won't trust motion model")
        recommendation = "Increase Q matrix values by factor of 2-10"
    else:
        print("   ✅ Process noise seems reasonable")
        recommendation = "Process noise OK"
    
    return recommendation

def diagnose_measurement_noise():
    """Diagnose measurement noise settings."""
    print("🔍 Diagnosing Measurement Noise...")
    
    ekf = EKFLocalization(dt=1.0/30.0)
    
    # Check GPS noise
    gps_std = np.sqrt(np.diag(ekf.R_gps))
    print(f"   GPS position noise std: {gps_std}")
    
    # Check IMU noise
    imu_std = np.sqrt(np.diag(ekf.R_imu))
    print(f"   IMU measurement noise std: {imu_std}")
    
    # Test GPS update effect
    initial_pos_var = ekf.covariance[0, 0]  # x position variance
    
    gps_data = {'position': [0.0, 0.0, 0.0]}
    ekf.update_gps(gps_data)
    
    final_pos_var = ekf.covariance[0, 0]
    improvement = (1 - final_pos_var / initial_pos_var) * 100
    
    print(f"   GPS update reduces position uncertainty by: {improvement:.1f}%")
    
    if improvement < 5:
        print("   ❌ GPS noise too high - measurements barely help")
        return "Reduce GPS noise (R_gps) by factor of 2-5"
    elif improvement > 95:
        print("   ❌ GPS noise too low - EKF trusts GPS too much")
        return "Increase GPS noise (R_gps) by factor of 2-5"
    else:
        print("   ✅ GPS measurement noise seems reasonable")
        return "GPS noise OK"

def diagnose_state_transition():
    """Diagnose state transition model."""
    print("🔍 Diagnosing State Transition Model...")
    
    ekf = EKFLocalization(dt=1.0/30.0)
    
    # Test if constant velocity motion works
    ekf.state[3:6] = [1.0, 0.0, 0.0]  # Set velocity to 1 m/s in x
    initial_pos = ekf.state[0:3].copy()
    
    # Predict with no acceleration
    imu_data = {'ang_vel': [0.0, 0.0, 0.0], 'lin_acc': [0.0, 0.0, 9.81]}
    ekf.predict(imu_data)
    
    final_pos = ekf.state[0:3]
    expected_pos = initial_pos + np.array([1.0/30.0, 0.0, 0.0])  # Should move dt * velocity
    
    position_error = np.linalg.norm(final_pos - expected_pos)
    
    print(f"   Position prediction error: {position_error:.6f}m")
    
    if position_error > 1e-3:
        print("   ❌ State transition model has issues")
        return "Check kinematic equations in predict() method"
    else:
        print("   ✅ State transition model working correctly")
        return "State transition OK"

def diagnose_initialization():
    """Diagnose initialization parameters."""
    print("🔍 Diagnosing Initialization...")
    
    ekf = EKFLocalization(dt=1.0/30.0)
    
    # Check initial covariance
    initial_std = np.sqrt(np.diag(ekf.covariance))
    
    print(f"   Initial position std: {initial_std[0:3]}")
    print(f"   Initial velocity std: {initial_std[3:6]}")
    print(f"   Initial attitude std: {initial_std[6:9]}")
    
    if np.any(initial_std > 10):
        print("   ❌ Initial uncertainty too high")
        return "Reduce initial covariance matrix values"
    elif np.any(initial_std < 0.001):
        print("   ❌ Initial uncertainty too low")
        return "Increase initial covariance matrix values"
    else:
        print("   ✅ Initial uncertainty seems reasonable")
        return "Initialization OK"

def main():
    """Main diagnosis function."""
    print("🩺 EKF ISSUE DIAGNOSIS")
    print("=" * 50)
    
    print("\nBased on validation results, your EKF has these issues:")
    print("• Poor accuracy on synthetic trajectory")
    print("• Uncertainty increases instead of decreasing")
    print("• Sensor fusion not working effectively")
    print("\nLet's diagnose the root causes...\n")
    
    # Run diagnostics
    recommendations = []
    
    recommendations.append(diagnose_process_noise())
    recommendations.append(diagnose_measurement_noise())
    recommendations.append(diagnose_state_transition())
    recommendations.append(diagnose_initialization())
    
    print("\n" + "=" * 50)
    print("📋 DIAGNOSIS SUMMARY & RECOMMENDATIONS")
    print("=" * 50)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n🔧 IMMEDIATE ACTIONS TO TAKE:")
    print("1. Tune noise parameters (Q and R matrices)")
    print("2. Check measurement update implementations")
    print("3. Verify state transition model correctness")
    print("4. Test with simpler scenarios first")
    
    print("\n⚡ QUICK FIXES TO TRY:")
    print("• Reduce process noise Q by factor of 10: ekf.Q *= 0.1")
    print("• Reduce GPS noise R_gps by factor of 2: ekf.R_gps *= 0.5") 
    print("• Check that GPS updates actually get called")
    print("• Verify IMU coordinate frame matches expectations")

if __name__ == '__main__':
    main()
