#!/usr/bin/env python3

"""
EKF Validation Script

Simple script to validate your EKF implementation using the comprehensive validation framework.
"""

import sys
from pathlib import Path

# Add unified package to path
sys.path.append(str(Path(__file__).parent.parent))

from tests.ekf_validation_framework import EKFValidator

def main():
    """
    Run EKF validation with user-friendly output.
    """
    print("🚗 Polaris EKF Validation")
    print("=" * 40)
    print("This script will validate your EKF implementation using multiple approaches:")
    print("1. Mathematical consistency checks")
    print("2. Synthetic data with known ground truth")
    print("3. Convergence properties")
    print("4. Sensor fusion validation")
    print("5. Numerical stability tests")
    print()
    
    # Initialize validator
    validator = EKFValidator(dt=1.0/30.0)  # 30 Hz
    
    try:
        # Run comprehensive validation
        all_passed, results = validator.run_all_validations()
        
        # Create plots
        output_dir = Path(__file__).parent.parent / 'results' / 'ekf_validation'
        validator.create_validation_plots(output_dir)
        
        # Final assessment
        print()
        if all_passed:
            print("🎉 CONGRATULATIONS!")
            print("Your EKF implementation passed all validation tests.")
            print("Key indicators of correct implementation:")
            print("✅ Mathematical properties are sound")
            print("✅ Accurately tracks synthetic trajectories") 
            print("✅ Converges properly with measurements")
            print("✅ Sensor fusion working correctly")
            print("✅ Numerically stable under various conditions")
        else:
            print("⚠️ VALIDATION ISSUES DETECTED")
            print("Some tests failed. Please review the detailed output above.")
            print("Common issues to check:")
            print("• Covariance matrix properties (symmetry, positive definiteness)")
            print("• State transition model correctness")
            print("• Measurement model implementation")
            print("• Noise parameter tuning")
            print("• Numerical precision and stability")
        
        print(f"\n📊 Detailed results and plots saved to: {output_dir}")
        print("\nNext steps:")
        if all_passed:
            print("• Test with real sensor data")
            print("• Compare with ML approach using compare_ekf_ml.py")
            print("• Optimize for FPGA implementation")
        else:
            print("• Review failed tests in detail")
            print("• Check EKF implementation against theory")
            print("• Run individual test components for debugging")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ ERROR during validation: {e}")
        print("This might indicate a serious issue with the EKF implementation.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
