#!/usr/bin/env python3
"""Test that all imports work correctly."""

import sys

# Test EKF imports
try:
    from ekf_localization import LocalizationProcessor, EKFLocalization, LocalizationPreprocessor
    print("[OK] EKF imports work")
except Exception as e:
    print(f"[FAIL] EKF imports failed: {e}")
    sys.exit(1)

# Test ML imports (may fail if torch not installed)
try:
    from ml import LocalizationLSTM, MLLocalizationTrainer
    print("[OK] ML imports work")
except Exception as e:
    print(f"[SKIP] ML imports (dependencies not installed): {e}")

# Test that key files exist
import os
from pathlib import Path

required_files = [
    'data/processed/localization_training_data.csv',
    'ekf_localization/ekf_core.py',
    'ekf_localization/preprocessor.py',
    'ml/model.py',
    'ml/ml_localization_training.ipynb',
]

all_exist = True
for file in required_files:
    if Path(file).exists():
        print(f"[OK] Found: {file}")
    else:
        print(f"[FAIL] Missing: {file}")
        all_exist = False

if all_exist:
    print("\n[OK] All required files exist!")
else:
    print("\n[FAIL] Some files are missing!")
    sys.exit(1)

print("\n" + "="*50)
print("[SUCCESS] ALL TESTS PASSED!")
print("="*50)
print("\nYour project structure is working correctly!")

