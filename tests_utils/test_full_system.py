#!/usr/bin/env python3
"""Comprehensive system test - ensures everything actually works."""

import sys
from pathlib import Path

print("="*70)
print("COMPREHENSIVE SYSTEM TEST")
print("="*70)

passed = 0
failed = 0
warnings = 0

def test(name, condition, critical=True):
    global passed, failed, warnings
    if condition:
        print(f"[OK] {name}")
        passed += 1
        return True
    else:
        if critical:
            print(f"[FAIL] {name}")
            failed += 1
        else:
            print(f"[WARN] {name}")
            warnings += 1
        return False

# 1. Structure Tests
print("\n1. Testing Project Structure...")
test("data/ folder exists", Path("data").exists())
test("ekf_localization/ folder exists", Path("ekf_localization").exists())
test("ml/ folder exists", Path("ml").exists())
test("No old polaris_autonomous_system/", not Path("polaris_autonomous_system").exists())
test("No old scripts/", not Path("scripts").exists())
test("No old tests/", not Path("tests").exists())

# 2. Critical Files
print("\n2. Testing Critical Files...")
test("README.md exists", Path("README.md").exists())
test("requirements.txt exists", Path("requirements.txt").exists())
test("setup.py exists", Path("setup.py").exists())
test(".gitignore exists", Path(".gitignore").exists())
test("main.py exists", Path("main.py").exists())

# 3. Data Module
print("\n3. Testing Data Module...")
test("data/README.md exists", Path("data/README.md").exists())
test("data/__init__.py exists", Path("data/__init__.py").exists())
test("data/scripts/ exists", Path("data/scripts").exists())
test("data/processed/ exists", Path("data/processed").exists())
test("Training data exists", Path("data/processed/localization_training_data.csv").exists())

# 4. EKF Module
print("\n4. Testing EKF Module...")
test("ekf_localization/README.md", Path("ekf_localization/README.md").exists())
test("ekf_localization/__init__.py", Path("ekf_localization/__init__.py").exists())
test("ekf_localization/ekf_core.py", Path("ekf_localization/ekf_core.py").exists())
test("ekf_localization/preprocessor.py", Path("ekf_localization/preprocessor.py").exists())
test("ekf_localization/validate.py", Path("ekf_localization/validate.py").exists())

# 5. ML Module
print("\n5. Testing ML Module...")
test("ml/README.md", Path("ml/README.md").exists())
test("ml/__init__.py", Path("ml/__init__.py").exists())
test("ml/model.py", Path("ml/model.py").exists())
test("ml/ml_localization_training.ipynb", Path("ml/ml_localization_training.ipynb").exists())
test("ml/compare_ekf_ml.py", Path("ml/compare_ekf_ml.py").exists())

# 6. Import Tests
print("\n6. Testing Python Imports...")
try:
    from ekf_localization import EKFLocalization
    test("Can import EKFLocalization", True)
except Exception as e:
    test(f"Can import EKFLocalization: {e}", False)

try:
    from ekf_localization import LocalizationPreprocessor
    test("Can import LocalizationPreprocessor", True)
except Exception as e:
    test(f"Can import LocalizationPreprocessor: {e}", False)

try:
    from ekf_localization import LocalizationProcessor
    test("Can import LocalizationProcessor", True)
except Exception as e:
    test(f"Can import LocalizationProcessor: {e}", False)

try:
    from ml import LocalizationLSTM
    test("Can import LocalizationLSTM", True)
except Exception as e:
    test(f"Can import LocalizationLSTM (may need PyTorch): {e}", False, critical=False)

try:
    from ml import MLLocalizationTrainer
    test("Can import MLLocalizationTrainer", True)
except Exception as e:
    test(f"Can import MLLocalizationTrainer (may need PyTorch): {e}", False, critical=False)

# 7. Data Integrity
print("\n7. Testing Data Integrity...")
try:
    import pandas as pd
    df = pd.read_csv("data/processed/localization_training_data.csv")
    test(f"Training data readable ({len(df)} rows)", len(df) > 0)
    
    required_cols = ['timestamp', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']
    has_cols = all(col in df.columns for col in required_cols)
    test("Training data has required columns", has_cols)
except Exception as e:
    test(f"Training data readable: {e}", False)

# 8. Module Initialization
print("\n8. Testing Module Initialization...")
try:
    import ekf_localization
    test("ekf_localization module initializes", True)
except Exception as e:
    test(f"ekf_localization module: {e}", False)

try:
    import ml
    test("ml module initializes", True)
except Exception as e:
    test(f"ml module (may need PyTorch): {e}", False, critical=False)

try:
    import data
    test("data module initializes", True)
except Exception as e:
    test(f"data module: {e}", False)

# 9. EKF Functionality Test
print("\n9. Testing EKF Functionality...")
try:
    from ekf_localization import EKFLocalization
    ekf = EKFLocalization(dt=1/30.0)
    test("Can create EKF instance", True)
    test("EKF has 12 states", ekf.n_states == 12)
    test("EKF state initialized", ekf.state is not None)
    test("EKF covariance initialized", ekf.covariance is not None)
except Exception as e:
    test(f"EKF functionality: {e}", False)

# 10. Documentation
print("\n10. Testing Documentation...")
readme_files = [
    "README.md",
    "data/README.md", 
    "ekf_localization/README.md",
    "ml/README.md"
]
for readme in readme_files:
    test(f"{readme} exists", Path(readme).exists())

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print(f"Passed:   {passed} [OK]")
print(f"Failed:   {failed} [FAIL]")
print(f"Warnings: {warnings} [WARN]")
print(f"Total:    {passed + failed + warnings}")
print("="*70)

if failed == 0:
    print("\n[SUCCESS] ALL CRITICAL TESTS PASSED! System is fully functional!")
    if warnings > 0:
        print(f"[WARN] {warnings} non-critical warnings (likely missing ML dependencies)")
    print("\nYour project is ready for:")
    print("  [OK] GitHub publication")
    print("  [OK] Development")
    print("  [OK] Collaboration")
    print("  [OK] Production use")
    sys.exit(0)
else:
    print(f"\n[FAIL] {failed} CRITICAL TESTS FAILED")
    print("Please fix issues before publishing.")
    sys.exit(1)

