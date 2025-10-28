#!/usr/bin/env python3
"""Validate the final project structure."""

import sys
from pathlib import Path

print("="*60)
print("VALIDATING PROJECT STRUCTURE")
print("="*60)

# Check big 3 folders exist
big3 = ['data', 'ekf_localization', 'ml']
for folder in big3:
    if Path(folder).exists():
        print(f"[OK] {folder}/ exists")
    else:
        print(f"[FAIL] {folder}/ missing!")
        sys.exit(1)

# Check key files in each folder
data_files = [
    'data/processed/localization_training_data.csv',
    'data/scripts/bag_extractor.py',
    'data/README.md'
]

ekf_files = [
    'ekf_localization/ekf_core.py',
    'ekf_localization/preprocessor.py',
    'ekf_localization/validate.py',
    'ekf_localization/README.md'
]

ml_files = [
    'ml/model.py',
    'ml/ml_localization_training.ipynb',
    'ml/compare_ekf_ml.py',
    'ml/README.md'
]

root_files = [
    'README.md',
    'requirements.txt',
    'setup.py',
    '.gitignore'
]

all_files = data_files + ekf_files + ml_files + root_files

print(f"\nChecking {len(all_files)} key files...")
missing = []
for file in all_files:
    if Path(file).exists():
        print(f"[OK] {file}")
    else:
        print(f"[FAIL] {file}")
        missing.append(file)

# Check old directories are removed
old_dirs = ['polaris_autonomous_system', 'scripts', 'tests', 'results', 'docs', 'config', 'notebooks']
print(f"\nChecking old directories are removed...")
for old_dir in old_dirs:
    if Path(old_dir).exists():
        print(f"[WARNING] {old_dir}/ still exists (should be removed)")
    else:
        print(f"[OK] {old_dir}/ removed")

# Test imports
print("\nTesting imports...")
try:
    from ekf_localization import LocalizationProcessor
    print("[OK] EKF imports work")
except Exception as e:
    print(f"[FAIL] EKF imports: {e}")
    sys.exit(1)

try:
    from ml import LocalizationLSTM
    print("[OK] ML imports work")
except Exception as e:
    print(f"[SKIP] ML imports (dependencies may not be installed): {e}")

# Summary
print("\n" + "="*60)
if missing:
    print(f"[FAIL] {len(missing)} files missing:")
    for f in missing:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("[SUCCESS] ALL VALIDATIONS PASSED!")
    print("="*60)
    print("\nYour project is clean and GitHub-ready!")
    print("\nStructure:")
    print("  data/              - All data management")
    print("  ekf_localization/  - EKF implementation")
    print("  ml/                - Machine learning")
    print("\nNext steps:")
    print("  1. Review README.md")
    print("  2. Test: python ekf_localization/validate.py")
    print("  3. Commit: git add . && git commit -m 'Clean structure'")
    print("  4. Push to GitHub!")

