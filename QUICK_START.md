# 🚀 Quick Start Guide

## Your Clean Project Structure

```
polaris-autonomous-system/
├── 📊 data/              # All data files and processing
├── 🎯 ekf_localization/  # EKF sensor fusion
└── 🧠 ml/                # Machine learning
```

---

## ⚡ Common Tasks

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup ML environment (if using ML)
cd ml && python setup_environment.py
```

### Process Data
```bash
# Extract from ROS2 bags
python data/scripts/bag_extractor.py --input data/raw/CAST/collect5

# Or use main CLI
python main.py ekf --data-file data/processed/localization_training_data.csv
```

### Run EKF Localization
```python
from ekf_localization import LocalizationProcessor

processor = LocalizationProcessor('data/processed/localization_training_data.csv')
processor.run_localization()
```

### Train ML Model
```bash
# Open Jupyter notebook
cd ml
jupyter notebook ml_localization_training.ipynb

# Or train via Python
python main.py ml-train --data-file data/processed/localization_training_data.csv
```

### Compare EKF vs ML
```bash
python ml/compare_ekf_ml.py --data-file data/processed/localization_training_data.csv
```

### Validate System
```bash
# Validate EKF
python ekf_localization/validate.py

# Validate structure
python validate_structure.py
```

---

## 📂 Where to Find Things

| What | Where |
|------|-------|
| Raw data | `data/raw/CAST/` |
| Processed data | `data/processed/` |
| EKF core | `ekf_localization/ekf_core.py` |
| ML model | `ml/model.py` |
| Training notebook | `ml/ml_localization_training.ipynb` |
| Documentation | `README.md` in each folder |
| Tests | `ekf_localization/validate.py` |
| Config | `ekf_localization/config.yaml` |

---

## 🎯 Project Workflow

```
1. DATA → 2. EKF → 3. ML
   ↓         ↓        ↓
  Raw     Fusion   Learning
```

1. **Data**: Extract and process sensor data
2. **EKF**: Run sensor fusion for ground truth
3. **ML**: Train neural network to replicate EKF

---

## 📖 Documentation

- **Root README.md** - Project overview
- **data/README.md** - Data management
- **ekf_localization/README.md** - EKF details
- **ml/README.md** - ML training

---

## 🆘 Troubleshooting

### Import errors?
```bash
python validate_structure.py
```

### Missing dependencies?
```bash
pip install -r requirements.txt
cd ml && python setup_environment.py
```

### Need ML dependencies?
```bash
pip install torch scikit-learn joblib
```

---

## 🎉 Your Project is Ready!

Everything is tested and working. You can now:

1. ✅ Develop in a clean structure
2. ✅ Find files instantly
3. ✅ Publish to GitHub
4. ✅ Collaborate easily

For complete details, see `FINAL_SUMMARY.md`

---

**Happy coding!** 🚗💨

