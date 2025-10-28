# 🚀 ML Localization Quick Reference

## One-Command Setup

```bash
# Install everything you need
pip install torch scikit-learn joblib

# Verify installation
python setup_ml_environment.py
```

## One-Command Training

```bash
# Train the ML model
python polaris_autonomous_system/ml_pipeline/ml_localization.py
```

## Key Commands

```bash
# 1. Setup environment
python setup_ml_environment.py

# 2. Train ML model
python polaris_autonomous_system/ml_pipeline/ml_localization.py

# 3. Compare EKF vs ML
python scripts/compare_ekf_ml.py

# 4. Run EKF validation (baseline)
python scripts/validate_ekf.py
```

## File Locations

| What | Where |
|------|-------|
| Training data | `data/processed/localization_training_data.csv` |
| ML code | `polaris_autonomous_system/ml_pipeline/ml_localization.py` |
| Comparison script | `scripts/compare_ekf_ml.py` |
| Trained models | `models/ml_localization_model.pth` |
| Training results | `results/ml_training/` |
| Comparison plots | `results/comparison/` |

## Expected Performance

| Metric | EKF | ML Target |
|--------|-----|-----------|
| Position RMSE | 0.62 m | < 1.0 m |
| Speed | ~4000 samples/sec | > 10000 samples/sec |
| Latency | ~0.25 ms | < 0.1 ms |
| R² Score | N/A | > 0.90 |

## Hyperparameters Cheat Sheet

### Model Architecture
```python
LocalizationLSTM(
    input_size=10,       # Fixed (number of sensors)
    hidden_size=128,     # Try: 64, 128, 256
    num_layers=2,        # Try: 1, 2, 3
    output_size=12       # Fixed (state dimensions)
)
```

### Training Parameters
```python
trainer.train(
    epochs=50,           # Try: 30, 50, 100
    batch_size=32,       # Try: 16, 32, 64
    learning_rate=0.001  # Try: 0.0001, 0.001, 0.01
)
```

### Data Parameters
```python
trainer.prepare_data(
    sequence_length=10,  # Try: 5, 10, 20
    test_size=0.2        # Try: 0.1, 0.2, 0.3
)
```

## Common Issues

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch
```

### Training too slow
- **CPU**: 20-30 minutes (normal)
- **GPU**: Install CUDA version of PyTorch
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### High training loss (> 0.05 after 10 epochs)
- Increase learning rate to 0.01
- Check that data is normalized
- Verify data file path is correct

### Model not learning (loss stays constant)
- Check for NaN values in data
- Reduce batch size to 16
- Increase learning rate

### "CUDA out of memory"
- Reduce batch size: `batch_size=16`
- Or use CPU: `device='cpu'`
- Or reduce sequence length: `sequence_length=5`

## Performance Tuning

### For Better Accuracy
1. ↑ Increase `hidden_size` (128 → 256)
2. ↑ Increase `num_layers` (2 → 3)
3. ↑ Increase `epochs` (50 → 100)
4. ↑ Increase `sequence_length` (10 → 20)

### For Faster Training
1. ↓ Decrease `hidden_size` (128 → 64)
2. ↓ Decrease `num_layers` (2 → 1)
3. ↑ Increase `batch_size` (32 → 64)
4. ↓ Decrease `epochs` (50 → 30)

### For Smaller Model
1. ↓ Decrease `hidden_size` (128 → 64)
2. ↓ Decrease `num_layers` (2 → 1)
3. ↓ Decrease `sequence_length` (10 → 5)

## Debugging Tips

### Check data loading
```python
import pandas as pd
df = pd.read_csv('data/processed/localization_training_data.csv')
print(df.head())
print(df.columns)
print(df.describe())
```

### Check model architecture
```python
from polaris_autonomous_system.ml_pipeline.ml_localization import LocalizationLSTM
model = LocalizationLSTM(input_size=10)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

### Check GPU availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Experiment Tracking

Create a log file to track your experiments:

```markdown
## Experiment Log

### Experiment 1 (Baseline)
- hidden_size: 128
- num_layers: 2
- epochs: 50
- learning_rate: 0.001
- **Result**: RMSE = 0.85m, R² = 0.92

### Experiment 2 (Larger model)
- hidden_size: 256
- num_layers: 3
- epochs: 50
- learning_rate: 0.001
- **Result**: RMSE = 0.72m, R² = 0.95

### Experiment 3 (More epochs)
...
```

## Visualization Tips

After training, check these plots:
1. **Training history**: Should show decreasing loss
2. **Prediction scatter**: Should be close to diagonal line
3. **Trajectory comparison**: ML should follow EKF closely
4. **Error distribution**: Should be centered near zero

## Next Steps After First Training

1. ✅ Verify RMSE < 1.0m
2. ✅ Check R² score > 0.90
3. ✅ Compare speed with EKF
4. 📊 Run `compare_ekf_ml.py` for detailed analysis
5. 🔧 Tune hyperparameters if needed
6. 📝 Document best configuration
7. 🚀 Test on new data (if available)

## Need Help?

- Check `ML_GETTING_STARTED.md` for detailed guide
- Check `docs/ml_localization_guide.md` for theory
- Check `polaris_autonomous_system/ml_pipeline/ml_localization.py` for code
- Check training plots in `results/ml_training/`

## Quick Sanity Checks

```bash
# Is data there?
ls -lh data/processed/localization_training_data.csv

# Are packages installed?
python -c "import torch, sklearn, joblib; print('All good!')"

# Can I import the ML code?
python -c "from polaris_autonomous_system.ml_pipeline.ml_localization import LocalizationLSTM; print('Code works!')"

# How many data points?
python -c "import pandas as pd; print(len(pd.read_csv('data/processed/localization_training_data.csv')))"
```

---

**Pro Tip**: Start with default parameters, train once, then iterate!

