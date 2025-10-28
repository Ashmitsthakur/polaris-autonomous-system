# 🧠 Getting Started with ML Localization

## 📋 Pre-Flight Checklist

Before starting ML development, let's verify everything is ready:

### ✅ **Current Status Assessment**

#### **1. Data Readiness** ✅
- [x] Processed training data available: `data/processed/localization_training_data.csv`
- [x] EKF baseline results computed (0.62m RMSE)
- [x] 6,480 samples at 30 Hz (216 seconds of driving data)
- [x] Ground truth coordinates in ENU frame

#### **2. Code Readiness** ✅
- [x] ML pipeline skeleton implemented: `polaris_autonomous_system/ml_pipeline/ml_localization.py`
- [x] LSTM neural network architecture defined
- [x] Training framework complete
- [x] Comparison script ready: `scripts/compare_ekf_ml.py`

#### **3. Dependencies** ❌ **NEEDS INSTALLATION**
- [ ] PyTorch (deep learning framework)
- [ ] scikit-learn (data preprocessing)
- [ ] joblib (model serialization)

---

## 🚀 Step-by-Step Setup Guide

### **Step 1: Install ML Dependencies**

Run these commands in order:

```bash
# Install PyTorch (CPU version - faster to download)
pip install torch torchvision torchaudio

# Or for GPU support (if you have NVIDIA GPU with CUDA):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install scikit-learn for preprocessing
pip install scikit-learn

# Install joblib for model saving
pip install joblib
```

**Expected time:** 5-10 minutes depending on internet speed

### **Step 2: Verify Installation**

```bash
# Test PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully!')"

# Test scikit-learn
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__} installed successfully!')"

# Test joblib
python -c "import joblib; print('joblib installed successfully!')"
```

### **Step 3: Create Model Output Directory**

```bash
# Create directory for saving trained models
mkdir -p models
mkdir -p results/ml_training
mkdir -p results/comparison
```

### **Step 4: Run Your First ML Training**

```bash
# Train the ML localization model
python polaris_autonomous_system/ml_pipeline/ml_localization.py
```

**What this does:**
- Loads your processed sensor data
- Trains LSTM neural network to replicate EKF behavior
- Takes ~10-30 minutes depending on your CPU
- Saves trained model to `models/ml_localization_model.pth`
- Generates training plots in `results/ml_training/`

**Expected output:**
```
🚀 Starting ML Localization Training
Using device: cpu (or cuda if GPU available)
Preparing ML training data...
Input features: 10
Target features: 12
Data points: 6480
Training samples: 5175
Test samples: 1295

Training ML localization model...
Epoch   0: Train Loss = 0.045231, Test Loss = 0.038452
Epoch  10: Train Loss = 0.012456, Test Loss = 0.010234
Epoch  20: Train Loss = 0.005678, Test Loss = 0.004892
...
Epoch  50: Train Loss = 0.001234, Test Loss = 0.001567

Evaluating ML localization model...
Evaluation Results:
  RMSE: 0.7532
  R² Score: 0.9645

Model saved to models/ml_localization_model.pth
🎉 ML Localization Training Complete!
```

### **Step 5: Compare EKF vs ML Performance**

```bash
# Run comprehensive comparison
python scripts/compare_ekf_ml.py
```

**What this generates:**
- Performance comparison plots
- Speed benchmarks (samples/second)
- Memory usage analysis
- Accuracy comparison
- Detailed report in `results/comparison/`

---

## 📊 What to Expect

### **ML Model Performance Targets**

| Metric | Target | Why This Target |
|--------|--------|-----------------|
| Position RMSE | < 1.0 m | Comparable to EKF (0.62m) |
| R² Score | > 0.90 | High correlation with ground truth |
| Inference Speed | > 1000 samples/sec | Much faster than EKF |
| Training Time | 10-30 min | Depends on CPU/GPU |
| Model Size | < 10 MB | Small enough for deployment |

### **Typical Training Progression**

1. **Epochs 0-10**: Loss drops rapidly (0.05 → 0.01)
2. **Epochs 10-30**: Steady improvement (0.01 → 0.003)
3. **Epochs 30-50**: Fine-tuning (0.003 → 0.001)
4. **Final Result**: RMSE ~0.7-1.0m, R² > 0.90

### **Common Issues and Solutions**

#### Issue 1: "ModuleNotFoundError: No module named 'torch'"
**Solution:** PyTorch not installed. Run Step 1 above.

#### Issue 2: Training is very slow
**Solution:** 
- Normal on CPU: 20-30 minutes
- With GPU (CUDA): 5-10 minutes
- Consider reducing epochs: `trainer.train(..., epochs=30)`

#### Issue 3: "CUDA out of memory"
**Solution:** 
- Reduce batch size: `trainer.train(..., batch_size=16)`
- Or use CPU: `device = 'cpu'`

#### Issue 4: High RMSE (> 2.0m)
**Solution:**
- Normal for first few epochs
- Should converge to < 1.0m after full training
- Check that data preprocessing is correct

---

## 🎯 ML Development Roadmap

### **Phase 1: Basic Training (This Week)**
- [x] Install dependencies
- [ ] Run first training
- [ ] Verify model trains successfully
- [ ] Check RMSE < 1.0m

### **Phase 2: Optimization (Next Week)**
- [ ] Experiment with hyperparameters
  - Learning rate (try 0.001, 0.0001)
  - Hidden size (try 64, 128, 256)
  - Number of layers (try 1, 2, 3)
  - Sequence length (try 5, 10, 20)
- [ ] Compare different architectures
- [ ] Improve accuracy to match or beat EKF

### **Phase 3: Comparison Analysis (Week 3)**
- [ ] Run comprehensive EKF vs ML comparison
- [ ] Analyze speed differences
- [ ] Document accuracy tradeoffs
- [ ] Create presentation plots

### **Phase 4: Advanced Features (Week 4+)**
- [ ] Real-time inference testing
- [ ] Model quantization for FPGA
- [ ] Uncertainty estimation
- [ ] Online learning / fine-tuning

---

## 💡 Understanding Your ML Pipeline

### **Input Features (10 dimensions)**
```python
sensor_data = [
    ang_vel_x, ang_vel_y, ang_vel_z,  # IMU angular velocity
    lin_acc_x, lin_acc_y, lin_acc_z,  # IMU linear acceleration
    latitude, longitude, altitude,     # GPS position
    speed                              # Vehicle odometry
]
```

### **Output Predictions (12 dimensions)**
```python
state_estimate = [
    x, y, z,                    # Position in ENU frame (meters)
    vx, vy, vz,                 # Velocity (m/s)
    roll, pitch, yaw,           # Attitude (radians)
    omega_x, omega_y, omega_z   # Angular velocity (rad/s)
]
```

### **Neural Network Architecture**
```
Input (10) 
    ↓
LSTM Layer 1 (128 hidden units)
    ↓
LSTM Layer 2 (128 hidden units)
    ↓
Dense Layer (256 units) + ReLU
    ↓
Dense Layer (128 units) + ReLU
    ↓
Dense Layer (64 units) + ReLU
    ↓
Output (12)
```

### **Why LSTM?**
- **Temporal Dependencies**: Vehicle motion has time dependencies
- **Sequence Learning**: Understands patterns over time
- **State Memory**: Maintains internal state like EKF
- **Proven Performance**: Works well for time-series prediction

---

## 🔬 Research Questions to Answer

1. **Can ML match EKF accuracy?**
   - Hypothesis: Yes, should achieve RMSE < 1.0m
   - Measurement: Compare RMSE on test set

2. **Is ML faster than EKF?**
   - Hypothesis: Yes, 5-10x faster inference
   - Measurement: Samples per second comparison

3. **Does ML need less tuning?**
   - Hypothesis: Yes, learns parameters automatically
   - Measurement: Compare setup time

4. **Is ML suitable for FPGA?**
   - Hypothesis: Yes, with quantization
   - Measurement: Model size and computation analysis

5. **Does sequence length matter?**
   - Hypothesis: Longer sequences improve accuracy
   - Measurement: Try 5, 10, 20 timestep sequences

---

## 📚 Additional Resources

### **Key Files to Study**
- `polaris_autonomous_system/ml_pipeline/ml_localization.py` - Main ML implementation
- `scripts/compare_ekf_ml.py` - Comparison framework
- `docs/ml_localization_guide.md` - Detailed ML guide

### **Hyperparameters to Experiment With**
```python
# In ml_localization.py, modify these:
model = LocalizationLSTM(
    input_size=10,
    hidden_size=128,      # Try: 64, 128, 256
    num_layers=2,         # Try: 1, 2, 3
    output_size=12
)

trainer.prepare_data(
    sequence_length=10,   # Try: 5, 10, 20
    test_size=0.2         # Try: 0.1, 0.2, 0.3
)

trainer.train(
    epochs=50,            # Try: 30, 50, 100
    batch_size=32,        # Try: 16, 32, 64
    learning_rate=0.001   # Try: 0.0001, 0.001, 0.01
)
```

### **Performance Tuning Tips**
1. Start with default parameters
2. If training loss doesn't decrease, increase learning rate
3. If validation loss increases, decrease learning rate
4. If overfitting (train << val loss), add more dropout
5. If underfitting (both losses high), increase model size

---

## ✅ Pre-Training Checklist

Before running training, verify:

- [ ] PyTorch installed: `python -c "import torch; print('OK')"`
- [ ] scikit-learn installed: `python -c "import sklearn; print('OK')"`
- [ ] Data file exists: `data/processed/localization_training_data.csv`
- [ ] Output directories created: `models/`, `results/ml_training/`
- [ ] Sufficient disk space: ~100 MB for models and results
- [ ] Sufficient time: 10-30 minutes for training

---

## 🎉 Ready to Start!

Once all dependencies are installed, simply run:

```bash
# Start training
python polaris_autonomous_system/ml_pipeline/ml_localization.py
```

Watch the training progress and wait for completion. The model will automatically:
- Load and preprocess your sensor data
- Train for 50 epochs (adjustable)
- Save the best model
- Generate performance plots
- Print final accuracy metrics

Good luck with your ML localization research! 🚀

