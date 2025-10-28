# 📊 Data Module

This folder contains all data management for the Polaris autonomous system.

## 📁 Structure

```
data/
├── raw/                    # Raw ROS2 bag files
│   └── CAST/              # Data collection runs
│       ├── collect5/
│       ├── follow5.1/
│       └── follow5.2/
├── processed/             # Processed CSV datasets  
│   ├── localization_training_data.csv
│   └── localization_training_data_metadata.yaml
├── examples/              # Sample data (optional)
└── scripts/               # Data processing utilities
    ├── bag_extractor.py   # Extract from ROS2 bags
    ├── test_pipeline.py   # Test data pipeline
    └── read_bag.py        # Inspect bag files
```

## 🚀 Usage

### Extract Data from ROS2 Bags

```python
from data.scripts.bag_extractor import BagExtractor

extractor = BagExtractor('data/raw/CAST/collect5')
extractor.connect_to_bag()
extractor.extract_messages()
# Process and save...
```

### Command Line

```bash
# Process raw bags
python data/scripts/bag_extractor.py \
    --input data/raw/CAST/collect5 \
    --output data/processed
```

## 📊 Data Format

### Raw Data
- **Format**: ROS2 bag files (`.db3`)
- **Topics**: IMU, GPS, odometry, control
- **Frequency**: 30 Hz (synchronized)

### Processed Data
- **Format**: CSV file
- **Columns**: 
  - Sensor data: IMU, GPS, speed
  - Derived: velocities, accelerations, ENU coordinates
  - Ground truth: position, attitude

## 📈 Data Statistics

| Dataset | Duration | Samples | Frequency |
|---------|----------|---------|-----------|
| collect5 | 216 sec | 6,480 | 30 Hz |
| follow5.1 | TBD | TBD | 30 Hz |
| follow5.2 | TBD | TBD | 30 Hz |

## 🔄 Data Pipeline

1. **Raw Collection** → ROS2 bags in `data/raw/`
2. **Extraction** → Use `bag_extractor.py`
3. **Preprocessing** → Sync, transform, clean
4. **Output** → CSV in `data/processed/`
5. **Ready for EKF/ML** → Use processed data

## 📝 Notes

- Raw bags are large (> 100MB each)
- Not committed to Git
- Provide download link if needed
- Processed data is smaller and included in repo

