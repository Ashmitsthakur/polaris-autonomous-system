# Documentation Images

This directory contains images and diagrams used in the documentation.

## Adding Images

When adding images to documentation:

1. Place the image in this directory
2. Use descriptive filenames (e.g., `ekf_architecture.png`, `sensor_fusion_diagram.png`)
3. Keep image sizes reasonable (< 1MB when possible)
4. Use PNG for diagrams and screenshots, JPEG for photos

## Current Images

- `system_overview.png` - High-level system architecture (placeholder)
- `ekf_flow.png` - EKF algorithm flowchart (placeholder)
- `validation_results.png` - Sample validation results (placeholder)

## Generating Images

Some images can be generated from the code:

```bash
# Generate localization results plot
python main.py --data-file data/localization_training_data.csv \
               --output-dir results --visualize

# Generate EKF validation plots
python scripts/validate_ekf.py

# Results will be in results/ekf_validation/
```

## Image Guidelines

- **Resolution**: At least 800px wide for main images
- **Format**: PNG for technical diagrams, JPEG for photos
- **Compression**: Use online tools to optimize size
- **Attribution**: Document source if using external images

