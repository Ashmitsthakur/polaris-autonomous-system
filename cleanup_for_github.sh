#!/bin/bash

# Cleanup script for preparing Polaris Autonomous System for GitHub
# This script removes unnecessary files and directories

echo "🧹 Cleaning up repository for GitHub publication..."

# Navigate to script directory
cd "$(dirname "$0")"

# Remove Python cache
echo "  → Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type f -name "*.pyd" -delete 2>/dev/null

# Remove build artifacts
echo "  → Removing build artifacts..."
rm -rf build/ dist/ *.egg-info/ 2>/dev/null

# Remove temporary files
echo "  → Removing temporary files..."
find . -type f -name "*.tmp" -delete 2>/dev/null
find . -type f -name "*.bak" -delete 2>/dev/null
find . -type f -name "*~" -delete 2>/dev/null
find . -type f -name ".DS_Store" -delete 2>/dev/null

# Remove test result directories (they'll be regenerated)
echo "  → Removing test result directories..."
rm -rf results/ml_test/ results/ml_test_fixed/ 2>/dev/null

# Remove validation reports (will be regenerated)
echo "  → Removing generated validation reports..."
rm -f config/validation_report.yaml 2>/dev/null

# Remove ROS2 build artifacts (if any in this directory)
echo "  → Removing ROS2 artifacts..."
rm -rf install/ log/ 2>/dev/null

# Create necessary directories if they don't exist
echo "  → Ensuring required directories exist..."
mkdir -p docs/images
mkdir -p data/examples
mkdir -p results
mkdir -p .github/workflows

# Create empty placeholder files for git
touch results/.gitkeep
touch data/examples/.gitkeep

# Replace README with new version
if [ -f "README_NEW.md" ]; then
    echo "  → Updating README.md..."
    mv README.md README_OLD.md 2>/dev/null
    mv README_NEW.md README.md
    echo "    ℹ️  Old README saved as README_OLD.md"
fi

# Show git status
echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 Repository status:"
echo ""
git status --short 2>/dev/null || echo "Git not initialized. Run 'git init' to start."
echo ""
echo "Next steps:"
echo "  1. Review changes with: git status"
echo "  2. Check what will be committed: git add . && git status"
echo "  3. Review .gitignore is working properly"
echo "  4. Make initial commit: git commit -m 'Initial commit'"
echo "  5. Follow steps in GITHUB_PREP.md"
echo ""
echo "📚 Read GITHUB_PREP.md for complete publication guide"

