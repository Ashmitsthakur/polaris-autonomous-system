#!/usr/bin/env python3
"""
Setup script for ML localization development environment.
Checks dependencies, creates directories, and verifies data availability.
"""

import sys
import subprocess
import os
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Python version: {version_str}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error("Python 3.8+ required")
        return False
    
    print_success("Python version compatible")
    return True

def check_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def check_dependencies():
    """Check required and optional dependencies."""
    print_header("Checking Dependencies")
    
    # Core dependencies (should be installed)
    core_deps = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'pyyaml': 'yaml'
    }
    
    # ML dependencies (need to be installed)
    ml_deps = {
        'torch': 'torch',
        'scikit-learn': 'sklearn',
        'joblib': 'joblib'
    }
    
    all_ok = True
    missing_core = []
    missing_ml = []
    
    print("Core dependencies:")
    for package, import_name in core_deps.items():
        if check_package(package, import_name):
            print_success(f"{package:20} installed")
        else:
            print_error(f"{package:20} MISSING")
            missing_core.append(package)
            all_ok = False
    
    print("\nML dependencies:")
    for package, import_name in ml_deps.items():
        if check_package(package, import_name):
            print_success(f"{package:20} installed")
        else:
            print_warning(f"{package:20} NOT INSTALLED (required for ML)")
            missing_ml.append(package)
    
    return all_ok, missing_core, missing_ml

def install_ml_dependencies(packages):
    """Install missing ML dependencies."""
    print_header("Installing ML Dependencies")
    
    if not packages:
        print_info("All ML dependencies already installed")
        return True
    
    print_info(f"Will install: {', '.join(packages)}")
    response = input("\nProceed with installation? [y/N]: ").strip().lower()
    
    if response != 'y':
        print_warning("Installation cancelled")
        return False
    
    # Map package names to pip install names
    pip_packages = []
    for pkg in packages:
        if pkg == 'torch':
            print_info("Installing PyTorch (CPU version)...")
            print_info("For GPU support, visit: https://pytorch.org/get-started/locally/")
            pip_packages.append('torch')
        elif pkg == 'scikit-learn':
            pip_packages.append('scikit-learn')
        elif pkg == 'joblib':
            pip_packages.append('joblib')
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install'
        ] + pip_packages)
        print_success("ML dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Installation failed: {e}")
        return False

def check_data_files():
    """Check if required data files exist."""
    print_header("Checking Data Files")
    
    required_files = [
        'data/processed/localization_training_data.csv',
        'data/processed/localization_training_data_metadata.yaml'
    ]
    
    all_exist = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print_success(f"{file_path:50} ({size_mb:.2f} MB)")
        else:
            print_error(f"{file_path:50} NOT FOUND")
            all_exist = False
    
    return all_exist

def create_directories():
    """Create necessary output directories."""
    print_header("Creating Output Directories")
    
    directories = [
        'models',
        'results/ml_training',
        'results/comparison',
        'results/visualizations'
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print_success(f"Created: {directory}")
        else:
            print_info(f"Exists:  {directory}")

def verify_ml_pipeline():
    """Verify ML pipeline code is present."""
    print_header("Verifying ML Pipeline Code")
    
    required_files = [
        'polaris_autonomous_system/ml_pipeline/ml_localization.py',
        'scripts/compare_ekf_ml.py'
    ]
    
    all_exist = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            lines = len(path.read_text().splitlines())
            print_success(f"{file_path:50} ({lines} lines)")
        else:
            print_error(f"{file_path:50} NOT FOUND")
            all_exist = False
    
    return all_exist

def print_next_steps(ml_deps_installed):
    """Print next steps for the user."""
    print_header("Next Steps")
    
    if ml_deps_installed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Environment is ready for ML development!{Colors.END}\n")
        
        print(f"{Colors.BOLD}To start training:{Colors.END}")
        print(f"  python polaris_autonomous_system/ml_pipeline/ml_localization.py\n")
        
        print(f"{Colors.BOLD}To compare EKF vs ML:{Colors.END}")
        print(f"  python scripts/compare_ekf_ml.py\n")
        
        print(f"{Colors.BOLD}For detailed guide:{Colors.END}")
        print(f"  See ML_GETTING_STARTED.md\n")
        
        print(f"{Colors.BOLD}Expected training time:{Colors.END}")
        print(f"  CPU: 20-30 minutes")
        print(f"  GPU: 5-10 minutes\n")
        
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ ML dependencies need to be installed{Colors.END}\n")
        
        print(f"{Colors.BOLD}Install manually:{Colors.END}")
        print(f"  pip install torch scikit-learn joblib\n")
        
        print(f"{Colors.BOLD}Then run this script again:{Colors.END}")
        print(f"  python setup_ml_environment.py\n")

def main():
    """Main setup function."""
    print_header("ML Localization Environment Setup")
    print("This script will check your environment and prepare for ML development\n")
    
    # Check Python version
    if not check_python_version():
        print_error("Please upgrade Python to version 3.8 or higher")
        sys.exit(1)
    
    # Check dependencies
    deps_ok, missing_core, missing_ml = check_dependencies()
    
    if missing_core:
        print_error(f"\nCore dependencies missing: {', '.join(missing_core)}")
        print_info("Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check data files
    data_ok = check_data_files()
    
    if not data_ok:
        print_error("\nProcessed data files not found!")
        print_info("Run preprocessing first:")
        print_info("  python main.py --data-file <input> --output-dir data/processed")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Verify ML pipeline
    pipeline_ok = verify_ml_pipeline()
    
    if not pipeline_ok:
        print_error("\nML pipeline files missing!")
        print_info("Make sure you're in the project root directory")
        sys.exit(1)
    
    # Install ML dependencies if needed
    ml_installed = True
    if missing_ml:
        print_header("ML Dependencies Status")
        print_warning(f"Missing ML packages: {', '.join(missing_ml)}")
        ml_installed = install_ml_dependencies(missing_ml)
    
    # Print next steps
    print_next_steps(ml_installed and not missing_ml)
    
    # Final summary
    print_header("Setup Summary")
    print(f"Python version:      {Colors.GREEN}✓{Colors.END}")
    print(f"Core dependencies:   {Colors.GREEN}✓{Colors.END}")
    print(f"Data files:          {Colors.GREEN}✓{Colors.END}")
    print(f"Output directories:  {Colors.GREEN}✓{Colors.END}")
    print(f"ML pipeline code:    {Colors.GREEN}✓{Colors.END}")
    
    if missing_ml and not ml_installed:
        print(f"ML dependencies:     {Colors.YELLOW}⚠ (need installation){Colors.END}")
        print(f"\n{Colors.YELLOW}Note: ML dependencies are required for training{Colors.END}")
    else:
        print(f"ML dependencies:     {Colors.GREEN}✓{Colors.END}")
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Ready to start ML development!{Colors.END}")

if __name__ == '__main__':
    main()

