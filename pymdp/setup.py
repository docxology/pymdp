#!/usr/bin/env python3
"""
Setup configuration for the RGM package.
Automatically installs all dependencies and prepares the environment.
"""

import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path
from setuptools import setup, find_packages

MIN_PYTHON_VERSION = (3, 8)

def clean_build_files():
    """Clean up build artifacts and temporary files."""
    patterns = [
        'build', 'dist', '*.egg-info', '**/__pycache__',
        '**/*.pyc', '**/*.pyo', '**/*.pyd', '.eggs'
    ]
    
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print(f"✓ Cleaned {path}")
            except Exception as e:
                print(f"Warning: Could not clean {path}: {e}")

def check_python_version():
    """Check if Python version meets minimum requirements."""
    if sys.version_info < MIN_PYTHON_VERSION:
        sys.exit(f"❌ Python {'.'.join(map(str, MIN_PYTHON_VERSION))} or higher is required")
    print(f"✓ Python version: {sys.version.split()[0]}")

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        # First upgrade pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--user"])
        
        # Install from requirements.txt if it exists
        if os.path.exists('requirements.txt'):
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--user"
            ])
        else:
            # Fallback to minimal requirements
            dependencies = [
                "torch>=1.7.0",
                "numpy>=1.19.0",
                "matplotlib>=3.3.0",
                "seaborn>=0.11.0",
                "pandas>=1.1.0",
                "scipy>=1.5.0",
                "tqdm>=4.50.0",
                "pyyaml>=5.3.0",
                "jsonschema>=3.2.0",
                "psutil>=5.7.0"
            ]
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--user"
            ] + dependencies)
            
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_cuda():
    """Check if CUDA is available and print version info."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️ CUDA not available, will use CPU")
    except ImportError:
        print("ℹ️ PyTorch not found, will be installed")
    except Exception as e:
        print(f"ℹ️ CUDA check failed (will use CPU): {str(e)}")

def create_default_directories():
    """Create default directories needed by RGM."""
    directories = [
        'configs',
        'experiments',
        'data',
        'rgm/utils',
        'rgm/models',
        'rgm/training'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_package():
    """Install the package in development mode."""
    try:
        # Override sys.argv to force install command
        sys.argv = [sys.argv[0], "install", "--user"]
        
        setup(
            name="rgm",
            version="0.1.0",
            description="Renormalization Generative Model (RGM)",
            author="RGM Team",
            packages=find_packages(include=['rgm', 'rgm.*']),
            python_requires=f">={'.'.join(map(str, MIN_PYTHON_VERSION))}",
            install_requires=[
                "torch>=1.7.0",
                "numpy>=1.19.0",
                "matplotlib>=3.3.0",
                "seaborn>=0.11.0",
                "pandas>=1.1.0",
                "scipy>=1.5.0",
                "tqdm>=4.50.0",
                "pyyaml>=5.3.0",
                "jsonschema>=3.2.0",
                "psutil>=5.7.0"
            ],
            include_package_data=True,
            zip_safe=False
        )
        print("✓ Package installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install package: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n🚀 Setting up RGM package...")
    print("=" * 50)
    
    print("\n🧹 Cleaning previous build files...")
    clean_build_files()
    
    print("\n🔍 Checking environment...")
    check_python_version()
    
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        sys.exit(1)
    
    print("\n🔍 Checking CUDA...")
    check_cuda()
    
    print("\n📦 Creating default directories...")
    create_default_directories()
    
    print("\n📦 Installing RGM package...")
    if install_package():
        print("\n✨ Setup completed successfully!")
        print("\nYou can now run RGM using:")
        print("python3 run_rgm.py")
    else:
        print("\n❌ Setup failed")
        sys.exit(1) 