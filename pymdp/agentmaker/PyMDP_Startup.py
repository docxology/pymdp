"""
Startup script for PyMDP - clones repo and sets up virtual environment
"""
import subprocess
import sys
import venv
import shutil
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Core dependencies
PACKAGES = [
    'numpy==1.23.5',
    'pandas',
    'matplotlib',
]

def check_environment():
    """Check if we're in a virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logger.error("Please deactivate any virtual environment before running this script")
        logger.error("Run: 'deactivate' and try again")
        return False
    return True

def clone_pymdp():
    """Clone the PyMDP repository"""
    pymdp_path = Path('pymdp')
    if pymdp_path.exists():
        logger.info("Removing existing PyMDP directory...")
        shutil.rmtree(pymdp_path)
    
    logger.info("Cloning PyMDP repository...")
    try:
        subprocess.run([
            'git', 'clone',
            'https://github.com/infer-actively/pymdp.git'
        ], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone PyMDP: {e.stderr}")
        return False

def get_python_executable():
    """Get the system Python executable path"""
    return sys.executable

def setup_environment():
    """Setup virtual environment and install dependencies"""
    venv_path = Path('venv').absolute()
    
    # Clean existing venv
    if venv_path.exists():
        logger.info("Removing existing virtual environment...")
        shutil.rmtree(venv_path)
    
    # Create new venv using system Python
    logger.info("Creating virtual environment...")
    python_exe = get_python_executable()
    try:
        subprocess.run([
            python_exe, '-m', 'venv', str(venv_path)
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e.stderr}")
        return False
    
    # Get pip path
    pip_path = venv_path / ('Scripts' if sys.platform == 'win32' else 'bin') / 'pip'
    
    # Upgrade pip
    logger.info("Upgrading pip...")
    subprocess.run([str(pip_path), 'install', '--upgrade', 'pip'], check=True)
    
    # Install packages
    logger.info("Installing dependencies...")
    for package in PACKAGES:
        try:
            subprocess.run([str(pip_path), 'install', package], check=True)
            logger.info(f"Installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}")
            return False
    
    # Install pymdp in editable mode
    logger.info("Installing PyMDP...")
    try:
        subprocess.run([
            str(pip_path), 'install', '-e', './pymdp'
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install PyMDP: {e.stderr}")
        return False
    
    # Print activation instructions
    activate_cmd = (
        f'source {venv_path}/bin/activate' if sys.platform != 'win32' 
        else f'.\\venv\\Scripts\\activate'
    )
    logger.info("\n" + "="*50)
    logger.info("Setup complete! To activate the environment, run:")
    logger.info(f"\n{activate_cmd}\n")
    logger.info("="*50)
    return True

def main():
    if not check_environment():
        return False
        
    try:
        if not clone_pymdp():
            return False
        if not setup_environment():
            return False
        return True
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False

if __name__ == "__main__":
    if not main():
        sys.exit(1)