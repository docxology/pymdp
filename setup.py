"""
PyMDP Setup
==========

Setup configuration for PyMDP package.
"""

from setuptools import setup, find_packages

setup(
    name="pymdp",
    version="0.1.0",
    description="Python implementation of Active Inference and Message Passing",
    author="PyMDP Contributors",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.7.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pandas>=1.1.0",
        "tqdm>=4.50.0",
        "jsonschema>=3.2.0",
        "pyyaml>=5.3.0",
        "psutil>=5.7.0"
    ],
    python_requires=">=3.7",
    include_package_data=True,
    package_data={
        'pymdp.rgm': [
            'configs/*.json',
            'models/*.gnn'
        ]
    }
)

