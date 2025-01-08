"""
Setup configuration for pymdp package
"""

from setuptools import setup, find_packages

setup(
    name="pymdp",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "psutil>=5.8.0",
        "pyyaml>=5.4.0"
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'mypy>=0.900'
        ]
    },
    author="VERSES Research Lab",
    author_email="research@verses.ai",
    description="Active Inference and RGM Implementation",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/versesresearch/pymdp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
) 