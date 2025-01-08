#!/bin/bash

# Install package in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev] 