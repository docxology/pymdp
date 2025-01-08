"""
Tests for RGM pipeline
"""

import pytest
from pathlib import Path
from pymdp.rgm import RGMPipelineRunner

def test_pipeline_initialization():
    """Test pipeline initialization"""
    runner = RGMPipelineRunner()
    assert runner is not None
    assert runner.config is not None

def test_pipeline_execution():
    """Test complete pipeline execution"""
    runner = RGMPipelineRunner()
    success = runner.run_pipeline()
    assert success is True

def test_error_handling():
    """Test error handling"""
    with pytest.raises(ValueError):
        runner = RGMPipelineRunner(config_path=Path("nonexistent.json")) 