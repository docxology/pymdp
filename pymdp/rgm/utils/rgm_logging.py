"""
RGM Logging Configuration
========================

Provides centralized logging configuration for the RGM pipeline.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

class RGMLogging:
    """Manages logging configuration for RGM pipeline"""
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def initialize(cls, log_dir: Optional[Path] = None):
        """Initialize logging configuration"""
        if cls._initialized:
            return
            
        # Set up basic logging format
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Create file handler if log directory provided
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"rgm_pipeline_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
        
        # Configure root logger
        root_logger = logging.getLogger("rgm")
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        
        if log_dir:
            root_logger.addHandler(file_handler)
            
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger with the given name"""
        if not cls._initialized:
            cls.initialize()
            
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
            
        return cls._loggers[name]
    
    @classmethod
    def set_level(cls, level: int):
        """Set logging level for all loggers"""
        for logger in cls._loggers.values():
            logger.setLevel(level)
            
    @classmethod
    def add_file_handler(cls, log_dir: Path):
        """Add file handler to all loggers"""
        if not cls._initialized:
            cls.initialize(log_dir)
            return
            
        # Create file handler
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"rgm_pipeline_{timestamp}.log"
        
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Add handler to all loggers
        for logger in cls._loggers.values():
            logger.addHandler(file_handler) 