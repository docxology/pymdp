"""
RGM Logging Configuration
=======================

Logging utilities for the Renormalization Generative Model (RGM).
Provides centralized logging configuration and management.
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

class RGMLogging:
    """Logging configuration for the Renormalization Generative Model."""
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger with the given name.
        
        Args:
            name: Name of the logger to retrieve
            
        Returns:
            Configured logger instance
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            
            if not cls._initialized:
                cls._setup_default_logging()
                
            cls._loggers[name] = logger
            
        return cls._loggers[name]
    
    @classmethod
    def _setup_default_logging(cls):
        """Set up default logging configuration."""
        if cls._initialized:
            return
            
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        
        cls._initialized = True
    
    @staticmethod
    def setup_logging(log_dir: Path, level: int = logging.INFO):
        """
        Set up logging with file output.
        
        Args:
            log_dir: Directory for log files
            level: Logging level
        """
        # Create log directory
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"rgm_{timestamp}.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(file_handler)
        
        # Log initial message
        root_logger.info(f"Logging initialized: {log_file}")
        
    @classmethod
    def set_level(cls, level: int):
        """
        Set logging level for all loggers.
        
        Args:
            level: New logging level
        """
        for logger in cls._loggers.values():
            logger.setLevel(level) 