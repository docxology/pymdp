"""
RGM Logging Configuration
=======================

Configures logging for the RGM pipeline with:
- Console output with color formatting
- File output with detailed formatting
- Error logging with traceback
- Hierarchical logger management
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

class ColorFormatter(logging.Formatter):
    """Custom formatter adding colors to log levels."""
    
    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;32m',     # Green
        'WARNING': '\033[0;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'CRITICAL': '\033[0;35m', # Magenta
        'RESET': '\033[0m',       # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        if hasattr(record, 'color'):
            record.color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        else:
            record.color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.reset = self.COLORS['RESET']
        return super().format(record)

class RGMLogging:
    """
    Handles logging configuration for the RGM pipeline.
    
    Features:
    - Hierarchical logger management
    - Color-formatted console output
    - Detailed file logging
    - Error state tracking
    - Log rotation
    
    Example:
        >>> logger = RGMLogging.setup_logging(log_dir="logs")
        >>> logger.info("Starting pipeline...")
        >>> try:
        ...     # Your code here
        ...     pass
        ... except Exception as e:
        ...     RGMLogging.log_error_state(logger, e, "errors")
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(cls, name: str = "rgm") -> logging.Logger:
        """
        Get or create a logger with the given name.
        
        Args:
            name: Logger name (default: "rgm")
            
        Returns:
            Logger instance
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        return cls._loggers[name]
    
    @classmethod
    def setup_logging(cls,
                     log_dir: Optional[Path] = None,
                     level: str = "INFO",
                     log_to_file: bool = True,
                     name: str = "rgm") -> logging.Logger:
        """
        Set up logging configuration.
        
        Args:
            log_dir: Directory to store log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to save logs to file
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        # Create or get logger
        logger = cls.get_logger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler with color formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_format = '%(color)s%(levelname)-8s%(reset)s %(message)s'
        console_handler.setFormatter(ColorFormatter(console_format))
        logger.addHandler(console_handler)
        
        if log_to_file and log_dir:
            # Create log directory if it doesn't exist
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # File handler with detailed formatting
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f'rgm_pipeline_{timestamp}.log'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_format = ('%(asctime)s [%(levelname)-8s] %(name)s - %(message)s '
                         '(%(filename)s:%(lineno)d)')
            file_handler.setFormatter(logging.Formatter(file_format))
            logger.addHandler(file_handler)
            
            logger.info(f"Log file created at: {log_file}")
        
        return logger
    
    @staticmethod
    def log_error_state(logger: logging.Logger,
                       error: Exception,
                       error_dir: Path,
                       state: Optional[dict] = None):
        """
        Log error state and save relevant information.
        
        Args:
            logger: Logger instance
            error: Exception that occurred
            error_dir: Directory to save error state
            state: Additional state information to save
        """
        import traceback
        
        # Log error with traceback
        logger.error("\n" + "="*80)
        logger.error("❌ Error occurred in RGM pipeline")
        logger.error(f"Error type: {type(error).__name__}")
        logger.error(f"Error message: {str(error)}")
        logger.error("\nTraceback:")
        logger.error(traceback.format_exc())
        
        # Save error state
        if error_dir:
            error_dir = Path(error_dir)
            error_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            error_file = error_dir / f'error_state_{timestamp}.log'
            
            with open(error_file, 'w') as f:
                f.write(f"Error type: {type(error).__name__}\n")
                f.write(f"Error message: {str(error)}\n")
                f.write("\nTraceback:\n")
                f.write(traceback.format_exc())
                
                if state:
                    f.write("\nState:\n")
                    for key, value in state.items():
                        f.write(f"{key}: {value}\n")
            
            logger.info(f"💾 Error state saved to: {error_file}")
        
        logger.error("="*80 + "\n") 