import logging
from pathlib import Path
from typing import Optional

def setup_experiment_logging(
    output_dir: Path,
    log_file: str = "experiment.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """Setup experiment-wide logging"""
    logger = logging.getLogger('biofirm')
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(levelname)s:%(name)s: %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(console_formatter)
    logger.addHandler(console)
    
    # File handler
    log_path = output_dir / log_file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger 