import logging
from typing import Optional

class LoggerFormatter:
    """Utility class for consistent and aesthetic logging formats"""
    
    HEADER = "="*80
    SECTION = "-"*60
    SUBSECTION = "·"*40
    
    COLORS = {
        'HEADER': '\033[95m',
        'INFO': '\033[94m',
        'SUCCESS': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'ENDC': '\033[0m',
        'BOLD': '\033[1m'
    }
    
    @staticmethod
    def format_header(text: str) -> str:
        """Format section header"""
        return f"\n{LoggerFormatter.HEADER}\n{text}\n{LoggerFormatter.HEADER}\n"
    
    @staticmethod
    def format_section(text: str) -> str:
        """Format subsection header"""
        return f"\n{LoggerFormatter.SECTION}\n{text}\n{LoggerFormatter.SECTION}\n"
    
    @staticmethod
    def format_subsection(text: str) -> str:
        """Format minor section header"""
        return f"\n{LoggerFormatter.SUBSECTION}\n{text}\n{LoggerFormatter.SUBSECTION}\n"
    
    @staticmethod
    def setup_logger(name: str, 
                    log_file: Optional[str] = None, 
                    level: int = logging.INFO) -> logging.Logger:
        """Setup logger with consistent formatting"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Console handler with color
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            fmt='%(asctime)s | %(levelname)8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                fmt='%(asctime)s | %(name)s | %(levelname)8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        return logger 