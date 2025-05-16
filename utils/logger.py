"""
utils/logger.py

Centralized logging setup for the Trading Performance Dashboard.
Configures a logger that can be used across all modules of the application.
"""
import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# --- Configuration Import Handling ---
# Attempt to import from the root-level config.py.
# This relies on the Python import path being set up correctly,
# which is usually the case when running `streamlit run app.py` from the project root.
try:
    # If utils/ is directly under the project root alongside config.py:
    # from ..config import LOG_FILE, LOG_LEVEL, LOG_FORMAT, APP_TITLE # This would be for utils as a sub-package of a larger app package
    # For a flat structure where app.py and config.py are at root, and utils is a subdir,
    # direct import from config might not work from within utils/logger.py without sys.path modification.
    # A common pattern is to pass config values or have a singleton config object.
    # However, for Streamlit, often the execution context allows this.
    # Let's assume app.py (which imports this) handles the root path correctly.
    # If this fails, the fallback below will be used.
    # The most straightforward way if app.py is the entry point:
    # config values are passed to setup_logger from app.py.
    # For now, we'll keep the try-except for default values if config isn't found
    # during direct execution of this file or if the import path isn't as expected.
    # This will be primarily driven by how app.py calls setup_logger.

    # The logger will be configured by app.py passing parameters from config.py.
    # So, we don't strictly need to import config here if setup_logger always receives them.
    # For fallback/default values if not passed:
    _DEFAULT_APP_TITLE = "TradingDashboard_Default"
    _DEFAULT_LOG_FILE = "logs/trading_dashboard_app_default.log"
    _DEFAULT_LOG_LEVEL = "INFO"
    _DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"

except ImportError:
    # This block is primarily for standalone testing of the logger module.
    # In the Streamlit app, config should be importable by app.py and values passed.
    print("Warning (logger.py): Could not import from project's config. Using default logger settings for standalone execution.", file=sys.stderr)
    _DEFAULT_APP_TITLE = "TradingDashboard_Fallback"
    _DEFAULT_LOG_FILE = "logs/trading_dashboard_app_fallback.log"
    _DEFAULT_LOG_LEVEL = "INFO"
    _DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"


_loggers = {} # To store configured loggers and prevent re-configuration

def setup_logger(
    logger_name: str = _DEFAULT_APP_TITLE,
    log_file: str = _DEFAULT_LOG_FILE,
    level: str = _DEFAULT_LOG_LEVEL,
    log_format: str = _DEFAULT_LOG_FORMAT,
    max_bytes: int = 10*1024*1024, # 10 MB
    backup_count: int = 5,
    console_output: bool = True # Control console output
) -> logging.Logger:
    """
    Configures and returns a logger instance. Ensures a logger is configured only once.

    Args:
        logger_name (str): The name for the logger.
        log_file (str): The path to the log file. If None, file logging is disabled.
        level (str): The logging level (e.g., "INFO", "DEBUG").
        log_format (str): The format string for log messages.
        max_bytes (int): Maximum size of the log file before rotation.
        backup_count (int): Number of backup log files to keep.
        console_output (bool): Whether to output logs to the console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    global _loggers

    if logger_name in _loggers:
        return _loggers[logger_name] # Return existing logger

    # Get the logger
    logger = logging.getLogger(logger_name)

    # Set the logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Clear existing handlers to prevent duplication if any were attached externally
    if logger.hasHandlers():
        logger.handlers.clear()

    if console_output:
        # Console Handler (for Streamlit's console output and local debugging)
        console_handler = logging.StreamHandler(sys.stdout) # Use stdout for Streamlit compatibility
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level) # Ensure handler respects the level
        logger.addHandler(console_handler)

    # File Handler (for persistent logs)
    if log_file:
        # Ensure the logs directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError as e:
                # Fallback to logging to console if directory creation fails
                logger.error(f"Error creating log directory {log_dir}: {e}. File logging will be disabled.", exc_info=True)
                log_file = None # Disable file logging for this call

        if log_file: # Check again in case it was disabled
            try:
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(numeric_level) # Ensure handler respects the level
                logger.addHandler(file_handler)
            except Exception as e:
                # Use a basic print here if logger itself is failing for file handler
                print(f"Critical Error: Failed to set up file handler for {log_file}: {e}", file=sys.stderr)
                # The console handler (if enabled) should still work.

    # Set propagation to False if this is the main app logger and you don't want
    # logs to go to the root logger if it has handlers.
    logger.propagate = False

    _loggers[logger_name] = logger # Store configured logger
    return logger


if __name__ == "__main__":
    # Test the logger setup
    # This will use the default config values defined in this file.
    main_app_logger = setup_logger(
        logger_name="MainAppTest",
        log_file="logs/main_app_test.log",
        level="DEBUG"
    )
    main_app_logger.debug("This is a debug message from MainAppTest logger.")
    main_app_logger.info("This is an info message from MainAppTest logger.")

    module_logger = setup_logger(
        logger_name="ModuleSpecificTest",
        log_file="logs/module_test.log",
        level="INFO",
        console_output=False # Test disabling console for this one
    )
    module_logger.info("This info message from ModuleSpecificTest should only go to its file.")
    module_logger.warning("This warning also only to file for ModuleSpecificTest.")

    # Attempting to get the same logger should return the configured one without re-adding handlers
    retrieved_main_logger = setup_logger(logger_name="MainAppTest")
    retrieved_main_logger.info("Info message from retrieved MainAppTest logger (should not have duplicate handlers).")

    print(f"Log files for testing should be at: logs/main_app_test.log and logs/module_test.log")

