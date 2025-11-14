import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import time


# Global console instance
console = Console()


class PerformanceLogger:
    """
    Context manager for logging performance metrics.

    Usage:
        with PerformanceLogger("Operation name"):
            # Your code here
            pass
    """

    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize performance logger.

        Args:
            operation_name: Name of the operation being timed
            logger: Optional logger instance. If None, uses root logger
        """
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if exc_type is None:
            self.logger.info(
                f"Completed: {self.operation_name} " f"(Duration: {duration:.2f}s)"
            )
        else:
            self.logger.error(
                f"Failed: {self.operation_name} "
                f"(Duration: {duration:.2f}s) - {exc_val}"
            )

        return False  # Don't suppress exceptions


class FormattedLogger:
    """
    Enhanced logger with Rich formatting capabilities.

    Provides methods for structured logging with colors, tables, and panels.
    """

    def __init__(self, name: str = __name__, level: int = logging.INFO):
        """
        Initialize formatted logger.

        Args:
            name: Logger name
            level: Logging level (default: INFO)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.console = console

    def header(self, text: str, style: str = "bold cyan") -> None:
        """
        Print a header with decorative borders.

        Args:
            text: Header text
            style: Rich style string
        """
        self.console.print()
        self.console.print(f"{'=' * 80}", style=style)
        self.console.print(Text(text.center(80), style=style))
        self.console.print(f"{'=' * 80}", style=style)
        self.console.print()

    def section(self, text: str, style: str = "bold yellow") -> None:
        """
        Print a section header.

        Args:
            text: Section text
            style: Rich style string
        """
        self.console.print()
        self.console.print(Text(f"▶ {text}", style=style))
        self.console.print(f"{'-' * 60}", style="dim")

    def success(self, message: str) -> None:
        """
        Log success message.

        Args:
            message: Success message
        """
        self.console.print(f"✓ {message}", style="bold green")
        self.logger.info(f"SUCCESS: {message}")

    def error(self, message: str) -> None:
        """
        Log error message.

        Args:
            message: Error message
        """
        self.console.print(f"✗ {message}", style="bold red")
        self.logger.error(message)

    def warning(self, message: str) -> None:
        """
        Log warning message.

        Args:
            message: Warning message
        """
        self.console.print(f"⚠ {message}", style="bold yellow")
        self.logger.warning(message)

    def info(self, message: str) -> None:
        """
        Log info message.

        Args:
            message: Info message
        """
        self.console.print(f"ℹ {message}", style="cyan")
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """
        Log debug message.

        Args:
            message: Debug message
        """
        self.console.print(f"🔍 {message}", style="dim")
        self.logger.debug(message)

    def panel(self, content: str, title: str = "", style: str = "blue") -> None:
        """
        Display content in a formatted panel.

        Args:
            content: Panel content
            title: Panel title
            style: Border style
        """
        self.console.print(Panel(content, title=title, border_style=style))

    def table(self, data: Dict[str, Any], title: str = "") -> None:
        """
        Display data in a formatted table.

        Args:
            data: Dictionary of key-value pairs
            title: Table title
        """
        table = Table(title=title, show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold cyan", width=30)
        table.add_column("Value", style="white")

        for key, value in data.items():
            table.add_row(str(key), str(value))

        self.console.print(table)

    def progress_bar(self, total: int, description: str = "Processing"):
        """
        Create a progress bar context.

        Args:
            total: Total number of items
            description: Description text

        Returns:
            Progress context manager
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        )


def setup_logging(
    name: str = "reSSNT",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up unified logging system.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional log file path
        console_output: Enable console output (default: True)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging("myapp", level=logging.DEBUG, log_file="app.log")
        >>> logger.info("Application started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with Rich formatting
    if console_output:
        console_handler = RichHandler(
            rich_tracebacks=True, console=console, show_time=True, show_path=False
        )
        console_handler.setLevel(level)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def disable_external_logging():
    """
    Disable logging from external libraries (MMEngine, etc).

    Call this at the start of your application to reduce noise.
    """
    logging.getLogger("mmengine").setLevel(logging.CRITICAL)
    logging.getLogger("mmseg").setLevel(logging.CRITICAL)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# Global logger instance
_default_logger = None


def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> FormattedLogger:
    """
    Get or create a formatted logger instance.

    Args:
        name: Logger name (default: uses calling module name)
        level: Logging level
        log_file: Optional log file path

    Returns:
        FormattedLogger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.success("Operation completed")
    """
    global _default_logger

    if _default_logger is None or name is not None:
        logger_name = name or "reSSNT"
        setup_logging(logger_name, level=level, log_file=log_file)
        _default_logger = FormattedLogger(logger_name, level=level)

    return _default_logger


# Convenience functions
def log_experiment_start(experiment_name: str, config: Dict[str, Any]) -> None:
    """
    Log experiment start with configuration.

    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration dictionary
    """
    logger = get_logger()
    logger.header(f"Starting Experiment: {experiment_name}")
    logger.table(config, title="Configuration")


def log_experiment_end(experiment_name: str, results: Dict[str, Any]) -> None:
    """
    Log experiment end with results.

    Args:
        experiment_name: Name of the experiment
        results: Experiment results dictionary
    """
    logger = get_logger()
    logger.section("Experiment Results")
    logger.table(results)
    logger.success(f"Experiment '{experiment_name}' completed successfully")


def log_model_info(model_name: str, dataset: str, model_type: str) -> None:
    """
    Log model information in a formatted way.

    Args:
        model_name: Model name
        dataset: Dataset name
        model_type: Model type (CNN/Transformer/Other)
    """
    logger = get_logger()
    logger.panel(
        f"[bold]Model:[/bold] {model_name}\n"
        f"[bold]Dataset:[/bold] {dataset}\n"
        f"[bold]Type:[/bold] {model_type}",
        title="Model Information",
        style="green",
    )


# Export public API
__all__ = [
    "console",
    "PerformanceLogger",
    "FormattedLogger",
    "setup_logging",
    "get_logger",
    "disable_external_logging",
    "log_experiment_start",
    "log_experiment_end",
    "log_model_info",
]