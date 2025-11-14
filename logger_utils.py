"""
Enhanced logging utilities for reSSNT with structured output and performance monitoring
"""
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager
import psutil
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.text import Text
import threading


class PerformanceLogger:
    """Enhanced logger with performance monitoring and structured output"""
    
    def __init__(self, name: str, log_dir: str = "./logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.console = Console()
        self._setup_logging()
        self._metrics = {}
        self._start_times = {}
        
    def _setup_logging(self):
        """Setup structured logging with file and console handlers"""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler for structured logs
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
    def log_structured(self, level: str, event: str, **kwargs):
        """Log structured data in JSON format"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            "process_id": psutil.Process().pid,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            **kwargs
        }
        
        # Add GPU memory if available
        if torch.cuda.is_available():
            log_entry["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            log_entry["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            
        self.logger.info(json.dumps(log_entry))
        
    def info(self, message: str, **kwargs):
        """Log info message with console output"""
        self.console.print(f"[blue]INFO[/blue] {message}")
        self.log_structured("INFO", message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message with console output"""
        self.console.print(f"[yellow]WARNING[/yellow] {message}")
        self.log_structured("WARNING", message, **kwargs)
        
    def error(self, message: str, **kwargs):
        """Log error message with console output"""
        self.console.print(f"[red]ERROR[/red] {message}")
        self.log_structured("ERROR", message, **kwargs)
        
    def success(self, message: str, **kwargs):
        """Log success message with console output"""
        self.console.print(f"[green]SUCCESS[/green] {message}")
        self.log_structured("SUCCESS", message, **kwargs)
        
    @contextmanager
    def timer(self, operation: str, **kwargs):
        """Context manager for timing operations"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        self.info(f"Starting {operation}", operation=operation, **kwargs)
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            
            self.success(
                f"Completed {operation} in {duration:.2f}s",
                operation=operation,
                duration_seconds=duration,
                memory_delta_mb=memory_delta,
                **kwargs
            )
            
    def log_model_info(self, model, model_name: str):
        """Log detailed model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            "model_name": model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        }
        
        # Display model info table
        table = Table(title=f"Model Information: {model_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Parameters", f"{total_params:,}")
        table.add_row("Trainable Parameters", f"{trainable_params:,}")
        table.add_row("Model Size", f"{model_info['model_size_mb']:.2f} MB")
        
        self.console.print(table)
        self.log_structured("MODEL_INFO", "model_analysis", **model_info)
        
    def log_coverage_metrics(self, metrics: Dict[str, float], coverage_type: str):
        """Log coverage metrics with visual representation"""
        
        # Create visual table
        table = Table(title=f"Coverage Metrics: {coverage_type}")
        table.add_column("Coverage Type", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Progress", style="green")
        
        for name, value in metrics.items():
            if isinstance(value, float) and 0 <= value <= 1:
                progress_bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
                table.add_row(name, f"{value:.4f}", f"{progress_bar} {value:.1%}")
            else:
                table.add_row(name, str(value), "N/A")
                
        self.console.print(table)
        self.log_structured("COVERAGE_METRICS", coverage_type, metrics=metrics)
        
    def log_system_info(self):
        """Log system information"""
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
            "disk_usage_gb": psutil.disk_usage('/').free / 1024 / 1024 / 1024,
            "python_version": psutil.Process().exe(),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
            })
            
        self.log_structured("SYSTEM_INFO", "startup", **system_info)
        self.info("System information logged")


class ProgressReporter:
    """Enhanced progress reporting with Rich"""
    
    def __init__(self, logger: PerformanceLogger):
        self.logger = logger
        self.active_progress = None
        
    def create_progress(self, description: str, total: int):
        """Create a progress bar"""
        self.active_progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=self.logger.console
        )
        
        task_id = self.active_progress.add_task(description, total=total)
        self.active_progress.start()
        return task_id
        
    def update_progress(self, task_id: int, advance: int = 1):
        """Update progress"""
        if self.active_progress:
            self.active_progress.update(task_id, advance=advance)
            
    def finish_progress(self):
        """Finish progress reporting"""
        if self.active_progress:
            self.active_progress.stop()
            self.active_progress = None


# Global logger instance
_global_logger = None
_lock = threading.Lock()


def get_logger(name: str = "reSSNT", log_dir: str = "./logs") -> PerformanceLogger:
    """Get global logger instance (singleton)"""
    global _global_logger
    with _lock:
        if _global_logger is None:
            _global_logger = PerformanceLogger(name, log_dir)
        return _global_logger


def log_function_call(func):
    """Decorator to log function calls with performance metrics"""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = f"{func.__module__}.{func.__name__}"
        
        with logger.timer(f"function_call_{func_name}"):
            result = func(*args, **kwargs)
            
        return result
    return wrapper