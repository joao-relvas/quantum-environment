import json
import time
import threading
from datetime import datetime
from typing import Any, Dict, Optional


# ============================================================
#  CENTRAL LOGGING CONFIGURATION
# ============================================================

class LoggerConfig:
    """
    Centralized logger configuration.
    """
    log_to_console: bool = True
    log_to_file: bool = False
    log_file_path: str = "quantum_platform.log"
    min_severity: str = "DEBUG"  # DEBUG < INFO < WARNING < ERROR

    severity_order = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
    }


# Global lock for thread safety
_log_lock = threading.Lock()


# ============================================================
#  HELPER: FORMAT A STRUCTURED LOG ENTRY
# ============================================================

def _make_entry(severity: str, message: str, module: Optional[str], extra: Dict[str, Any]):
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "severity": severity,
        "module": module or "unknown",
        "message": message,
        "data": extra if extra else None,
    }


# ============================================================
#  HELPER: EMIT A LOG ENTRY
# ============================================================

def _emit(entry: Dict[str, Any]):
    """
    Emit the entry to console and/or file according to config.
    """

    if not _should_emit(entry["severity"]):
        return

    line = json.dumps(entry, ensure_ascii=False)

    with _log_lock:

        # Console output
        if LoggerConfig.log_to_console:
            print(line)

        # File output
        if LoggerConfig.log_to_file:
            with open(LoggerConfig.log_file_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


# ============================================================
#  HELPER: CHECK MINIMUM SEVERITY
# ============================================================

def _should_emit(severity: str) -> bool:
    return (
        LoggerConfig.severity_order[severity]
        >= LoggerConfig.severity_order[LoggerConfig.min_severity]
    )


# ============================================================
#  PUBLIC LOGGING FUNCTIONS
# ============================================================

def log_debug(message: str, module: Optional[str] = None, **extra):
    entry = _make_entry("DEBUG", message, module, extra)
    _emit(entry)


def log_info(message: str, module: Optional[str] = None, **extra):
    entry = _make_entry("INFO", message, module, extra)
    _emit(entry)


def log_warning(message: str, module: Optional[str] = None, **extra):
    entry = _make_entry("WARNING", message, module, extra)
    _emit(entry)


def log_error(message: str, module: Optional[str] = None, **extra):
    entry = _make_entry("ERROR", message, module, extra)
    _emit(entry)


# ============================================================
#  DECORATOR FOR TIMING ANY FUNCTION
# ============================================================

def log_timing(module: str = None):
    """
    Decorator that logs execution time for any function.
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                log_debug(
                    f"{func.__name__} executed",
                    module=module or func.__module__,
                    duration_ms=round(duration_ms, 3),
                )

        return wrapper

    return decorator
