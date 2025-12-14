from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from core.errors import ConfigError


# ===============================================================
#  GLOBAL PLATFORM CONFIGURATION
# ===============================================================

class GlobalConfig(BaseModel):
    """
    Global, platform-wide configuration values.
    Available to all modules.
    """

    # numerical settings
    precision: Literal["float32", "float64"] = "float64"
    dtype: Literal["complex64", "complex128"] = "complex128"

    # execution settings
    num_threads: int = 1
    debug: bool = False

    # simulator behavior
    noise_model: Optional[str] = None

    # logger verbosity (DEBUG, INFO, WARNING, ERROR)
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @validator("num_threads")
    def validate_num_threads(cls, v):
        if v <= 0:
            raise ConfigError("num_threads must be >= 1")
        return v


# ===============================================================
#  PER-MODULE CONFIGURATION
# ===============================================================

class CompilerConfig(BaseModel):
    """
    Compiler-related settings.
    """

    optimization_level: int = Field(1, ge=0, le=3)
    enable_fusion: bool = True
    remove_identities: bool = True


class SimulatorConfig(BaseModel):
    """
    Simulator backend settings.
    """

    backend: Literal["statevector", "density", "stabilizer"] = "statevector"
    enforce_unitarity: bool = True
    max_qubits: int = 30


class LoggingConfig(BaseModel):
    """
    Settings specifically for the logging system.
    """

    log_to_console: bool = True
    log_to_file: bool = False
    log_file_path: str = "quantum_platform.log"


# ===============================================================
#  MASTER CONFIG OBJECT
# ===============================================================

class PlatformConfig(BaseModel):
    """
    Root configuration object.
    All modules will receive this or individual sub-configs.
    """

    global_config: GlobalConfig = GlobalConfig()
    compiler: CompilerConfig = CompilerConfig()
    simulator: SimulatorConfig = SimulatorConfig()
    logging: LoggingConfig = LoggingConfig()

    def print(self):
        """Pretty-print all configuration sections."""
        import json
        print(json.dumps(self.dict(), indent=4))


# ===============================================================
#  INSTANCE USED BY THE ENTIRE PLATFORM
# ===============================================================

# This is what your entire platform will import.
CONFIG = PlatformConfig()
