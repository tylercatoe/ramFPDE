"""
rampde: Mixed precision Neural ODE solvers for PyTorch.

This package provides seamless drop-in replacements for torchdiffeq with automatic
mixed precision support via PyTorch's autocast. Key features:

- Drop-in compatibility with torchdiffeq API
- Automatic mixed precision handling (float16, bfloat16, float32)
- Dynamic loss scaling for stable gradient computation
- High-performance fixed grid solvers (Euler, RK4)
- Support for both tensor and tuple inputs following torchdiffeq conventions

Main API:
    odeint: Main integration function with automatic solver selection
    
Increment functions:
    Euler, RK4: Explicit integration schemes (extensible, so feel free to add more)
    
Solvers:
    FixedGridODESolverUnscaled: Optimal performance variant (default for float32, bfloat16)
    FixedGridODESolverDynamic: Dynamic scaling variant (default for float16)
    FixedGridODESolverUnscaledSafe: Exception handling variant (use this for float16 in combination with GradScaler)
    
Mixed precision:
    DynamicScaler: Dynamic loss scaling for mixed precision training
"""

from ._version import __version__
from .odeint import odeint
from .increment import Euler, RK4, INCREMENTS
from .fixed_grid_unscaled_uniform import FixedGridODESolverUnscaledUniform
from .fixed_grid_dynamic_uniform import FixedGridODESolverDynamicUniform
from .fixed_grid_unscaled_safe_uniform import FixedGridODESolverUnscaledSafeUniform
from .loss_scalers import DynamicScaler
from .utils import _is_any_infinite

__all__ = [
    "__version__",
    "odeint", 
    "Euler", "RK4", "INCREMENTS",
    "FixedGridODESolverUnscaledUniform", "FixedGridODESolverDynamicUniform", "FixedGridODESolverUnscaledSafeUniform",
    "DynamicScaler", "_is_any_infinite"
]
