"""
Base class for fixed grid ODE solvers.

This module provides the shared forward pass implementation that is identical
across all fixed grid solver variants. Only the backward pass differs between
variants to handle different scaling and exception handling strategies.
"""

from typing import Any, Optional, Tuple, Union
import torch
from torch.amp import autocast
from math import gamma

# Import custom_fwd and custom_bwd from torch.cuda.amp
try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    from torch.cuda.amp import custom_fwd, custom_bwd



class FixedGridODESolverBase(torch.autograd.Function):
    """
    Base class for fixed grid ODE solvers with shared forward pass.
    
    This class implements the forward pass that is identical across all variants:
    - Unscaled (optimal performance)
    - Dynamic (with scaling loop)
    - Unscaled Safe (with exception handling)
    
    Subclasses only need to implement the backward pass according to their
    specific scaling and exception handling strategy.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any, 
        increment_func: torch.nn.Module, 
        ode_func: torch.nn.Module, 
        z0: torch.Tensor, 
        beta: torch.Tensor, 
        t: torch.Tensor, 
        loss_scaler: Any, 
        *params: torch.Tensor
    ) -> torch.Tensor:
        """
        Shared forward pass implementation.
        
        This method is identical across all solver variants and implements
        the fixed grid forward integration using the specified increment function.
        
        Args:
            ctx: PyTorch autograd context for saving information for backward pass
            increment_func: Increment function (Euler, RK4, etc.)
            ode_func: ODE function f(t, y)
            z0: Initial condition tensor
            beta: ODE order tensor in (0,1]
            t: Time points tensor
            loss_scaler: Loss scaler for mixed precision (DynamicScaler or NoScaler)
            *params: Parameters of the ODE function
            
        Returns:
            zt: Solution tensor at all time points
        """
        with torch.no_grad():
            # Determine precision levels
            dtype_hi = z0.dtype
            dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else dtype_hi
            
            # Initialize solution storage
            N = t.shape[0]
            z = z0
            zt = torch.zeros(N, *z.shape, dtype=dtype_low, device=z.device)
            df = torch.zeros(N, *z.shape, dtype=dtype_low, device=z.device)  # Store increments for reuse
            zt[0] = z0.to(dtype_low)

            # Caluclate Gamma(beta) once
            gamma_beta = gamma(beta.item())
            
            # Forward integration loop
            for k in range(1, N):
                dt = t[k] - t[k - 1]
                zp = z0
                
                # Compute Predictor/Corrector in low precision
                with autocast(device_type='cuda', dtype=dtype_low):

                    # Compute Predictor
                    for j in range(k-1):
                        df_j = df[j] # Reuse previously computed increment
                        mu_jk = dt ** beta / beta * ((k-j) ** beta - (k-j-1) ** beta)
                        zp = zp + (1/gamma_beta * mu_jk * df_j).to(dtype_hi) # Accumulate in high precision
                        
                    j = k - 1
                    df_j = increment_func(ode_func, z, t[j], dt) # Compute new increment
                    df[j] = df_j  # Store for reuse
                    mu_jk = dt ** beta / beta * ((k-j) ** beta - (k-j-1) ** beta) / gamma_beta
                    zp = zp + (1/gamma_beta * mu_jk * df_j).to(dtype_hi) # Accumulate in high precision

                    # Compute Corrector
                    z = z0 
                    dfP = increment_func(ode_func, zp, t[k], dt) # Predictor increment
                    nu_00 = dt ** beta / (beta * (beta + 1)) * ((k-1) ** (beta + 1) - (k-1-beta) * k ** beta ) 
                    z = z + (1/gamma_beta * nu_00 * df_j[0]).to(dtype_hi)
                    for j in range(1, k):
                        nu_jk = dt ** beta / (beta * (beta + 1)) * ((k-j+1)**(beta+1) + (k-j-1)**(beta+1) - 2*(k-j)**(beta+1))
                        z = z + (1/gamma_beta * nu_jk * df[j]).to(dtype_hi)
                    nu_kk = dt ** beta / (beta * (beta + 1))
                    z = z + (1/gamma_beta * nu_kk * dfP).to(dtype_hi)

                zt[k] = z.to(dtype_low)
        
        # Save information for backward pass
        ctx.save_for_backward(zt, beta, *params)
        ctx.increment_func = increment_func
        ctx.ode_func = ode_func
        ctx.t = t
        ctx.dtype_hi = dtype_hi
        ctx.loss_scaler = loss_scaler
        
        return zt
    
    @staticmethod
    def backward(ctx: Any, at: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Abstract backward method - must be implemented by subclasses.
        
        Each subclass implements this method according to its specific
        scaling and exception handling strategy:
        - Unscaled: Simple, fast backward pass
        - Dynamic: Backward pass with scaling loop
        - Unscaled Safe: Backward pass with exception handling
        
        Args:
            ctx: PyTorch autograd context with saved tensors and attributes
            at: Gradient tensor from subsequent operations
            
        Returns:
            Tuple of gradients for all inputs to forward pass
        """
        raise NotImplementedError(
            "Subclasses must implement the backward method according to their "
            "specific scaling and exception handling strategy."
        )