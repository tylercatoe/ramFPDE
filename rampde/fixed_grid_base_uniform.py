"""
Base class for fixed grid ODE solvers. (Uniform grid, ABM Predictor-Corrector)

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
            device_type = z0.device.type
            try:
                autocast_enabled = torch.is_autocast_enabled(device_type)
            except TypeError:
                autocast_enabled = torch.is_autocast_enabled() if device_type == "cuda" else False

            if beta == 1.0:
                beta = 1.0 - 0.0001

            # Determine precision levels
            dtype_hi = z0.dtype
            dtype_low = torch.get_autocast_dtype(device_type) if autocast_enabled else dtype_hi
            
            # Initialize solution storage
            N = t.shape[0]
            zt = torch.zeros(N, *z0.shape, dtype=dtype_low, device=z0.device)
            f_func = torch.zeros(N, *z0.shape, dtype=dtype_low, device=z0.device)
            zt[0] = z0.to(dtype_low)

            # Calculate Gamma(beta) once
            beta_scalar = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
            gamma_beta = torch.tensor(gamma(beta_scalar), dtype=dtype_hi, device=zt.device)

            h = t[1] - t[0]  # Assuming uniform grid for simplicity
            predictor_factor = h ** beta / beta
            corrector_factor = h ** beta / (beta * (beta + 1))
            # Reuse index buffer to avoid per-step arange allocations.
            j_full = torch.arange(max(N - 1, 1), device=z0.device, dtype=dtype_low) 
            
            # Forward integration loop
            for k in range(0, N-1): 
                zk1P = torch.zeros_like(z0, dtype=dtype_hi, device=z0.device)
                zk1 = torch.zeros_like(z0, dtype=dtype_hi, device=z0.device)
                with autocast(device_type=device_type, dtype=dtype_low, enabled=autocast_enabled):
                    if k > 0:
                        # Vectorized history accumulation over j=0,...,k-1.
                        j_idx = j_full[:k]
                        hist = f_func[:k].to(dtype_hi)
                        view_shape = (k,) + (1,) * (hist.dim() - 1)

                        mu = predictor_factor * ((k + 1 - j_idx) ** beta - (k - j_idx) ** beta)
                        zk1P = torch.sum(mu.view(view_shape) * hist, dim=0)

                        eta = corrector_factor * ((k + 2 - j_idx) ** (beta + 1) + (k - j_idx) ** (beta + 1) - 2 * (k + 1 - j_idx) ** (beta + 1))
                        eta[0] = corrector_factor * (k ** (beta + 1) - (k - beta) * ((k + 1) ** beta))
                        zk1 = torch.sum(eta.view(view_shape) * hist, dim=0)
                    
                    # j = k term
                    f_func_k = increment_func(ode_func, (zt[k]).to(dtype_low), t[k], 0.0)
                    f_func[k] = f_func_k.to(dtype_low)
                    mu_k_k1 = predictor_factor
                    zk1P = z0 + 1/gamma_beta * (zk1P + (mu_k_k1 * f_func_k.to(dtype_hi)))

                    if k == 0:
                        eta_j_k1 =  corrector_factor * (beta)
                        zk1 = zk1 + eta_j_k1 * f_func_k.to(dtype_hi)
                    else:
                        eta_j_k1 = corrector_factor * ((2) ** (beta+1) - 2)
                        zk1 = zk1 + eta_j_k1 * f_func_k.to(dtype_hi)

                    # final corrector step
                    eta_k1_k1 = corrector_factor
                    f_func_pred = increment_func(ode_func, zk1P, t[k+1], 0.0)
                    zk1 = z0 + 1/gamma_beta * (zk1 + (eta_k1_k1 * f_func_pred.to(dtype_hi)))

                zt[k+1] = zk1.to(dtype_low)
                    
        
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
