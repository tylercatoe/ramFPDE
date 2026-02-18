"""
Unscaled safe fixed grid ODE solver with exception handling.

This variant provides exception handling for overflow protection without 
dynamic scaling infrastructure. It's designed to work with PyTorch's 
GradScaler when overflow protection is needed but dynamic scaling is not.

Performance: Minor overhead compared to unscaled variant due to exception
handling infrastructure. Compatible with PyTorch's GradScaler.
"""

from typing import Any, Optional, Tuple
import torch
from torch.amp import autocast
from .fixed_grid_base import FixedGridODESolverBase
from .utils import _is_any_infinite
from math import gamma

# Import custom_fwd and custom_bwd from torch.cuda.amp
try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    from torch.cuda.amp import custom_fwd, custom_bwd


class FixedGridODESolverUnscaledSafe(FixedGridODESolverBase):
    """
    Unscaled safe fixed grid ODE solver with exception handling.
    
    This variant provides exception handling for overflow protection without
    the full dynamic scaling infrastructure. It includes:
    - Exception handling for overflow scenarios
    - Compatible with PyTorch's GradScaler
    - No dynamic scaling loops
    - Parameter restoration via finally block
    
    Use this variant when:
    - Overflow protection is needed
    - Dynamic scaling is not required
    - Working with PyTorch's GradScaler
    - Precision is float16 but no DynamicScaler
    """

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any, at: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Unscaled safe backward pass with exception handling.
        
        This implementation provides exception handling for overflow scenarios
        while avoiding the full dynamic scaling infrastructure. It catches
        overflow exceptions and returns inf gradients for compatibility with
        PyTorch's GradScaler.
        
        Args:
            ctx: PyTorch autograd context with saved tensors and attributes
            at: Gradient tensor from subsequent operations
            
        Returns:
            Tuple of gradients: (None, None, grad_y0, grad_t, None, *grad_params)
            Returns inf gradients if overflow occurs
        """
        # Retrieve saved tensors and context
        zt, beta, *params = ctx.saved_tensors
        increment_func = ctx.increment_func
        ode_func = ctx.ode_func
        t = ctx.t
        dtype_hi = ctx.dtype_hi
        scaler = ctx.loss_scaler
        
        # Determine precision
        dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else dtype_hi
        
        # Initialize gradients
        N = t.shape[0]
        params = tuple(params)
        
        a = at[-1].to(dtype_hi)
        grad_theta = [torch.zeros_like(param) for param in params]
        grad_t = None if not t.requires_grad else torch.zeros_like(t)
        
        # Store original parameters for restoration
        old_params = {name: param.data.clone() for name, param in ode_func.named_parameters()}
        
        # Fast path check - skip parameter gradients if not needed
        any_param_requires_grad = any(p.requires_grad for p in params) if params else False
        
        # Exception handling for overflow protection
        try:
            with torch.no_grad():
                for k in reversed(range(1, N)):
                    dtk = t[k] - t[k - 1]
                    
                    da = 0.0 

                    # Prepare current state - directly from saved tensor
                    z = zt[k].detach().requires_grad_(True)
                    
                    # Prepare time variables - no unnecessary cloning
                    tk = t[k].detach()
                    dtk_local = dtk.detach()
                    if t.requires_grad:
                        tk.requires_grad_(True)
                        dtk_local.requires_grad_(True)

                    for j in range((N-k), N):
                        b_jk1 = dtk_local ** beta / beta * ((j - (N - k - 1)) ** beta - (j - (N - k))** beta) 

                        # Rebuild computational graph
                        with torch.enable_grad():
                            dz = increment_func(ode_func, z, tk, dtk_local)
                    
                        # Simple overflow checking without scaling loop
                        if _is_any_infinite((a, dz)):
                            raise OverflowError(f"Overflow detected in gradients at time step i={k}")
                    
                        # Compute gradients - optimized for different cases
                        if t.requires_grad and any_param_requires_grad:
                            # Full gradient computation
                            grads = torch.autograd.grad(
                                dz, (z, tk, dtk_local, *params), a,
                                create_graph=False, allow_unused=True
                            )
                            da, gtk, gdtk, *dparams = grads
                            
                            # Handle None gradients
                            gtk = gtk.to(dtype_hi) if gtk is not None else torch.zeros_like(tk)
                            gdtk = gdtk.to(dtype_hi) if gdtk is not None else torch.zeros_like(dtk)
                            gdtk2 = torch.sum(a * dz, dim=-1)
                        elif t.requires_grad:
                            # Only time gradients needed
                            grads = torch.autograd.grad(
                                dz, (z, tk, dtk_local), a,
                                create_graph=False, allow_unused=True
                            )
                            da, gtk, gdtk = grads
                            dparams = [torch.zeros_like(p) for p in params]
                            
                            # Handle None gradients
                            gtk = gtk.to(dtype_hi) if gtk is not None else torch.zeros_like(tk)
                            gdtk = gdtk.to(dtype_hi) if gdtk is not None else torch.zeros_like(dtk)
                            gdtk2 = torch.sum(a * dz, dim=-1)
                        elif any_param_requires_grad:
                            # Only parameter gradients needed
                            grads = torch.autograd.grad(
                                dz, (z, *params), a,
                                create_graph=False, allow_unused=True
                            )
                            da, *dparams = grads
                            gtk = gdtk = gdtk2 = None
                            
                            # Handle None gradients for parameters
                            dparams = [d if d is not None else torch.zeros_like(p) 
                                    for d, p in zip(dparams, params)]
                        else:
                            # Only adjoint gradient needed
                            da = torch.autograd.grad(dz, z, a, create_graph=False)[0]
                            dparams = [torch.zeros_like(p) for p in params]
                            gtk = gdtk = gdtk2 = None
                    
                            # Accumulate da with gdtk2 term only if not None
                        if gdtk2 is not None:
                            da = da.to(dtype_hi) + b_jk1 * gdtk2.to(dtype_hi)

                    # Check for overflow in computed gradients (after j loop)
                    if _is_any_infinite((da, gtk, gdtk, dparams)):
                        raise OverflowError(f"Overflow detected in computed gradients at time step k={k}")
                    
                    # Update gradients - optimized with in-place operations
                    # Convert da once and reuse
                    da_hi = da.to(dtype_hi)
                    a.add_(dtk * da_hi).add_(at[k].to(dtype_hi))
                    
                    if any_param_requires_grad:
                        # Use in-place operations for parameter gradient accumulation
                        for g, d in zip(grad_theta, dparams):
                            if d is not None:
                                g.add_(dtk * d.to(g.dtype))
                    
                    if grad_t is not None and gdtk2 is not None:
                        gdtk2_hi = gdtk2.to(dtype_hi)
                        grad_t[k].add_(dtk * (gtk - gdtk)).sub_(gdtk2_hi)
                        if k + 1 < len(grad_t):
                            grad_t[k + 1].add_(dtk * gdtk).add_(gdtk2_hi)
                        
                    
                    # Check for overflow in accumulated gradients
                    if _is_any_infinite((a, grad_t, grad_theta)):
                        raise OverflowError(f"Overflow detected in accumulated gradients at time step k={k}")
            a_inf = torch.full_like(a, float('inf'))
            grad_theta_inf = [torch.full_like(grad, float('inf')) for grad in grad_theta]
            grad_t_inf = torch.full_like(t, float('inf')) if t.requires_grad else None
            
            return (None, None, a_inf, grad_t_inf, None, *grad_theta_inf)
        
        finally:
            # Always restore original parameters
            for name, param in ode_func.named_parameters():
                param.data = old_params[name].data
        
        # Return gradients for all inputs to forward pass
        # (increment_func, ode_func, z0, beta, t, loss_scaler, *params)
        return (None, None, a, None, grad_t, None, *grad_theta)