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
        
        # Calculate Gamma(beta) once
        gamma_beta = gamma(beta)

        # Initialize adjoint storage
        at_history = torch.zeros_like(zt)
        at_history[-1] = at[-1].to(dtype_hi) 

        # Exception handling for overflow protection
        try:
            with torch.no_grad():
                for k in range(N-1):
                    da = torch.zeros_like(a)
                    dtk = t[N-1-k] - t[N-2-k]
                    t_Ik = t[N-2-k]                    
                    
                    # Prepare time variables - no unnecessary cloning
                    tk = t[k].detach()
                    dtk_local = dtk.detach()
                    if t.requires_grad:
                        tk.requires_grad_(True)
                        dtk_local.requires_grad_(True)

                    for j in range(k+1):
                        # Prepare current state - directly from saved tensor
                        a_ind = at_history[N-1-k+j]
                        z_ind = zt[N-1-k+j].detach().requires_grad_(True)
                        
                        t_ind = t[N-1-k+j].detach()
                        dt_ind = t[N-1-k+j] - t[N-1-k+j-1]
                        dt_ind_local = dt_ind.detach()
                        c_jk = ((t_Ik - t[N-2-k+j]) ** beta - (t_Ik - t_ind) ** beta) / beta

                        if t.requires_grad:
                            t_ind.requires_grad_(True)
                            dt_ind_local.requires_grad_(True)

                        # Rebuild computational graph
                        with torch.enable_grad():
                            dz = increment_func(ode_func, z_ind, t_ind, 0.0)
                    
                        # Simple overflow checking without scaling loop
                        if _is_any_infinite((a_ind, dz)):
                            raise OverflowError(f"Overflow detected in gradients at time step i={k}")
                    
                        # Compute gradients - optimized for different cases
                        if t.requires_grad and any_param_requires_grad:
                            # Full gradient computation
                            grads = torch.autograd.grad(
                                dz, (z_ind, t_ind, dt_ind_local, *params), a_ind,
                                create_graph=False, allow_unused=True
                            )
                            da_ind, gtj, gdtj, *dparams = grads
                            
                            # Handle None gradients
                            gtj = gtj.to(dtype_hi) if gtj is not None else torch.zeros_like(t_ind)
                            gdtj = gdtj.to(dtype_hi) if gdtj is not None else torch.zeros_like(dt_ind_local)
                            gdtj2 = torch.sum(a_ind * dz, dim=-1)
                        elif t.requires_grad:
                            # Only time gradients needed
                            grads = torch.autograd.grad(
                                dz, (z_ind, t_ind, dt_ind_local), a_ind,
                                create_graph=False, allow_unused=True
                            )
                            da_ind, gtj, gdtj = grads
                            dparams = [torch.zeros_like(p) for p in params]
                            
                            # Handle None gradients
                            gtj = gtj.to(dtype_hi) if gtj is not None else torch.zeros_like(t_ind)
                            gdtj = gdtj.to(dtype_hi) if gdtj is not None else torch.zeros_like(dt_ind_local)
                            gdtj2 = torch.sum(a_ind * dz, dim=-1)
                        elif any_param_requires_grad:
                            # Only parameter gradients needed
                            grads = torch.autograd.grad(
                                dz, (z_ind, *params), a_ind,
                                create_graph=False, allow_unused=True
                            )
                            da_ind, *dparams = grads
                            gtj = gdtj = gdtj2 = None
                            
                            # Handle None gradients for parameters
                            dparams = [d if d is not None else torch.zeros_like(p) 
                                    for d, p in zip(dparams, params)]
                        else:
                            # Only adjoint gradient needed
                            da_ind = torch.autograd.grad(dz, z_ind, a_ind, create_graph=False)[0]
                            dparams = [torch.zeros_like(p) for p in params]
                            gtj = gdtj = gdtj2 = None

                        
                        da += c_jk * da_ind.to(dtype_hi)
                    
                    # Compute and store new adjoint, Update gradients
                    da_hi = da.to(dtype_hi)
                    at_history[N-2-k] = at_history[N-1] + (1 / gamma_beta) * da_hi

                    if any_param_requires_grad:
                        # Use in-place operations for parameter gradient accumulation
                        for g, d in zip(grad_theta, dparams):
                            if d is not None:
                                #vjp = torch.sum(at_history[N-2-k] * d, dim=-1)
                                g.add_(dtk * d.to(g.dtype))
                    
                    # Compute time gradients for the current backward step (index N-2-k)
                    if t.requires_grad:
                        # Use backward indexing consistent with the rest of the algorithm
                        idx = N - 2 - k
                        z_idx = zt[idx].detach().requires_grad_(True)
                        t_idx = t[idx].detach().requires_grad_(True)
                        dt_local = dtk.detach().requires_grad_(True)
                        
                        # Rebuild computational graph for dz at backward step idx
                        with torch.enable_grad():
                            dz_idx = increment_func(ode_func, z_idx, t_idx, 0.0)
                        
                        # Compute gradients using the adjoint we just computed
                        grads = torch.autograd.grad(
                            dz_idx, (z_idx, t_idx, dt_local), dtk * at_history[idx],
                            create_graph=False, allow_unused=True
                        )
                        da_ind, gtj, gdtj = grads
                        
                        # Handle None gradients
                        gtj = gtj.to(dtype_hi) if gtj is not None else torch.zeros_like(t_idx)
                        gdtj = gdtj.to(dtype_hi) if gdtj is not None else torch.zeros_like(dt_local)
                        gdtj2 = torch.sum(dtk * at_history[idx] * dz_idx, dim=-1)
                    else:
                        gtj = gdtj = gdtj2 = None

                    # Update time gradients at the backward position idx = N-2-k
                    if grad_t is not None and gdtj2 is not None:
                        idx = N - 2 - k
                        gdtj2_hi = gdtj2.to(dtype_hi)
                        grad_t[idx].add_(dtk * (gtj - gdtj)).sub_(gdtj2_hi)
                        if idx + 1 < len(grad_t):
                            grad_t[idx + 1].add_(dtk * gdtj).add_(gdtj2_hi)

                    # Check for overflow in computed gradients (after j loop)
                    if _is_any_infinite((da, gtj, gdtj, dparams)):
                        raise OverflowError(f"Overflow detected in computed gradients at time step k={k}")

        except OverflowError:
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
        return (None, None, at_history[0], None, grad_t, None, *grad_theta)