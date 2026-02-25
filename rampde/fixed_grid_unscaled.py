"""
Unscaled fixed grid ODE solver - optimal performance variant.

This variant provides the fastest performance by eliminating all scaling
infrastructure. It should be used as the default for float32 and bfloat16
precision where overflow is not a concern.

Performance: Optimal performance baseline - significantly faster than variants
with scaling or exception handling overhead.
"""

from typing import Any, Optional, Tuple
import torch
from torch.amp import autocast
from .fixed_grid_base import FixedGridODESolverBase
from math import gamma

# Import custom_fwd and custom_bwd from torch.cuda.amp
try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    from torch.cuda.amp import custom_fwd, custom_bwd


class FixedGridODESolverUnscaled(FixedGridODESolverBase):
    """
    Unscaled fixed grid ODE solver for optimal performance.
    
    This variant eliminates all scaling infrastructure to provide the fastest
    possible performance. It performs simple gradient computation without:
    - Scaling loops
    - Parameter dtype conversion
    - Overflow checking
    - Exception handling
    
    Use this variant when:
    - Precision is float32 or bfloat16
    - No overflow concerns
    - Maximum performance is needed
    """

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any, at: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Unscaled backward pass - optimal performance.
        
        This implementation provides the fastest backward pass by eliminating
        all scaling infrastructure. It performs direct gradient computation
        without any overflow protection or scaling loops.
        
        Args:
            ctx: PyTorch autograd context with saved tensors and attributes
            at: Gradient tensor from subsequent operations
            
        Returns:
            Tuple of gradients: (None, None, grad_y0, grad_t, None, *grad_params)
        """
        # Retrieve saved tensors and context
        zt, beta, *params = ctx.saved_tensors
        increment_func = ctx.increment_func
        ode_func = ctx.ode_func
        t = ctx.t
        dtype_hi = ctx.dtype_hi
        
        # Determine precision
        dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else dtype_hi
        
        # Initialize gradients
        N = t.shape[0]
        params = tuple(params)
        
        a = at[-1].to(dtype_hi)
        grad_theta = [torch.zeros_like(param) for param in params]
        grad_t = None if not t.requires_grad else torch.zeros_like(t)
        
        # Fast path check - skip parameter gradients if not needed
        any_param_requires_grad = any(p.requires_grad for p in params) if params else False
        
        # Calculate Gamma(beta) once
        gamma_beta = gamma(beta)

        # Initialize adjoint storage
        at_history = torch.zeros_like(zt)
        at_history[-1] = at[-1].to(dtype_hi) 

        # Backward pass loop - no scaling, no exceptions
        with torch.no_grad():
            for k in range(N-1):
                da = torch.zeros_like(a)
                dtk = t[N-1-k] - t[N-2-k]
                t_Ik = t[N-2-k]
                
                for j in range(k+1):
                    # Prepare current state - directly from saved tensor
                    a_ind = at_history[N-1-k+j]
                    z_ind = zt[N-1-k+j].detach().requires_grad_(True) #####

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
                        gdtj2 = torch.sum(a * dz, dim=-1)
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
                
                # Compute and store new adjoint
                da_hi = da.to(dtype_hi)
                at_history[N-2-k] = at_history[N-1] + (1 / gamma_beta) * da_hi
                    
                # Update gradients - optimized with in-place operations
                # Convert da once and reuse
                #da_hi = da.to(dtype_hi)
                #a = a - dtk / gamma_beta * da_hi 
                

                ######## What's happening here? ########
                if any_param_requires_grad:
                    # Use in-place operations for parameter gradient accumulation
                    for g, d in zip(grad_theta, dparams):
                        if d is not None:
                            vjp = torch.sum(at_history[N-2-k] * d, dim=-1)
                            g.add_(dtk * vjp.to(g.dtype))
                
                # I dont understand what's happening here
                if grad_t is not None and gdtj2 is not None:
                    gdtj2_hi = gdtj2.to(dtype_hi)
                    grad_t[k].add_(dtk * (gtj - gdtj)).sub_(gdtj2_hi)
                    if k + 1 < len(grad_t):
                        grad_t[k + 1].add_(dtk * gdtj).add_(gdtj2_hi)
        
        # Return gradients for all inputs to forward pass
        # (increment_func, ode_func, z0, beta, t, loss_scaler, *params)
        return (None, None, at_history[0], None, grad_t, None, *grad_theta)