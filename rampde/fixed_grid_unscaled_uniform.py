"""
Unscaled fixed uniform grid ODE solver - optimal performance variant.

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


class FixedGridODESolverUnscaledUniform(FixedGridODESolverBase):
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
        
        # Fast path check - skip parameter gradients if not needed
        any_param_requires_grad = any(p.requires_grad for p in params) if params else False
        
        # Calculate Gamma(beta) once
        gamma_beta = gamma(beta.item())

        # Initialize adjoint storage
        at_history = torch.zeros_like(zt)
        at_history[-1] = at[-1].to(dtype_hi) 

        # Backward pass loop - no scaling, no exceptions
        with torch.no_grad():
            h = t[1] - t[0]  # Assuming uniform grid for simplicity

            for k in reversed(range(1, N)): # k = N-1, N-2, ..., 1, we calculate at_history[k-1] at each iteration
                da = torch.zeros_like(a)
                
                for j in range(k-1, N-1):
                    # Prepare current state - directly from saved tensor
                    a_ind = at_history[j+1]
                    z_ind = zt[j+1].detach().requires_grad_(True) #####
                    tj = j*h
                    
                    nu_jk1 = h ** beta / beta * ((j + 2 - k) ** beta - (j - k + 1) ** beta)

                    # Rebuild computational graph
                    with torch.enable_grad():
                        df = increment_func(ode_func, z_ind, tj, 0.0)
                    
                    # Compute gradients using the adjoint
                    if any_param_requires_grad:
                        grads = torch.autograd.grad(
                            df, (z_ind, *params), a_ind,
                            create_graph=False, allow_unused=True
                        )
                        da_ind, *dparams = grads

                        # Handle None gradients (unused inputs)
                        if da_ind is None:
                            da_ind = torch.zeros_like(z_ind)
                        dparams = [d if d is not None else torch.zeros_like(p) 
                                for d, p in zip(dparams, params)]
                        
                    else:
                        # only adjoint gradient needed
                        da_ind = torch.autograd.grad(df, z_ind, a_ind, create_graph=False, allow_unused=True)[0]
                        if da_ind is None:
                            da_ind = torch.zeros_like(z_ind)
                        dparams = [torch.zeros_like(p) for p in params]
                    
                    da += nu_jk1 * da_ind.to(dtype_hi)

                    if any_param_requires_grad:
                        for i, d in enumerate(dparams):
                            if d is not None:
                                grad_theta[i].add_((-1) * h * d.to(grad_theta[i].dtype))

                da = a + 1 / gamma_beta * da
                at_history[k-1] = da.to(dtype_hi)


                # if any_param_requires_grad:
                #     for g, d in zip(grad_theta, dparams):
                #         da_hi = da.to(dtype_hi)
                #         if d is not None:
                #             vjp = torch.sum(da_hi * d, dim=-1)
                #             g.add_((-1) * h * vjp.to(g.dtype))
        
        # Return gradients for all inputs to forward pass
        # (increment_func, ode_func, z0, beta, t, loss_scaler, *params)
        return (None, None, at_history[0], None, None, None, *grad_theta)