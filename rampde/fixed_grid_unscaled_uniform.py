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
from .fixed_grid_base_uniform import FixedGridODESolverBase
from math import gamma

# Import custom_fwd and custom_bwd from torch.cuda.amp
try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    from torch.cuda.amp import custom_fwd, custom_bwd


class FixedGridODESolverUnscaledUniform(FixedGridODESolverBase):
    """
    Unscaled fixed grid ODE solver for optimal performance.

    Backward pass 
    
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

        if beta == 1.0:
            beta = 1.0 - 0.0001  # Avoid edge case in Gamma function for beta=1
        
        # Determine precision
        dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else dtype_hi
        
        # Initialize gradients
        N = t.shape[0]
        params = tuple(params)

        a = at[-1].to(dtype_hi)
        grad_theta = [torch.zeros_like(param, dtype=dtype_hi) for param in params]
        
        # Fast path check - skip parameter gradients if not needed
        any_param_requires_grad = any(p.requires_grad for p in params) if params else False
        
        # Calculate Gamma(beta) once
        gamma_beta = gamma(beta.item())

        # Initialize adjoint storage
        at_history = torch.zeros_like(zt, dtype = dtype_hi)
        at_history[-1] = at[-1].to(dtype_hi) 

        # Backward pass loop - no scaling, no exceptions
        
        h = t[1] - t[0]  # Assuming uniform grid for simplicity
        nu_factor = h ** beta / beta
        j_full = torch.arange(max(N - 1, 1), device=zt.device, dtype=dtype_low)

        for k in reversed(range(1, N)): # k = N-1, N-2, ..., 1, we calculate at_history[k-1] at each iteration
            with torch.no_grad():
                da = torch.zeros_like(a)

                # Cache all j-dependent coefficients for this k.
                j_idx = j_full[k - 1:N - 1]
                nu_vec = nu_factor * ((j_idx + 2 - k) ** beta - (j_idx - k + 1) ** beta)

                for offset, j in enumerate(range(k - 1, N - 1)):
                    # Prepare current state - directly from saved tensor
                    a_ind = at_history[j + 1].to(dtype_low)
                    z_ind = zt[j + 1].to(dtype_low).detach().requires_grad_(True)
                    tj = t[j]
                    nu_jk1 = nu_vec[offset]

                    # Rebuild computational graph
                    with torch.enable_grad():
                        df = increment_func(ode_func, z_ind, tj, 0.0)

                    # Inner j-loop only needs d(df)/dz; parameter grads are accumulated below.
                    da_ind = torch.autograd.grad(
                        df, z_ind, a_ind, create_graph=False, allow_unused=True
                    )[0]
                    if da_ind is None:
                        da_ind = torch.zeros_like(z_ind)
                    
                    da += nu_jk1 * da_ind.to(dtype_hi)

                da = a + 1 / gamma_beta * da
                at_history[k-1] = da.to(dtype_hi)

            if any_param_requires_grad:
                z_k = zt[k].to(dtype_low).detach().requires_grad_(True)
                z_km1 = zt[k-1].to(dtype_low).detach().requires_grad_(True)

                with torch.enable_grad():
                    dfk = increment_func(ode_func, z_k, t[k], 0.0)
                grads = torch.autograd.grad(
                    dfk, params, at_history[k],
                    create_graph=False, allow_unused=True
                )
                dparams_k = [d if d is not None else torch.zeros_like(p)
                                for d, p in zip(grads, params)]

                with torch.enable_grad():
                    dfk1 = increment_func(ode_func, z_km1, t[k-1], 0.0)
                grads = torch.autograd.grad(
                    dfk1, params, at_history[k-1],
                    create_graph=False, allow_unused=True
                )
                dparams_km1 = [d if d is not None else torch.zeros_like(p)
                                for d, p in zip(grads, params)]
                
                trap_updates = [0.5 * h * (d_k.to(dtype_hi) + d_km1.to(dtype_hi)) for d_k, d_km1 in zip(dparams_k, dparams_km1)]
                torch._foreach_add_(grad_theta, trap_updates) 
                
                # for i, (d_k, d_km1) in enumerate(zip(dparams_k, dparams_km1)):
                #     trap_update = 0.5 * h * (d_k.to(dtype_hi) + d_km1.to(dtype_hi))
                #     grad_theta[i].add_(trap_update.to(grad_theta[i].dtype))
        
        # Return gradients for all inputs to forward pass
        # (increment_func, ode_func, z0, beta, t, loss_scaler, *params)
        return (None, None, at_history[0], None, None, None, *grad_theta)