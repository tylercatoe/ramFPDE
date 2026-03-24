"""
Dynamic scaling fixed grid ODE solver.

This variant includes dynamic scaling infrastructure to handle mixed precision
training with DynamicScaler. It includes scaling loops, parameter dtype conversion,
and overflow checking but no exception handling.

Performance: Moderate overhead compared to unscaled variant due to scaling loops
and overflow checking. Required when using DynamicScaler for mixed precision.
"""

from typing import Any, Optional, Tuple
import torch
from torch.amp import autocast
from .fixed_grid_base_uniform import FixedGridODESolverBase
from .utils import _is_any_infinite
from math import gamma

# Import custom_fwd and custom_bwd from torch.cuda.amp
try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    from torch.cuda.amp import custom_fwd, custom_bwd


class FixedGridODESolverDynamicUniform(FixedGridODESolverBase):
    """
    Dynamic scaling fixed grid ODE solver for optimal performance. Backward pass 
    
    This variant includes dynamic scaling infrastructure to handle mixed precision training with
    DynamicScaler. It includes:
    - Scaling loops for overflow protection
    - Parameter dtype conversion
    - Overflow checking and scalar updates
    - No exception handling (uses RuntimeError on failure)
    
    Use this variant when:
    - DynamicScaler is being used
    - Mixed precision with float16
    - Dynamic scaling is required
    """

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any, at: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Dynamic scaling backward pass with uniform time grid
        
        This implementation includes dynamic scaling infrastructure to handle
        mixed precision training with DynamicScaler. It performs gradient
        computation with scaling loops and overflow checking. 
        
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
        scaler = ctx.loss_scaler

        if beta == 1.0:
            beta = 1.0 - 0.0001  # Avoid edge case in Gamma function for beta=1
        
        # Determine precision
        dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else dtype_hi
        
        # Initialize gradients
        N = t.shape[0]
        params = tuple(params)

        # Initialize the dynamic scaler
        if scaler.S is None:
            scaler.init_scaling(at[-1])
        
        a = at[-1].to(dtype_hi)
        grad_theta = [torch.zeros_like(param, dtype=dtype_hi) for param in params]
        #grad_t = None if not t.requires_grad else torch.zeros_like(t)
        
        # Parameter dtype conversion for scaling
        old_params = {name: param.data.clone() for name, param in ode_func.named_parameters()}
        for name, param in ode_func.named_parameters():
            param.data = param.data.to(dtype_low)
        try:
            # Fast path check - skip parameter gradients if not needed
            any_param_requires_grad = any(p.requires_grad for p in params) if params else False

            # Compute Gamma(beta) once
            gamma_beta = torch.tensor(gamma(beta.item()), dtype=dtype_hi, device=zt.device)

            # Initialize adjoint storage
            at_history = torch.zeros_like(zt, dtype=dtype_hi)
            at_history[-1] = at[-1].to(dtype_hi)

            # Backward pass loop - no scaling, no exceptions
            h = t[1] - t[0]  # Assuming uniform grid for simplicity
            nu_factor = h ** beta / beta
            j_full = torch.arange(max(N - 1, 1), device=zt.device, dtype=dtype_low)

            for k in reversed(range(1, N)): # k = N-1, N-2, ..., 1, we calculate at_history[k-1] at each iteration
                with torch.no_grad():
                    da = torch.zeros_like(a, dtype=dtype_hi)

                    # Cache all j-dependent coefficients for this k.
                    j_idx = j_full[k - 1:N - 1]
                    nu_vec = nu_factor * ((j_idx + 2 - k) ** beta - (j_idx - k + 1) ** beta)

                    for offset, j in enumerate(range(k - 1, N - 1)):
                        # Prepare current state - directly from saved tensor
                        a_ind = at_history[j + 1].to(dtype_low)
                        z_ind = zt[j + 1].to(dtype_low).detach().requires_grad_(True)
                        tj = t[j]
                        nu_jk1 = nu_vec[offset]

                        attempts = 0
                        while attempts < scaler.max_attempts:
                            # Check for overflow in scaled gradients
                            if _is_any_infinite((scaler.S * a_ind,)):
                                scaler.update_on_overflow()
                                attempts += 1
                                continue

                            # Rebuild computational graph
                            with torch.enable_grad():
                                df = increment_func(ode_func, z_ind, tj, 0.0)

                            if any_param_requires_grad:
                                grads = torch.autograd.grad(
                                    df, (z_ind, *params), scaler.S * a_ind, create_graph=False, allow_unused=True
                                )
                                da_ind, *dparams = grads

                                if da_ind is None:
                                    da_ind = torch.zeros_like(z_ind)

                                # Handle None gradients for parameters
                                dparams = [d if d is not None else torch.zeros_like(p) for d, p in zip(dparams, params)]
                            else:
                                # Only adjoint gradient needed
                                da_ind = torch.autograd.grad(
                                    df, z_ind, scaler.S * a_ind, create_graph=False, allow_unused=True
                                )[0]
                                if da_ind is None:
                                    da_ind = torch.zeros_like(z_ind)
                                dparams = [torch.zeros_like(p) for p in params]

                            # Check for overflow in computed gradients
                            if _is_any_infinite((da_ind, dparams)):
                                scaler.update_on_overflow()
                                attempts += 1
                                continue
                            else:
                                break

                        # Check if we exceeded maximum attempts
                        if attempts >= scaler.max_attempts:
                            raise RuntimeError(
                                f"Reached maximum number of {scaler.max_attempts} attempts "
                                f"in backward pass at time step k={k}"
                            )

                        da += nu_jk1 * da_ind.to(dtype_hi)

                    da = a + 1 / (scaler.S * gamma_beta) * da
                    at_history[k-1] = da.to(dtype_hi)

                if any_param_requires_grad:
                    z_k = zt[k].to(dtype_low).detach().requires_grad_(True)
                    z_km1 = zt[k-1].to(dtype_low).detach().requires_grad_(True)

                    with torch.enable_grad():
                        dfk = increment_func(ode_func, z_k, t[k], 0.0)

                    grads = torch.autograd.grad(
                        dfk, params, scaler.S * at_history[k],
                        create_graph=False, allow_unused=True
                    )
                    dparams_k = [d if d is not None else torch.zeros_like(p)
                                    for d, p in zip(grads, params)]

                    with torch.enable_grad():
                        dfk1 = increment_func(ode_func, z_km1, t[k-1], 0.0)
                    grads = torch.autograd.grad(
                        dfk1, params, scaler.S * at_history[k-1],
                        create_graph=False, allow_unused=True
                    )
                    dparams_km1 = [d if d is not None else torch.zeros_like(p)
                                    for d, p in zip(grads, params)]

                    trap_updates = [0.5 * h / scaler.S * (d_k.to(dtype_hi) + d_km1.to(dtype_hi)) for d_k, d_km1 in zip(dparams_k, dparams_km1)]
                    torch._foreach_add_(grad_theta, trap_updates)

                # Check for overflow in accumulated gradients with enhanced error reporting
                if _is_any_infinite((da, grad_theta)):

                    # Collect diagnostic information
                    error_details = []
                    if not da.isfinite().all():
                        n_inf = torch.isinf(da).sum().item()
                        n_nan = torch.isnan(da).sum().item()
                        error_details.append(f"adjoint: {n_inf} inf, {n_nan} nan")

                    if any(not g.isfinite().all() for g in grad_theta):
                        bad_params = sum(1 for g in grad_theta if not g.isfinite().all())
                        error_details.append(f"param_grads: {bad_params}/{len(grad_theta)} tensors")

                    # Enhanced error message with actionable suggestions
                    error_msg = (
                        f"Gradients became non-finite at time step {k}/{len(t)-1}.\n"
                        f"Scale factor: {scaler.S:.2e}, attempt: {attempts}/{scaler.max_attempts}\n"
                        f"Non-finite: {', '.join(error_details)}\n"
                        f"Try: reduce learning rate, gradient clipping, check ODE stability, or use float32"
                    )
                    raise RuntimeError(error_msg)

                # Adjust upward scaling if the norm is too small
                if attempts == 0 and scaler.check_for_increase(da):
                    scaler.update_on_small_grad()

            # Return gradients for all inputs to forward pass
            # (increment_func, ode_func, z0, beta, t, loss_scaler, *params)
            return (None, None, at_history[0], None, None, None, *grad_theta)
        finally:
            # Restore original parameter dtypes even if backward exits via exception.
            for name, param in ode_func.named_parameters():
                param.data = old_params[name]