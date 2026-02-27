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
from .fixed_grid_base import FixedGridODESolverBase
from .utils import _is_any_infinite
from math import gamma

# Import custom_fwd and custom_bwd from torch.cuda.amp
try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    from torch.cuda.amp import custom_fwd, custom_bwd


class FixedGridODESolverDynamic(FixedGridODESolverBase):
    """
    Dynamic scaling fixed grid ODE solver.
    
    This variant includes dynamic scaling infrastructure to handle mixed precision
    training with DynamicScaler. It includes:
    - Scaling loops for overflow handling
    - Parameter dtype conversion
    - Overflow checking and scaler updates
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
        Dynamic scaling backward pass.
        
        This implementation includes dynamic scaling infrastructure to handle
        mixed precision training with DynamicScaler. It performs gradient
        computation with scaling loops and overflow checking.
        
        Args:
            ctx: PyTorch autograd context with saved tensors and attributes
            at: Gradient tensor from subsequent operations
            
        Returns:
            Tuple of gradients: (None, None, grad_z0, grad_t, None, *grad_params)
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
        
        # Initialize the dynamic scaler
        if scaler.S is None:
            scaler.init_scaling(at[-1])
        
        a = at[-1].to(dtype_hi)
        grad_theta = [torch.zeros_like(param) for param in params]
        grad_t = None if not t.requires_grad else torch.zeros_like(t)
        
        # Parameter dtype conversion for scaling
        old_params = {name: param.data.clone() for name, param in ode_func.named_parameters()}
        for name, param in ode_func.named_parameters():
            param.data = param.data.to(dtype_low)
        
        # Fast path check - skip parameter gradients if not needed
        any_param_requires_grad = any(p.requires_grad for p in params) if params else False
        
        # Compute Gamma(beta) once
        gamma_beta = gamma(beta)
        
        # Initialize adjoint storage
        at_history = torch.zeros_like(zt)
        at_history[-1] = at[-1].to(dtype_hi)

        # Backward pass loop with dynamic scaling
        with torch.no_grad():
            for k in range(N-1):
                da = torch.zeros_like(a)
                dtk = t[N-1-k] - t[N-2-k]
                t_Ik = t[N-2-k]

                # Prepare time variables
                tk = t[k].detach()
                dtk_local = dtk.detach()
                if t.requires_grad:
                    tk.requires_grad_(True)
                    dtk_local.requires_grad_(True)

                for j in range(k+1):
                    a_ind = at_history[N-1-k+j]
                    z_ind = zt[N-1-k+j].detach().requires_grad_(True)

                    t_ind = t[N-1-k+j].detach()
                    dt_ind = t[N-1-k+j] - t[N-1-k+j-1]
                    dt_ind_local = dt_ind.detach()
                    c_jk = ((t_Ik - t[N-2-k+j]) ** beta - (t_Ik - t_ind) ** beta) / beta

                    if t.requires_grad:
                        t_ind.requires_grad_(True)
                        dt_ind_local.requires_grad_(True)

                    # Dynamic scaling loop
                    attempts = 0
                    while attempts < scaler.max_attempts:
                        # Check for overflow in scaled gradients
                        if _is_any_infinite((scaler.S * a_ind,)):
                            scaler.update_on_overflow()
                            attempts += 1
                            continue
                        
                        # Rebuild computational graph (moved inside loop for recomputation on scale change)
                        with torch.enable_grad():
                            dz = increment_func(ode_func, z_ind, t_ind, 0.0)
                        
                        # Compute gradients with scaling - optimized for different cases
                        if t.requires_grad and any_param_requires_grad:
                            # Full gradient computation
                            grads = torch.autograd.grad(
                                dz, (z_ind, t_ind, dt_ind_local, *params), scaler.S * a_ind,
                                create_graph=False, allow_unused=True
                            )
                            da_ind, gtj, gdtj, *dparams = grads
                            
                            # Handle None gradients
                            gtj = gtj.to(dtype_hi) if gtj is not None else torch.zeros_like(t_ind)
                            gdtj = gdtj.to(dtype_hi) if gdtj is not None else torch.zeros_like(dt_ind_local)
                            gdtj2 = torch.sum(scaler.S * a_ind * dz, dim=-1)
                        elif t.requires_grad:
                            # Only time gradients needed
                            grads = torch.autograd.grad(
                                dz, (z_ind, t_ind, dt_ind_local), scaler.S * a_ind,
                                create_graph=False, allow_unused=True
                            )
                            da_ind, gtj, gdtj = grads
                            dparams = [torch.zeros_like(p) for p in params]
                            
                            # Handle None gradients
                            gtj = gtj.to(dtype_hi) if gtj is not None else torch.zeros_like(t_ind)
                            gdtj = gdtj.to(dtype_hi) if gdtj is not None else torch.zeros_like(dt_ind_local)
                            gdtj2 = torch.sum(scaler.S * a_ind * dz, dim=-1)
                        elif any_param_requires_grad:
                            # Only parameter gradients needed
                            grads = torch.autograd.grad(
                                dz, (z_ind, *params), scaler.S * a_ind,
                                create_graph=False, allow_unused=True
                            )
                            da_ind, *dparams = grads
                            gtj = gdtj = gdtj2 = None
                            
                            # Handle None gradients for parameters
                            dparams = [d if d is not None else torch.zeros_like(p) 
                                    for d, p in zip(dparams, params)]
                        else:
                            # Only adjoint gradient needed
                            da_ind = torch.autograd.grad(dz, z_ind, scaler.S * a_ind, create_graph=False)[0]
                            dparams = [torch.zeros_like(p) for p in params]
                            gtj = gdtj = gdtj2 = None
                        
                        # Check for overflow in computed gradients
                        if _is_any_infinite((da_ind, gtj, gdtj, dparams)):
                            scaler.update_on_overflow()
                            attempts += 1
                            continue
                        else:
                            break

                    
                    da += c_jk * da_ind.to(dtype_hi)
                    
                
                # Update gradients with descaling - optimized with in-place operations
                # Convert da once and reuse, compute scale factor once
                da_hi = da.to(dtype_hi)
                scale_factor = dtk / scaler.S # should this be 1/S or dt_k/S? 
                at_history[N-2-k] = at_history[N-1] + (scale_factor / gamma_beta) * da_hi

                if any_param_requires_grad:
                    # Use in-place operations for parameter gradient accumulation
                    for g, d in zip(grad_theta, dparams):
                        if d is not None:                           
                            vjp = torch.sum(at_history[N-2-k] * d, dim=-1)
                            g.add_(scale_factor * vjp.to(g.dtype)) 
                            #g.add_(scale_factor * d.to(g.dtype))
                
                # Compute time gradients for the current k step
                if t.requires_grad:
                    # Rebuild dz with correct inputs for gradient computation
                    j = k
                    z_j = zt[j].detach().requires_grad_(True)
                    t_j = t[j].detach().requires_grad_(True)
                    dt = t[j+1] - t[j]
                    dt_local = dt.detach().requires_grad_(True)
                    
                    # Rebuild computational graph for dz at step j
                    with torch.enable_grad():
                        dz_j = increment_func(ode_func, z_j, t_j, 0.0)
                    
                    # Compute gradients
                    grads = torch.autograd.grad(
                        dz_j, (z_j, t_j, dt_local), scaler.S * at_history[j],
                        create_graph=False, allow_unused=True
                    )
                    da_ind, gtj, gdtj = grads
                    
                    # Handle None gradients
                    gtj = gtj.to(dtype_hi) if gtj is not None else torch.zeros_like(t[j])
                    gdtj = gdtj.to(dtype_hi) if gdtj is not None else torch.zeros_like(dt_local)
                    gdtj2 = torch.sum(scaler.S * at_history[j] * dz_j, dim=-1)
                else:
                    gtj = gdtj = gdtj2 = None
                        
                # Update time gradients
                if grad_t is not None and gdtj2 is not None:
                    gdtj2_hi = gdtj2.to(dtype_hi)
                    grad_t[k].add_(scale_factor * (gtj - gdtj)).sub_(gdtj2_hi)
                    if k + 1 < len(grad_t):
                        grad_t[k + 1].add_(scale_factor  * gdtj).add_(gdtj2_hi)
                
                # Check for overflow in accumulated gradients with enhanced error reporting
                if _is_any_infinite((a, grad_t, grad_theta)):
                    # Collect diagnostic information
                    error_details = []
                    if not a.isfinite().all():
                        n_inf = torch.isinf(a).sum().item()
                        n_nan = torch.isnan(a).sum().item()
                        error_details.append(f"adjoint: {n_inf} inf, {n_nan} nan")

                    if grad_t is not None:
                        if not grad_t[k].isfinite().all():
                            n_inf = torch.isinf(grad_t[k]).sum().item()
                            n_nan = torch.isnan(grad_t[k]).sum().item()
                            error_details.append(f"time_grad[{k}]: {n_inf} inf, {n_nan} nan")
                        if k + 1 < grad_t.shape[0] and not grad_t[k + 1].isfinite().all():
                            n_inf = torch.isinf(grad_t[k + 1]).sum().item()
                            n_nan = torch.isnan(grad_t[k + 1]).sum().item()
                            error_details.append(f"time_grad[{k + 1}]: {n_inf} inf, {n_nan} nan")

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
                if attempts == 0 and scaler.check_for_increase(a):
                    scaler.update_on_small_grad()
        
        # Restore original parameter dtypes
        for name, param in ode_func.named_parameters():
            param.data = old_params[name].data
        
        # Return gradients for all inputs to forward pass
        # (increment_func, ode_func, z0, beta, t, loss_scaler, *params)
        return (None, None, at_history[0], None, grad_t, None, *grad_theta)