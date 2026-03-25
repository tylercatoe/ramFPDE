"""
Unscaled safe fixed uniform grid ODE solver with exception handling.

This variant mirrors the uniform unscaled backward pass but adds overflow
checks and safe fallback behavior for compatibility with external GradScaler
flows when DynamicScaler is not used.
"""

from typing import Any, Optional, Tuple
import torch
from .fixed_grid_base_uniform import FixedGridODESolverBase
from .utils import _is_any_infinite
from math import gamma

# Import custom_fwd and custom_bwd from torch.cuda.amp
try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    from torch.cuda.amp import custom_fwd, custom_bwd


class FixedGridODESolverUnscaledSafeUniform(FixedGridODESolverBase):
    """
    Unscaled safe fixed uniform-grid ODE solver.

    This variant keeps the unscaled algorithm (no dynamic scaling loop) while
    adding finite-value checks and overflow fallback behavior.
    """

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any, at: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Unscaled safe backward pass with overflow handling.

        Returns:
            Tuple of gradients: (None, None, grad_z0, None, None, None, *grad_params)
            If overflow occurs, returns inf gradients to signal failure safely.
        """
        # Retrieve saved tensors and context
        zt, beta, *params = ctx.saved_tensors
        increment_func = ctx.increment_func
        ode_func = ctx.ode_func
        t = ctx.t
        dtype_hi = ctx.dtype_hi

        # Determine precision
        device_type = zt.device.type
        try:
            autocast_enabled = torch.is_autocast_enabled(device_type)
        except TypeError:
            autocast_enabled = torch.is_autocast_enabled() if device_type == "cuda" else False
        dtype_low = torch.get_autocast_dtype(device_type) if autocast_enabled else dtype_hi

        # Initialize gradients
        N = t.shape[0]
        params = tuple(params)
        a = at[-1].to(dtype_hi)
        grad_theta = [torch.zeros_like(param, dtype=dtype_hi) for param in params]
        any_param_requires_grad = any(p.requires_grad for p in params) if params else False

        # Compute Gamma(beta) once
        beta_scalar = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
        gamma_beta = torch.tensor(gamma(beta_scalar), dtype=dtype_hi, device=zt.device)

        # Initialize adjoint storage
        at_history = torch.zeros_like(zt, dtype=dtype_hi)
        at_history[-1] = at[-1].to(dtype_hi)

        h = t[1] - t[0]  # Assuming uniform grid for simplicity
        nu_factor = h ** beta / beta
        j_full = torch.arange(max(N - 1, 1), device=zt.device, dtype=dtype_low)

        try:
            for k in reversed(range(1, N)):
                with torch.no_grad():
                    da = torch.zeros_like(a, dtype=dtype_hi)

                    # Cache all j-dependent coefficients for this k.
                    j_idx = j_full[k - 1:N - 1]
                    nu_vec = nu_factor * ((j_idx + 2 - k) ** beta - (j_idx - k + 1) ** beta)

                    for offset, j in enumerate(range(k - 1, N - 1)):
                        a_ind = at_history[j + 1].to(dtype_low)
                        z_ind = zt[j + 1].to(dtype_low).detach().requires_grad_(True)
                        tj = t[j]
                        nu_jk1 = nu_vec[offset]

                        with torch.enable_grad():
                            df = increment_func(ode_func, z_ind, tj, 0.0)

                        if _is_any_infinite((a_ind, df)):
                            raise OverflowError(
                                f"Overflow detected while building adjoint at k={k}, j={j}"
                            )

                        if any_param_requires_grad:
                            grads = torch.autograd.grad(
                                df, (z_ind, *params), a_ind, create_graph=False, allow_unused=True
                            )
                            da_ind, *dparams = grads
                            if da_ind is None:
                                da_ind = torch.zeros_like(z_ind)
                            dparams = [d if d is not None else torch.zeros_like(p) for d, p in zip(dparams, params)]
                        else:
                            da_ind = torch.autograd.grad(
                                df, z_ind, a_ind, create_graph=False, allow_unused=True
                            )[0]
                            if da_ind is None:
                                da_ind = torch.zeros_like(z_ind)
                            dparams = [torch.zeros_like(p) for p in params]

                        if _is_any_infinite((da_ind, dparams)):
                            raise OverflowError(
                                f"Overflow detected in local gradients at k={k}, j={j}"
                            )

                        da += nu_jk1 * da_ind.to(dtype_hi)

                    da = a + 1 / gamma_beta * da
                    at_history[k - 1] = da.to(dtype_hi)

                if any_param_requires_grad:
                    z_k = zt[k].to(dtype_low).detach().requires_grad_(True)
                    z_km1 = zt[k - 1].to(dtype_low).detach().requires_grad_(True)

                    with torch.enable_grad():
                        dfk = increment_func(ode_func, z_k, t[k], 0.0)
                    grads = torch.autograd.grad(
                        dfk, params, at_history[k], create_graph=False, allow_unused=True
                    )
                    dparams_k = [d if d is not None else torch.zeros_like(p) for d, p in zip(grads, params)]

                    with torch.enable_grad():
                        dfk1 = increment_func(ode_func, z_km1, t[k - 1], 0.0)
                    grads = torch.autograd.grad(
                        dfk1, params, at_history[k - 1], create_graph=False, allow_unused=True
                    )
                    dparams_km1 = [d if d is not None else torch.zeros_like(p) for d, p in zip(grads, params)]

                    trap_updates = [
                        0.5 * h * (d_k.to(dtype_hi) + d_km1.to(dtype_hi))
                        for d_k, d_km1 in zip(dparams_k, dparams_km1)
                    ]

                    if _is_any_infinite((dparams_k, dparams_km1, trap_updates)):
                        raise OverflowError(
                            f"Overflow detected in parameter gradients at k={k}"
                        )

                    torch._foreach_add_(grad_theta, trap_updates)

                    if _is_any_infinite((da, grad_theta)):
                        raise OverflowError(
                            f"Overflow detected in accumulated gradients at k={k}"
                        )

        except OverflowError:
            grad_z0_inf = torch.full_like(at_history[0], float("inf"))
            grad_theta_inf = [torch.full_like(g, float("inf")) for g in grad_theta]
            return (None, None, grad_z0_inf, None, None, None, *grad_theta_inf)

        # Return gradients for all inputs to forward pass
        # (increment_func, ode_func, z0, beta, t, loss_scaler, *params)
        return (None, None, at_history[0], None, None, None, *grad_theta)
