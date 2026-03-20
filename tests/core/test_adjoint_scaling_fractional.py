import os
import sys
import unittest
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.amp import autocast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rampde import DynamicScaler, odeint


torch.set_default_dtype(torch.float32)


class FractionalLinearODE(nn.Module):
    """Simple fractional test ODE: D^beta y = -lambda * y."""

    def __init__(self, lam: float, lam_dtype: torch.dtype = torch.float32):
        super().__init__()
        # Keep lambda as a buffer so this test isolates dy0 adjoint behavior only.
        self.register_buffer('lam', torch.tensor(lam, dtype=lam_dtype))

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if torch.is_autocast_enabled():
            work_dtype = torch.get_autocast_dtype('cuda')
            y = y.to(work_dtype)
            lam = self.lam.to(work_dtype)
        else:
            lam = self.lam
        out = -lam * y
        if not torch.isfinite(out).all():
            raise ValueError(f"Non-finite RHS detected: {out}")
        return out


def _solve_fractional_l1(
    model: nn.Module,
    y0: torch.Tensor,
    t: torch.Tensor,
    beta: float,
    working_dtype: torch.dtype,
    scaler: Any,
) -> torch.Tensor:
    if scaler is False:
        loss_scaler = False
    elif scaler is None:
        loss_scaler = None
    else:
        loss_scaler = scaler(working_dtype)

    # Autocast only for low precision dtypes.
    use_autocast = working_dtype in (torch.float16, torch.bfloat16)
    with autocast(device_type='cuda', dtype=working_dtype, enabled=use_autocast):
        return odeint(model, y0, t, method='l1', beta=beta, loss_scaler=loss_scaler)


def _run_case(
    device: torch.device,
    lam: float,
    beta: float,
    t: torch.Tensor,
    y0_value: float,
    work_dtype: torch.dtype,
    scaler: Any,
) -> Dict[str, Any]:
    model = FractionalLinearODE(lam=lam, lam_dtype=torch.float32).to(device)
    y0 = torch.tensor([y0_value], device=device, dtype=work_dtype).requires_grad_(True)

    try:
        sol = _solve_fractional_l1(
            model=model,
            y0=y0,
            t=t,
            beta=beta,
            working_dtype=work_dtype,
            scaler=scaler,
        )
        loss = 0.5 * sol[-1].pow(2).sum()
        loss.backward()

        yT = sol[-1].detach().to(torch.float64)
        if y0.grad is None:
            return {
                'ok': False,
                'yT': None,
                'grad_y0': None,
                'error': 'missing y0 gradient',
            }

        grad_y0 = y0.grad.detach().to(torch.float64)

        is_finite = bool(torch.isfinite(yT).all() and torch.isfinite(grad_y0).all())
        return {
            'ok': is_finite,
            'yT': yT,
            'grad_y0': grad_y0,
            'error': None if is_finite else 'non-finite outputs',
        }
    except (RuntimeError, ValueError) as exc:
        return {
            'ok': False,
            'yT': None,
            'grad_y0': None,
            'error': str(exc),
        }


def _reference_fd(
    device: torch.device,
    lam: float,
    beta: float,
    t: torch.Tensor,
    y0_value: float,
    eps_rel: float = 1e-6,
) -> Dict[str, Any]:
    """Compute a robust float64 reference using forward solves and finite difference."""
    model = FractionalLinearODE(lam=lam, lam_dtype=torch.float64).to(device)
    y0 = torch.tensor([y0_value], device=device, dtype=torch.float64)
    dy0 = max(abs(y0_value) * eps_rel, eps_rel)

    try:
        y_plus = torch.tensor([y0_value + dy0], device=device, dtype=torch.float64)
        y_minus = torch.tensor([y0_value - dy0], device=device, dtype=torch.float64)

        sol0 = _solve_fractional_l1(model, y0, t, beta, torch.float64, False)
        solp = _solve_fractional_l1(model, y_plus, t, beta, torch.float64, False)
        solm = _solve_fractional_l1(model, y_minus, t, beta, torch.float64, False)

        yT0 = sol0[-1].detach().to(torch.float64)
        d_yT_dy0 = (solp[-1].detach().to(torch.float64) - solm[-1].detach().to(torch.float64)) / (2.0 * dy0)
        grad_y0 = yT0 * d_yT_dy0  # d/dy0 [0.5 * y(T)^2]

        is_finite = bool(torch.isfinite(yT0).all() and torch.isfinite(grad_y0).all())
        return {
            'ok': is_finite,
            'yT': yT0,
            'grad_y0': grad_y0,
            'error': None if is_finite else 'non-finite reference outputs',
        }
    except (RuntimeError, ValueError) as exc:
        return {
            'ok': False,
            'yT': None,
            'grad_y0': None,
            'error': str(exc),
        }


class TestFractionalAdjointScalingCrossDtype(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest('CUDA required for fractional mixed-precision tests.')

        self.device = torch.device('cuda:0')
        self.beta = 0.5
        # Mild regime to keep forward/backward finite across dtypes.
        self.lam = 0.25
        self.T = 1.0
        self.n_time = 64
        self.y0_value = 0.7
        self.t64 = torch.linspace(0.0, self.T, self.n_time, device=self.device, dtype=torch.float64)

    def test_fractional_cross_dtype_behavior(self) -> None:
        # High-precision numerical reference via forward-only finite difference.
        ref = _reference_fd(
            device=self.device,
            lam=self.lam,
            beta=self.beta,
            t=self.t64,
            y0_value=self.y0_value,
        )
        self.assertTrue(ref['ok'], f"Reference run failed: {ref['error']}")

        cases: List[Tuple[torch.dtype, Any, str]] = [
            (torch.float32, False, 'False'),
            (torch.float32, DynamicScaler, 'DynamicScaler'),
            (torch.float16, False, 'False'),
            (torch.float16, DynamicScaler, 'DynamicScaler'),
            (torch.bfloat16, False, 'False'),
            (torch.bfloat16, DynamicScaler, 'DynamicScaler'),
        ]

        rows = []
        rel_errors: Dict[Tuple[torch.dtype, str], Dict[str, Any]] = {}

        for dtype, scaler, scaler_name in cases:
            t_case = self.t64.to(dtype)
            out = _run_case(
                device=self.device,
                lam=self.lam,
                beta=self.beta,
                t=t_case,
                y0_value=self.y0_value,
                work_dtype=dtype,
                scaler=scaler,
            )

            if not out['ok']:
                rows.append((str(dtype), scaler_name, 'fail', 'fail', out['error']))
                rel_errors[(dtype, scaler_name)] = {'ok': False}
                continue

            rel_state = (torch.linalg.norm(out['yT'] - ref['yT']) / torch.linalg.norm(ref['yT'])).item()
            rel_grad = (
                torch.linalg.norm(out['grad_y0'] - ref['grad_y0']) / torch.linalg.norm(ref['grad_y0'])
            ).item()

            rows.append((str(dtype), scaler_name, f"{rel_state:.8e}", f"{rel_grad:.8e}", 'ok'))
            rel_errors[(dtype, scaler_name)] = {
                'ok': True,
                'rel_state': rel_state,
                'rel_grad': rel_grad,
            }

        quiet = os.environ.get('RAMPDE_TEST_QUIET', '0') == '1'
        if not quiet:
            print("| dtype | scaler | relerr y(T) | relerr dL/dy0 | status |")
            print("|-------|--------|-------------|---------------|--------|")
            for r in rows:
                print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")

        # Basic correctness checks against float64 L1 reference.
        f32_no_scaler = rel_errors[(torch.float32, 'False')]
        self.assertTrue(f32_no_scaler['ok'], 'float32 without scaler should succeed')
        self.assertLess(f32_no_scaler['rel_state'], 1e-4)
        self.assertLess(f32_no_scaler['rel_grad'], 1e-4)

        bf16_no_scaler = rel_errors[(torch.bfloat16, 'False')]
        self.assertTrue(bf16_no_scaler['ok'], 'bfloat16 without scaler should succeed')
        self.assertLess(bf16_no_scaler['rel_state'], 5e-2)
        self.assertLess(bf16_no_scaler['rel_grad'], 5e-2)

        # Cross-dtype behavior check for fp16 + DynamicScaler.
        fp16_dynamic = rel_errors[(torch.float16, 'DynamicScaler')]
        self.assertTrue(fp16_dynamic['ok'], 'float16 with DynamicScaler should produce finite outputs')


if __name__ == '__main__':
    unittest.main()
