"""
Discrete-sensitivity tests for the uniform L1 custom backward path.

This file focuses on checking:
1) Directional checks at a base point and along perturbation paths.
2) Grid-refinement trend checks for the manufactured constant-forcing case.
3) Reference-grid consistency checks for a nonlinear tanh-forcing case.

For the manufactured scalar problem

    _0^C D_t^beta z(t) = theta,    z(0) = z0,

the implemented terminal-loss sensitivity convention is

    dL/dz0    = dL/dz(T)
    dL/dtheta = T * dL/dz(T),  where T = t[-1] - t[0].

The directional tests verify autograd gradients against this rule.
"""

import math
import os
import sys
import unittest
import warnings

import torch
import torch.nn as nn
from torch.amp import autocast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from rampde.increment import L1
from rampde.fixed_grid_unscaled_uniform import FixedGridODESolverUnscaledUniform
from rampde.fixed_grid_dynamic_uniform import FixedGridODESolverDynamicUniform
from rampde.loss_scalers import DynamicScaler


_BACKWARD_VARIANT = os.getenv("RAMPDE_UNIFORM_BACKWARD_VARIANT", "unscaled").strip().lower()
if _BACKWARD_VARIANT not in {"unscaled", "dynamic"}:
    raise ValueError(
        "RAMPDE_UNIFORM_BACKWARD_VARIANT must be 'unscaled' or 'dynamic', "
        f"got '{_BACKWARD_VARIANT}'"
    )
_IS_DYNAMIC = _BACKWARD_VARIANT == "dynamic"


class ConstantParameterForcing(nn.Module):
    """f(t, z; theta) = theta (independent of z)."""

    def __init__(self, theta_init: float):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float64))

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.theta * torch.ones_like(z)


class TanhParameterForcing(nn.Module):
    """f(t, z; theta) = theta * tanh(z)."""

    def __init__(self, theta_init: float):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float64))

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.theta * torch.tanh(z)


def _run_solver(func: nn.Module, z0: torch.Tensor, t: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # zt = Phi_L1(func, z0, beta, t) using the custom autograd path.
    params = tuple(func.parameters())
    func = func.to(z0.device)

    if _BACKWARD_VARIANT == "dynamic":
        dtype_low = torch.float16
        solver_class = FixedGridODESolverDynamicUniform
        # Use a conservative scaler setup for test stability in fp16 dynamic mode.
        loss_scaler = DynamicScaler(dtype_low, target_factor=256.0, max_attempts=200)
    else:
        solver_class = FixedGridODESolverUnscaledUniform
        loss_scaler = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if _IS_DYNAMIC:
            with autocast(device_type='cuda', dtype=dtype_low):
                zt = solver_class.apply(L1(), func, z0, beta, t, loss_scaler, *params)
        else:
            zt = solver_class.apply(L1(), func, z0, beta, t, loss_scaler, *params)
    return zt


def _safe_rate(prev_err: float, curr_err: float, prev_n: int, curr_n: int) -> float:
    """Return observed step-doubling rate, or NaN when not well-defined."""
    if prev_err <= 0.0 or curr_err <= 0.0:
        return float('nan')
    if curr_n <= prev_n:
        return float('nan')
    return math.log(prev_err / curr_err) / math.log(curr_n / prev_n)


class TestL1UniformDiscreteDirectionalSensitivity(unittest.TestCase):

    def setUp(self):
        #print(f"Running: {self._testMethodName}")
        # Use stricter dtype/tolerance for unscaled and practical mixed-precision settings for dynamic.
        self.device = torch.device("cuda")
        self.dtype = torch.float32 if _IS_DYNAMIC else torch.float64
        self.directional_places = 4 if _IS_DYNAMIC else 7
        self.floor_atol = 1e-6 if _IS_DYNAMIC else 1e-10
        self.jitter_rel = 0.20 if _IS_DYNAMIC else 0.05
        self.beta = torch.tensor(0.6, dtype=self.dtype, device=self.device)
        self.t = torch.linspace(0.0, 1.0, 192, dtype=self.dtype, device=self.device)
        # T = t[-1] - t[0] in dL/dtheta = T * dL/dz(T).
        self.total_time = (self.t[-1] - self.t[0]).item()

    def _directional_pair(self, theta0: float, z00: float, v_z0: float, v_theta: float, loss_kind: str):
        func = ConstantParameterForcing(theta0).to(device=self.device, dtype=self.dtype)
        z0 = torch.tensor([z00], dtype=self.dtype, device=self.device, requires_grad=True)

        zt = _run_solver(func, z0, self.t, self.beta)
        z_terminal = zt[-1, 0]

        if loss_kind == "linear":
            loss = z_terminal
            # dL/dz(T) for L = z(T).
            dL_dzT = 1.0
        elif loss_kind == "quadratic":
            loss = 0.5 * z_terminal ** 2
            # dL/dz(T) for L = 0.5 z(T)^2.
            dL_dzT = z_terminal.item()
        elif loss_kind == "cubic":
            loss = (z_terminal ** 3) / 3.0
            # dL/dz(T) for L = z(T)^3 / 3.
            dL_dzT = z_terminal.item() ** 2
        else:
            raise ValueError(f"Unknown loss_kind: {loss_kind}")

        loss.backward()
        grad_z0 = z0.grad.item()
        grad_theta = func.theta.grad.item()

        # Directional derivative from returned gradients: Jv_auto = <grad, v>.
        jv_auto = grad_z0 * v_z0 + grad_theta * v_theta
        # Expected directional rule:
        # dL/dz0 = dL/dz(T), dL/dtheta = T * dL/dz(T).
        jv_expected = dL_dzT * v_z0 + (self.total_time * dL_dzT) * v_theta
        return jv_auto, jv_expected

    def _terminal_grads_for_grid_tanh(self, N: int, theta0: float, z00: float, beta_val: float, loss_kind: str):
        t = torch.linspace(0.0, 1.0, N, dtype=self.dtype, device=self.device)
        beta = torch.tensor(beta_val, dtype=self.dtype, device=self.device)

        func = TanhParameterForcing(theta0).to(device=self.device, dtype=self.dtype)
        z0 = torch.tensor([z00], dtype=self.dtype, device=self.device, requires_grad=True)

        zt = _run_solver(func, z0, t, beta)
        if loss_kind == "linear":
            loss = zt[-1, 0]
        elif loss_kind == "quadratic":
            loss = 0.5 * zt[-1, 0] ** 2
        elif loss_kind == "cubic":
            loss = (zt[-1, 0] ** 3) / 3.0
        else:
            raise ValueError(f"Unknown loss_kind: {loss_kind}")

        loss.backward()

        return z0.grad.item(), func.theta.grad.item()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for uniform L1 discrete-sensitivity tests")
    def test_directional_sensitivity_matches_rule_at_basepoint(self):
        print()
        print("-" * 60 + f"\nTesting directional sensitivity at z_0, theta_0\n" + "-" * 60)
        print()

        torch.manual_seed(42)

        theta0 = 1.25
        z00 = -0.35
        # Random direction v = (v_z0, v_theta), analogous to Taylor-test directions.
        v_z0 = torch.randn((), device=self.device, dtype=self.dtype).item()
        v_theta = torch.randn((), device=self.device, dtype=self.dtype).item()

        for loss_kind in ("linear", "quadratic", "cubic"):
            with self.subTest(loss_kind=loss_kind):
                jv_auto, jv_expected = self._directional_pair(theta0, z00, v_z0, v_theta, loss_kind)
                try:
                    self.assertAlmostEqual(jv_auto, jv_expected, places=self.directional_places)
                    print(" " * 10 + f"Testing {loss_kind} loss:   [PASS]")
                except AssertionError:
                    print(" " * 10 + f"Testing {loss_kind} loss:   [FAIL]")
                    raise

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for uniform L1 discrete-sensitivity tests")
    def test_directional_sensitivity_matches_rule_along_perturbation_path(self):
        print()
        print("-" * 60 + f"\nTesting directional sensitivity along perturbation paths\n" + "-" * 60)
        print()
        torch.manual_seed(7)

        theta0 = 0.9
        z00 = 0.2
        v_z0 = torch.randn((), device=self.device, dtype=self.dtype).item()
        v_theta = torch.randn((), device=self.device, dtype=self.dtype).item()

        # Probe the same direction along a perturbed path:
        # z0(h) = z00 + h v_z0, theta(h) = theta0 + h v_theta.
        h_vals = [0.0, 0.05, 0.1, 0.2, 0.3]
        for loss_kind in ("linear", "quadratic", "cubic"):
            for h in h_vals:
                with self.subTest(loss_kind=loss_kind, h=h):
                    jv_auto, jv_expected = self._directional_pair(
                        theta0 + h * v_theta,
                        z00 + h * v_z0,
                        v_z0,
                        v_theta,
                        loss_kind,
                    )
                    try:
                        self.assertAlmostEqual(jv_auto, jv_expected, places=self.directional_places)
                        print(" " * 10 + f"Testing {loss_kind} loss with h = {h} [PASS]")
                    except AssertionError:
                        print(" " * 10 + f"Testing {loss_kind} loss with h = {h} [FAIL]")
                        raise
            print()


    def _sensitivity_errors_for_grid(self, N: int, theta0: float, z00: float, beta_val: float, loss_kind: str):
        t = torch.linspace(0.0, 1.0, N, dtype = self.dtype, device = self.device)
        beta = torch.tensor(beta_val, dtype=self.dtype, device=self.device)
        total_time = (t[-1] - t[0]).item()
        
        func = ConstantParameterForcing(theta0).to(device=self.device, dtype=self.dtype)
        z0 = torch.tensor([z00], dtype=self.dtype, device=self.device, requires_grad = True)

        zt = _run_solver(func, z0, t, beta)
        z_terminal = zt[-1, 0]

        if loss_kind == "linear":
            loss = z_terminal
            dL_dzT = 1.0
        elif loss_kind == "quadratic":
            loss = 0.5 * z_terminal ** 2
            dL_dzT = z_terminal.item()
        elif loss_kind == "cubic":
            loss = (z_terminal ** 3) / 3.0
            dL_dzT = z_terminal.item() ** 2
        else:
            raise ValueError(f"Unknown loss_kind: {loss_kind}")
        
        loss.backward()
        grad_z0 = z0.grad.item()
        grad_theta = func.theta.grad.item()

        err_z0 = abs(grad_z0 - dL_dzT)
        err_theta = abs(grad_theta - (total_time * dL_dzT))
        return err_z0, err_theta
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for uniform L1 discrete-sensitivity tests")
    def test_sensitivity_errors_decrease_with_grid_refinement(self):
        print()
        print("-" * 60 + f"\nTesting sensitivity errors, convergence rates with various loss forms:\n" + "-" * 60)
        print()
        N_vals = [2, 4, 8, 16, 32, 64, 128, 256]
        theta0 = 1.1
        z00 = -0.2
        beta_val = 0.6
        loss_kinds = ["linear", "quadratic", "cubic"]
        # If errors are already at numerical floor, do not require further decreases.
        atol_z0 = self.floor_atol
        atol_theta = self.floor_atol
        jitter_rel = self.jitter_rel

        for loss_kind in loss_kinds:
            print()
            print(" " * 10 + "-" * 50 + f"\n " + " " * 10 + f"{loss_kind} loss" + f"\n" + " " * 10 + "-" * 50)
            print()
            errs_z0 = []
            errs_theta = []
            for i in range(len(N_vals)):
                N = N_vals[i]
                e_z0, e_th = self._sensitivity_errors_for_grid(N, theta0, z00, beta_val, loss_kind)
                errs_z0.append(e_z0)
                errs_theta.append(e_th)
                if i > 0:
                    rate_z0 = _safe_rate(errs_z0[-2], errs_z0[-1], N_vals[i - 1], N)
                    rate_theta = _safe_rate(errs_theta[-2], errs_theta[-1], N_vals[i - 1], N)
                else:
                    rate_z0 = float('nan')
                    rate_theta = float('nan')
                print(" " * 10 + f'N = {N}: err_z0 = {e_z0:.2e}, err_theta = {e_th:.2e}, rate_z0 = {rate_z0:.2e}, rate_theta = {rate_theta:.2e}')

            # Only enforce improvement for transitions that are above a numerical floor.
            active_z0 = [
                i for i in range(1, len(errs_z0))
                if max(errs_z0[i], errs_z0[i - 1]) > atol_z0
            ]
            active_theta = [
                i for i in range(1, len(errs_theta))
                if max(errs_theta[i], errs_theta[i - 1]) > atol_theta
            ]

            improve_z0 = sum(
                1
                for i in active_z0
                if errs_z0[i] <= errs_z0[i - 1] * (1.0 + jitter_rel) + atol_z0
            )
            improve_theta = sum(
                1
                for i in active_theta
                if errs_theta[i] <= errs_theta[i - 1] * (1.0 + jitter_rel) + atol_theta
            )

            # If we are already near floor (<=1 active transition), skip strict checks.
            # Otherwise require majority improvement, not near-monotone improvement.
            req_z0 = 0 if len(active_z0) <= 1 else max(1, len(active_z0) // 2)
            req_theta = 0 if len(active_theta) <= 1 else max(1, len(active_theta) // 2)

            self.assertGreaterEqual(
                improve_z0,
                req_z0,
                (
                    f"z0 sensitivity error did not decrease sufficiently for {loss_kind}: "
                    f"errors = {errs_z0}, active_steps = {len(active_z0)}, improvements = {improve_z0}"
                ),
            )
            self.assertGreaterEqual(
                improve_theta,
                req_theta,
                (
                    f"theta sensitivity error did not decrease sufficiently for {loss_kind}: "
                    f"errors = {errs_theta}, active_steps = {len(active_theta)}, improvements = {improve_theta}"
                ),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @unittest.skip("Skip for now takes long")
    def test_tanh_forcing(self):
        # Test with a nontrivial case, 
        # f(t, z(t); theta) = theta * tanh(z(t)), various loss kinds
        print()
        print("-" * 60 + f"\nTesting sensitivity errors and convergence rates for tanh forcing\n" + "-" * 60)
        print()

        theta0 = 1.1
        z00 = -0.2
        beta_val = 0.6
        loss_kinds = ["linear", "quadratic", "cubic"]
        N_vals = [4, 8, 16, 32, 64, 128, 256]
        N_ref = 2048
        for loss_kind in loss_kinds:
            err_z0 = []
            err_theta = []
            print()
            print(" " * 10 + "-" * 20 + f"\n" + " " * 10 + f"{loss_kind} loss" + f"\n" + " " * 10 + "-" * 20)
            print(" " * 10 + "Building reference solution...")
            g_z0_ref, g_theta_ref = self._terminal_grads_for_grid_tanh(N_ref, theta0, z00, beta_val, loss_kind=loss_kind)
            print(" " * 10 + f"Reference N = { N_ref}: grad_z0 = {g_z0_ref:.4e}, grad_theta = {g_theta_ref:.4e}")
            print()

            for i, N in enumerate(N_vals):
                g_z0, g_theta = self._terminal_grads_for_grid_tanh(N, theta0, z00, beta_val, loss_kind=loss_kind)
                e_z0 = abs(g_z0 - g_z0_ref)
                e_theta = abs(g_theta - g_theta_ref)
                err_z0.append(e_z0)
                err_theta.append(e_theta)
                if i > 0:
                    rate_z0 = _safe_rate(prev_e_z0, e_z0, N_vals[i - 1], N)
                    rate_theta = _safe_rate(prev_e_theta, e_theta, N_vals[i - 1], N)
                else:
                    rate_z0 = float('nan')
                    rate_theta = float('nan')
                print(" " * 10 + f"N = {N:4d}: grad_z0 = {g_z0:.4e}, grad_theta = {g_theta:.4e}, err_z0 = {e_z0:.2e}, err_theta = {e_theta:.2e}, rate_z0 = {rate_z0:.2f}, rate_theta = {rate_theta:.2f}")
                prev_e_z0 = e_z0
                prev_e_theta = e_theta
            print()
            
        improve_z0 = sum(1 for i in range(1, len(err_z0)) if err_z0[i] <= err_z0[i - 1])
        improve_theta = sum(1 for i in range(1, len(err_theta)) if err_theta[i] <= err_theta[i - 1])

        # Require trend toward reference without enforcing strict monotonicity.
        self.assertGreaterEqual(improve_z0, 3, f"Loss kind: {loss_kind}, z0 gradient errors did not improve enough: {err_z0}")
        self.assertGreaterEqual(improve_theta, 3, f"Loss kind: {loss_kind}, theta gradient errors did not improve enough: {err_theta}")

        # Also require that final error is smaller than coarse-grid error unless already at floor.
        atol = 1e-6 if _IS_DYNAMIC else 1e-12
        if err_z0[0] > atol:
            self.assertLessEqual(err_z0[-1], err_z0[0], f"Loss kind: {loss_kind}, z0 final error did not improve: {err_z0}")
        if err_theta[0] > atol:
            self.assertLessEqual(err_theta[-1], err_theta[0], f"Loss kind: {loss_kind}, theta final error did not improve: {err_theta}")


if __name__ == "__main__":
    unittest.main(verbosity=1)