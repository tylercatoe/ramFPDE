"""
Correctness tests for the ABM predictor-corrector forward pass in
rampde/fixed_grid_base.py.

This file mirrors tests/core/test_l1_uniform_correctness.py, using the same
manufactured ODE examples:

1. Constant forcing: f(t, z) = 1, z(0) = 0
2. Polynomial forcing: f(t, z) = 2/Gamma(3-beta) * t^(2-beta), z(0) = 0
3. beta=1 limit with linear decay: f(t, z) = -z, z(0) = 1

but runs them through the ABM forward pass implementation.
"""

import os
import sys
import unittest
import warnings

import torch
import torch.nn as nn
from math import gamma as gamma_fn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rampde.fixed_grid_base import FixedGridODESolverBase
from rampde.increment import L1


# ---------------------------------------------------------------------------
# Pure-Python reference: ABM-L1 predictor-corrector on a uniform grid
# ---------------------------------------------------------------------------

def _C_left(A, B, A_minus_B, beta):
    return (beta * A ** (beta + 1) - (beta + 1) * B * A ** beta + B ** (beta + 1)) / (
        A_minus_B * beta * (beta + 1)
    )


def _C_right(A, B, A_minus_B, beta):
    return (A ** (beta + 1) - (beta + 1) * A * B ** beta + beta * B ** (beta + 1)) / (
        A_minus_B * beta * (beta + 1)
    )


def reference_abm_l1(func, z0_val, t_vals, beta_val):
    """Reference ABM-L1 predictor-corrector implementation."""
    N = len(t_vals)
    t = t_vals
    b = beta_val
    g = gamma_fn(b)

    zs = [0.0] * N
    fs = [0.0] * N
    zs[0] = z0_val

    for k in range(N - 1):
        # Predictor
        zp = z0_val
        for j in range(k):
            bj = (1.0 / b) * ((t[k + 1] - t[j]) ** b - (t[k + 1] - t[j + 1]) ** b)
            zp += (1.0 / g) * bj * fs[j]

        fs[k] = func(t[k], zs[k])
        bj = (1.0 / b) * ((t[k + 1] - t[k]) ** b - (t[k + 1] - t[k + 1]) ** b)
        zp += (1.0 / g) * bj * fs[k]

        # Corrector
        zc = z0_val

        # j = 0
        A = t[k + 1] - t[0]
        B = t[k + 1] - t[1]
        h01 = t[1] - t[0]
        a0 = _C_left(A, B, h01, b)
        zc += (1.0 / g) * a0 * fs[0]

        # j = 1..k-1
        for j in range(1, k):
            A_r = t[k + 1] - t[j - 1]
            B_r = t[k + 1] - t[j]
            h_r = t[j] - t[j - 1]
            A_l = t[k + 1] - t[j]
            B_l = t[k + 1] - t[j + 1]
            h_l = t[j + 1] - t[j]
            aj = _C_right(A_r, B_r, h_r, b) + _C_left(A_l, B_l, h_l, b)
            zc += (1.0 / g) * aj * fs[j]

        # j = k
        if k >= 1:
            A_r = t[k + 1] - t[k - 1]
            B_r = t[k + 1] - t[k]
            h_r = t[k] - t[k - 1]
            c_r = _C_right(A_r, B_r, h_r, b)
            c_l = (t[k + 1] - t[k]) ** b / (b + 1)
            zc += (1.0 / g) * (c_r + c_l) * fs[k]

        # predictor contribution
        a_pred = (t[k + 1] - t[k]) ** b / (b * (b + 1))
        f_pred = func(t[k + 1], zp)
        zs[k + 1] = zc + (1.0 / g) * a_pred * f_pred

    return zs


# ---------------------------------------------------------------------------
# ODE function nn.Modules
# ---------------------------------------------------------------------------

class ConstantForcing(nn.Module):
    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.full_like(z, self.c)


class PolyForcing(nn.Module):
    def __init__(self, coeff: float, exponent: float):
        super().__init__()
        self.coeff = coeff
        self.exponent = exponent

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        tv = t.item() if t.dim() == 0 else float(t)
        val = self.coeff * tv ** self.exponent if tv > 0.0 else 0.0
        return torch.full_like(z, val)


class LinearDecay(nn.Module):
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return -z


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _solver_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_forward(ode_func, z0, t, beta):
    params = list(ode_func.parameters())
    ode_func = ode_func.to(z0.device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            zt = FixedGridODESolverBase.apply(L1(), ode_func, z0, beta, t, None, *params)
    return zt


class TestABMForwardPass(unittest.TestCase):
    """Forward-pass correctness tests for ABM predictor-corrector."""

    def test_constant_forcing_exact_solution(self):
        for beta_val in [0.3, 0.5, 0.75, 0.9, 1.0]:
            with self.subTest(beta=beta_val):
                device = _solver_device()
                N, T = 200, 1.0
                t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
                z0 = torch.zeros(1, dtype=torch.float64, device=device)
                beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

                zt = _run_forward(ConstantForcing(1.0), z0, t, beta)
                exact = (t ** beta_val / gamma_fn(beta_val + 1)).unsqueeze(1)
                max_err = (zt - exact).abs().max().item()

                self.assertLess(max_err, 1e-6, f"beta={beta_val}, max_err={max_err:.3e}")

    def test_polynomial_manufactured_solution(self):
        for beta_val in [0.5, 0.75]:
            with self.subTest(beta=beta_val):
                device = _solver_device()
                N, T = 500, 1.0
                t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
                z0 = torch.zeros(1, dtype=torch.float64, device=device)
                beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

                coeff = 2.0 / gamma_fn(3.0 - beta_val)
                exponent = 2.0 - beta_val
                zt = _run_forward(PolyForcing(coeff, exponent), z0, t, beta)

                exact = (t ** 2.0).unsqueeze(1)
                max_err = (zt - exact).abs().max().item()
                self.assertLess(max_err, 2e-2, f"beta={beta_val}, max_err={max_err:.3e}")

    def test_beta_one_linear_decay(self):
        device = _solver_device()
        N, T = 2000, 1.0
        t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
        z0 = torch.ones(1, dtype=torch.float64, device=device)
        beta = torch.tensor(1.0, dtype=torch.float64, device=device)

        zt = _run_forward(LinearDecay(), z0, t, beta)
        exact = torch.exp(-t).unsqueeze(1)
        max_err = (zt - exact).abs().max().item()

        self.assertLess(max_err, 5e-3, f"beta=1, max_err={max_err:.3e}")

    def test_convergence_under_refinement(self):
        beta_val, T = 0.75, 1.0
        exact_T = T ** 2
        device = _solver_device()

        coeff = 2.0 / gamma_fn(3.0 - beta_val)
        exponent = 2.0 - beta_val
        beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

        errors = []
        grid_sizes = [50, 100, 200, 400, 800]
        for N in grid_sizes:
            t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
            z0 = torch.zeros(1, dtype=torch.float64, device=device)
            zt = _run_forward(PolyForcing(coeff, exponent), z0, t, beta)
            errors.append(abs(zt[-1, 0].item() - exact_T))

        for i in range(len(errors) - 1):
            self.assertLess(
                errors[i + 1],
                errors[i],
                f"Error did not decrease: N={grid_sizes[i]} -> N={grid_sizes[i + 1]}"
            )

    def test_matches_pure_python_reference(self):
        beta_val, N, T = 0.6, 80, 1.0
        device = _solver_device()
        t_list = [i * T / (N - 1) for i in range(N)]

        for label, py_func, torch_func in [
            ("constant f=1", lambda tt, z: 1.0, ConstantForcing(1.0)),
            (
                "poly f=D^b t^2",
                lambda tt, z: (2.0 / gamma_fn(3.0 - beta_val)) * tt ** (2.0 - beta_val) if tt > 0.0 else 0.0,
                PolyForcing(2.0 / gamma_fn(3.0 - beta_val), 2.0 - beta_val),
            ),
        ]:
            with self.subTest(forcing=label):
                ref = reference_abm_l1(py_func, 0.0, t_list, beta_val)

                t = torch.tensor(t_list, dtype=torch.float64, device=device)
                z0 = torch.zeros(1, dtype=torch.float64, device=device)
                beta = torch.tensor(beta_val, dtype=torch.float64, device=device)
                zt = _run_forward(torch_func, z0, t, beta)

                for i in range(N):
                    self.assertAlmostEqual(
                        zt[i, 0].item(),
                        ref[i],
                        places=9,
                        msg=(
                            f"[{label}] mismatch at i={i}: "
                            f"torch={zt[i,0].item():.14f}, ref={ref[i]:.14f}"
                        ),
                    )

    def test_initial_condition_preserved(self):
        for beta_val in [0.3, 0.7, 1.0]:
            with self.subTest(beta=beta_val):
                device = _solver_device()
                z0_val = 3.14159
                z0 = torch.tensor([z0_val], dtype=torch.float64, device=device)
                t = torch.linspace(0, 1, 50, dtype=torch.float64, device=device)
                beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

                zt = _run_forward(LinearDecay(), z0, t, beta)

                self.assertAlmostEqual(zt[0, 0].item(), z0_val, places=14)

    def test_output_stays_on_input_device(self):
        device = _solver_device()
        t = torch.linspace(0, 1, 32, dtype=torch.float64, device=device)
        z0 = torch.zeros(1, dtype=torch.float64, device=device)
        beta = torch.tensor(0.5, dtype=torch.float64, device=device)

        zt = _run_forward(ConstantForcing(1.0), z0, t, beta)

        self.assertEqual(zt.device.type, device.type)


if __name__ == "__main__":
    unittest.main(verbosity=2)
