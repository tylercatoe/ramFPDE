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
import math

import torch
import torch.nn as nn
from math import gamma as gamma_fn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rampde.fixed_grid_base_uniform import FixedGridODESolverBase
from rampde.increment import L1

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
        print()
        print("-" * 60 + f"\nTesting constant forcing exact solution\n" + "-" * 60)
        print()
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
                try:
                    self.assertLess(max_err, 1e-12, f"beta={beta_val}, max_err={max_err:.3e}")
                    print(" " * 5 + f"beta={beta_val:.2f}, max_err={max_err:.3e} [PASS]")
                except AssertionError:
                    print(" " * 5 + f"beta={beta_val:.2f}, max_err={max_err:.3e} [FAIL]")
                    raise

    def test_polynomial_manufactured_solution(self):
        print()
        print("-" * 60 + f"\nTesting polynomial manufactured solution\n" + "-" * 60)
        print()
        for beta_val in [0.3, 0.5, 0.75, 0.9, 1.0]:
            with self.subTest(beta=beta_val):
                device = _solver_device()
                N, T = 200, 1.0
                t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
                z0 = torch.zeros(1, dtype=torch.float64, device=device)
                beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

                coeff = 2.0 / gamma_fn(3.0 - beta_val)
                exponent = 2.0 - beta_val
                zt = _run_forward(PolyForcing(coeff, exponent), z0, t, beta)

                exact = (t ** 2.0).unsqueeze(1)
                max_err = (zt - exact).abs().max().item()
                h = T / (N-1)
                tol = 3 * (h ** (1.0 + beta_val))
                try:
                    self.assertLess(max_err, tol, f"beta={beta_val}, max_err={max_err:.3e}")
                    print(" " * 5 + f"beta={beta_val:.2f}, max_err={max_err:.3e}, tol={tol:.3e} [PASS]")
                except AssertionError:
                    print(" " * 5 + f"beta={beta_val:.2f}, max_err={max_err:.3e}, tol={tol:.3e} [FAIL]")
                    raise

    def test_beta_one_linear_decay(self):
        print()
        print("-" * 60 + f"\nTesting beta=1 linear decay\n" + "-" * 60)
        print()
        device = _solver_device()
        N, T = 200, 1.0
        t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
        z0 = torch.ones(1, dtype=torch.float64, device=device)
        beta = torch.tensor(1.0, dtype=torch.float64, device=device)

        zt = _run_forward(LinearDecay(), z0, t, beta)
        exact = torch.exp(-t).unsqueeze(1)
        max_err = (zt - exact).abs().max().item()
        try:
            self.assertLess(max_err, 1e-5, f"beta=1, max_err={max_err:.3e}")
            print(" " * 5 + f"beta=1.00, max_err={max_err:.3e} [PASS]")
        except AssertionError:
            print(" " * 5 + f"beta=1.00, max_err={max_err:.3e} [FAIL]")
            raise

    def test_convergence_under_refinement(self):
        print()
        print("-" * 60 + f"\nTesting convergence under grid refinement\n" + "-" * 60)
        print()
        beta_val, T = 0.75, 1.0
        exact_T = T ** 2
        device = _solver_device()

        coeff = 2.0 / gamma_fn(3.0 - beta_val)
        exponent = 2.0 - beta_val
        beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

        errors = []
        grid_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        prev_error = None
        prev_N = None
        for N in grid_sizes:
            t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
            z0 = torch.zeros(1, dtype=torch.float64, device=device)
            zt = _run_forward(PolyForcing(coeff, exponent), z0, t, beta)
            err = abs(zt[-1, 0].item() - exact_T)
            errors.append(err)

            if prev_error is not None and err > 0.0 and prev_error > 0.0:
                observed_rate = math.log(prev_error / err) / math.log(float(N) / float(prev_N))
                print(f'N = {N}, error = {err:.3e}, observed_rate = {observed_rate:.3f}')
            else:
                print(f'N = {N}, error = {err:.3e}, observed_rate = n/a')

            prev_error = err
            prev_N = N

        for i in range(len(errors) - 1):
            n0 = grid_sizes[i]
            n1 = grid_sizes[i + 1]
            try:
                self.assertLess(
                    errors[i + 1],
                    errors[i],
                    f"Error did not decrease: N={n0} -> N={n1}"
                )
                print(" " * 5 + f"N={n0} -> N={n1} error decrease [PASS]")
            except AssertionError:
                print(" " * 5 + f"N={n0} -> N={n1} error decrease [FAIL]")
                raise


    def test_initial_condition_preserved(self):
        print()
        print("-" * 60 + f"\nTesting initial condition preservation\n" + "-" * 60)
        print()
        for beta_val in [0.3, 0.7, 1.0]:
            with self.subTest(beta=beta_val):
                device = _solver_device()
                z0_val = 3.14159
                z0 = torch.tensor([z0_val], dtype=torch.float64, device=device)
                t = torch.linspace(0, 1, 50, dtype=torch.float64, device=device)
                beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

                zt = _run_forward(LinearDecay(), z0, t, beta)
                try:
                    self.assertAlmostEqual(zt[0, 0].item(), z0_val, places=14)
                    print(" " * 5 + f"beta={beta_val:.2f}, z(0) preserved [PASS]")
                except AssertionError:
                    print(" " * 5 + f"beta={beta_val:.2f}, z(0) preserved [FAIL]")
                    raise

    def test_output_stays_on_input_device(self):
        print()
        print("-" * 60 + f"\nTesting output device consistency\n" + "-" * 60)
        print()
        device = _solver_device()
        t = torch.linspace(0, 1, 32, dtype=torch.float64, device=device)
        z0 = torch.zeros(1, dtype=torch.float64, device=device)
        beta = torch.tensor(0.5, dtype=torch.float64, device=device)

        zt = _run_forward(ConstantForcing(1.0), z0, t, beta)
        try:
            self.assertEqual(zt.device.type, device.type)
            print(" " * 5 + f"device={device.type}, output device match [PASS]")
        except AssertionError:
            print(" " * 5 + f"device={device.type}, output device match [FAIL]")
            raise


if __name__ == "__main__":
    unittest.main(verbosity=1)
