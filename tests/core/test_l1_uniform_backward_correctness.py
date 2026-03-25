"""
Regression tests for the uniform-grid L1 custom backward.

This suite targets rampde/fixed_grid_unscaled_uniform.py directly and checks
that gradients from the custom backward follow the solver's intended discrete
sensitivity convention.

These tests verify that the custom backward matches forward-map parameter
sensitivities for manufactured cases.

We use the manufactured Caputo IVP

    _0^C D_t^beta z(t) = f(t, z(t); theta),    z(0) = 0,

with constant forcing

    f(t, z; theta) = theta.

Forward still matches the known closed form

    z(t) = theta * t^beta / Gamma(beta + 1).

Because f is independent of z, the uniform backward's terminal adjoint is
constant across the reverse sweep for terminal-only losses. On a uniform grid
of length T,

    dL/dz0     = dL/dz(T)
    dL/dtheta  = T * dL/dz(T)

for any scalar loss L that depends only on z(T).
"""

import os
import sys
import unittest
import warnings

import torch
import torch.nn as nn
from math import gamma as gamma_fn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from rampde.increment import L1
from rampde.fixed_grid_unscaled_uniform import FixedGridODESolverUnscaledUniform


class ConstantParamForcing(nn.Module):
    """f(t, z; theta) = theta (independent of z)."""

    def __init__(self, theta_init: float):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float64))

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.theta * torch.ones_like(z)


def _solver_device() -> torch.device:
    return torch.device("cuda")


def _run_solver(func: nn.Module, z0: torch.Tensor, t: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    params = tuple(func.parameters())
    func = func.to(z0.device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zt = FixedGridODESolverUnscaledUniform.apply(L1(), func, z0, beta, t, None, *params)
    return zt


class TestL1UniformBackwardCorrectness(unittest.TestCase):

    @staticmethod
    def _terminal_convention_expectations(
        terminal_grad: float,
        t: torch.Tensor,
    ) -> tuple[float, float]:
        total_time = (t[-1] - t[0]).item()
        return terminal_grad, total_time * terminal_grad

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for uniform L1 backward tests")
    def test_forward_closed_form_constant_forcing(self):
        """Forward solution matches z(t)=theta*t^beta/Gamma(beta+1)."""
        device = _solver_device()
        theta0 = 1.7
        beta_val = 0.6
        N, T = 200, 1.0

        func = ConstantParamForcing(theta0).to(device)
        z0 = torch.zeros(1, dtype=torch.float64, device=device)
        t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
        beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

        zt = _run_solver(func, z0, t, beta)
        exact = (theta0 * t ** beta_val / gamma_fn(beta_val + 1)).unsqueeze(1)

        max_err = (zt - exact).abs().max().item()
        self.assertLess(max_err, 1e-8, f"Forward closed-form mismatch: max_err={max_err:.3e}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for uniform L1 backward tests")
    def test_backward_exact_terminal_loss_gradients(self):
        """
        With L = z(T), check the current discrete terminal sensitivity rule:
          dL/dz0 = dL/dz(T) = 1,
                    dL/dtheta = T * dL/dz(T).
        """
        device = _solver_device()
        theta0 = 2.3
        beta_val = 0.5
        N, T = 200, 1.0

        func = ConstantParamForcing(theta0).to(device)
        z0 = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
        t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
        beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

        zt = _run_solver(func, z0, t, beta)
        loss = zt[-1, 0]
        expected_grad_z0, expected_grad_theta = self._terminal_convention_expectations(1.0, t)
        loss.backward()

        grad_z0 = z0.grad.item()
        grad_theta = func.theta.grad.item()

        self.assertAlmostEqual(grad_z0, expected_grad_z0, places=8)
        self.assertAlmostEqual(grad_theta, expected_grad_theta, places=8)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for uniform L1 backward tests")
    def test_backward_quadratic_loss_closed_form_theta_grad(self):
        """
        With L = 0.5 * z(T)^2, the current convention gives
        dL/dtheta = T * dL/dz(T) = T * z(T).
        """
        device = _solver_device()
        theta0 = 1.2
        beta_val = 0.75
        N, T = 200, 1.0

        func = ConstantParamForcing(theta0).to(device)
        z0 = torch.zeros(1, dtype=torch.float64, device=device)
        t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
        beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

        zt = _run_solver(func, z0, t, beta)
        loss = 0.5 * zt[-1, 0] ** 2
        _, expected_grad_theta = self._terminal_convention_expectations(zt[-1, 0].item(), t)
        loss.backward()

        grad_theta = func.theta.grad.item()

        self.assertAlmostEqual(grad_theta, expected_grad_theta, places=7)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for uniform L1 backward tests")
    def test_backward_theta_matches_standard_sensitivity_formula(self):
        """Quadratic terminal loss follows dL/dtheta = T * dL/dz(T)."""
        device = _solver_device()
        theta0 = 1.1
        beta_val = 0.55
        N, T = 160, 1.0

        func = ConstantParamForcing(theta0).to(device)
        z0 = torch.zeros(1, dtype=torch.float64, device=device)
        t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
        beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

        zt = _run_solver(func, z0, t, beta)
        loss = 0.5 * zt[-1, 0] ** 2
        _, expected_grad = self._terminal_convention_expectations(zt[-1, 0].item(), t)
        loss.backward()
        grad_auto = func.theta.grad.item()

        self.assertAlmostEqual(grad_auto, expected_grad, places=7)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for uniform L1 backward tests")
    def test_backward_cubic_terminal_loss_matches_terminal_sensitivity_rule(self):
        """Cubic terminal loss still follows the same terminal adjoint convention."""
        device = _solver_device()
        theta0 = 0.9
        beta_val = 0.65
        N, T = 192, 1.0

        func = ConstantParamForcing(theta0).to(device)
        z0 = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
        t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
        beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

        zt = _run_solver(func, z0, t, beta)
        loss = (zt[-1, 0] ** 3) / 3.0
        terminal_grad = zt[-1, 0].item() ** 2
        expected_grad_z0, expected_grad_theta = self._terminal_convention_expectations(terminal_grad, t)
        loss.backward()

        self.assertAlmostEqual(z0.grad.item(), expected_grad_z0, places=7)
        self.assertAlmostEqual(func.theta.grad.item(), expected_grad_theta, places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
