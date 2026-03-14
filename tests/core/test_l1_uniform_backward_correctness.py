"""
Backward-pass correctness tests for the uniform-grid L1 solver.

This test suite targets `rampde/fixed_grid_unscaled_uniform.py` directly and
checks that gradients from the custom backward are consistent with known
fractional-IVP structure.

We test the forward Caputo IVP

    _0^C D_t^beta z(t) = f(t, z(t); theta),    z(0) = 0,

and backward/parameter sensitivities with a manufactured case where
closed-form expressions are available.

Manufactured ODE used here
--------------------------
Choose

    f(t, z; theta) = theta,   (independent of z)

Then

    z(t) = theta * t^beta / Gamma(beta + 1)

exactly for the Caputo IVP with z(0)=0.

For a terminal loss L = z(T), we have

    lambda(T) = dL/dz(T) = 1

and since df/dz = 0, the adjoint equation implies lambda is constant,
so dL/dz(0) = 1.

Also

    dL/dtheta = T^beta / Gamma(beta + 1)

which provides an exact gradient target.
"""

import math
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
    # Keep auto selection; tests run in float64 regardless of device.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_solver(func: nn.Module, z0: torch.Tensor, t: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    params = tuple(func.parameters())
    func = func.to(z0.device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zt = FixedGridODESolverUnscaledUniform.apply(L1(), func, z0, beta, t, None, *params)
    return zt


class TestL1UniformBackwardCorrectness(unittest.TestCase):

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

    def test_backward_exact_terminal_loss_gradients(self):
        """
        With L = z(T), check exact gradients:
          dL/dz0 = 1,
          dL/dtheta = T^beta / Gamma(beta+1).
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
        loss.backward()

        grad_z0 = z0.grad.item()
        grad_theta = func.theta.grad.item()

        expected_grad_z0 = 1.0
        expected_grad_theta = T ** beta_val / gamma_fn(beta_val + 1)

        self.assertAlmostEqual(grad_z0, expected_grad_z0, places=8)
        self.assertAlmostEqual(grad_theta, expected_grad_theta, places=8)

    def test_backward_quadratic_loss_closed_form_theta_grad(self):
        """
        With L = 0.5 * z(T)^2 and z(T)=theta*C, C=T^beta/Gamma(beta+1),
        exact gradient is dL/dtheta = theta * C^2.
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
        loss.backward()

        grad_theta = func.theta.grad.item()
        C = T ** beta_val / gamma_fn(beta_val + 1)
        expected_grad_theta = theta0 * (C ** 2)

        self.assertAlmostEqual(grad_theta, expected_grad_theta, places=7)

    def test_backward_theta_matches_finite_difference(self):
        """Autograd theta gradient agrees with central finite differences."""
        device = _solver_device()
        theta0 = 1.1
        beta_val = 0.55
        N, T = 160, 1.0
        eps = 1e-6

        def compute_loss(theta_val: float) -> float:
            func_fd = ConstantParamForcing(theta_val).to(device)
            z0_fd = torch.zeros(1, dtype=torch.float64, device=device)
            t_fd = torch.linspace(0, T, N, dtype=torch.float64, device=device)
            beta_fd = torch.tensor(beta_val, dtype=torch.float64, device=device)
            zt_fd = _run_solver(func_fd, z0_fd, t_fd, beta_fd)
            return (0.5 * zt_fd[-1, 0] ** 2).item()

        # Autograd gradient
        func = ConstantParamForcing(theta0).to(device)
        z0 = torch.zeros(1, dtype=torch.float64, device=device)
        t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
        beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

        zt = _run_solver(func, z0, t, beta)
        loss = 0.5 * zt[-1, 0] ** 2
        loss.backward()
        grad_auto = func.theta.grad.item()

        # Central finite-difference gradient
        loss_p = compute_loss(theta0 + eps)
        loss_m = compute_loss(theta0 - eps)
        grad_fd = (loss_p - loss_m) / (2.0 * eps)

        rel_err = abs(grad_auto - grad_fd) / max(1e-12, abs(grad_fd))
        self.assertLess(rel_err, 1e-4, f"Autograd/FD mismatch: rel_err={rel_err:.3e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
