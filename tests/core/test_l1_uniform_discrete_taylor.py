"""
Discrete-sensitivity directional tests for the uniform L1 backward path.

These tests mirror the directional structure of Taylor checks (random directions
and perturbed base points), but validate the architecture-2 sensitivity rule
directly instead of forward-map Taylor convergence.

For the manufactured scalar problem

    _0^C D_t^beta z(t) = theta,    z(0) = z0,

the implemented uniform backward convention for terminal-only losses is

    dL/dz0    = dL/dz(T)
    dL/dtheta = -T * dL/dz(T),  where T = t[-1] - t[0].

We verify that directional sensitivities from autograd gradients match this
rule at the base point and across perturbed points.
"""

import math
import os
import sys
import unittest
import warnings

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from rampde.increment import L1
from rampde.fixed_grid_unscaled_uniform import FixedGridODESolverUnscaledUniform


class ConstantParameterForcing(nn.Module):
    """f(t, z; theta) = theta (independent of z)."""

    def __init__(self, theta_init: float):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float64))

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.theta * torch.ones_like(z)


def _run_solver(func: nn.Module, z0: torch.Tensor, t: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # zt = Phi_L1(func, z0, beta, t) using the custom autograd path.
    params = tuple(func.parameters())
    func = func.to(z0.device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zt = FixedGridODESolverUnscaledUniform.apply(L1(), func, z0, beta, t, None, *params)
    return zt


class TestL1UniformDiscreteDirectionalSensitivity(unittest.TestCase):

    def setUp(self):
        # Use float64 so directional sensitivity comparisons are numerically stable.
        self.device = torch.device("cuda")
        self.dtype = torch.float64
        self.beta = torch.tensor(0.6, dtype=self.dtype, device=self.device)
        self.t = torch.linspace(0.0, 1.0, 192, dtype=self.dtype, device=self.device)
        # T = t[-1] - t[0] in dL/dtheta = -T * dL/dz(T).
        self.total_time = (self.t[-1] - self.t[0]).item()

    def _directional_pair(self, theta0: float, z00: float, v_z0: float, v_theta: float, loss_kind: str):
        func = ConstantParameterForcing(theta0).to(self.device)
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
        # Expected architecture-2 directional rule:
        # dL/dz0 = dL/dz(T), dL/dtheta = -T * dL/dz(T).
        jv_expected = dL_dzT * v_z0 + (-self.total_time * dL_dzT) * v_theta
        return jv_auto, jv_expected

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for uniform L1 discrete-sensitivity tests")
    def test_directional_sensitivity_matches_rule_at_basepoint(self):
        torch.manual_seed(42)

        theta0 = 1.25
        z00 = -0.35
        # Random direction v = (v_z0, v_theta), analogous to Taylor-test directions.
        v_z0 = torch.randn((), device=self.device, dtype=self.dtype).item()
        v_theta = torch.randn((), device=self.device, dtype=self.dtype).item()

        for loss_kind in ("linear", "quadratic", "cubic"):
            with self.subTest(loss_kind=loss_kind):
                jv_auto, jv_expected = self._directional_pair(theta0, z00, v_z0, v_theta, loss_kind)
                self.assertAlmostEqual(jv_auto, jv_expected, places=7)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for uniform L1 discrete-sensitivity tests")
    def test_directional_sensitivity_matches_rule_along_perturbation_path(self):
        torch.manual_seed(7)

        theta0 = 0.9
        z00 = 0.2
        v_z0 = torch.randn((), device=self.device, dtype=self.dtype).item()
        v_theta = torch.randn((), device=self.device, dtype=self.dtype).item()

        # Probe the same direction along a perturbed path:
        # z0(h) = z00 + h v_z0, theta(h) = theta0 + h v_theta.
        h_vals = [0.0, 0.05, 0.1, 0.2, 0.3]
        for loss_kind in ("quadratic", "cubic"):
            for h in h_vals:
                with self.subTest(loss_kind=loss_kind, h=h):
                    jv_auto, jv_expected = self._directional_pair(
                        theta0 + h * v_theta,
                        z00 + h * v_z0,
                        v_z0,
                        v_theta,
                        loss_kind,
                    )
                    self.assertAlmostEqual(jv_auto, jv_expected, places=7)


    def _sensitivity_errors_for_grid(self, N: int, theta0: float, z00: float, beta_val: float, loss_kind: str):
        t = torch.linspace(0.0, 1.0, N, dtype = self.dtype, device = self.device)
        beta = torch.tensor(beta_val, dtype=self.dtype, device=self.device)
        total_time = (t[-1] - t[0]).item()
        
        func = ConstantParameterForcing(theta0).to(self.device)
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
        err_theta = abs(grad_theta - (-total_time * dL_dzT))
        return err_z0, err_theta
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for uniform L1 discrete-sensitivity tests")
    def test_sensitivity_errors_decrease_with_grid_refinement(self):
        N_vals = [32, 64, 128, 256, 512]
        theta0 = 1.1
        z00 = -0.2
        beta_val = 0.6
        loss_kinds = ["linear", "quadratic", "cubic"]

        for loss_kind in loss_kinds:
            print("-" * 60)
            print(f"Loss kind: {loss_kind}")
            print("-" * 60)
            print()
            errs_z0 = []
            errs_theta = []
            for i in range(len(N_vals)):
                N = N_vals[i]
                e_z0, e_th = self._sensitivity_errors_for_grid(N, theta0, z00, beta_val, loss_kind)
                errs_z0.append(e_z0)
                errs_theta.append(e_th)
                if i > 0 and errs_z0[-1] > 0 and errs_theta[-1] > 0:
                    rate_z0 = math.log(errs_z0[-2] / errs_z0[-1]) / math.log(N / N_vals[i - 1])
                    rate_theta = math.log(errs_theta[-2] / errs_theta[-1]) / math.log(N / N_vals[i - 1])
                else:
                    rate_z0 = float('nan')
                    rate_theta = float('nan')
                print(f'N = {N}: err_z0 = {e_z0:.2e}, err_theta = {e_th:.2e}, rate_z0 = {rate_z0:.2e}, rate_theta = {rate_theta:.2e}')

            improve_z0 = sum(1 for i in range(1, len(errs_z0)) if errs_z0[i] <= errs_z0[i - 1])
            improve_theta = sum(1 for i in range(1, len(errs_theta)) if errs_theta[i] <= errs_theta[i - 1])

            self.assertGreaterEqual(
                improve_z0,
                3,
                f"z0 sensitivity error did not decrease sufficiently for {loss_kind}: {errs_z0}",
            )
            self.assertGreaterEqual(
                improve_theta,
                3,
                f"theta sensitivity error did not decrease sufficiently for {loss_kind}: {errs_theta}",
            )
        
    
if __name__ == "__main__":
    unittest.main(verbosity=2)