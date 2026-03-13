"""
Correctness tests for the uniform-grid L1 forward pass in
rampde/fixed_grid_base_uniform.py.

The fractional Caputo IVP

    D^beta z = f(t, z),   z(0) = z0,   beta in (0, 1]

is reformulated as the Volterra integral equation

    z(t) = z0 + (1/Gamma(beta)) * int_0^t (t-s)^{beta-1} f(s, z(s)) ds

and discretised with a left-rectangle (piecewise-constant) quadrature on a
uniform grid of step h = T/(N-1):

    z_{k+1} = z0 + (1/Gamma(beta)) * sum_{j=0}^{k} mu_{j,k+1} * f(t_j, z_j)

    mu_{j,k+1} = h^beta / beta * [(k+1-j)^beta - (k-j)^beta]   (j = 0,...,k-1)
    mu_{k,k+1} = h^beta / beta

Known analytical solutions used for verification
-------------------------------------------------
1. Constant forcing  f(t,z) = c,  z(0) = 0
       Exact: z(t) = c * t^beta / Gamma(beta + 1)
   The quadrature weights telescope algebraically → scheme is *exact* for
   constant f (up to floating-point rounding).

2. Polynomial forcing  f(t) = [2/Gamma(3-beta)] * t^{2-beta},  z(0) = 0
       Exact: z(t) = t^2
   Uses the Caputo identity  D^beta[t^n] = Gamma(n+1)/Gamma(n+1-beta) * t^{n-beta}.

3. beta = 1  (reduces to the standard ODE d z/dt = f):
       f(t,z) = -z,  z(0) = 1  =>  z(t) = e^{-t}
   The L1 scheme degenerates to the forward-Euler method.
"""

import math
import sys
import os
import unittest
import warnings

import torch
import torch.nn as nn
from math import gamma as gamma_fn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from rampde.fixed_grid_base_uniform import FixedGridODESolverBase
from rampde.increment import L1


# ---------------------------------------------------------------------------
# Pure-Python reference: L1 uniform scheme (no torch.autograd.Function)
# ---------------------------------------------------------------------------

def reference_l1_uniform(func, z0, t_vals, beta):
    """
    Reference implementation of the uniform-grid L1 scheme.

    Args:
        func:    callable f(t: float, z: float) -> float
        z0:      initial condition (float)
        t_vals:  list of N uniformly-spaced time points
        beta:    fractional order in (0, 1]

    Returns:
        list of N solution values z[k] at each t_vals[k]
    """
    N = len(t_vals)
    h = t_vals[1] - t_vals[0]
    b = beta
    g = gamma_fn(b)

    f = [0.0] * N
    z = [0.0] * N
    z[0] = z0

    for k in range(N - 1):
        acc = 0.0
        for j in range(k):
            mu = h**b / b * ((k + 1 - j)**b - (k - j)**b)
            acc += mu * f[j]
        f[k] = func(t_vals[k], z[k])
        z[k + 1] = z0 + (1.0 / g) * (acc + (h**b / b) * f[k])

    return z


# ---------------------------------------------------------------------------
# ODE function nn.Modules
# ---------------------------------------------------------------------------

class ConstantForcing(nn.Module):
    """f(t, z) = c  (constant, independent of t and z)."""

    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.full_like(z, self.c)


class PolyForcing(nn.Module):
    """f(t, z) = coeff * t^exponent  (independent of z)."""

    def __init__(self, coeff: float, exponent: float):
        super().__init__()
        self.coeff = coeff
        self.exponent = exponent

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        tv = t.item() if t.dim() == 0 else float(t)
        val = self.coeff * tv**self.exponent if tv > 0.0 else 0.0
        return torch.full_like(z, val)


class LinearDecay(nn.Module):
    """f(t, z) = -z  (linear decay; gives z(t) = exp(-t) for beta=1)."""

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return -z


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _solver_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _run_forward(ode_func, z0, t, beta):
    """Run FixedGridODESolverBase forward pass on the input device."""
    params = list(ode_func.parameters())
    ode_func = ode_func.to(z0.device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress CUDA-autocast warnings on CPU
        with torch.no_grad():
            zt = FixedGridODESolverBase.apply(
                L1(), ode_func, z0, beta, t, None, *params
            )
    return zt   # shape [N, *z0.shape]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestL1UniformForward(unittest.TestCase):
    """Forward-pass correctness tests for the uniform-grid L1 scheme."""

    # ------------------------------------------------------------------
    # 1. Constant forcing  f = c,  z(0) = 0
    #    Exact: z(t) = c * t^beta / Gamma(beta + 1)
    #
    #    Proof: the quadrature weights telescope:
    #      sum_{j=0}^{k-1} [(k+1-j)^b - (k-j)^b] = (k+1)^b - 1
    #    so the total weight is h^b/b * (k+1)^b = t_{k+1}^b / b.
    #    Dividing by Gamma(b) gives t_{k+1}^b / Gamma(b+1).
    #    => scheme is algebraically exact for constant f.
    # ------------------------------------------------------------------
    def test_constant_forcing_exact(self):
        """
        For constant f=1, z(0)=0 the L1 scheme is algebraically exact:
            z_num(t) == t^beta / Gamma(beta+1)
        to floating-point precision.
        """
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

                self.assertLess(
                    max_err, 1e-8,
                    f"beta={beta_val}: constant-forcing algebraic exactness "
                    f"violated, max_err={max_err:.3e}"
                )

    # ------------------------------------------------------------------
    # 2. Polynomial manufactured solution  z(t) = t^2
    #    Forcing: f(t) = Gamma(3)/Gamma(3-beta) * t^{2-beta}
    #              = 2/Gamma(3-beta) * t^{2-beta}
    #    (Caputo identity: D^beta[t^2] = 2/Gamma(3-beta) * t^{2-beta})
    # ------------------------------------------------------------------
    def test_polynomial_manufactured_solution(self):
        """
        Manufactured-solution test using the Caputo identity
            D^beta(t^2) = 2 / Gamma(3-beta) * t^{2-beta}.
        Exact solution: z(t) = t^2.  Tests that the scheme converges to
        the exact solution (error O(h) for smooth f).
        """
        for beta_val in [0.5, 0.75]:
            with self.subTest(beta=beta_val):
                device = _solver_device()
                N, T = 500, 1.0
                t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
                z0 = torch.zeros(1, dtype=torch.float64, device=device)
                beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

                coeff = 2.0 / gamma_fn(3.0 - beta_val)
                exponent = 2.0 - beta_val
                ode_func = PolyForcing(coeff, exponent)

                zt = _run_forward(ode_func, z0, t, beta)

                exact = (t ** 2.0).unsqueeze(1)
                max_err = (zt - exact).abs().max().item()

                self.assertLess(
                    max_err, 5e-2,
                    f"beta={beta_val}: polynomial manufactured-solution test "
                    f"failed, max_err={max_err:.3e}"
                )

    # ------------------------------------------------------------------
    # 3. beta = 1  =>  L1 scheme degenerates to Euler's method
    #    f(t, z) = -z,  z(0) = 1  =>  z(t) = exp(-t)
    # ------------------------------------------------------------------
    def test_beta_one_is_euler_method(self):
        """
        When beta=1, D^1 z = dz/dt (standard ODE) and the L1 scheme
        is the forward-Euler method.  f=-z, z(0)=1 => z=exp(-t).
        """
        device = _solver_device()
        N, T = 2000, 1.0
        t = torch.linspace(0, T, N, dtype=torch.float64, device=device)
        z0 = torch.ones(1, dtype=torch.float64, device=device)
        beta = torch.tensor(1.0, dtype=torch.float64, device=device)

        zt = _run_forward(LinearDecay(), z0, t, beta)

        exact = torch.exp(-t).unsqueeze(1)
        max_err = (zt - exact).abs().max().item()

        self.assertLess(
            max_err, 1e-2,
            f"beta=1 (Euler limit): max_err={max_err:.3e}"
        )

    # ------------------------------------------------------------------
    # 4. Convergence: error at t=T decreases as N increases
    #    (polynomial forcing, beta=0.75)
    # ------------------------------------------------------------------
    def test_convergence_under_refinement(self):
        """
        For the polynomial manufactured solution with beta=0.75, the
        error at t=T should decrease monotonically as the grid is refined.
        """
        beta_val, T = 0.75, 1.0
        exact_T = T ** 2  # exact final value
        device = _solver_device()

        coeff   = 2.0 / gamma_fn(3.0 - beta_val)
        exp_val = 2.0 - beta_val
        beta    = torch.tensor(beta_val, dtype=torch.float64, device=device)

        errors, grid_sizes = [], [50, 100, 200, 400, 800]
        for N in grid_sizes:
            t   = torch.linspace(0, T, N, dtype=torch.float64, device=device)
            z0  = torch.zeros(1, dtype=torch.float64, device=device)
            zt  = _run_forward(PolyForcing(coeff, exp_val), z0, t, beta)
            err = abs(zt[-1, 0].item() - exact_T)
            errors.append(err)

        # Print convergence table for manual inspection
        print("\n  Convergence table (beta=0.75, f ~ t^{1.25}, exact z=t^2):")
        print(f"  {'N':>6}  {'h':>10}  {'error':>12}  {'ratio':>8}")
        for i, (N, err) in enumerate(zip(grid_sizes, errors)):
            h     = T / (N - 1)
            ratio = errors[i - 1] / err if i > 0 else float("nan")
            print(f"  {N:>6}  {h:>10.5f}  {err:>12.3e}  {ratio:>8.2f}")

        for i in range(len(errors) - 1):
            self.assertLess(
                errors[i + 1], errors[i],
                f"Error did not decrease: N={grid_sizes[i]} "
                f"{errors[i]:.3e} -> N={grid_sizes[i+1]} {errors[i+1]:.3e}"
            )

    # ------------------------------------------------------------------
    # 5. Matches pure-Python reference implementation (bit-for-bit)
    # ------------------------------------------------------------------
    def test_matches_pure_python_reference(self):
        """
        The solver output in float64 must match the pure-Python
        reference implementation to 10 decimal places.
        Covers both constant and polynomial forcing.
        """
        beta_val, N, T = 0.6, 80, 1.0
        device = _solver_device()
        t_list = [i * T / (N - 1) for i in range(N)]

        for label, py_func, torch_func in [
            ("constant f=1", lambda t, z: 1.0,        ConstantForcing(1.0)),
            ("poly f=D^b t^2", lambda t, z: (2.0 / gamma_fn(3.0 - beta_val)) * t ** (2.0 - beta_val) if t > 0.0 else 0.0,
             PolyForcing(2.0 / gamma_fn(3.0 - beta_val), 2.0 - beta_val)),
        ]:
            with self.subTest(forcing=label):
                ref = reference_l1_uniform(py_func, 0.0, t_list, beta_val)

                t    = torch.tensor(t_list, dtype=torch.float64, device=device)
                z0   = torch.zeros(1, dtype=torch.float64, device=device)
                beta = torch.tensor(beta_val, dtype=torch.float64, device=device)
                zt   = _run_forward(torch_func, z0, t, beta)

                for i in range(N):
                    self.assertAlmostEqual(
                        zt[i, 0].item(), ref[i], places=10,
                        msg=f"[{label}] Mismatch at i={i}: "
                            f"torch={zt[i,0].item():.14f}, ref={ref[i]:.14f}"
                    )

    # ------------------------------------------------------------------
    # 6. Initial condition is preserved exactly
    # ------------------------------------------------------------------
    def test_initial_condition_preserved(self):
        """z_num[0] must equal z0 regardless of f, beta, or N."""
        for beta_val in [0.3, 0.7, 1.0]:
            with self.subTest(beta=beta_val):
                device = _solver_device()
                z0_val = 3.14159
                z0   = torch.tensor([z0_val], dtype=torch.float64, device=device)
                t    = torch.linspace(0, 1, 50, dtype=torch.float64, device=device)
                beta = torch.tensor(beta_val, dtype=torch.float64, device=device)

                zt = _run_forward(LinearDecay(), z0, t, beta)

                self.assertAlmostEqual(
                    zt[0, 0].item(), z0_val, places=14,
                    msg=f"Initial condition not preserved for beta={beta_val}"
                )

    def test_output_stays_on_input_device(self):
        """The forward pass should return its output on the same device as z0."""
        device = _solver_device()
        t = torch.linspace(0, 1, 32, dtype=torch.float64, device=device)
        z0 = torch.zeros(1, dtype=torch.float64, device=device)
        beta = torch.tensor(0.5, dtype=torch.float64, device=device)

        zt = _run_forward(ConstantForcing(1.0), z0, t, beta)

        self.assertEqual(zt.device.type, device.type)


if __name__ == "__main__":
    unittest.main(verbosity=2)
