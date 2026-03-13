"""
Correctness tests for the Adams-Bashforth-Moulton Predictor-Corrector L1 forward
pass in rampde/fixed_grid_base.py.

The fractional Caputo IVP
    D^beta z = f(t, z),  z(0) = z0,  beta in (0, 1]
reduces to the classical ODE  dz/dt = f(t,z)  when beta == 1.

Choosing f(t, z) = z and z(0) = 1 gives the exact solution z(t) = e^t.
A correct ABM corrector should converge at rate O(h^{1+beta}) to this solution.
For beta == 1 the expected rate is ~2 (second-order).

Two bugs in the current corrector (see detailed notes below) cause the solver to
*diverge* as the grid is refined.  These tests document expected behaviour and
can be used to verify a correct implementation.

Known corrector issues (as of the time this test was written)
-------------------------------------------------------------
Bug 1 – Wrong j=0 weight:
    The code computes
        a_{0,k+1} = (t[k+1] - t[0])^beta / beta
    which grows linearly with k.  The correct piecewise-linear ABM weight is
        a_{0,k+1} = C_{0,left}
    where, with A = t[k+1]-t[0] and B = t[k+1]-t[1]:
        C_{0,left} = (beta*A^{beta+1} - (beta+1)*B*A^beta + B^{beta+1})
                     / ((t[1]-t[0]) * beta * (beta+1))
    For beta=1 this equals h/2 (trapezoidal weight), not h*(k+1) (code).

Bug 2 – Missing j=k history term:
    The corrector loop runs for j in range(1, k), skipping j=k entirely.
    The weight a_{k,k+1} = C_{k-1,right} + C_{k,left}|_{B=0} is never added.
    For beta=1 this weight equals h (interior trapezoidal weight), so every
    step after the first discards one full h*f(t_k, y_k) contribution.

Correct corrector structure (for step k):
    zc = z0
    + (1/Gamma(beta)) * sum_{j=0}^{k}  a_{j,k+1} * f(t_j, z_j)   # history
    + (1/Gamma(beta)) * a_{k+1,k+1}   * f(t_{k+1}, z_P)           # predictor
where the weights are derived from the piecewise-linear interpolant:
    j=0:      a_{0,k+1} = C_{0,left}(A=t[k+1]-t[0], B=t[k+1]-t[1])
    j=1..k-1: a_{j,k+1} = C_{j-1,right} + C_{j,left}   (already correct in code)
    j=k:      a_{k,k+1} = C_{k-1,right} + C_{k,left}|_{B=0}       (MISSING)
    j=k+1:    a_{k+1,k+1} = C_{k,right}|_{B=0}                     (correct in code)
with
    C_{m,left}  = (beta*A^{b+1} - (b+1)*B*A^b + B^{b+1}) / ((A-B)*b*(b+1))
    C_{m,right} = (A^{b+1} - (b+1)*A*B^b + beta*B^{b+1}) / ((A-B)*b*(b+1))
    A = t[k+1]-t[m],  B = t[k+1]-t[m+1],  A-B = t[m+1]-t[m]
"""

import math
import sys
import os
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from rampde import odeint


# ---------------------------------------------------------------------------
# Reference implementation of the corrected ABM-L1 corrector weights
# ---------------------------------------------------------------------------

def _C_left(A, B, A_minus_B, beta):
    """Piecewise-linear left weight: contribution of the left node of [t_m, t_{m+1}]."""
    return (beta * A**(beta+1) - (beta+1)*B*A**beta + B**(beta+1)) / (A_minus_B * beta * (beta+1))

def _C_right(A, B, A_minus_B, beta):
    """Piecewise-linear right weight: contribution of the right node of [t_m, t_{m+1}]."""
    return (A**(beta+1) - (beta+1)*A*B**beta + beta*B**(beta+1)) / (A_minus_B * beta * (beta+1))

def reference_abm_l1(func, z0_val, t_vals, beta_val):
    """
    Pure-Python reference implementation of the corrected ABM-L1 predictor-corrector.

    Implements the Volterra form:
        z(t_{k+1}) = z0 + (1/Gamma(beta)) * integral_0^{t_{k+1}} (t_{k+1}-s)^{beta-1} f(s,z(s)) ds

    using the Adams-Bashforth predictor (piecewise-constant f) and Adams-Moulton
    corrector (piecewise-linear f).

    Args:
        func:      callable f(t, z) -> float
        z0_val:    initial condition (float)
        t_vals:    list/array of time points
        beta_val:  fractional order (float, in (0, 1])

    Returns:
        list of z values at each t in t_vals
    """
    from math import gamma as gamma_fn
    N = len(t_vals)
    t = t_vals
    b = beta_val
    g = gamma_fn(b)

    zs = [0.0] * N
    fs = [0.0] * N
    zs[0] = z0_val

    for k in range(N - 1):
        z0 = z0_val

        # ---- Predictor (Adams-Bashforth, piecewise-constant) ----
        zp = z0
        for j in range(k):
            bj = (1/b) * ((t[k+1]-t[j])**b - (t[k+1]-t[j+1])**b)
            zp += (1/g) * bj * fs[j]
        j = k
        fs[k] = func(t[k], zs[k])
        bj = (1/b) * ((t[k+1]-t[j])**b - (t[k+1]-t[j+1])**b)
        zp += (1/g) * bj * fs[k]

        # ---- Corrector (Adams-Moulton, piecewise-linear) ----
        zc = z0

        # j=0: only C_{0,left}
        A = t[k+1] - t[0];  B = t[k+1] - t[1];  h01 = t[1] - t[0]
        a0 = _C_left(A, B, h01, b)
        zc += (1/g) * a0 * fs[0]

        # j=1,...,k-1: C_{j-1,right} + C_{j,left}
        for j in range(1, k):
            A_r = t[k+1]-t[j-1]; B_r = t[k+1]-t[j]; h_r = t[j]-t[j-1]
            A_l = t[k+1]-t[j];   B_l = t[k+1]-t[j+1]; h_l = t[j+1]-t[j]
            aj = _C_right(A_r, B_r, h_r, b) + _C_left(A_l, B_l, h_l, b)
            zc += (1/g) * aj * fs[j]

        # j=k (k>=1): C_{k-1,right} + C_{k,left}|_{B=0}
        if k >= 1:
            A_r = t[k+1]-t[k-1]; B_r = t[k+1]-t[k]; h_r = t[k]-t[k-1]
            c_r = _C_right(A_r, B_r, h_r, b)
            c_l = (t[k+1]-t[k])**b / (b + 1)          # C_{k,left} with B=0
            zc += (1/g) * (c_r + c_l) * fs[k]

        # j=k+1: predictor via C_{k,right}|_{B=0}
        a_pred = (t[k+1]-t[k])**b / (b * (b + 1))
        f_pred = func(t[k+1], zp)
        zs[k+1] = zc + (1/g) * a_pred * f_pred

    return zs


# ---------------------------------------------------------------------------
# ODE function:  dz/dt = z  ->  z(t) = exp(t)
# ---------------------------------------------------------------------------

class LinearGrowth(torch.nn.Module):
    """f(t, z) = z  =>  exact solution z(t) = e^t for z(0)=1."""
    def forward(self, t, z):
        return z


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_solver(N, t_end=1.0, beta=1.0, dtype=torch.float64):
    t = torch.linspace(0.0, t_end, N + 1, dtype=dtype)
    y0 = torch.ones(1, dtype=dtype)
    beta_t = torch.tensor(beta, dtype=dtype)
    func = LinearGrowth()
    yt = odeint(func, y0, t, method='l1', beta=beta_t, loss_scaler=False)
    return yt, t


def _run_reference(N, t_end=1.0, beta=1.0):
    t = [t_end * i / N for i in range(N + 1)]
    zs = reference_abm_l1(lambda t_val, z: z, 1.0, t, beta)
    return zs, t


class TestABML1ForwardPassBeta1(unittest.TestCase):
    """
    Tests for ABM-L1 forward pass with beta=1 against z(t)=e^t.

    These tests expose the two corrector bugs described in the module docstring.
    A correct implementation PASSES all tests; the current implementation FAILS.
    """

    # ------------------------------------------------------------------ #
    # 1.  Single-step accuracy (k=0)                                      #
    # ------------------------------------------------------------------ #
    def test_single_step_accuracy(self):
        """
        k=0: z(h) should approximate e^h.

        Exact ABM corrector at k=0 (beta=1, h=0.1):
            z_1 = z_0 + h/2 * f(0, z_0) + h/2 * f(h, z_P)
                = 1 + 0.05 + 0.05*(1+h) = 1.1050...
        Code (bug 1): uses a_{0,1}=h instead of h/2, giving 1.155.
        Expected error: < 1e-3 for h=0.1 (O(h^2) method).
        """
        h = 0.1
        yt, _ = _run_solver(N=1, t_end=h, beta=1.0)
        err = abs(yt[1].item() - math.exp(h))
        self.assertLess(
            err, 1e-3,
            msg=f"Single-step error {err:.4e} too large (expected < 1e-3). "
                f"Got z(h)={yt[1].item():.8f}, exact={math.exp(h):.8f}. "
                "Likely Bug 1: wrong a_{0,k+1} weight."
        )

    # ------------------------------------------------------------------ #
    # 2.  Multi-step solution stays bounded and close to e^t              #
    # ------------------------------------------------------------------ #
    def test_multistep_accuracy(self):
        """
        With N=200 steps on [0,1], the maximum pointwise error vs e^t
        should be small (< 0.01 for any reasonable O(h) or better method).
        """
        N = 200
        yt, t = _run_solver(N=N, beta=1.0)
        t_np = t.tolist()
        max_err = max(abs(yt[i].item() - math.exp(t_np[i])) for i in range(N + 1))
        self.assertLess(
            max_err, 0.01,
            msg=f"Max pointwise error {max_err:.4e} is too large for N={N}. "
                "The solver may be diverging (Bug 1 + Bug 2 compound)."
        )

    # ------------------------------------------------------------------ #
    # 3.  Convergence: error must DECREASE as N increases                 #
    # ------------------------------------------------------------------ #
    def test_convergence_monotone(self):
        """
        Doubling N must strictly reduce the error at t=1.
        If errors grow, the corrector is diverging (Bugs 1 & 2 combined).
        """
        errors = []
        for N in [10, 20, 40, 80]:
            yt, _ = _run_solver(N=N, beta=1.0)
            errors.append(abs(yt[-1].item() - math.e))

        for i in range(len(errors) - 1):
            self.assertLess(
                errors[i + 1], errors[i],
                msg=f"Error did not decrease when N doubled: "
                    f"err(N={10*2**i})={errors[i]:.3e}, "
                    f"err(N={10*2**(i+1)})={errors[i+1]:.3e}. "
                    "Solver is diverging instead of converging."
            )

    # ------------------------------------------------------------------ #
    # 4.  Convergence rate ~2 for beta=1                                  #
    # ------------------------------------------------------------------ #
    def test_convergence_rate_beta1(self):
        """
        The ABM method is order 1+beta = 2 for beta=1; observed rate
        from halving h should be in [1.5, 2.5].
        """
        errors = []
        for N in [50, 100, 200, 400]:
            yt, _ = _run_solver(N=N, beta=1.0)
            errors.append(abs(yt[-1].item() - math.e))

        # Measure rate from the last two refinements (asymptotic regime)
        if errors[-1] > 0 and errors[-2] > 0:
            rate = math.log(errors[-2] / errors[-1]) / math.log(2)
            self.assertGreater(
                rate, 1.5,
                msg=f"Convergence rate {rate:.2f} too low; expected ~2 for beta=1. "
                    "Bug 2 (missing j=k term) reduces the order."
            )
            self.assertLess(
                rate, 3.0,
                msg=f"Convergence rate {rate:.2f} unexpectedly high."
            )

    # ------------------------------------------------------------------ #
    # 5.  Reference implementation agrees with e^t                        #
    # ------------------------------------------------------------------ #
    def test_reference_implementation_accuracy(self):
        """
        The corrected reference implementation must produce small errors,
        confirming the fixed formulas are mathematically right.
        """
        for N in [50, 100, 200]:
            zs, t_vals = _run_reference(N=N, beta=1.0)
            err = abs(zs[-1] - math.e)
            self.assertLess(
                err, 1.0 / N,        # O(h) worst case; expect O(h^2)
                msg=f"Reference impl error {err:.4e} too large for N={N}."
            )

    # ------------------------------------------------------------------ #
    # 6.  Reference converges at rate ~2                                  #
    # ------------------------------------------------------------------ #
    def test_reference_convergence_rate(self):
        """
        The corrected reference formulas should achieve order ~2 for beta=1.
        """
        errors = []
        for N in [50, 100, 200, 400]:
            zs, _ = _run_reference(N=N, beta=1.0)
            errors.append(abs(zs[-1] - math.e))

        if errors[-1] > 0 and errors[-2] > 0:
            rate = math.log(errors[-2] / errors[-1]) / math.log(2)
            self.assertGreater(
                rate, 1.5,
                msg=f"Reference convergence rate {rate:.2f} too low; expected ~2."
            )


class TestABML1ForwardPassFractional(unittest.TestCase):
    """
    Sanity checks for beta < 1 using the reference implementation.

    For D^beta z = lambda * z, the exact solution is z(t) = E_beta(lambda * t^beta)
    where E_beta is the Mittag-Leffler function.  Here we only test that:
      - the reference solver converges as N increases, and
      - the observed rate is consistent with O(h^{1+beta}).
    """

    def _mittag_leffler(self, x, beta, n_terms=50):
        """Approximate E_beta(x) = sum_{k=0}^inf x^k / Gamma(k*beta + 1)."""
        result = 0.0
        for k in range(n_terms):
            result += x**k / math.gamma(k * beta + 1)
        return result

    def test_reference_fractional_convergence(self):
        """
        Reference solver converges for beta=0.5 on D^0.5 z = z, z(0)=1,
        exact solution z(t) = E_{0.5}(t^0.5).
        """
        beta = 0.5
        t_end = 0.5    # stay in a range where Mittag-Leffler sum converges well
        exact = self._mittag_leffler(t_end**beta, beta)

        errors = []
        for N in [50, 100, 200]:
            zs, _ = _run_reference(N=N, t_end=t_end, beta=beta)
            errors.append(abs(zs[-1] - exact))

        # Just require monotone convergence
        for i in range(len(errors) - 1):
            self.assertLess(
                errors[i + 1], errors[i],
                msg=f"Reference fractional solver not converging for beta={beta}. "
                    f"Errors: {errors}"
            )


# ---------------------------------------------------------------------------
# Diagnostic helper (not a test) – run directly for a convergence table
# ---------------------------------------------------------------------------

def print_convergence_table():
    """Print a convergence table for visual inspection."""
    print("\n" + "="*70)
    print("ABM-L1 Convergence Study  |  beta=1,  dz/dt=z,  z(0)=1")
    print("="*70)

    print("\n--- rampde solver (current) ---")
    print(f"{'N':>6}  {'y(1)':>14}  {'|y(1)-e|':>12}  {'rate':>8}")
    prev = None
    for N in [10, 20, 40, 80, 160, 320]:
        yt, _ = _run_solver(N, beta=1.0)
        err = abs(yt[-1].item() - math.e)
        rate = f"{math.log(prev/err)/math.log(2):.2f}" if prev and err > 0 else "---"
        print(f"{N:>6}  {yt[-1].item():>14.8f}  {err:>12.3e}  {rate:>8}")
        prev = err

    print("\n--- reference (corrected formulas) ---")
    print(f"{'N':>6}  {'y(1)':>14}  {'|y(1)-e|':>12}  {'rate':>8}")
    prev = None
    for N in [10, 20, 40, 80, 160, 320]:
        zs, _ = _run_reference(N, beta=1.0)
        err = abs(zs[-1] - math.e)
        rate = f"{math.log(prev/err)/math.log(2):.2f}" if prev and err > 0 else "---"
        print(f"{N:>6}  {zs[-1]:>14.8f}  {err:>12.3e}  {rate:>8}")
        prev = err

    print("\n--- k=0 single step detail (h=0.1) ---")
    yt, _ = _run_solver(N=1, t_end=0.1, beta=1.0)
    zs, _ = _run_reference(N=1, t_end=0.1, beta=1.0)
    exact_1 = math.exp(0.1)
    print(f"  rampde:    z(0.1) = {yt[1].item():.10f},  error = {abs(yt[1].item()-exact_1):.3e}")
    print(f"  reference: z(0.1) = {zs[1]:.10f},  error = {abs(zs[1]-exact_1):.3e}")
    print(f"  exact:     z(0.1) = {exact_1:.10f}")


if __name__ == "__main__":
    # Print the convergence tables first for visual inspection
    print_convergence_table()

    # Then run the formal unit tests
    print("\n" + "="*70)
    print("Running unit tests ...")
    print("="*70 + "\n")
    unittest.main(argv=[''], verbosity=2, exit=False)
