"""
Motivated by the forward pass analysis of the mixed-precision method, this test
investiages the potential accuracy gain of the mixed-precision method compared 
to stricly using the lower precision method. 

We use the same manufactured ODE examples as in test_forward_uniform.py. 
"""

import os, sys
import unittest
import torch
import torch.nn as nn
import warnings
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
        warnings.filterwarnings("ignore", category=UserWarning, module="rampde")
        with torch.no_grad():
            zt = FixedGridODESolverBase.apply(L1(), ode_func, z0, beta, t, None, *params)
        
    return zt[-1]


class TestPrecisionGain(unittest.TestCase):
    def test_constant_forcing(self):
        print()
        print("-" * 60 + "Testing Constant Forcing ODE" + "-" * 60)
        print()

        device = _solver_device()
        N, T = 100, torch.tensor(torch.pi, dtype = torch.float64, device=device)  
        c = 1.0
        ode_func = ConstantForcing(c)        
        t = torch.linspace(0.0, T, N, device=device)
        z0 = torch.zeros(1, device=device)

        for beta_val in [0.3, 0.5, 0.7, 0.9, 1.0]:
            beta = beta_val
            exact_final = torch.full_like(z0, c * (t[-1] ** beta) / gamma_fn(beta + 1)).to(torch.float64)

            with self.subTest(beta=beta_val):
                print(" " * 5 + f"Testing beta = {beta_val}")
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                    z_final_mp = _run_forward(ode_func, z0.to(torch.float32), t, beta)
                z_final_lp = _run_forward(ode_func, z0.to(torch.float16), t, beta)
                mp_error = (z_final_mp.to(torch.float64) - torch.as_tensor(exact_final, device=device, dtype=torch.float64)).abs()
                lp_error = (z_final_lp.to(torch.float64) - torch.as_tensor(exact_final, device=device, dtype=torch.float64)).abs()
                print(" " * 10 + f"MP final dtype: {z_final_mp.dtype}, LP final dtype: {z_final_lp.dtype}")
                print(" " * 10 + f"Constant Forcing - MP Error: {mp_error.item():.6e}, LP Error: {lp_error.item():.6e}")
                print()
                self.assertGreater(lp_error.item() + 1e-10, mp_error.item(), f"Expected MP error to be less than LP error for beta={beta}, but got MP error {mp_error.item():.6e} and LP error {lp_error.item():.6e}")

        
    def test_poly_forcing(self):
        print()
        print("-" * 60 + "Testing Polynomial Forcing ODE" + "-" * 60)
        print()

        device = _solver_device()
        N, T = 100, torch.tensor(torch.pi, dtype = torch.float64, device=device)      
        t = torch.linspace(0.0, T, N, device=device)
        z0 = torch.zeros(1, device=device)

        for beta_val in [0.3, 0.5, 0.7, 0.9, 1.0]:
            beta = beta_val
            exact_final = T ** 2
            coeff = 2.0 / gamma_fn(3.0 - beta)
            exponent = 2.0 - beta_val
            ode_func = PolyForcing(coeff, exponent)

            with self.subTest(beta=beta_val):
                print(" " * 5 + f"Testing beta = {beta_val}")
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                    z_final_mp = _run_forward(ode_func, z0.to(torch.float32), t, beta)
                z_final_lp = _run_forward(ode_func, z0.to(torch.float16), t, beta)
                mp_error = (z_final_mp.to(torch.float64) - torch.as_tensor(exact_final, device=device, dtype=torch.float64)).abs()
                lp_error = (z_final_lp.to(torch.float64) - torch.as_tensor(exact_final, device=device, dtype=torch.float64)).abs()
                print(" " * 10 + f"MP final dtype: {z_final_mp.dtype}, LP final dtype: {z_final_lp.dtype}")
                print(" " * 10 + f"Polynomial Forcing - MP Error: {mp_error.item():.6e}, LP Error: {lp_error.item():.6e}")
                print()
                self.assertGreater(lp_error.item() + 1e-10, mp_error.item(), f"Expected MP error to be less than LP error for beta={beta}, but got MP error {mp_error.item():.6e} and LP error {lp_error.item():.6e}")



    def test_linear_decay(self):
        print()
        print("-" * 60 + "Testing Linear Decay ODE" + "-" * 60)
        print()

        device = _solver_device()
        N, T = 100, torch.tensor(torch.pi, dtype = torch.float64, device=device)
        t = torch.linspace(0.0, T, N, device=device)
        z0 = torch.ones(1, device=device)

        beta = 1.0
        
        exact_final = torch.exp(-T)
        ode_func = LinearDecay()

        with self.subTest(beta=beta):
            print(" " * 5 + f"Testing beta = {beta}")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                z_final_mp = _run_forward(ode_func, z0.to(torch.float32), t, beta)
            z_final_lp = _run_forward(ode_func, z0.to(torch.float16), t, beta)
            mp_error = (z_final_mp.to(torch.float64) - torch.as_tensor(exact_final, device=device, dtype=torch.float64)).abs()
            lp_error = (z_final_lp.to(torch.float64) - torch.as_tensor(exact_final, device=device, dtype=torch.float64)).abs()
            print(" " * 10 + f"MP final dtype: {z_final_mp.dtype}, LP final dtype: {z_final_lp.dtype}")
            print(" " * 10 + f"Linear Decay - MP Error: {mp_error.item():.6e}, LP Error: {lp_error.item():.6e}")
            print()
            self.assertGreater((lp_error.item() + 1e-10), mp_error.item(), f"Expected MP error to be less than LP error for beta={beta}, but got MP error {mp_error.item():.6e} and LP error {lp_error.item():.6e}")


if __name__ == "__main__":
    unittest.main(verbosity=1)
