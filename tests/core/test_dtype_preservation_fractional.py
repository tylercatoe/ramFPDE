"""
Test dtype preservatio for the fractional setting without autocast.

This test verififes that rampFDE correctly preserves data types (float32, float64, bfloat16)
throughout fODE computations when autocast is not used. It ensures that all intermediates
maintain the expected precision.
"""

import unittest
import torch
import torch.nn as nn
import random
import numpy as np
from rampde import odeint

class SimpleDtypeCheckingFODE(nn.Module):
    """fODE that checks dtype of all intermediates and outputs"""

    def __init__(self, dim = 10, target_dtype = torch.float32, seed=None):
        super().__init__()
        self.dim = dim
        self.target_dtype = target_dtype
        if seed is not None:
            torch.manual_seed(seed)

        # Initialize weights in target dtype
        self.W1 = nn.Parameter(torch.randn(64, dim, dtype=target_dtype) * 0.1)
        self.W2 = nn.Parameter(torch.randn(dim, 64, dtype=target_dtype) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(64, dtype=target_dtype))
        self.b2 = nn.Parameter(torch.zeros(dim, dtype=target_dtype))

    def forward(self, t: torch.Tensor, z: torch.Tensor):
        # Check input dtype
        assert t.dtype == self.target_dtype, f"t has dtype {t.dtype}, expected {self.target_dtype}"
        assert z.dtype == self.target_dtype, f"z has dtype {z.dtype}, expected {self.target_dtype}, device {z.device}"

        # First layer 
        h = torch.matmul(z, self.W1.t()) + self.b1
        assert h.dtype == self.target_dtype, f"hidden layer has dtype {h.dtype}, expected {self.target_dtype}"

        # ReLU
        h = torch.relu(h)
        assert h.dtype == self.target_dtype, f"after ReLU has dtype {h.dtype}, expected {self.target_dtype}"

        # Second layer
        out = torch.matmul(h, self.W2.t()) + self.b2
        assert out.dtype == self.target_dtype, f"output has dtype {out.dtype}, expected {self.target_dtype}"

        return out
    
class TestDtypePreservationFractional(unittest.TestCase):

    def setUp(self):
        self.seed = 42
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _test_dtype_preservation(self, dtype, device='cpu'):

        # Skip if dtype not supported on device
        if device == 'cuda' and not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if dtype == torch.bfloat16 and device == 'cuda' and not torch.cuda.is_bf16_supported():
            self.skipTest("bfloat16 not supported on this GPU")

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        
        # Create RHS func with target dtype 
        func = SimpleDtypeCheckingFODE(dim = 10, target_dtype=dtype, seed=self.seed).to(device)

        # create inputs with target dtype
        z0 = torch.randn(10, dtype=dtype, device=device)
        t = torch.linspace(0, 1, steps = 10, dtype = dtype, device = device)
        beta = torch.tensor(0.5, dtype=dtype, device=device)

        # Run solver - should not raise any dtype assertion errors
        zt = odeint(func, z0, t, beta=beta, method='l1', loss_scaler=False)

        # Check output dtype
        self.assertEqual(zt.dtype, dtype, f"Output zt has dtype {zt.dtype}, expected {dtype}")

    def test_float32_cpu(self):
        self._test_dtype_preservation(torch.float32, device='cpu')
    
    def test_float64_cpu(self):
        self._test_dtype_preservation(torch.float64, device='cpu')
    
    def test_bfloat16_cpu(self):
        self._test_dtype_preservation(torch.bfloat16, device='cpu')

    def test_float32_cuda(self):
        self._test_dtype_preservation(torch.float32, device='cuda')

    def test_float64_cuda(self):
        self._test_dtype_preservation(torch.float64, device='cuda')

    def test_bfloat16_cuda(self):
        self._test_dtype_preservation(torch.bfloat16, device='cuda')

    def test_float64_gradients(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for gradient test")
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        func = SimpleDtypeCheckingFODE(dim = 10, target_dtype=torch.float64, seed=self.seed).cuda()
        z0 = torch.randn(10, dtype=torch.float64, device='cuda', requires_grad=True)
        t = torch.linspace(0, 1, steps = 10, dtype=torch.float64, device='cuda')
        beta = torch.tensor(0.5, dtype=torch.float64, device='cuda')

        soln = odeint(func, z0, t, beta=beta, method = 'l1', loss_scaler=False)
        loss = soln[-1].sum()
        loss.backward()

        # Check gradients dtype
        self.assertEqual(z0.grad.dtype, torch.float64, f"Gradient of z0 has dtype {z0.grad.dtype}, expected torch.float64")
        self.assertEqual(func.W1.grad.dtype, torch.float64, f"Gradient of W1 has dtype {func.W1.grad.dtype}, expected torch.float64")
        self.assertEqual(func.W2.grad.dtype, torch.float64, f"Gradient of W2 has dtype {func.W2.grad.dtype}, expected torch.float64")
        
    def test_float16_with_dynamic_scaling(self):
        """Test float16 with DynamicScaler"""

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for float16 test")
        
        from rampde.loss_scalers import DynamicScaler

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        func = SimpleDtypeCheckingFODE(dim = 10, target_dtype=torch.float16, seed=self.seed).cuda()

        z0 = torch.randn(10, dtype=torch.float16, device = 'cuda', requires_grad=True)
        t = torch.linspace(0, 1, steps = 10, dtype=torch.float16, device = 'cuda')
        beta = torch.tensor(0.5, dtype=torch.float16, device = 'cuda')

        scaler = DynamicScaler(dtype_low = torch.float16)

        # Run solver with dynamic scaler 
        soln = odeint(func, z0, t, beta=beta, method='l1', loss_scaler=scaler)
        
        # Check output dtype
        self.assertEqual(soln.dtype, torch.float16, f"Output zt has dtype {soln.dtype}, expected torch.float16")

        # Compute gradients to trigger scaler usage
        loss = soln[-1].sum()
        loss.backward()

        # Check gradients computed 
        self.assertIsNotNone(z0.grad, "Gradient of z0 is None, not computed")
        self.assertEqual(z0.grad.dtype, torch.float16, f"Gradient of z0 has dtype {z0.grad.dtype}, expected torch.float16")
        self.assertIsNotNone(func.W1.grad, "Gradient of W1 is None, not computed")
        self.assertEqual(func.W1.grad.dtype, torch.float16, f"Gradient of W1 has dtype {func.W1.grad.dtype}, expected torch.float16")

    def test_float16_no_scaler(self):
        """Test float16 without scaler """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for float16 test")

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        func = SimpleDtypeCheckingFODE(dim = 10, target_dtype=torch.float16, seed=self.seed).cuda()

        z0 = torch.randn(10, dtype=torch.float16, device = 'cuda', requires_grad=True)
        t = torch.linspace(0, 1, steps = 10, dtype=torch.float16, device = 'cuda')
        beta = torch.tensor(0.5, dtype=torch.float16, device = 'cuda')

        # Run solver without scaler - may produce non-finite values but should preserve dtype
        soln = odeint(func, z0, t, beta=beta, method='l1', loss_scaler=False)

        # Check output dtype
        self.assertEqual(soln.dtype, torch.float16, f"Output zt has dtype {soln.dtype}, expected torch.float16")

        # Compute gradients to trigger scaler usage
        loss = soln[-1].sum()
        loss.backward()

        # Check gradients computed 
        self.assertIsNotNone(z0.grad, "Gradient of z0 is None, not computed")
        self.assertEqual(z0.grad.dtype, torch.float16, f"Gradient of z0 has dtype {z0.grad.dtype}, expected torch.float16")
        self.assertIsNotNone(func.W1.grad, "Gradient of W1 is None, not computed")
        self.assertEqual(func.W1.grad.dtype, torch.float16, f"Gradient of W1 has dtype {func.W1.grad.dtype}, expected torch.float16")

    def test_solver_selection(self):
        """Test that correct solver varients are selected based on dtype and scaler."""

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for solver selection test")

        from rampde.odeint import _select_ode_solver
        from rampde.loss_scalers import DynamicScaler
        from rampde.fixed_grid_unscaled_uniform import FixedGridODESolverUnscaledUniform
        from rampde.fixed_grid_dynamic_uniform import FixedGridODESolverDynamicUniform
        # from rampde.fixed_grid_unscaled_safe_uniform import FixedGridODESolverUnscaledSafeUniform --- IGNORE ---

        # Test float64
        solver, _ = _select_ode_solver(None, torch.float64)
        self.assertEqual(solver, FixedGridODESolverUnscaledUniform, "float64 should use unscaled solver")

        # Test float32
        solver, _ = _select_ode_solver(None, torch.float32)
        self.assertEqual(solver, FixedGridODESolverUnscaledUniform, "float32 should use unscaled solver")

        # Test bfloat16
        solver, _ = _select_ode_solver(None, torch.bfloat16)
        self.assertEqual(solver, FixedGridODESolverUnscaledUniform, "bfloat16 should use unscaled solver")

        # Test float16 with None
        solver, scaler = _select_ode_solver(None, torch.float16)
        self.assertEqual(solver, FixedGridODESolverDynamicUniform, "float16 with None scaler should use dynamic solver")
    
        # Test float16 with explicit False
        #solver, scaler = _select_ode_solver(False, torch.float16)
        #self.assertEqual(solver, FixedGridODESolveUnscaledUniformSafe, "float16 with False should be unscaled safe")

        # Test float16 with DynamicScaler
        scaler = DynamicScaler(
            dtype_low=torch.float16,
            target_factor=128.0,
            increase_factor=1.0,
            decrease_factor=0.125,
            max_attempts=150,
            verbose=False,
        )
        solver, _ = _select_ode_solver(scaler, torch.float16)
        self.assertEqual(solver, FixedGridODESolverDynamicUniform, "float16 with DynamicScaler should use dynamic solver")

if __name__ == '__main__':
    unittest.main()



