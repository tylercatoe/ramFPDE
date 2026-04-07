## Mixed Precision Training of Fractional Neural ODEs (MPT-FNDE), MNIST Test Script

import os, sys
job_id = os.environ.get('SLURM_JOB_ID', '')
import argparse, time, datetime, random, torch, csv, shutil, logging
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST
from torch.amp import autocast

# Add parent directory to path for common imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_runtime import (
    RunningAverageMeter,
    RunningMaximumMeter,
    setup_environment,
    get_precision_dtype, 
    determine_scaler,
    setup_experiment
)

def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser()
    #parser.add_argument('--tol', type=float, default=1e-3, help='Tolerance for FODE solver')
    parser.add_argument('--nepochs', type=int, default=160, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--test_batch_size', type=int, default=100, help='Batch size for testing')
    parser.add_argument('--beta', type=float, default=0.6, help='Beta parameter for L1 method')

    parser.add_argument('--precision', type=str, default='float32', choices=['float16', 'bfloat16', 'float32', 'tfloat32'], help='Precision for training')
    parser.add_argument('--method', type=str, default='l1', choices=['l1'], help='Integration method')
    parser.add_argument('--odeint', type=str, default='rampde', choices=['rampde'], help='ODE solver backend')
    parser.add_argument('--unstable', action='store_true', help='Use unstable ODE formulation (default: stable)')
    parser.add_argument('--no_grad_scaler', action='store_true', help='Disable GradScaler for float16')
    parser.add_argument('--no_dynamic_scaler', action='store_true', help='Disable DynamicScaler for rampde float16')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--test_freq', type=int, default=1, help='evaluate / log every N training steps')
    parser.add_argument('--width', type=int, default=64, help='Base channel width (default: 64)')
    parser.add_argument('--h', type = float, default = 0.1, help = 'Grid spacing')
    parser.add_argument('--T', type=float, default = 1.0, help ='Final time')

    parser.add_argument("--generate-plots", default=True, action="store_true", help="Whether to generate plots after training completes.")
    return parser



 
class ConcatLinear(nn.Module): 
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out, bias=bias)

    def forward(self, t, x):
        tt = torch.ones(x.size(0), 1, device = x.device, dtype = x.dtype) * t
        return self._layer(torch.cat([tt, x], dim = 1))


def norm(ch: int) -> nn.LayerNorm:
    return nn.LayerNorm(ch)

class ODEFunc_f(nn.Module):
    def __init__(self, ch, t_grid, act=nn.Tanh()):
        super().__init__()
        self.nfe = 0

        self.norm1 = norm(ch)
        self.act = act
        self.fc1 = ConcatLinear(ch, ch)
        self.norm2 = norm(ch)
        self.fc2 = ConcatLinear(ch, ch)
        self.norm3 = norm(ch)

    def forward(self, t: torch.Tensor, z0: torch.Tensor):
        self.nfe += 1

        z = self.act(self.norm1(z0))
        z = self.fc1(t, z)
        z = self.act(self.norm2(z))
        z = self.fc2(t, z)
        z = self.norm3(z)
        return z

class FODEBlock(nn.Module):
    def __init__(self, func, t_grid, beta = 0.6, loss_scaler = None, odeint_func = None):
        super().__init__()
        self.func = func
        self.register_buffer('t_grid', t_grid)
        self.beta = beta.item() if isinstance(beta, torch.Tensor) else beta
        self.loss_scaler = loss_scaler
        self.odeint_func = odeint_func

    def forward(self, z):
        # if not args.adjoint:
        #     out = self.odeint_func(self.func, z, beta = self.beta, t = args.T, step_size = args.h, method = 'corrector')
        #     return out
        # else:
        if self.loss_scaler is not None:
            out = self.odeint_func(self.func, z, self.t_grid, method = 'l1', beta = self.beta, loss_scaler = self.loss_scaler)
        else:
            out = self.odeint_func(self.func, z, self.t_grid, method = 'l1', beta = self.beta)
        return out[-1]
    
class MPNFODE_MNIST(nn.Module):
    def __init__(self, width, args, precision, odeint_func, ScalerClass, dynamic_scaler_enabled = False, grad_scaler_enabled = False):
        super().__init__()

        N = int(round(args.T / args.h))

        # Create t_grid on appropriate device
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        t_grid = torch.linspace(0, args.T, N + 1, device=device)
    
        # Set scaler
        if args.odeint == 'rampde' and dynamic_scaler_enabled and ScalerClass is not None:
            S1 = ScalerClass(precision)
        elif args.odeint == 'rampde' and args.precision == 'float16' and not dynamic_scaler_enabled:
            # Explicitly disable internal scaler when using external GradScaler  
            S1 = False
        else:
            S1 = None
        
        proj_in_layers = [
            nn.Flatten(),
            nn.Linear(784, width),
            nn.ReLU(inplace=True),
        ]
        # if not args.adjoint:
        #     from torchfde import fdeint as odeint_func
        
            
        feature_layers = [
            FODEBlock(ODEFunc_f(width, t_grid), t_grid, beta = args.beta, loss_scaler=S1, odeint_func=odeint_func),
        ]
        proj_out_layer = [
            norm(width),
            nn.ReLU(inplace=True),
            nn.Linear(width, 10),
        ]
        self.proj_in_layers = nn.Sequential(*proj_in_layers)
        self.feature_layers = nn.Sequential(*feature_layers)
        self.proj_out_layer = nn.Sequential(*proj_out_layer)

        self.model = nn.Sequential(*proj_in_layers, *feature_layers, *proj_out_layer)

    def forward(self, x):
        return self.model(x)
    

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed()
    np.random.seed(worker_seed)

def get_mnist_loaders(batch_size = 128, 
                      test_batch_size = 1000, 
                      seed = None):
    
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: x), 
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = MNIST(root='.data/mnist', train=True, download=True, transform=transform_train)
    test_dataset = MNIST(root='.data/mnist', train=False, download=True, transform=transform_test)
    train_eval_dataset = MNIST(root='.data/mnist', train=True, download=True, transform=transform_test)

    train_laoder = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers = 2, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=False
    )
    train_eval_loader = DataLoader(
        train_eval_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=False
    )

    return train_laoder, test_loader, train_eval_loader

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def accuracy(model, dataset_loader, device):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates, lr):
    initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for x,y in dataloader:
        x = x.to(device)
        logits = model(x)
        predictions = logits.argmax(dim=1).cpu()
        correct += (predictions == y).sum().item()
        total += y.size(0)
    model.train()
    return correct/total


@torch.no_grad()
def evaluate_inference_metrics(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device=device)
    start = time.time()
    correct = 0
    total = 0
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device=device) / (1024.0 ** 2)
    else:
        peak_mb = float("nan")
    elapsed = time.time() - start
    model.train()
    acc = correct / max(total, 1)
    return acc, elapsed, float(peak_mb)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def main():

    parser = create_parser()
    args = parser.parse_args()

    grad_scaler_enabled = not args.no_grad_scaler
    dynamic_scaler_enabled = not args.no_dynamic_scaler

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    makedirs('./results')
    makedirs('./logs')
    
    log_name = f"ode_mnist_{'adjoint'}"
    if job_id:
        log_name = f"{log_name}_{job_id}"
    log_path = os.path.join("logs", f"{log_name}.log")
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(args)

    
    logger.info("Using adjoint method for gradient computation.")
    args.odeint = 'rampde'

    odeint_func, DynamicScaler = setup_environment(args.odeint, base_dir)

    precision = get_precision_dtype(args.precision)
    loss_scaler, scaler_name, loss_scaler_for_odeint = determine_scaler(
        args.odeint,
        args.precision,
        grad_scaler_enabled,
        dynamic_scaler_enabled,
        DynamicScaler
    )

    result_dir, ckpt_path, folder_name, device, log_file = setup_experiment(
        args.results_dir,
        "ode_mnist",
        "mnist",
        args.precision,
        args.odeint,
        args.method,
        args.seed,
        args.gpu,
        scaler_name,
        extra_params={"width": args.width, "h": args.h, "T": args.T},
        args=args
    )

    try:
        model = MPNFODE_MNIST(
            args.width, 
            args, 
            precision,
            odeint_func,
            DynamicScaler,
            dynamic_scaler_enabled=dynamic_scaler_enabled,
            grad_scaler_enabled=grad_scaler_enabled
        ).to(device)

        logger.info(f"Model architecture:\n{model}")
        logger.info(f'Number of parameters: {count_parameters(model)}')

        criterion = nn.CrossEntropyLoss()
        train_loader, test_loader, train_eval_loader = get_mnist_loaders(args.batch_size, args.test_batch_size, args.seed)

        data_gen = inf_generator(train_loader)
        batches_per_epoch = len(train_loader)
        

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepochs*batches_per_epoch, eta_min=1e-4)

        best_acc = 0.0
        batch_time_meter = RunningAverageMeter()
        fwd_time_meter = RunningAverageMeter()
        bwd_time_meter = RunningAverageMeter()
        train_loss_meter = RunningAverageMeter()
        f_nfe_meter = RunningAverageMeter()
        b_nfe_meter = RunningAverageMeter()
        mem_meter = RunningMaximumMeter()
        train_start_time = time.time()
        


        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device)
        


        for iter in range(args.nepochs * batches_per_epoch):
            iter_start = time.perf_counter()

            model.train()
            
            optimizer.zero_grad()
            x, y = data_gen.__next__()
            x, y = x.to(device), y.to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            fwd_start = time.perf_counter()
            with autocast(device_type='cuda', dtype=precision):
                logits = model(x)
                loss = criterion(logits.float(), y)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            fwd_time = time.perf_counter() - fwd_start

            
            nfe_forward = model.feature_layers[0].func.nfe
            model.feature_layers[0].func.nfe = 0

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            bwd_start = time.perf_counter()
            
            # Handle backward pass with or without loss scaling
            if loss_scaler is not None:
                # Track loss scale before step
                old_scale = loss_scaler.get_scale()
                
                # Use gradient scaling for torchdiffeq with float16
                loss_scaler.scale(loss).backward()
                loss_scaler.step(optimizer)
                loss_scaler.update()
                
                # Track loss scale after step and log changes
                new_scale = loss_scaler.get_scale()
                if old_scale != new_scale:
                    print(f"Iteration {iter}: Loss scale changed from {old_scale} to {new_scale} (gradient overflow detected)")
                elif iter < 20 or iter % 100 == 0:  # Log scale periodically for first 20 iterations or every 100
                    print(f"Iteration {iter}: Loss scale = {new_scale} (no overflow)")
                
                # Only step scheduler if no overflow occurred (scale didn't change)
                if old_scale == new_scale:
                    scheduler.step()
                else:
                    print(f"Iteration {iter}: Skipping scheduler step due to gradient overflow")
            else:
                # Standard backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            bwd_time = time.perf_counter() - bwd_start

            peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024.0 ** 2) if torch.cuda.is_available() else float("nan")

            nfe_backward = model.feature_layers[0].func.nfe
            model.feature_layers[0].func.nfe = 0

            if not torch.isfinite(loss).all():
                print(f"Training stopped at iteration {iter}: Loss is {'NaN' if torch.isnan(loss).any() else 'infinite'}")
                print(f"Loss value: {loss.item()}")
                print("Saving current model state before stopping...")
                torch.save({
                    'state_dict': model.state_dict(), 
                    'args': args,
                    'iteration': iter,
                    'loss': loss.item()
                }, ckpt_path.replace('.pth', '_emergency_stop.pth'))
                return  # Exit the training function
            
            # Check for NaN gradients (outside timed zone)
            # Only stop training for NaN gradients if we're not using gradient scaling
            # When using GradScaler, NaN/inf gradients are expected and handled automatically
            if loss_scaler is None:
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"Training stopped at iteration {iter}: NaN/infinite gradient detected in parameter '{name}'")
                        print(f"Gradient stats - min: {param.grad.min().item()}, max: {param.grad.max().item()}")
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print("Saving current model state before stopping...")
                    torch.save({
                        'state_dict': model.state_dict(), 
                        'args': args,
                        'iteration': iter,
                        'loss': loss.item()
                    }, ckpt_path.replace('.pth', '_gradient_nan_stop.pth'))
                    return  # Exit the training function
            else:
                # When using gradient scaling, just log if we encounter NaN gradients
                # but don't stop training as GradScaler handles this automatically
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"NaN/inf gradients detected in '{name}' at iteration {iter} - GradScaler will handle this")
                        has_nan_grad = True
                        break
                
                # Also log if we have finite gradients for comparison
                if not has_nan_grad and (iter < 5 or iter % 50 == 0):
                    print(f"Iteration {iter}: All gradients are finite")

            fwd_time_meter.update(fwd_time)
            bwd_time_meter.update(bwd_time)
            train_loss_meter.update(loss.item())
            mem_meter.update(peak_memory)
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
            batch_time_meter.update(time.perf_counter() - iter_start)

            if iter % batches_per_epoch == 0:
                with torch.no_grad():
                    with autocast(device_type='cuda', dtype=precision):
                        train_acc = accuracy(model, train_eval_loader, device)
                        val_acc = accuracy(model, test_loader, device)
                        if val_acc > best_acc:
                            ckpt = {'state_dict': model.state_dict(), 'args': args}
                            torch.save(ckpt, ckpt_path)
                            torch.save(ckpt, os.path.join(result_dir, "model.pth"))
                            best_acc = val_acc
                        current_lr = optimizer.param_groups[0]['lr']
                        logger.info(
                            "Epoch {:04d} | LR {:.4f} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | Loss {:.6f} | "
                            "Train Acc {:.4f} | Test Acc {:.4f}".format(
                                iter // batches_per_epoch, current_lr, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                                b_nfe_meter.avg, float(loss.item()), train_acc, val_acc
                            )
                        )
        train_time_sec = time.time() - train_start_time
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            train_peak_mem_mb = torch.cuda.max_memory_allocated(device=device) / (1024.0 ** 2)
        else:
            train_peak_mem_mb = float("nan")
        inference_test_acc, inference_time_sec, inference_peak_mem_mb = evaluate_inference_metrics(
            model, test_loader, device
        )
        test_error_pct = 100.0 * (1.0 - inference_test_acc)
        logger.info(
        "FINAL_METRICS | Method {} | TestErrorPct {:.4f} | TrainGPUMemMB {:.2f} | "
        "TrainTimeSec {:.3f} | InferenceGPUMemMB {:.2f} | InferenceTimeSec {:.3f} | "
        "InferenceTestAcc {:.4f}".format(
            'MP Adjoint',
            test_error_pct,
            float(train_peak_mem_mb),
            float(train_time_sec),
            float(inference_peak_mem_mb),
            float(inference_time_sec),
            float(inference_test_acc),
            )
        )



    finally:
        if 'log_file' in locals() and log_file:
            log_file.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()
