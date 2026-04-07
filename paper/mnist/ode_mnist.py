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
    setup_environment,
    get_precision_dtype, 
    determine_scaler,
    setup_experiment
)

def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--tol', type=float, default=1e-3, help='Tolerance for FODE solver')
    parser.add_argument('--nepochs', type=int, default=160, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--test_batch_size', type=int, default=100, help='Batch size for testing')
    parser.add_argument('--beta', type=float, default=0.6, help='Beta parameter for L1 method')

    parser.add_argument('--precision', type=str, default='float32', choices=['float16', 'bfloat16', 'float32', 'tfloat32'], help='Precision for training')
    parser.add_argument('--method', type=str, default='l1', choices=['l1'], help='Integration method')
    parser.add_argument('--odeint', type=str, default='rampde', choices=['rampde', 'torchfde','torchdiffeq'], help='ODE solver backend')
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
    parser.add_argument('--adjoint', action='store_true', help='Use adjoint method for gradient computation')
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
        if not args.adjoint:
            out = self.odeint_func(self.func, z, beta = self.beta, t = args.T, step_size = args.h, method = 'predictor')
        else:
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
    
        # Set scaler to false for now (only high precision)
        S1 = None
        proj_in_layers = [
            nn.Flatten(),
            nn.Linear(784, width),
            nn.ReLU(inplace=True),
        ]
        if args.adjoint:
            from torchfde import fdeint as odeint_func
            

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

def accuracy(model, dataset_loader):
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


if __name__ == "__main__": 

    parser = create_parser()
    args = parser.parse_args()

    grad_scaler_enabled = not args.no_grad_scaler
    dynamic_scaler_enabled = not args.no_dynamic_scaler

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    makedirs('./results')
    logger = get_logger(logpath=os.path.join('logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    if args.adjoint: 
        logger.info("Using adjoint method for gradient computation.")
        args.odeint = 'rampde'
    else:
        args.odeint = 'torchfde'
        logger.info("Using backpropagation for gradient computation (from torchfde).")


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

    lr_fn = learning_rate_with_decay(
        batch_size=args.batch_size,
        batch_denom=128,
        batches_per_epoch=batches_per_epoch,
        boundary_epochs=[60,100,140],
        decay_rates=[1.0, 0.1, 0.01, 0.001],
        lr=args.lr
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_acc = 0.0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    for iter in range(args.nepochs * batches_per_epoch):
        model.train()
        lr = lr_fn(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        
        nfe_forward = model.feature_layers[0].func.nfe
        model.feature_layers[0].func.nfe = 0

        loss.backward()
        optimizer.step()

        nfe_backward = model.feature_layers[0].func.nfe
        model.feature_layers[0].func.nfe = 0

        batch_time_meter.update(time.time() - end)

        f_nfe_meter.update(nfe_forward)
        b_nfe_meter.update(nfe_backward)
        end = time.time()

        if iter % batches_per_epoch == 0:
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader)
                val_acc = accuracy(model, test_loader)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join('model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Test Acc {:.4f}".format(
                        iter // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc
                    )
                )
