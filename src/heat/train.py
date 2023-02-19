import os
import torch.distributed as dist

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from .equation import HeatEquation

from .derivative_wrapper import build_wrapper

from ..model import MLP
from .data import HeatDataset
from src.utils.glob import setup_logging, config
from src.utils import build_lr
import torch.multiprocessing as mp

def train(rank, world_size, config):
    cfg = config.heat
    setup(rank, world_size)
    logger = setup_logging()
    logger.info(f"starting {rank}")

    layers = [cfg.equation.x_dim+1] + [cfg.model.width]*(cfg.model.depth-1) + [1]
    g = MLP(layers).to(rank)

    if world_size == 1:
        ddp_g = g
    else:
        ddp_g = DDP(g, device_ids=[rank])

    f = build_wrapper(cfg, ddp_g)

    dataset = HeatDataset(domain_bsz=cfg.train.batch.domain_size, \
        init_bsz=cfg.train.batch.initial_size, \
        spatial_bound_bsz=cfg.train.batch.spatial_boundary_size, \
        xdim=cfg.equation.x_dim, T=cfg.equation.T, rank=rank \
        )
    heat = HeatEquation(cfg.equation.x_dim)

    # test set
    if rank == 0:
        test_X, test_Y = generate_test_set(cfg, heat, rank)

    optimizer, scheduler = build_lr(ddp_g, cfg.train, cfg.train.iteration)

    for i in range(cfg.train.iteration):
        f.train()
        optimizer.zero_grad()

        domain_X, init_X, spatial_boundary_X = dataset.get_online_data()
        
        dloss = heat.domain_loss(domain_X, f)
        iloss = heat.initial_loss(init_X, f)
        sloss = heat.spatial_boundary_loss(spatial_boundary_X, f)
        loss = cfg.train.loss.domain*dloss + cfg.train.loss.initial*iloss + cfg.train.loss.spatial_boundary*sloss

        if rank == 0:
            logger.info(f'iteration {i}\t| loss {loss.detach().cpu().item():.5f}\t| '
                f'domain {dloss.detach().cpu().item():.5f}\t|'
                f'initial {iloss.detach().cpu().item():.5f}\t|'
                f'spatial boundary {sloss.detach().cpu().item():.5f}\t'
            )

        if cfg.model.derivative != 'gt':
            loss.backward()
            optimizer.step()
            scheduler.step()

        if rank==0 and (i+1)%cfg.test.step==0:
            # test the model, only test in one thread.
            i_avg_err, i_rel_err = test(cfg, f, test_X, test_Y, rank, norm_type='l1')
            logger.info(f'L1 test error: average {i_avg_err}, relative {i_rel_err}')
            i_avg_err, i_rel_err = test(cfg, f, test_X, test_Y, rank, norm_type='l2')
            logger.info(f'L2 test error: average {i_avg_err}, relative {i_rel_err}')

    cleanup(rank, world_size)

def pgd(x, f, loss_func, step_cnt=5, step_size=0.2, t_lower_bound=0.0, t_upper_bound=1.0):
    for _ in range(step_cnt):
        x.requires_grad_()
        loss = loss_func(x, f)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + step_size * torch.sign(grad.detach())
        x[:,-1] = torch.clamp(x[:,-1], t_lower_bound, t_upper_bound)
    return x

def generate_test_set(cfg, heat: HeatEquation, rank):
    x = torch.randn((cfg.test.total_size, heat.xdim), device=rank)
    x = x / (1e-6 + x.norm(dim=-1, keepdim=True))
    x_norm = torch.rand((cfg.test.total_size, 1), device=rank) ** (1/heat.xdim)
    x = x * x_norm
    test_X = torch.concat(
        [x, # x ~ U(B(0,1)), where B(0,1) denotes the unit ball
         torch.rand((cfg.test.total_size, 1), device=rank)*cfg.equation.T, # t ~ U(0,T)
        ],
        dim=1
    )
    test_Y = heat.ground_truth(test_X)
    return test_X, test_Y

def test(cfg, f, X, Y, rank, norm_type='l1'):
    if norm_type == 'l1':
        return test_l1(cfg, f, X, Y, rank)
    elif norm_type == 'l2':
        return test_l2(cfg, f, X, Y, rank)
    else:
        raise NotImplementedError
    
def test_l1(cfg, f, X, Y, rank):
    with torch.no_grad():
        f.eval()
        dataloader = DataLoader(TensorDataset(X, Y), batch_size=cfg.test.batch_size)
        tot_err, tot_norm = 0, 0
        for x, y in dataloader:
            x, y = x.to(rank), y.to(rank)
            pred_y = f(x).squeeze()
            err = (pred_y - y).abs().sum()
            y_norm = y.abs().sum()
            tot_err += err.cpu().item()
            tot_norm += y_norm
        avg_err = tot_err/X.shape[0]
        rel_err = tot_err/tot_norm
    return avg_err, rel_err

def test_l2(cfg, f, X, Y, rank):
    with torch.no_grad():
        f.eval()
        dataloader = DataLoader(TensorDataset(X, Y), batch_size=cfg.test.batch_size)
        tot_err, tot_norm = 0, 0
        for x, y in dataloader:
            x, y = x.to(rank), y.to(rank)
            pred_y = f(x).squeeze()
            err = ((pred_y - y)**2).sum()
            y_norm = (y**2).sum()
            tot_err += err.cpu().item()
            tot_norm += y_norm
        tot_err, tot_norm = tot_err**0.5, tot_norm**0.5
        avg_err = tot_err/(X.shape[0]**0.5)
        rel_err = tot_err/tot_norm
    return avg_err, rel_err

def setup(rank, world_size):
    if world_size<=1:
        return
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup(rank, world_size):
    if world_size<=1:
        return
    dist.destroy_process_group()

def heat_training():
    if config.heat.gpu_cnt == 1:
        train(0, 1, config)
    else:
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= config.heat.gpu_cnt, \
            f"Requires at least {config.heat.gpu_cnt} GPUs to run, but got {n_gpus}"
        world_size = n_gpus

        mp.spawn(train,
                args=(world_size, config),
                nprocs=world_size,
                join=True)
