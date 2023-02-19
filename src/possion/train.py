import os
import torch.distributed as dist

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from .equation import PossionEquation

from .derivative_wrapper import build_wrapper

from ..model import MLP
from .data import BoundaryPossionDataset, DomainPossionDataset
from src.utils.glob import setup_logging, config
from src.utils import MultiSetDataLoader, build_lr
import torch.multiprocessing as mp
import random

def train(rank, world_size, config):
    cfg = config.possion
    setup(rank, world_size)
    logger = setup_logging()
    logger.info(f"starting {rank}")
    
    g = MLP([cfg.equation.x_dim, 256, 256, 256, 1]).to(rank)
    if world_size == 1:
        ddp_g = g
    else:
        ddp_g = DDP(g, device_ids=[rank])

    f = build_wrapper(cfg, ddp_g)

    # x in [0, 1]
    # T in [0, 1]
    # online samplers
    domain_dataset = DomainPossionDataset(xdim=cfg.equation.x_dim, rank=0)
    boundary_dataset = BoundaryPossionDataset(xdim=cfg.equation.x_dim, rank=0)

    possion = PossionEquation(cfg.equation.x_dim)

    if rank == 0:
        # construct validation set
        test_domain_X = torch.rand([cfg.test.domain_size,  cfg.equation.x_dim], dtype=torch.float32)
        test_boundary_X = torch.rand([cfg.test.domain_size,  cfg.equation.x_dim], dtype=torch.float32)
        for i in range(cfg.test.domain_size):
            test_boundary_X[i, random.choice(range(cfg.equation.x_dim))] = random.choice([0,1])

    dataloader = MultiSetDataLoader(datasets=[domain_dataset, boundary_dataset], 
                                    batch_sizes=[cfg.train.batch.domain_size, cfg.train.batch.boundary_size])
    optimizer, scheduler = build_lr(ddp_g, cfg.train, cfg.train.iteration)

    data_iter = iter(dataloader)
    for i in range(cfg.train.iteration):
        f.train()
        domain_X, boundary_X = next(data_iter)
        optimizer.zero_grad()
        domain_X = domain_X.to(rank)
        boundary_X = boundary_X.to(rank)
        iloss = possion.domain_loss(domain_X, f, sample_cnt=cfg.train.batch.domain_sample_cnt)
        bloss = possion.boundary_loss(boundary_X, f, sample_cnt=cfg.train.batch.boundary_sample_cnt)
        
        loss = cfg.train.loss.domain*iloss + cfg.train.loss.boundary*bloss
        loss.backward()

        if rank == 0:
            logger.info(f'thread {rank}, iteration {i}, loss {loss.detach().cpu().item()}')

        if cfg.model.derivative != 'gt':
            optimizer.step()
            scheduler.step()

        if rank==0 and (i+1)%cfg.test.step==0:
            # calculate validation loss
            i_loss, b_loss = valid_loss(cfg, f, possion, test_domain_X, test_boundary_X, rank)
            validation_loss = cfg.train.loss.domain*i_loss + cfg.train.loss.boundary*b_loss
            logger.info(f'test loss: domain {i_loss:.7f}, boundary: {b_loss:.7f}, total: {validation_loss:.7f}')

            # calculate test error
            domain_err, domain_rel_err = test_error(cfg, f, possion, rank, norm_type='l1')
            logger.info(f'L1 err: {domain_err:.5f}, rel_err: {domain_rel_err:.5f}')
            domain_err, domain_rel_err = test_error(cfg, f, possion, rank, norm_type='l2')
            logger.info(f'L2 err: {domain_err:.5f}, rel_err: {domain_rel_err:.5f}')

    cleanup(rank, world_size)
   
def valid_loss(cfg, f, possion:PossionEquation, domain_X, boundary_X, rank):
    with torch.set_grad_enabled(cfg.model.derivative!='steins'):
        f.eval()
        dataloader = DataLoader(TensorDataset(domain_X), batch_size=cfg.test.batch_size)
        domain_loss = 0
        for x, in dataloader:
            x = x.to(rank)
            iloss = possion.domain_loss(x, f)
            domain_loss += iloss.detach()*x.shape[0]
        domain_loss /= domain_X.shape[0]
        
        dataloader = DataLoader(TensorDataset(boundary_X), batch_size=cfg.test.batch_size)
        boundary_loss = 0
        for x, in dataloader:
            x = x.to(rank)
            bloss = possion.boundary_loss(x, f)
            boundary_loss += bloss.detach()*x.shape[0]
        boundary_loss /= boundary_X.shape[0]
        return domain_loss.cpu().item(), boundary_loss.cpu().item()

def test_error(cfg, f, possion:PossionEquation, rank, test_data=None, norm_type='l1'):
    if norm_type == 'l1':
        return test_l1(cfg, f, possion, rank, test_data)
    elif norm_type == 'l2':
        return test_l2(cfg, f, possion, rank, test_data)
    else:
        raise NotImplementedError

def test_l1(cfg, f, possion:PossionEquation, rank, test_data=None):
    if test_data is None:
        test_data = torch.rand([cfg.test.domain_size,  cfg.equation.x_dim], dtype=torch.float32)
    with torch.no_grad():
        dataloader = DataLoader(TensorDataset(test_data), batch_size=cfg.test.batch_size)
        tot_err, tot_norm = 0, 0
        for x, in dataloader:
            x = x.to(rank)
            pred_y, y = possion.ground_truth(x), f(x).squeeze()
            err = (pred_y - y).abs().sum()
            y_norm = y.abs().sum()
            tot_err += err.cpu().item()
            tot_norm += y_norm
        avg_err = tot_err/test_data.shape[0]
        rel_err = tot_err/(tot_norm+1e-6)
    return avg_err, rel_err

def test_l2(cfg, f, possion:PossionEquation, rank, test_data=None):
    if test_data is None:
        test_data = torch.rand([cfg.test.domain_size,  cfg.equation.x_dim], dtype=torch.float32)
    with torch.no_grad():
        dataloader = DataLoader(TensorDataset(test_data), batch_size=cfg.test.batch_size)
        tot_err, tot_norm = 0, 0
        for x, in dataloader:
            x = x.to(rank)
            pred_y, y = possion.ground_truth(x), f(x).squeeze()
            err = ((pred_y - y)**2).sum()
            y_norm = (y**2).sum()
            tot_err += err.cpu().item()
            tot_norm += y_norm
        tot_err, tot_norm = tot_err**0.5, tot_norm**0.5
        avg_err = tot_err/(test_data.shape[0]**0.5)
        rel_err = tot_err/(tot_norm+1e-6)
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

def possion_training():
    if config.possion.gpu_cnt == 1:
        train(0, 1, config)
    else:
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus

        mp.spawn(train,
                args=(world_size, config),
                nprocs=world_size,
                join=True)
