from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import IterableDataset

class OnlineDataset(IterableDataset):
    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError
    

class MultiSetDataLoader:
    def __init__(self, datasets:List, batch_sizes:List, shuffle=False) -> None:
        self.dataloaders = []
        for d, b in zip(datasets, batch_sizes):
            # If the dataset is iterable(possibly the OnlineDataset), it cannot be shuffled.
            s = False if isinstance(d, IterableDataset) else shuffle
            self.dataloaders.append(DataLoader(dataset=d, batch_size=b, shuffle=s))

        self.iters = None

    def __iter__(self):
        self.iters = [iter(dataloader) for dataloader in self.dataloaders]
        return self

    def __next__(self):
        '''
        If the iterator on any dataset causes a StopException, it will stop.
        '''
        rlt = [next(i) for i in self.iters]
        return rlt 

def build_lr(model, cfg, tot_epoch=-1):
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.scheduler.name == 'linear':
        if tot_epoch == -1:
            tot_epoch = cfg.epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR( \
            optimizer_adam, \
            lr_lambda=lambda epoch:(1-epoch/tot_epoch), \
        )
    elif cfg.scheduler.name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=cfg.scheduler.step_size, 
                                                    gamma=cfg.scheduler.gamma)
    else:
        # Default is constant learning rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_adam, lr_lambda=lambda epoch:1)
    return optimizer_adam, scheduler


def gaussian_augment(x:torch.Tensor, std, N_sample):
    sample_x = torch.cat(N_sample*[x])
    e = torch.normal(mean=0, std=std, size=sample_x.shape, device=sample_x.device)
    return sample_x+e, e
