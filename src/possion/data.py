import torch
from src.utils import OnlineDataset
import random


class DomainPossionDataset(OnlineDataset):
    def __init__(self, xdim, rank=0) -> None:
        super().__init__()
        self.rank = rank
        self.xdim = xdim

    def __next__(self):
        sample = torch.rand((self.xdim, ))
        return sample
    
class BoundaryPossionDataset(OnlineDataset):
    def __init__(self, xdim, rank=0) -> None:
        super().__init__()
        self.rank = rank
        self.xdim = xdim

    def __next__(self):
        sample = torch.rand((self.xdim, ))
        sample[random.choice(range(self.xdim))] = random.choice([0,1])
        return sample
    