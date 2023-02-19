import torch

class HeatDataset():
    def __init__(self, domain_bsz, init_bsz, spatial_bound_bsz, xdim=100, T=1.0, rank=0):
        self.domain_bsz = domain_bsz
        self.init_bsz = init_bsz
        self.spatial_bound_bsz = spatial_bound_bsz
        self.xdim = xdim
        self.T = T
        self.rank = rank

    def get_online_data(self):

        x = torch.randn((self.domain_bsz + self.init_bsz, self.xdim), device=self.rank)
        x = x / (1e-6 + x.norm(dim=-1, keepdim=True))
        x_norm = torch.rand((self.domain_bsz + self.init_bsz, 1), device=self.rank) ** (1/self.xdim)
        x = x * x_norm

        domain_X = torch.concat(
            [x[:self.domain_bsz], # x ~ U(B(0,1)), where B(0,1) denotes the unit ball
             torch.rand((self.domain_bsz, 1), device=self.rank)*self.T, # t ~ U(0,T)
            ],
            dim=1
        )
        
        init_X = torch.concat(
            [x[-self.init_bsz:],  # x ~ U(B(0,1)), where B(0,1) denotes the unit ball
             torch.zeros((self.init_bsz, 1), device=self.rank), # t = 0
            ],
            dim=1
        )

        x = torch.randn((self.spatial_bound_bsz, self.xdim), device=self.rank)
        x = x / (1e-6 + x.norm(dim=-1, keepdim=True))

        spatial_boundary_X = torch.concat(
            [x,  # x ~ U(\partial B(0,1)), where B(0,1) denotes the unit ball
             torch.rand((self.spatial_bound_bsz, 1), device=self.rank)*self.T, # t ~ U(0,T)
            ],
            dim=1
        )

        return domain_X, init_X, spatial_boundary_X