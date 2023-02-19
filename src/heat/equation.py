import torch

class HeatEquation:
    def __init__(self, x_dim) -> None:
        self.xdim = x_dim

    def domain_loss(self, X, f, sample_cnt=None):
        # dt: (batch, 1)
        # dx: (batch, 100)
        # dx2: (batch, 100)
        if hasattr(f, 'dE'):
            dt, dx, dx2 = f.dE(X, sample_cnt)
        else:
            dt = f.dt(X, sample_cnt)
            dx2 = f.dx2(X, sample_cnt)
            # dx = f.dx(X, sample_cnt)

        residual = dt.squeeze(1) - torch.sum(dx2, dim=1)
        loss = torch.mean(residual**2)
        return loss

    def initial_loss(self, X, f, sample_cnt=None):
        # u(x, 0) = ||x||^2 / 2d
       
        y = f(X, sample_cnt=sample_cnt).squeeze(1)
        gt = self.ground_truth(X)
        
        # Alternatively
        # x = X[:, :-1]
        # gt = torch.sum(x**2, dim=1) / (2*x.shape[1])

        return torch.mean((y-gt)**2)

    def spatial_boundary_loss(self, X, f, sample_cnt=None):
        # u(x, t) = 1/2d + t ( ||x||=1 )
        y = f(X, sample_cnt=sample_cnt).squeeze(1)
        gt = self.ground_truth(X)
        return torch.mean((y-gt)**2)

    def ground_truth(self, X):
        _, total_dim = X.shape
        # x.shape: (batch_size, total_dim-1)
        # t.shape: (batch_size)
        x, t = X[:, :-1], X[:, -1]
        gt = (x**2).sum(dim=-1)/(2*(total_dim-1)) + t
        return gt
