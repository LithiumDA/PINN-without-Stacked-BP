import torch

class PossionEquation:
    def __init__(self, x_dim) -> None:
        self.xdim = x_dim

    def domain_loss(self, X, f, sample_cnt=None):
        # dt: (batch, 1)
        # dx: (batch, 100)
        # dx2: (batch, 100)
        if hasattr(f, 'dE'):
            dt, dx, dx2 = f.dE(X, sample_cnt)
        else:
            dx2 = f.dx2(X, sample_cnt)

        residual = torch.sum(dx2, dim=1) + torch.sin(X.sum(dim=-1))
        loss = torch.mean(residual**2)
        return loss

    def boundary_loss(self, X, f, sample_cnt=None):
        # terminal state: g(x) = log((1+x**2)/2)
        # (batch, 1)
        y = f(X, sample_cnt=sample_cnt).squeeze(1)
        gt = torch.sin(X.sum(dim=-1)) / self.xdim
        return torch.mean((y-gt)**2)

    def ground_truth(self, X):
        return torch.sin(X.sum(dim=-1)) / self.xdim