'''
Classes to wrap the neural network function g(x) with the expectation:
f = E(g(x+epsilon)) 

Supporting the following derivatives:
df_dt
df_dx
d2f_dx2
'''
from typing import Any
import torch
from torch import nn
from src.utils import gaussian_augment


def build_wrapper(cfg, g):
    cfg.model.derivative = cfg.model.derivative
    if cfg.model.derivative == 'pinn':
        return PinnWrapper(g, cfg.equation.x_dim)
    
    t_std, x_std = cfg.model.t_std, cfg.model.x_std
    train_sample_cnt, eval_sample_cnt = cfg.train.model_sample_cnt, cfg.test.model_sample_cnt
    if cfg.model.derivative == 'steins':
        wrapper = SteinsWrapper(g, t_std=t_std, x_std=x_std, train_sample_cnt=train_sample_cnt, 
                                         eval_sample_cnt=eval_sample_cnt)
    elif cfg.model.derivative == 'bp':
        wrapper = BackpropWrapper(g, t_std=t_std, x_std=x_std, train_sample_cnt=train_sample_cnt, 
                                         eval_sample_cnt=eval_sample_cnt)
    elif cfg.model.derivative == 'improved-steins':
        wrapper = ImprovedSteinsWrapper(g, t_std=t_std, x_std=x_std, train_sample_cnt=train_sample_cnt, 
                                         eval_sample_cnt=eval_sample_cnt)
    else:
        raise NotImplementedError
    return wrapper
    
class Wrapper:
    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class GaussianWrapper(Wrapper):
    def __init__(self, g, t_std, x_std, train_sample_cnt=8, eval_sample_cnt=16) -> None:
        self.g = g
        self.train_sample_cnt = train_sample_cnt
        self.eval_sample_cnt = eval_sample_cnt
        self.t_std = t_std
        self.x_std = x_std
        # default is the training setting.
        self.training = True

    def eval(self):
        self.training = False
        self.g.eval()
    
    def train(self):
        self.training = True
        self.g.train()

    def __call__(self, X, sample_cnt=None):
        if sample_cnt is None:
            sample_cnt = self.train_sample_cnt if self.training else self.eval_sample_cnt

        # To save memory, disable the grad in evaluation mode.
        with torch.set_grad_enabled(self.training):
            batch_size = X.shape[0]
            x, t = X[:, :-1], X[:, -1:]
            sample_x, e_x = gaussian_augment(x, self.x_std, sample_cnt)
            sample_t, e_t = gaussian_augment(t, self.t_std, sample_cnt)
            sample_X = torch.cat([sample_x, sample_t], dim=1)

            sample_u = self.g(sample_X).reshape(sample_cnt, batch_size, 1)
            # uv.shape: (batch, 2)
            u = torch.mean(sample_u, dim=0)
            return u

            
class PinnWrapper(Wrapper):
    def __init__(self, g:nn.Module, x_dim) -> None:
        super().__init__()
        self.g:nn.Module = g
        self.x_dim = x_dim    

    def eval(self):
        self.g.eval()

    def train(self):
        self.g.train()

    def __call__(self, X, sample_cnt=None):
        with torch.set_grad_enabled(self.g.training):
            return self.g(X)
    
    def dx(self, X, sample_cnt=None):
        # x.shape: (batch_size, 2)
        x, t = X[:, :-1], X[:, -1:]
        x.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.g(X)
        df_dx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        return df_dx

    def dt(self, X, sample_cnt=None):
        # x.shape: (batch_size, 2)
        x, t = X[:, :-1], X[:, -1:]
        t.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.g(X)
        df_dx = torch.autograd.grad(f.sum(), t, create_graph=True)[0]
        return df_dx

    def dx2(self, X, sample_cnt=None):
        # x.shape: (batch_size, 2)
        x, t = X[:, :-1], X[:, -1:]
        x.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.g(X)

        df_dx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        d2f_dx2 = []
        for i in range(self.x_dim):
            # (batch_size, 1)
            d2f_dxidxi = torch.autograd.grad(df_dx[:, i].sum(), x, create_graph=True)[0][:, i:i+1]
            d2f_dx2.append(d2f_dxidxi)
        # (batch_size, x_dim)
        d2f_dx2 = torch.concat(d2f_dx2, dim=1)
        return d2f_dx2
     
class BackpropWrapper(GaussianWrapper):
    def dx(self, X, sample_cnt=None):
        if sample_cnt is None:
            sample_cnt = self.train_sample_cnt if self.training else self.eval_sample_cnt
        
        batch_size = X.shape[0]
        # x.shape: (batch_size, 2)
        x, t = X[:, :-1], X[:, -1:]
        x.requires_grad_()
        sample_x, e_x = gaussian_augment(x, self.x_std, sample_cnt)
        sample_t, e_t = gaussian_augment(t, self.t_std, sample_cnt)
        sample_X = torch.cat([sample_x, sample_t], dim=1)
        
        sample_g = self.g(sample_X).reshape(sample_cnt, batch_size, 1)
        # f.shape: (batch_size, 1)
        f = torch.mean(sample_g, dim=0)
        # df_dx.shape: (batch_size, 2)
        df_dx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        return df_dx
    
    def dt(self, X, sample_cnt=None):
        if sample_cnt is None:
            sample_cnt = self.train_sample_cnt if self.training else self.eval_sample_cnt
        
        batch_size = X.shape[0]
        # t.shape: (batch_size, 1)
        x, t = X[:, :-1], X[:, -1:]
        t.requires_grad_()
        sample_x, e_x = gaussian_augment(x, self.x_std, sample_cnt)
        sample_t, e_t = gaussian_augment(t, self.t_std, sample_cnt)
        sample_X = torch.cat([sample_x, sample_t], dim=1)
        
        sample_g = self.g(sample_X).reshape(sample_cnt, batch_size, 1)
        # f.shape: (batch_size, 1)
        f = torch.mean(sample_g, dim=0)
        # df_dx.shape: (batch_size, 1)
        df_dt = torch.autograd.grad(f.sum(), t, create_graph=True)[0]
        return df_dt

    def dx2(self, X, sample_cnt=None):
        if sample_cnt is None:
            sample_cnt = self.train_sample_cnt if self.training else self.eval_sample_cnt
        
        batch_size = X.shape[0]
        # x.shape: (batch_size, 2)
        x, t = X[:, :-1], X[:, -1:]
        x_dim = x.shape[1]
        x.requires_grad_()
        sample_x, e_x = gaussian_augment(x, self.x_std, sample_cnt)
        sample_t, e_t = gaussian_augment(t, self.t_std, sample_cnt)
        sample_X = torch.cat([sample_x, sample_t], dim=1)
        
        sample_g = self.g(sample_X).reshape(sample_cnt, batch_size, 1)
        # f.shape: (batch_size, 1)
        f = torch.mean(sample_g, dim=0)
        # df_dx.shape: (batch_size, 2)
        df_dx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        d2f_dx2 = []
        for i in range(x_dim):
            # (batch_size, 1)
            d2f_dxidxi = torch.autograd.grad(df_dx[:, i].sum(), x, create_graph=True)[0][:, i:i+1]
            d2f_dx2.append(d2f_dxidxi)
        # (batch_size, x_dim)
        d2f_dx2 = torch.concat(d2f_dx2, dim=1)
        return d2f_dx2
        
class SteinsWrapper(GaussianWrapper):

    def dx(self, X, sample_cnt=None):
        if sample_cnt is None:
            sample_cnt = self.train_sample_cnt if self.training else self.eval_sample_cnt

        batch_size = X.shape[0]
        # x.shape: (batch_size, 100)
        x, t = X[:, :-1], X[:, -1:]

        # sample_x.shape: (sample_cnt*batch_size, 100)
        # e_x.shape: (sample_cnt*batch_size, 100)
        sample_x, e_x = gaussian_augment(x, self.x_std, sample_cnt)
        sample_t, e_t = gaussian_augment(t, self.t_std, sample_cnt)
        sample_X = torch.cat([sample_x, sample_t], dim=1)

        # sample_u.shape: (sample_cnt, batch, 1)
        sample_u = self.g(sample_X).reshape(sample_cnt, batch_size, 1)
        # e_x.shape: (sample_cnt, batch, 100)
        e_x = e_x.reshape(sample_cnt, batch_size, -1)

        # (batch, 100)
        df_dx = torch.mean((sample_u*e_x)/self.x_std**2, dim=0)

        return df_dx
    
    def dt(self, X, sample_cnt=None):
        if sample_cnt is None:
            sample_cnt = self.train_sample_cnt if self.training else self.eval_sample_cnt

        batch_size = X.shape[0]
        x, t = X[:, :-1], X[:, -1:]

        sample_x, e_x = gaussian_augment(x, self.x_std, sample_cnt)
        # e_t.shape: (sample_cnt*batch, 1)
        sample_t, e_t = gaussian_augment(t, self.t_std, sample_cnt)
        sample_X = torch.cat([sample_x, sample_t], dim=1)

        # sample_u.shape: (sample_cnt, batch, 1)
        sample_u = self.g(sample_X).reshape(sample_cnt, batch_size, 1)
        e_t = e_t.reshape(sample_cnt, batch_size, 1)
        
        # (batch_size, 1)
        df_dt = torch.mean((sample_u*e_t)/self.t_std**2, dim=0)
        return df_dt

    def dx2(self, X, sample_cnt=None):
        '''
        laplacian of f on x (100 dim).
        '''
        if sample_cnt is None:
            sample_cnt = self.train_sample_cnt if self.training else self.eval_sample_cnt

        batch_size = X.shape[0]
        x, t = X[:, :-1], X[:, -1:]

        # e_x.shape: (sample_cnt*batch, 100)
        sample_x, e_x = gaussian_augment(x, self.x_std, sample_cnt)
        sample_t, e_t = gaussian_augment(t, self.t_std, sample_cnt)
        sample_X = torch.cat([sample_x, sample_t], dim=1)

        # sample_u.shape: (sample_cnt, batch, 1)
        sample_u = self.g(sample_X).reshape(sample_cnt, batch_size, 1)
        e_x = e_x.reshape(sample_cnt, batch_size, -1)
        
        # (batch_size, 100)
        d2f_dx2 = torch.mean((sample_u*(e_x**2 - self.x_std**2)/self.x_std**4), dim=0)
        return d2f_dx2

class ImprovedSteinsWrapper(GaussianWrapper):
    def dE(self, X, sample_cnt=None):
        # calculate every derivative with the same bunch of samples
        if sample_cnt is None:
            sample_cnt = self.train_sample_cnt if self.training else self.eval_sample_cnt

        sample_cnt = sample_cnt//2
        batch_size = X.shape[0]
        x, t = X[:, :-1], X[:, -1:]

        sample_t, e_t = gaussian_augment(t, self.t_std, sample_cnt)
        sample_t_plus = sample_t
        sample_t_minus = sample_t - 2*e_t
        sample_x, e_x = gaussian_augment(x, self.x_std, sample_cnt)
        sample_x_plus = sample_x
        sample_x_minus = sample_x - 2*e_x

        sample_X_plus = torch.cat([sample_x_plus, sample_t_plus], dim=1)
        sample_X_minus = torch.cat([sample_x_minus, sample_t_minus], dim=1)
        
        # sample_u_plus/minus.shape: (sample_cnt, batch, 1)
        sample_u_plus = self.g(sample_X_plus).reshape(sample_cnt, batch_size, 1)
        sample_u_minus = self.g(sample_X_minus).reshape(sample_cnt, batch_size, 1)

        e_t = e_t.reshape(sample_cnt, batch_size, 1)
        e_x = e_x.reshape(sample_cnt, batch_size, -1)

        df_dt = ((sample_u_plus-sample_u_minus)*e_t)/(2*self.t_std*self.t_std)
        df_dt = torch.mean(df_dt, dim=0)

        df_dx = ((sample_u_plus-sample_u_minus)*e_x)/(2*self.x_std*self.x_std)
        df_dx = torch.mean(df_dx, dim=0)
        
        u = self.g(X).reshape(1, batch_size, 1)
        d2f_dx2 = (e_x**2-self.x_std**2)*(sample_u_plus+sample_u_minus-2*u)/(2*(self.x_std**4))
        # (batch_size, 100)
        d2f_dx2 = torch.mean(d2f_dx2, dim=0)
        return df_dt, df_dx, d2f_dx2