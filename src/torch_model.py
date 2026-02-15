import torch
import torch.nn as nn
import numpy as np

class TorchRoughSABR_FMM(nn.Module):
    def __init__(self, tenors, F0, alpha_f, rho_f, nu_f, H, device='cpu'):
        super().__init__()
        self.device = device
        self.dtype = torch.float64
        
        # Buffers for simulation constants
        self.register_buffer('T', torch.tensor(tenors, device=device, dtype=self.dtype))
        self.register_buffer('tau', torch.tensor(np.diff(tenors), device=device, dtype=self.dtype))
        self.N = len(F0)
        
        # Parameters for AAD (Requires Grad)
        self.F0 = nn.Parameter(torch.tensor(F0, device=device, dtype=self.dtype))
        # Initial Alphas as parameters to enable the Vega Ladder
        self.alphas = nn.Parameter(torch.tensor(alpha_f(tenors[:-1]), device=device, dtype=self.dtype))
        
        # Static Parameters (Buffers)
        self.register_buffer('rhos', torch.tensor(rho_f(tenors[:-1]), device=device, dtype=self.dtype))
        self.register_buffer('nus', torch.tensor(nu_f(tenors[:-1]), device=device, dtype=self.dtype))
        self.register_buffer('H', torch.tensor(H, device=device, dtype=self.dtype))

    def generate_rough_shocks(self, n_paths, time_grid, seed=56):
        """ Fully vectorized unit shock driver (One-Factor) """
        n_steps = len(time_grid) - 1
        dt = (time_grid[1] - time_grid[0]).to(self.dtype)
        
        # Quasi-Monte Carlo (Sobol) for stable gradients
        sobol = torch.quasirandom.SobolEngine(dimension=n_steps * 2, scramble=True, seed=seed)
        u = sobol.draw(n_paths).to(self.device).to(self.dtype)
        
        # Inverse Gaussian Transform
        from torch.distributions import Normal
        dist = Normal(torch.tensor(0.0, device=self.device, dtype=self.dtype), 
                      torch.tensor(1.0, device=self.device, dtype=self.dtype))
        z = dist.icdf(u).view(n_paths, n_steps, 2)
        
        dW_v = z[:, :, 0] * torch.sqrt(dt)
        dW_r = (self.rhos[0] * z[:, :, 0] + torch.sqrt(1.0 - self.rhos[0]**2) * z[:, :, 1]) * torch.sqrt(dt)
        
        # Volterra Kernel integration (Roughness logic)
        t, s = time_grid[1:].to(self.dtype), time_grid[:-1].to(self.dtype)
        dt_mat = t[:, None] - s[None, :]
        kernel = torch.where(dt_mat > 0, dt_mat**(self.H - 0.5), torch.tensor(0.0, device=self.device, dtype=self.dtype))
        
        fBm = torch.matmul(dW_v, kernel.T)
        var_comp = 0.5 * (self.nus[0]**2) * (t**(2*self.H))
        
        # Unit volatility driver
        unit_vols = torch.exp(self.nus[0] * fBm - var_comp)
        
        ones = torch.zeros(n_paths, 1, device=self.device, dtype=self.dtype)
        return torch.cumsum(torch.cat([ones, unit_vols * dW_r], dim=1), dim=1)

    def get_terminal_bond(self):
        """ Differentiable P(0, Tn) bond """
        dfs = 1.0 / (1.0 + self.tau * self.F0)
        return torch.prod(dfs)