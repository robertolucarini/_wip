import torch
import torch.nn as nn
import numpy as np

class TorchRoughSABR_FMM(nn.Module):
    def __init__(self, tenors, F0, alpha_f, rho_f, nu_f, H, beta=0.05, n_factors=3, device='cpu'):
        super().__init__()
        self.device = device
        self.dtype = torch.float64
        self.n_factors = n_factors
        
        self.register_buffer('T', torch.tensor(tenors, device=device, dtype=self.dtype))
        self.register_buffer('tau', torch.tensor(np.diff(tenors), device=device, dtype=self.dtype))
        self.N = len(F0)
        
        self.F0 = nn.Parameter(torch.tensor(F0, device=device, dtype=self.dtype))
        self.alphas = nn.Parameter(torch.tensor(alpha_f(tenors[:-1]), device=device, dtype=self.dtype))
        
        self.register_buffer('rhos', torch.tensor(rho_f(tenors[:-1]), device=device, dtype=self.dtype))
        self.register_buffer('nus', torch.tensor(nu_f(tenors[:-1]), device=device, dtype=self.dtype))
        self.register_buffer('H', torch.tensor(H, device=device, dtype=self.dtype))

        # PCA Setup
        T_mat_i, T_mat_j = self.T[:-1].unsqueeze(1), self.T[:-1].unsqueeze(0)
        # rate-rate corr
        spatial_corr = torch.exp(-beta * torch.abs(T_mat_i - T_mat_j))
        # eigen decomposition
        evals, evecs = torch.linalg.eigh(spatial_corr)
        # sort
        idx = torch.argsort(evals, descending=True)
        # pca loadings
        loadings = evecs[:, idx[:n_factors]] * torch.sqrt(torch.clamp(evals[idx[:n_factors]], min=0.0)).unsqueeze(0)

        self.register_buffer('loadings', loadings / torch.sqrt(torch.sum(loadings**2, dim=1, keepdim=True)))
        self.register_buffer('Lambda_upper', torch.triu(self.loadings @ self.loadings.T, diagonal=1))

    def get_terminal_bond(self):
        return torch.prod(1.0 / (1.0 + self.tau * self.F0))

    def simulate_forward_curve(self, n_paths, time_grid, seed=56, freeze_drift=True, use_checkpoint=False):
        from torch.distributions import Normal

        n_steps = len(time_grid) - 1
        dt = (time_grid[1] - time_grid[0]).to(self.dtype)
        
        # 1. Pre-calculate Kernel & Martingale Correction
        # t > s
        t, s = time_grid[1:].to(self.dtype), time_grid[:-1].to(self.dtype)
        # Gamma func
        gamma_const = torch.exp(torch.lgamma(self.H + 0.5))
        # control for negative times
        dt_mat = torch.clamp(t[:, None] - s[None, :], min=0.0)
        # power-low decay: dt**(H - 0.5) 
        # kernel at midpoint of the integration interval: t_i - s_j - 1/2*dt
        kernel = torch.pow(torch.clamp(dt_mat - 0.5 * dt, min=1e-12), self.H - 0.5)
        # diagonal distances indexes
        diag_idx = torch.arange(n_steps, device=self.device)
        # integrate kernel analytically over interval [0, dt]
        kernel[diag_idx, diag_idx] = torch.pow(dt, self.H - 0.5) / (self.H + 0.5)
        # divide kernel matrix by gamma func and zero-out the upper-triangle 
        kernel = torch.where(t[:, None] > s[None, :], kernel / gamma_const, 0.0)
        
        # Discrete Martingale Correction (remains N x Steps)
        var_comp = 0.5 * (self.nus**2).unsqueeze(-1) * torch.sum(kernel**2, dim=1).unsqueeze(0) * dt

        # sobol generator
        sobol = torch.quasirandom.SobolEngine(dimension=n_steps * (self.n_factors + 1), scramble=True, seed=seed)
        # draws
        u = sobol.draw(n_paths).to(self.device).to(self.dtype)
        # from uniform to standard normal distribution
        dist = Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        # shocks
        z = dist.icdf(torch.clamp(u, 1e-7, 1-1e-7)).view(n_paths, n_steps, self.n_factors + 1)
        
        # time-scaled shocks
        # macro factors
        dz_curve = z[..., :self.n_factors] * torch.sqrt(dt)
        # idio factor
        dz_perp = z[..., self.n_factors:self.n_factors+1] * torch.sqrt(dt)
        
        # dZ @ L.T  
        dW_r = torch.matmul(dz_curve, self.loadings.T)

        # 3. Define Simulation Block
        def _simulate(dz_c, dz_p, wr, kern, vcomp, f0, alphas):
            # SPEED OPTIMIZATION: Factorized Rough Volatility
            # Calculate 4 rough drivers (3 factors + 1 orthogonal) instead of 31
            # shocks -> (n_paths, n_steps, n_factors)
            dz_all = torch.cat([dz_c, dz_p], dim=-1)
            # Kernel * dZ
            # (n_paths, n_steps, n_factors)
            fbm_4 = torch.matmul(dz_all.transpose(1, 2), kern.T).transpose(1, 2)
            
            # Reconstruct tenor fBms: fBm_i = rho_i * (sum L_ik * fBm_k) + sqrt(1-rho^2) * fBm_p
            fbm_curve = torch.matmul(fbm_4[..., :-1], self.loadings.T)
            fbm = self.rhos.view(1, 1, -1) * fbm_curve + torch.sqrt(1.0 - self.rhos**2).view(1, 1, -1) * fbm_4[..., -1:]
            
            # Stochastic Volatility
            unit_vols = torch.exp(self.nus.view(1, 1, -1) * fbm - vcomp.T.unsqueeze(0))
            v2_dt = (unit_vols ** 2 * dt)
            
            if freeze_drift:
                mu_0 = -alphas * torch.matmul(self.Lambda_upper, (self.tau * alphas) / (1.0 + self.tau * f0))
                # Use in-place ops for memory speed
                dF = mu_0.view(1, 1, -1) * v2_dt
                dF.add_(wr * (unit_vols * alphas.view(1, 1, -1)))
                return torch.cat([f0.expand(n_paths, 1, -1), f0 + torch.cumsum(dF, dim=1)], dim=1)
            else:
                F_t = f0.expand(n_paths, self.N).clone()
                res = [F_t.unsqueeze(1)]
                for i in range(n_steps):
                    omega = (self.tau * alphas) / (1.0 + self.tau * F_t)
                    drift_weights = torch.einsum('ij,pj->pi', self.Lambda_upper, omega)
                    drift = -alphas * drift_weights * v2_dt[:, i]
                    F_t = F_t + drift + wr[:, i] * (unit_vols[:, i] * alphas)
                    res.append(F_t.unsqueeze(1))
                return torch.cat(res, dim=1)

        if use_checkpoint and torch.is_grad_enabled():
            import torch.utils.checkpoint as cp
            return cp.checkpoint(_simulate, dz_curve, dz_perp, dW_r, kernel, var_comp, self.F0, self.alphas, use_reentrant=False)
        else:
            return _simulate(dz_curve, dz_perp, dW_r, kernel, var_comp, self.F0, self.alphas)