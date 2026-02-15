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
        spatial_corr = torch.exp(-beta * torch.abs(T_mat_i - T_mat_j))
        evals, evecs = torch.linalg.eigh(spatial_corr)
        idx = torch.argsort(evals, descending=True)
        loadings = evecs[:, idx[:n_factors]] * torch.sqrt(torch.clamp(evals[idx[:n_factors]], min=0.0)).unsqueeze(0)
        self.register_buffer('loadings', loadings / torch.sqrt(torch.sum(loadings**2, dim=1, keepdim=True)))
        self.register_buffer('Lambda_upper', torch.triu(self.loadings @ self.loadings.T, diagonal=1))

    def get_terminal_bond(self):
        return torch.prod(1.0 / (1.0 + self.tau * self.F0))

    def simulate_forward_curve(self, n_paths, time_grid, seed=56, freeze_drift=True, use_checkpoint=False):
        n_steps = len(time_grid) - 1
        dt = (time_grid[1] - time_grid[0]).to(self.dtype)
        
        # 1. PRE-CALCULATE VOLTERRA KERNEL (Hybrid Scheme - Bennedsen et al. 2017)
        t, s = time_grid[1:].to(self.dtype), time_grid[:-1].to(self.dtype)
        gamma_const = torch.exp(torch.lgamma(self.H + 0.5))
        
        # Distances: t_i - s_j
        dt_mat = torch.clamp(t[:, None] - s[None, :], min=0.0)
        
        # Hybrid Scheme Discretization:
        # Step A: Use Midpoint power-law for the 'smooth' part (k > 1)
        # Weight = ((k - 0.5) * dt)^(H - 0.5)
        kernel = torch.pow(torch.clamp(dt_mat - 0.5 * dt, min=1e-12), self.H - 0.5)
        
        # Step B: Use Integrated singular bin for the 'singular' part (k = 1)
        # Weight = (1/dt) * integral_{0}^{dt} x^(H-0.5) dx = dt^(H-0.5) / (H + 0.5)
        diag_idx = torch.arange(n_steps, device=self.device)
        kernel[diag_idx, diag_idx] = torch.pow(dt, self.H - 0.5) / (self.H + 0.5)
        
        # Apply Gamma normalization and causal mask
        kernel = torch.where(t[:, None] > s[None, :], kernel / gamma_const, 0.0)
        
        # Step C: Discrete Martingale Correction
        # Using the discrete sum of squared weights ensures E[e^X] = 1.0 exactly
        var_comp = 0.5 * (self.nus[0]**2) * torch.sum(kernel**2, dim=1) * dt

        # 2. GENERATE SHOCKS
        sobol = torch.quasirandom.SobolEngine(dimension=n_steps * (self.n_factors + 1), scramble=True, seed=seed)
        u = sobol.draw(n_paths).to(self.device).to(self.dtype)
        from torch.distributions import Normal
        dist = Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        z = dist.icdf(torch.clamp(u, 1e-7, 1-1e-7)).view(n_paths, n_steps, self.n_factors + 1)
        
        dZ_c = z[..., :self.n_factors] * torch.sqrt(dt)
        dW_v = self.rhos[0] * dZ_c[..., 0] + torch.sqrt(1.0 - self.rhos[0]**2) * z[..., self.n_factors] * torch.sqrt(dt)
        dW_r = torch.einsum('psk,nk->psn', dZ_c, self.loadings)

        # 3. DEFINE SIMULATION BLOCK
        def _simulate(zv, wr, kern, vcomp, f0, alphas):
            fBm = torch.matmul(zv, kern.T)
            unit_vols = torch.exp(self.nus[0] * fBm - vcomp)
            v2_dt = (unit_vols ** 2 * dt).unsqueeze(-1)
            
            if freeze_drift:
                mu_0 = -alphas * torch.matmul(self.Lambda_upper, (self.tau * alphas) / (1.0 + self.tau * f0))
                dF = mu_0.view(1, 1, -1) * v2_dt
                dF.add_(wr * (unit_vols.unsqueeze(-1) * alphas))
                return torch.cat([f0.expand(n_paths, 1, -1), f0 + torch.cumsum(dF, dim=1)], dim=1)
            else:
                # --- UNFROZEN EXACT DRIFT ---
                F_t = f0.expand(n_paths, self.N).clone()
                res = [F_t.unsqueeze(1)]
                for i in range(n_steps):
                    # Corrected weight calculation for path-wise broadcasting
                    omega = (self.tau * alphas) / (1.0 + self.tau * F_t)
                    
                    # Use einsum to handle (N, N) matrix @ (n_paths, N) vectors
                    drift_weights = torch.einsum('ij,pj->pi', self.Lambda_upper, omega)
                    
                    # Apply volatility and step size
                    drift = -alphas * drift_weights * v2_dt[:, i]
                    
                    # Euler-Maruyama step
                    F_t = F_t + drift + wr[:, i] * (unit_vols[:, i:i+1] * alphas)
                    res.append(F_t.unsqueeze(1))
                    
                return torch.cat(res, dim=1)


        if use_checkpoint and torch.is_grad_enabled():
            import torch.utils.checkpoint as cp
            return cp.checkpoint(_simulate, dW_v, dW_r, kernel, var_comp, self.F0, self.alphas, use_reentrant=False)
        else:
            return _simulate(dW_v, dW_r, kernel, var_comp, self.F0, self.alphas)