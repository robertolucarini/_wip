import torch
import torch.nn as nn
import numpy as np
from src.utils import build_rapisarda_correlation_matrix


class TorchRoughSABR_FMM(nn.Module):
    def __init__(self, tenors, F0, alpha_f, rho_f, nu_f, H, beta_decay=0.05, beta_sabr=0.5, shift=0.0, n_factors=3, correlation_mode='pca', omega_matrix=None, device='cpu'):
        super().__init__()
        self.device = device
        self.dtype = torch.float64
        self.correlation_mode = correlation_mode.lower()
        
        self.register_buffer('T', torch.tensor(tenors, device=device, dtype=self.dtype))
        self.register_buffer('tau', torch.tensor(np.diff(tenors), device=device, dtype=self.dtype))
        self.N = len(F0)
        
        self.F0 = nn.Parameter(torch.tensor(F0, device=device, dtype=self.dtype))
        self.register_buffer('beta_sabr', torch.tensor(beta_sabr, device=device, dtype=self.dtype))
        self.register_buffer('shift', torch.tensor(shift, device=device, dtype=self.dtype))

        # CORRECT SCALING: alpha_f is the ATM Normal Volatility.
        # SABR base alpha = NormalVol / (F0^beta)
        base_normal_vols = torch.tensor(alpha_f(tenors[:-1]), device=device, dtype=self.dtype)
        eta_F0 = torch.pow(torch.abs(self.F0 + self.shift), self.beta_sabr)
        self.alphas = nn.Parameter(base_normal_vols / eta_F0)
        
        self.register_buffer('rhos', torch.tensor(rho_f(tenors[:-1]), device=device, dtype=self.dtype))
        self.register_buffer('nus', torch.tensor(nu_f(tenors[:-1]), device=device, dtype=self.dtype))
        self.register_buffer('H', torch.tensor(H, device=device, dtype=self.dtype))

# Spatial Correlation Matrix Setup
        if self.correlation_mode == 'pca':
            self.n_factors = n_factors
            T_mat_i, T_mat_j = self.T[:-1].unsqueeze(1), self.T[:-1].unsqueeze(0)
            spatial_corr = torch.exp(-beta_decay * torch.abs(T_mat_i - T_mat_j))
            evals, evecs = torch.linalg.eigh(spatial_corr)
            idx = torch.argsort(evals, descending=True)
            loadings = evecs[:, idx[:self.n_factors]] * torch.sqrt(torch.clamp(evals[idx[:self.n_factors]], min=0.0)).unsqueeze(0)
            
            self.register_buffer('loadings', loadings / torch.sqrt(torch.sum(loadings**2, dim=1, keepdim=True)))
            self.register_buffer('Lambda_upper', torch.triu(self.loadings @ self.loadings.T, diagonal=1))
            
        elif self.correlation_mode == 'full':
            self.n_factors = self.N + 1  # Full rank (N forward rates + 1 Vol driver)
            
            if omega_matrix is not None:
                # Use Adachi's Rapisarda Hyperspherical Angles
                omega_tensor = torch.tensor(omega_matrix, device=device, dtype=self.dtype)
                spatial_corr = build_rapisarda_correlation_matrix(omega_tensor)
            else:
                # Fallback to standard exponential decay if no angles provided
                T_mat_i, T_mat_j = self.T[:-1].unsqueeze(1), self.T[:-1].unsqueeze(0)
                spatial_corr_N = torch.exp(-beta_decay * torch.abs(T_mat_i - T_mat_j))
                spatial_corr = torch.eye(self.N + 1, device=device, dtype=self.dtype)
                spatial_corr[1:, 1:] = spatial_corr_N
                
            # Add a tiny jitter to the diagonal for numerical stability of Cholesky
            jitter = torch.eye(self.N + 1, device=device, dtype=self.dtype) * 1e-10
            loadings = torch.linalg.cholesky(spatial_corr + jitter)
            
            self.register_buffer('loadings', loadings / torch.sqrt(torch.sum(loadings**2, dim=1, keepdim=True)))
            # Lambda_upper must strictly be NxN for the forward rates drift computation
            corr_rates = self.loadings[1:, :] @ self.loadings[1:, :].T
            self.register_buffer('Lambda_upper', torch.triu(corr_rates, diagonal=1))
            
        else:
            raise ValueError(f"Unknown correlation_mode: {self.correlation_mode}")


    def get_terminal_bond(self):
        return torch.prod(1.0 / (1.0 + self.tau * self.F0))


    def simulate_forward_curve(self, n_paths, time_grid, seed=56, freeze_drift=True, use_checkpoint=False, scheme='hybrid', kappa=1, b_eval='optimal'):
        from torch.distributions import Normal

        n_steps = len(time_grid) - 1
        dt = (time_grid[1] - time_grid[0]).to(self.dtype)
        
        # 1. Pre-calculate Kernel & Martingale Correction
        t, s = time_grid[1:].to(self.dtype), time_grid[:-1].to(self.dtype)
        gamma_const = torch.exp(torch.lgamma(self.H + 0.5))
        alpha = self.H - 0.5
        diag_idx = torch.arange(n_steps, device=self.device)
        
        if scheme == 'hybrid':
            # Hybrid Scheme (Bennedsen et al. 2017)
            # k_mat represents the lag step k = 1, 2, ...
            k_mat = torch.clamp(diag_idx[:, None] - diag_idx[None, :] + 1, min=0.0).to(self.dtype)

            # 1. Exact integration of power function near zero (for k <= kappa)
            exact_weights = (torch.pow(k_mat, alpha + 1.0) - torch.pow(torch.clamp(k_mat - 1.0, min=0.0), alpha + 1.0)) / (alpha + 1.0)
            
            # 2. Step function evaluation (Riemann sum) elsewhere (for k > kappa)
            if b_eval == 'optimal':
                # Optimal evaluation points b_k^* from Proposition 2.8
                riemann_weights = exact_weights
            else:
                # Forward Riemann evaluation points (b_k = k)
                riemann_weights = torch.pow(k_mat, alpha)
                
            weights = torch.where(k_mat <= kappa, exact_weights, riemann_weights)
            kernel = weights * torch.pow(dt, alpha)
            
        else:
            # Modified Riemann (Original implementation)
            dt_mat = torch.clamp(t[:, None] - s[None, :], min=0.0)
            kernel = torch.pow(torch.clamp(dt_mat - 0.5 * dt, min=1e-12), alpha)
            kernel[diag_idx, diag_idx] = torch.pow(dt, alpha) / (alpha + 1.0)

        # divide kernel matrix by gamma func and zero-out the upper-triangle 
        kernel = torch.where(t[:, None] > s[None, :], kernel / gamma_const, 0.0)
        
        # Discrete Martingale Correction (remains N x Steps)
        var_comp = 0.5 * (self.nus**2).unsqueeze(-1) * torch.sum(kernel**2, dim=1).unsqueeze(0) * dt

        n_dims = self.n_factors + 1 if self.correlation_mode == 'pca' else self.n_factors
        
        # sobol generator
        sobol = torch.quasirandom.SobolEngine(dimension=n_steps * n_dims, scramble=True, seed=seed)
        # draws
        u = sobol.draw(n_paths).to(self.device).to(self.dtype)
        # from uniform to standard normal distribution
        dist = Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        # shocks
        z = dist.icdf(torch.clamp(u, 1e-7, 1-1e-7)).view(n_paths, n_steps, n_dims)
        
        # time-scaled shocks
        dz_all = z * torch.sqrt(dt)


        # 3. Define Simulation Block
        def _simulate(dz_all_in, kern, vcomp, f0, alphas):
            if self.correlation_mode == 'pca':
                dz_c = dz_all_in[..., :-1]
                dz_p = dz_all_in[..., -1:]
                wr = torch.matmul(dz_c, self.loadings.T)
                
                fbm_4 = torch.matmul(dz_all_in.transpose(1, 2), kern.T).transpose(1, 2)
                fbm_curve = torch.matmul(fbm_4[..., :-1], self.loadings.T)
                fbm = self.rhos.view(1, 1, -1) * fbm_curve + torch.sqrt(1.0 - self.rhos**2).view(1, 1, -1) * fbm_4[..., -1:]
            else:
                # Full Mode (Adachi N+1 Matrix)
                wr_all = torch.matmul(dz_all_in, self.loadings.T)
                wr = wr_all[..., 1:] # Forward rates are rows 1 to N
                
                fbm_indep = torch.matmul(dz_all_in.transpose(1, 2), kern.T).transpose(1, 2)
                fbm_all = torch.matmul(fbm_indep, self.loadings.T)
                fbm = fbm_all[..., 0:1] # Rough Volatility Z(t) is Row 0
                
            # Stochastic Volatility
            unit_vols = torch.exp(self.nus.view(1, 1, -1) * fbm - vcomp.T.unsqueeze(0))
            v2_dt = (unit_vols ** 2 * dt)

            
            if freeze_drift:
                # 1. Calculate the shifted SABR local vol component for the frozen initial curve
                eta_f0 = torch.pow(torch.abs(f0 + self.shift), self.beta_sabr)
                vol_scale = alphas * eta_f0
                
                # 2. Apply it to the drift (omega)
                omega_0 = (self.tau * vol_scale) / (1.0 + self.tau * f0)
                mu_0 = -vol_scale * torch.matmul(self.Lambda_upper, omega_0)
                
                dF = mu_0.view(1, 1, -1) * v2_dt
                # 3. Apply it to the diffusion (shock)
                dF.add_(wr * (unit_vols * vol_scale.view(1, 1, -1)))
                return torch.cat([f0.expand(n_paths, 1, -1), f0 + torch.cumsum(dF, dim=1)], dim=1)
            
            else:
                F_t = f0.expand(n_paths, self.N).clone()
                res = [F_t.unsqueeze(1)]
                for i in range(n_steps):
                    # 1. Dynamic local vol component evaluated at current F_t
                    eta_F = torch.pow(torch.abs(F_t + self.shift), self.beta_sabr)
                    vol_scale = alphas * eta_F
                    
                    # 2. Dynamic drift adjustment
                    omega = (self.tau * vol_scale) / (1.0 + self.tau * F_t)
                    drift_weights = torch.einsum('ij,pj->pi', self.Lambda_upper, omega)
                    drift = -vol_scale * drift_weights * v2_dt[:, i]
                    
                    # 3. Dynamic diffusion adjustment
                    F_t = F_t + drift + wr[:, i] * (unit_vols[:, i] * vol_scale)
                    res.append(F_t.unsqueeze(1))
                return torch.cat(res, dim=1)
            

        if use_checkpoint and torch.is_grad_enabled():
            import torch.utils.checkpoint as cp
            return cp.checkpoint(_simulate, dz_all, kernel, var_comp, self.F0, self.alphas, use_reentrant=False)
        else:
            return _simulate(dz_all, kernel, var_comp, self.F0, self.alphas)