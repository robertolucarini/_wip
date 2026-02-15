import torch
import torch.nn as nn
import numpy as np

class TorchRoughSABR_FMM(nn.Module):
    def __init__(self, tenors, F0, alpha_f, rho_f, nu_f, H, beta=0.05, n_factors=3, device='cpu'):
        super().__init__()
        self.device = device
        self.dtype = torch.float64
        self.n_factors = n_factors
        
        # Buffers for simulation constants
        self.register_buffer('T', torch.tensor(tenors, device=device, dtype=self.dtype))
        self.register_buffer('tau', torch.tensor(np.diff(tenors), device=device, dtype=self.dtype))
        self.N = len(F0)
        
        # Parameters for AAD (Requires Grad)
        self.F0 = nn.Parameter(torch.tensor(F0, device=device, dtype=self.dtype))
        self.alphas = nn.Parameter(torch.tensor(alpha_f(tenors[:-1]), device=device, dtype=self.dtype))
        
        # Static Parameters (Buffers)
        self.register_buffer('rhos', torch.tensor(rho_f(tenors[:-1]), device=device, dtype=self.dtype))
        self.register_buffer('nus', torch.tensor(nu_f(tenors[:-1]), device=device, dtype=self.dtype))
        self.register_buffer('H', torch.tensor(H, device=device, dtype=self.dtype))

        # -----------------------------------------------------------------
        # SPATIAL PCA FOR CURVE DECORRELATION
        # -----------------------------------------------------------------
        # 1. Parametric Spatial Correlation Matrix: rho_ij = exp(-beta * |T_i - T_j|)
        T_mat_i = self.T[:-1].unsqueeze(1)
        T_mat_j = self.T[:-1].unsqueeze(0)
        spatial_corr = torch.exp(-beta * torch.abs(T_mat_i - T_mat_j))
        
        # 2. Eigenvalue Decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(spatial_corr)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        
        # 3. Compute Factor Loadings (Keep top n_factors)
        top_evals = torch.clamp(eigenvalues[:n_factors], min=0.0)
        top_evecs = eigenvectors[:, :n_factors]
        loadings = top_evecs * torch.sqrt(top_evals).unsqueeze(0)
        
        # Normalize rows to ensure total variance per tenor remains exactly 1.0
        row_norms = torch.sqrt(torch.sum(loadings**2, dim=1, keepdim=True))
        self.register_buffer('loadings', loadings / row_norms)
        
        # 4. Precompute Lambda matrix for FMM Drift (Covariance between tenors)
        Lambda = self.loadings @ self.loadings.T
        # We only need the strict upper triangle for the drift summation (j > i)
        self.register_buffer('Lambda_upper', torch.triu(Lambda, diagonal=1))

    def get_terminal_bond(self):
        """ Differentiable P(0, Tn) bond """
        dfs = 1.0 / (1.0 + self.tau * self.F0)
        return torch.prod(dfs)

    def simulate_forward_curve(self, n_paths, time_grid, seed=56, freeze_drift=True):
        """ 
        Multi-Factor Arbitrage-Free Forward Market Model Simulation.
        """
        n_steps = len(time_grid) - 1
        dt = (time_grid[1] - time_grid[0]).to(self.dtype)
        
        # 1. Generate QMC Sobol Shocks
        # We need n_factors for the curve, plus 1 orthogonal shock for the rough vol
        dimension = n_steps * (self.n_factors + 1)
        sobol = torch.quasirandom.SobolEngine(dimension=dimension, scramble=True, seed=seed)
        u = sobol.draw(n_paths).to(self.device).to(self.dtype)
        u = torch.clamp(u, min=1e-7, max=1.0 - 1e-7)
        
        from torch.distributions import Normal
        dist = Normal(torch.tensor(0.0, device=self.device, dtype=self.dtype), 
                      torch.tensor(1.0, device=self.device, dtype=self.dtype))
        
        # Shape: (n_paths, n_steps, n_factors + 1)
        z = dist.icdf(u).view(n_paths, n_steps, self.n_factors + 1)
        
        # Curve Drivers (n_factors)
        dZ_curve = z[..., :self.n_factors] * torch.sqrt(dt)
        # Volatility Driver (Correlated to the 1st Principal Component to maintain Skew)
        dZ_vol_perp = z[..., self.n_factors] * torch.sqrt(dt)
        dW_v = self.rhos[0] * dZ_curve[..., 0] + torch.sqrt(1.0 - self.rhos[0]**2) * dZ_vol_perp
        
        # 2. Rough Volterra Kernel (Applied to Volatility factor)
        t, s = time_grid[1:].to(self.dtype), time_grid[:-1].to(self.dtype)
        dt_mat = torch.clamp(t[:, None] - s[None, :], min=1e-12)
        
        import math 
        gamma_factor = math.gamma(self.H.item() + 0.5)
        kernel = torch.where(t[:, None] - s[None, :] > 0, dt_mat**(self.H - 0.5) / gamma_factor, 
                             torch.tensor(0.0, device=self.device, dtype=self.dtype))
        
        fBm = torch.matmul(dW_v, kernel.T)
        var_comp = 0.5 * (self.nus[0]**2) * (t**(2*self.H)) / (2.0 * self.H * (gamma_factor**2))
        unit_vols = torch.exp(self.nus[0] * fBm - var_comp)
        
        # 3. Base process components
        V2_dt = (unit_vols ** 2) * dt  # Variance scaled by dt
        
        # Map K independent curve shocks to N correlated tenors using PCA Loadings
        # Shape: (n_paths, n_steps, N)
        dW_r = torch.einsum('psk,nk->psn', dZ_curve, self.loadings)
        dM = unit_vols.unsqueeze(2) * dW_r  # Martingale increments
        
        # 4. No-Arbitrage FMM Drift & Simulation
        if freeze_drift:
            # --- FROZEN DRIFT (Fast Matrix Multiplication) ---
            omega = (self.tau * self.alphas) / (1.0 + self.tau * self.F0)
            
            # The upper triangular dot product exactly computes sum(Lambda_ij * omega_j) for j > i
            drift_weights = torch.matmul(self.Lambda_upper, omega)
            mu_0 = -self.alphas * drift_weights 
            
            drift_term = mu_0.unsqueeze(0).unsqueeze(0) * V2_dt.unsqueeze(-1)
            martingale_term = self.alphas * dM
            
            dF = drift_term + martingale_term
            F_0_expanded = self.F0.expand(n_paths, 1, self.N)
            F_paths = torch.cat([F_0_expanded, self.F0 + torch.cumsum(dF, dim=1)], dim=1)
            
        else:
            # --- UNFROZEN EXACT DRIFT ---
            F_t = self.F0.expand(n_paths, self.N).clone()
            F_paths = [F_t.unsqueeze(1)]
            
            for step in range(n_steps):
                omega = (self.tau * self.alphas) / (1.0 + self.tau * F_t)
                
                # Dynamic batch matrix multiplication
                drift_weights = torch.einsum('ij,pj->pi', self.Lambda_upper, omega)
                mu_t = -self.alphas * drift_weights * V2_dt[:, step].unsqueeze(1)
                
                dF_t = mu_t + self.alphas * dM[:, step, :]
                F_t = F_t + dF_t
                F_paths.append(F_t.unsqueeze(1))
                
            F_paths = torch.cat(F_paths, dim=1)
            
        return F_paths