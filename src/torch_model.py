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
        
        # FIX 1: Clamp uniform draws to prevent inf / -inf from the ICDF
        u = torch.clamp(u, min=1e-7, max=1.0 - 1e-7)
        
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
        
        import math 
        gamma_factor = math.gamma(self.H.item() + 0.5)
        
        # FIX 2: Clamp dt_mat before the fractional power to prevent 0**(-0.4) = inf
        dt_mat_safe = torch.clamp(dt_mat, min=1e-12)
        
        kernel = torch.where(dt_mat > 0, dt_mat_safe**(self.H - 0.5) / gamma_factor, 
                             torch.tensor(0.0, device=self.device, dtype=self.dtype))
        
        fBm = torch.matmul(dW_v, kernel.T)
        
        # (Your correctly patched compensator)
        var_comp = 0.5 * (self.nus[0]**2) * (t**(2*self.H)) / (2.0 * self.H * (gamma_factor**2))
        
        # Unit volatility driver
        unit_vols = torch.exp(self.nus[0] * fBm - var_comp)
        
        ones = torch.zeros(n_paths, 1, device=self.device, dtype=self.dtype)
        return torch.cumsum(torch.cat([ones, unit_vols * dW_r], dim=1), dim=1)


    def get_terminal_bond(self):
        """ Differentiable P(0, Tn) bond """
        dfs = 1.0 / (1.0 + self.tau * self.F0)
        return torch.prod(dfs)


    def simulate_forward_curve(self, n_paths, time_grid, seed=56, freeze_drift=True):
        """ 
        Simulates the full arbitrage-free Forward Market Model (FMM) curve under 
        the Terminal Swap Measure P(., T_N).
        """
        n_steps = len(time_grid) - 1
        dt = (time_grid[1] - time_grid[0]).to(self.dtype)
        
        # 1. Generate QMC Sobol Shocks (Safely Clamped)
        sobol = torch.quasirandom.SobolEngine(dimension=n_steps * 2, scramble=True, seed=seed)
        u = sobol.draw(n_paths).to(self.device).to(self.dtype)
        u = torch.clamp(u, min=1e-7, max=1.0 - 1e-7)
        
        from torch.distributions import Normal
        dist = Normal(torch.tensor(0.0, device=self.device, dtype=self.dtype), 
                      torch.tensor(1.0, device=self.device, dtype=self.dtype))
        z = dist.icdf(u).view(n_paths, n_steps, 2)
        
        dW_v = z[:, :, 0] * torch.sqrt(dt)
        dW_r = (self.rhos[0] * z[:, :, 0] + torch.sqrt(1.0 - self.rhos[0]**2) * z[:, :, 1]) * torch.sqrt(dt)
        
        # 2. Rough Volterra Kernel
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
        dM = unit_vols * dW_r          # Martingale increments
        
        # 4. No-Arbitrage FMM Drift & Simulation
        if freeze_drift:
            # --- FROZEN DRIFT (Fast, Deterministic Weights) ---
            # omega_j = (tau_j * alpha_j) / (1 + tau_j * F_j(0))
            omega = (self.tau * self.alphas) / (1.0 + self.tau * self.F0)
            
            # Vectorized right-to-left sum for j = i+1 to N-1
            omega_shifted = torch.cat([omega[1:], torch.zeros(1, device=self.device, dtype=self.dtype)])
            drift_weights = torch.flip(torch.cumsum(torch.flip(omega_shifted, dims=[0]), dim=0), dims=[0])
            
            # Deterministic Drift Multiplier: mu_i(0)
            mu_0 = -self.alphas * drift_weights 
            
            # Expand to (n_paths, n_steps, N)
            drift_term = mu_0.unsqueeze(0).unsqueeze(0) * V2_dt.unsqueeze(-1)
            martingale_term = self.alphas.unsqueeze(0).unsqueeze(0) * dM.unsqueeze(-1)
            
            dF = drift_term + martingale_term
            
            # Integrate over time
            F_0_expanded = self.F0.expand(n_paths, 1, self.N)
            F_paths = torch.cat([F_0_expanded, self.F0 + torch.cumsum(dF, dim=1)], dim=1)
            
        else:
            # --- UNFROZEN EXACT DRIFT (Slower, Perfect Arbitrage-Free) ---
            F_t = self.F0.expand(n_paths, self.N).clone()
            F_paths = [F_t.unsqueeze(1)]
            
            for step in range(n_steps):
                # omega dynamically depends on stochastic F_t
                omega = (self.tau * self.alphas) / (1.0 + self.tau * F_t)
                
                # Vectorized right-to-left sum for each path
                omega_shifted = torch.cat([omega[:, 1:], torch.zeros(n_paths, 1, device=self.device, dtype=self.dtype)], dim=1)
                drift_weights = torch.flip(torch.cumsum(torch.flip(omega_shifted, dims=[1]), dim=1), dims=[1])
                
                # Stochastic Drift
                mu_t = -self.alphas * drift_weights * V2_dt[:, step].unsqueeze(1)
                
                # Step forward
                dF_t = mu_t + self.alphas * dM[:, step].unsqueeze(1)
                F_t = F_t + dF_t
                F_paths.append(F_t.unsqueeze(1))
                
            F_paths = torch.cat(F_paths, dim=1)
            
        return F_paths