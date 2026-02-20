import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
import torch
from torch.distributions import Normal
import math
from scipy.stats import norm


def mc_rough_bergomi_pricer(K_flat, T_flat, alpha_flat, rho_flat, nu, H, n_paths=32768, dt=1.0/50.0, kappa_hybrid=1, device='cpu'):
    """
    Ultra-fast PyTorch Monte Carlo Engine for 1D Rough Bergomi pricing.
    Uses the Bennedsen Hybrid Scheme and Bachelier Control Variates for massive variance reduction.
    """
    dtype = torch.float64
    
    # 1. Setup Time Grid (simulate once up to the longest expiry)
    max_T = float(torch.max(T_flat))
    n_steps = int(math.ceil(max_T / dt))
    dt_tensor = torch.tensor(dt, dtype=dtype, device=device)
    
    # 2. Vectorized Hybrid Scheme Kernel (Bennedsen et al. 2017)
    H_tensor = torch.tensor(H, dtype=dtype, device=device)
    gamma_const = torch.exp(torch.lgamma(H_tensor + 0.5))
    alpha_H = H - 0.5
    diag_idx = torch.arange(n_steps, device=device)
    k_mat = torch.clamp(diag_idx[:, None] - diag_idx[None, :] + 1, min=0.0).to(dtype)
    
    exact_weights = (torch.pow(k_mat, alpha_H + 1.0) - torch.pow(torch.clamp(k_mat - 1.0, min=0.0), alpha_H + 1.0)) / (alpha_H + 1.0)
    weights = torch.where(k_mat <= kappa_hybrid, exact_weights, exact_weights) # optimal evaluation
    kernel = weights * torch.pow(dt_tensor, alpha_H) / gamma_const
    
    # Zero-out upper triangle for causality
    t_idx, s_idx = diag_idx[:, None], diag_idx[None, :]
    kernel = torch.where(t_idx >= s_idx, kernel, torch.tensor(0.0, dtype=dtype, device=device))
    
    # Variance compensator for exact discrete martingale
    var_comp = 0.5 * (nu**2) * torch.sum(kernel**2, dim=1) * dt
    
    # 3. Sobol Random Draws (Anti-thetic / Scrambled)
    sobol = torch.quasirandom.SobolEngine(dimension=n_steps * 2, scramble=True, seed=42)
    u = sobol.draw(n_paths).to(device).to(dtype)
    dist = Normal(torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype))
    z = dist.icdf(torch.clamp(u, 1e-7, 1-1e-7)).view(n_paths, n_steps, 2)
    
    dz_vol = z[..., 0] * torch.sqrt(dt_tensor)
    dz_perp = z[..., 1] * torch.sqrt(dt_tensor)
    
    # 4. Global Volatility Process Generation
    fbm = torch.matmul(dz_vol, kernel.T)
    unit_vols = torch.exp(nu * fbm - var_comp.unsqueeze(0))
    
    # SHIFT volatility by 1 step to make it predictable (Euler-Maruyama strict adherence)
    # This prevents forward leakage and the massive spurious negative drift.
    unit_vols_shifted = torch.cat([torch.ones(n_paths, 1, device=device, dtype=dtype), unit_vols[:, :-1]], dim=1)
    
    # Pre-map expiries to step indices
    step_indices = torch.clamp(torch.round(T_flat / dt).long() - 1, 0, n_steps - 1)
    prices = torch.zeros(len(T_flat), dtype=dtype, device=device)
    
    # 5. Fast Pricing Loop with Control Variates
    for i in range(len(T_flat)):
        idx = step_indices[i]
        a = alpha_flat[i]
        r = rho_flat[i]
        k = K_flat[i]
        T = T_flat[i]
        
        # Spot process correlated shocks
        dz_spot = r * dz_vol[:, :idx+1] + torch.sqrt(1.0 - r**2) * dz_perp[:, :idx+1]
        
        # Target Path: Rough Bergomi (Normal / Bachelier scaling)
        vol_path = unit_vols_shifted[:, :idx+1]
        dS = a * vol_path * dz_spot
        S_T = torch.sum(dS, dim=1)
        
        # Shadow Path: Standard Bachelier (Constant Volatility)
        dS_cv = a * dz_spot
        S_T_cv = torch.sum(dS_cv, dim=1)
        
        # Call Payoffs
        payoff_rb = torch.clamp(S_T - k, min=0.0)
        payoff_cv = torch.clamp(S_T_cv - k, min=0.0)
        
        # True Analytical Control Variate Price (Bachelier)
        std = a * torch.sqrt(T)
        d = -k / std if std > 1e-12 else torch.tensor(0.0, device=device, dtype=dtype)
        true_cv_price = std * (dist.cdf(d) * d + torch.exp(dist.log_prob(d)))
        
        # Dynamic Beta calculation for Variance Reduction
        cov = torch.mean((payoff_rb - torch.mean(payoff_rb)) * (payoff_cv - torch.mean(payoff_cv)))
        var = torch.var(payoff_cv)
        beta = cov / var if var > 1e-12 else 0.0
        
        # Final noise-reduced MC Price
        mc_price = torch.mean(payoff_rb - beta * (payoff_cv - true_cv_price))
        prices[i] = mc_price
        
    return prices


def bachelier_iv_newton(target_prices, K, T, initial_guess_vol, max_iter=20, tol=1e-8):
    """
    Vectorized Newton-Raphson root finder to invert Bachelier prices to Normal Implied Volatility.
    """
    k = np.array(K)
    # Safeguard: Option price MUST be >= Intrinsic value, otherwise IV is mathematically undefined.
    # Prevents Newton solver from diverging to infinity if MC noise pushes price slightly too low.
    intrinsic = np.maximum(-k, 0.0)
    P = np.maximum(np.array(target_prices), intrinsic + 1e-12)
    t = np.array(T)
    sigma = np.array(initial_guess_vol) # starting guess (typically the base alpha)
    
    sqrt_t = np.sqrt(t)
    
    for _ in range(max_iter):
        std = sigma * sqrt_t
        # Guard against division by zero
        std = np.where(std < 1e-12, 1e-12, std)
        d = -k / std
        
        # Calculate Bachelier Price with current sigma
        pdf = norm.pdf(d)
        cdf = norm.cdf(d)
        price_current = std * (d * cdf + pdf)
        
        # Calculate Bachelier Vega
        vega = sqrt_t * pdf
        vega = np.where(vega < 1e-12, 1e-12, vega)
        
        # Newton-Raphson step
        diff = price_current - P
        step = diff / vega
        sigma = sigma - step
        
        # Break early if all points converge
        if np.max(np.abs(diff)) < tol:
            break
            
    return np.abs(sigma)


class RoughSABRCalibrator:
    def __init__(self, vol_matrix):
        self.vol_matrix = vol_matrix
        # Ensure expiries and strikes are floats for math operations
        self.expiries = self.vol_matrix.index.values.astype(float)
        self.strike_offsets = self.vol_matrix.columns.values.astype(float)
        self.n_exp = len(self.expiries)
        
        # Pre-calculate ATM Term Structure: alpha(T)
        atm_idx = np.argmin(np.abs(self.strike_offsets))
        self.alpha_ts = PchipInterpolator(self.expiries, self.vol_matrix.iloc[:, atm_idx].values, extrapolate=True)
        
        # Flatten market data for the global optimizer
        self.market_vols = self.vol_matrix.values.flatten()
        T_grid, K_grid = np.meshgrid(self.expiries, self.strike_offsets, indexing='ij')
        self.T_flat, self.K_flat = T_grid.flatten(), K_grid.flatten()
        
        # Clean out NaNs if any exist in the market data
        valid = ~np.isnan(self.market_vols)
        self.market_vols = self.market_vols[valid]
        self.T_flat = self.T_flat[valid]
        self.K_flat = self.K_flat[valid]


    def rough_sabr_vol(self, k, T, alpha, rho, nu, H):
        """ Core Rough SABR asymptotic expansion formula """
        # Skew term scales with T^(H-0.5)
        skew = (rho * nu) / (2.0 * alpha * (H + 0.5)) * k * (T**(H - 0.5))
        # Smile term scales with T^(2H-1)
        smile = (2.0 - 3.0*rho**2)/24.0 * (nu/alpha)**2 * (k**2) * (T**(2.0*H - 1.0))
        # ATM Drift scales with T^(2H)
        drift = (2.0 - 3.0*rho**2)/24.0 * nu**2 * (T**(2.0*H))
        
        return alpha * (1.0 + skew + smile + drift)


    def rough_sabr_vol_ode(self, k, T, alpha, rho, nu, H):
        """ 
        Advanced Rough SABR closed-form approximation using the ODE solution 
        from Fukasawa & Gatheral (2022).
        """
        # Calculate the scaled moneyness y
        y = (nu * (T**(H - 0.5)) * k) / alpha
        
        # Clip rho for numerical stability in log and arctan operations
        rho_safe = np.clip(rho, -0.9999, 0.9999)
        
        def G_half(z):
            inner = np.sqrt(1.0 + rho_safe * z + z**2 / 4.0) - rho_safe - z / 2.0
            return 4.0 * (np.log(inner / (1.0 - rho_safe)))**2
            
        def G_zero(z):
            term1 = np.log(1.0 + 2.0 * rho_safe * z + z**2)
            denom = np.sqrt(1.0 - rho_safe**2)
            term2 = (2.0 * rho_safe / denom) * (np.arctan(rho_safe / denom) - np.arctan((z + rho_safe) / denom))
            return term1 + term2

        # Avoid division by zero at exactly ATM (k=0)
        safe_y = np.where(np.abs(y) < 1e-12, 1e-12, y)
        
        z0 = safe_y / (2.0 * H + 1.0)
        z_half = 2.0 * safe_y / (2.0 * H + 1.0)
        
        w0 = 3.0 * (1.0 - 2.0 * H) / (2.0 * H + 3.0)
        w_half = 2.0 * H / (2.0 * H + 3.0)
        
        # Interpolate between H=0 and H=1/2 extreme solutions
        G_A = ((2.0 * H + 1.0)**2) * (w0 * G_zero(z0) + w_half * G_half(z_half))
        
        # Prevent floating-point negatives and division-by-zero
        G_A_safe = np.clip(G_A, a_min=1e-14, a_max=None)
        
        # The limit of |y|/sqrt(G_A) as y -> 0 is 1.0
        ratio = np.where(np.abs(y) < 1e-12, 1.0, np.abs(safe_y) / np.sqrt(G_A_safe))

        
        return alpha * ratio


    def _obj(self, p, H, method):
        # Unpack params: [rho_1...rho_n, nu_global]
        rhos, nu_global = p[:self.n_exp], p[-1]
        
        # Interpolate rho(T) term structure
        r_ts = PchipInterpolator(self.expiries, rhos, extrapolate=True)
        
        # Select the pricing method
        if method == 'polynomial':
            v = self.rough_sabr_vol(
                self.K_flat, self.T_flat, 
                self.alpha_ts(self.T_flat), 
                r_ts(self.T_flat), 
                nu_global, 
                H
            )
        elif method == 'ODE':
            v = self.rough_sabr_vol_ode(
                self.K_flat, self.T_flat, 
                self.alpha_ts(self.T_flat), 
                r_ts(self.T_flat), 
                nu_global, 
                H
            )
        elif method == 'MC':
            v = self.rough_sabr_vol_mc(
                self.K_flat, self.T_flat, 
                self.alpha_ts(self.T_flat), 
                r_ts(self.T_flat), 
                nu_global, 
                H
            )
        else:
            raise ValueError(f"Unknown calibration method: {method}")
            
        return (v - self.market_vols) * 10000.0


    def calibrate(self, method='MC', H_grid=np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])):
        print("\n" + "="*60)
        print(f"{f'ROUGH SABR CALIBRATION (GLOBAL NU | {method.upper()})':^60}")
        print("="*60)
        
        best_rmse = np.inf
        best_H = None
        best_res = None
        
        # Initial guess: [Rhos, Nu_global]
        guess = np.concatenate([np.full(self.n_exp, -0.1), [0.4]])
        low_bounds = np.concatenate([np.full(self.n_exp, -0.99), [0.001]])
        high_bounds = np.concatenate([np.full(self.n_exp, 0.99), [5.0]])
        
        # Discrete Grid Search over Hurst exponent
        for H in H_grid:
            res = least_squares(
                self._obj, guess, args=(H, method),
                bounds=(low_bounds, high_bounds), 
                ftol=1e-10, xtol=1e-10
            )
            rmse = np.sqrt(np.mean(res.fun**2))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_H = H
                best_res = res
                
        print(f"Status : SUCCESS")
        print(f"Global Hurst (H): {best_H:.6f}")
        print(f"Global Nu       : {best_res.x[-1]:.4f}")
        print(f"RMSE            : {best_rmse:.4f} bps")
        
        best_rhos = best_res.x[:self.n_exp]
        best_nu = best_res.x[-1]
        
        return {
            'alpha_func': self.alpha_ts, 
            'H': best_H, 
            'rmse_bps': best_rmse,
            'rho_func': PchipInterpolator(self.expiries, best_rhos, extrapolate=True),
            # Returning constant interpolator for nu to keep downstream FMM compatibility 
            'nu_func': PchipInterpolator(self.expiries, np.full(self.n_exp, best_nu), extrapolate=True)
        }
    

    def rough_sabr_vol_mc(self, k, T, alpha, rho, nu, H):
        """ 
        True Adachi et al. (2025) calibration mapping. 
        Uses the 1D Rough Bergomi PyTorch engine and inverts to Bachelier IV.
        """
        # Convert numpy arrays to torch tensors for the engine
        device = 'cpu'
        K_t = torch.tensor(k, device=device, dtype=torch.float64)
        T_t = torch.tensor(T, device=device, dtype=torch.float64)
        alpha_t = torch.tensor(alpha, device=device, dtype=torch.float64)
        rho_t = torch.tensor(rho, device=device, dtype=torch.float64)
        
        # Run the massive PyTorch Monte Carlo engine
        mc_prices_t = mc_rough_bergomi_pricer(
            K_t, T_t, alpha_t, rho_t, nu, H, 
            n_paths=16384, # high enough for optimizer stability, low enough for speed
            dt=1.0/24.0,   # roughly bi-weekly steps
            kappa_hybrid=1, 
            device=device
        )
        
        # Bring prices back to CPU numpy
        mc_prices = mc_prices_t.cpu().numpy()
        
        # Invert prices back to Implied Volatility
        ivs = bachelier_iv_newton(mc_prices, k, T, initial_guess_vol=alpha)
        
        return ivs