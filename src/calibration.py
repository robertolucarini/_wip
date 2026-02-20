import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
import torch
from torch.distributions import Normal
import math
from scipy.stats import norm
from src.utils import build_rapisarda_correlation_matrix
from src.pricers import mapped_smm_pricer
import time
import pandas as pd



def mc_rough_bergomi_pricer(K_flat, T_flat, alpha_flat, rho_flat, nu, H, n_paths=(32768/2), dt=1.0/50.0, kappa_hybrid=1, device='cpu'):
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

        # --- NEW AMMO CODE: Pre-computations for the PyTorch Jacobian ---
        # Map each point in the flattened grid directly to its expiry index
        self.t_indices = torch.tensor([np.where(self.expiries == t)[0][0] for t in self.T_flat], dtype=torch.long)
        self.K_flat_t = torch.tensor(self.K_flat, dtype=torch.float64)
        self.T_flat_t = torch.tensor(self.T_flat, dtype=torch.float64)
        self.alpha_flat_t = torch.tensor(self.alpha_ts(self.T_flat), dtype=torch.float64)


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
        import time
        print("\n" + "="*60)
        print(f"{f'ROUGH SABR 1D CALIBRATION (GLOBAL NU | {method.upper()})':^60}")
        print("="*60)
        
        best_rmse = np.inf
        best_H = None
        best_res_x = None
        best_alpha_vals = None
        
        # 1. Isolate ATM market data for exact explicit matching
        atm_idx = np.argmin(np.abs(self.strike_offsets))
        atm_val_strike = self.strike_offsets[atm_idx]
        base_market_alphas = self.vol_matrix.iloc[:, atm_idx].values.copy()
        
        # 2. REDUCED PARAMETER SPACE: ONLY [rhos (n_exp), nu_global (1)]
        guess = np.concatenate([np.full(self.n_exp, -0.1), [0.4]])
        low_bounds = np.concatenate([np.full(self.n_exp, -0.99), [0.001]])
        high_bounds = np.concatenate([np.full(self.n_exp, 0.99), [5.0]])
        
        start_time_total = time.time()
        
        for i, H in enumerate(H_grid):
            step_start_time = time.time()
            print(f"Grid {i+1:2d}/{len(H_grid)} | Testing Hurst (H) = {H:.3f} | ", end="", flush=True)
            
            # Fast ODE Helper (alphas are now passed externally)
            def run_ode(alphas_in, rhos_in, nu_in):
                a_ts = PchipInterpolator(self.expiries, alphas_in, extrapolate=True)
                r_ts = PchipInterpolator(self.expiries, rhos_in, extrapolate=True)
                return self.rough_sabr_vol_ode(self.K_flat, self.T_flat, a_ts(self.T_flat), r_ts(self.T_flat), nu_in, H)

            # Fast Polynomial Helper
            def run_poly(alphas_in, rhos_in, nu_in):
                a_ts = PchipInterpolator(self.expiries, alphas_in, extrapolate=True)
                r_ts = PchipInterpolator(self.expiries, rhos_in, extrapolate=True)
                return self.rough_sabr_vol(self.K_flat, self.T_flat, a_ts(self.T_flat), r_ts(self.T_flat), nu_in, H)

            if method == 'polynomial':
                # For pure polynomial, alpha IS the market ATM
                current_alphas = base_market_alphas.copy()
                def obj_poly(p): 
                    return (run_poly(current_alphas, p[:-1], p[-1]) - self.market_vols) * 10000.0
                res = least_squares(obj_poly, guess, bounds=(low_bounds, high_bounds), method='trf')
                rmse = np.sqrt(np.mean(res.fun**2))
                current_p = res.x
                final_alphas = current_alphas

            elif method == 'ODE':
                # For pure ODE, alpha IS the market ATM
                current_alphas = base_market_alphas.copy()
                def obj_ode(p): 
                    return (run_ode(current_alphas, p[:-1], p[-1]) - self.market_vols) * 10000.0
                res = least_squares(obj_ode, guess, bounds=(low_bounds, high_bounds), method='trf')
                rmse = np.sqrt(np.mean(res.fun**2))
                current_p = res.x
                final_alphas = current_alphas
            
            elif method == 'MC':
                # A. Base Predictor: Pre-optimize ODE to get excellent starting rhos and nu
                current_alphas = base_market_alphas.copy()
                def obj_ode(p): 
                    return (run_ode(current_alphas, p[:-1], p[-1]) - self.market_vols) * 10000.0
                res_ode = least_squares(obj_ode, guess, bounds=(low_bounds, high_bounds), method='trf')
                current_p = res_ode.x
                
                best_mc_rmse = np.inf
                best_mc_p = current_p.copy()
                best_mc_alphas = current_alphas.copy()
                
                # B. AMMO Outer Loop (Option 1: Implicit Alpha)
                for ammo_iter in range(3):
                    rhos = current_p[:-1]
                    nu_val = current_p[-1]
                    
                    # Evaluate High-Fidelity (MC) and Low-Fidelity (ODE) at current state
                    a_ts = PchipInterpolator(self.expiries, current_alphas, extrapolate=True)
                    r_ts = PchipInterpolator(self.expiries, rhos, extrapolate=True)
                    
                    v_mc = self.rough_sabr_vol_mc(self.K_flat, self.T_flat, a_ts(self.T_flat), r_ts(self.T_flat), nu_val, H)
                    v_ode = run_ode(current_alphas, rhos, nu_val)
                    
                    # Record True Accuracy against the raw market
                    mc_rmse = np.sqrt(np.mean(((v_mc - self.market_vols)*10000.0)**2))
                    if mc_rmse < best_mc_rmse:
                        best_mc_rmse = mc_rmse
                        best_mc_p = current_p.copy()
                        best_mc_alphas = current_alphas.copy()
                        
                    # Calculate the exact AMMO convexity defect
                    delta_k = v_mc - v_ode
                    
                    # IMPLICIT ALPHA UPDATE: 
                    # Shift alphas perfectly by the ATM defect so Surrogate(ATM) = Market_ATM exactly
                    atm_defects = np.zeros(self.n_exp)
                    for exp_idx, exp_t in enumerate(self.expiries):
                        idx_mask = (np.abs(self.T_flat - exp_t) < 1e-6) & (np.abs(self.K_flat - atm_val_strike) < 1e-6)
                        if np.any(idx_mask):
                            idx = np.where(idx_mask)[0][0]
                            atm_defects[exp_idx] = delta_k[idx]
                            
                    # The new locked alphas for the inner loop
                    current_alphas = base_market_alphas - atm_defects
                    
                    # C. Inner Surrogate Optimization (Alphas are mathematically FIXED!)
                    def ammo_surrogate(p_inner):
                        # p_inner only contains [rhos, nu], so dimensionality is low and fast
                        v_surr = run_ode(current_alphas, p_inner[:-1], p_inner[-1]) + delta_k
                        return (v_surr - self.market_vols) * 10000.0

                    res_surr = least_squares(
                        ammo_surrogate, current_p, bounds=(low_bounds, high_bounds), method='trf'
                    )
                    current_p = res_surr.x
                
                rmse = best_mc_rmse
                current_p = best_mc_p
                final_alphas = best_mc_alphas

            step_time = time.time() - step_start_time
            print(f"Done! RMSE: {rmse:6.2f} bps | Time: {step_time:5.2f}s")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_H = H
                best_res_x = current_p
                best_alpha_vals = final_alphas
                
        total_time = time.time() - start_time_total
        print(f"\nStatus : SUCCESS")
        print(f"Global Hurst (H): {best_H:.6f}")
        print(f"Global Nu       : {best_res_x[-1]:.4f}")
        print(f"Best RMSE       : {best_rmse:.4f} bps")
        print(f"1D Calibration Total Time: {total_time:.2f}s")
        print("="*60)
        
        # Save finalized parameters
        best_rhos = best_res_x[:-1]
        best_nu = best_res_x[-1]
        self.alpha_ts = PchipInterpolator(self.expiries, best_alpha_vals, extrapolate=True)
        
        return {
            'alpha_func': self.alpha_ts, 
            'H': best_H, 
            'rmse_bps': best_rmse,
            'rho_func': PchipInterpolator(self.expiries, best_rhos, extrapolate=True),
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
    
    
    def rough_sabr_vol_ode_torch(self, k, T, alpha, rho, nu, H):
        """ Pure PyTorch version of the ODE solution for AMMO Autograd """
        y = (nu * (T**(H - 0.5)) * k) / alpha
        rho_safe = torch.clamp(rho, -0.9999, 0.9999)
        
        def G_half(z):
            inner = torch.sqrt(1.0 + rho_safe * z + z**2 / 4.0) - rho_safe - z / 2.0
            return 4.0 * (torch.log(inner / (1.0 - rho_safe)))**2
            
        def G_zero(z):
            term1 = torch.log(1.0 + 2.0 * rho_safe * z + z**2)
            denom = torch.sqrt(1.0 - rho_safe**2)
            term2 = (2.0 * rho_safe / denom) * (torch.atan(rho_safe / denom) - torch.atan((z + rho_safe) / denom))
            return term1 + term2

        safe_y = torch.where(torch.abs(y) < 1e-12, torch.tensor(1e-12, dtype=y.dtype, device=y.device), y)
        
        z0 = safe_y / (2.0 * H + 1.0)
        z_half = 2.0 * safe_y / (2.0 * H + 1.0)
        
        w0 = 3.0 * (1.0 - 2.0 * H) / (2.0 * H + 3.0)
        w_half = 2.0 * H / (2.0 * H + 3.0)
        
        G_A = ((2.0 * H + 1.0)**2) * (w0 * G_zero(z0) + w_half * G_half(z_half))
        G_A_safe = torch.clamp(G_A, min=1e-14)
        
        ratio = torch.where(torch.abs(y) < 1e-12, torch.tensor(1.0, dtype=y.dtype, device=y.device), torch.abs(safe_y) / torch.sqrt(G_A_safe))
        
        return alpha * ratio
    

    def _get_ammo_jacobian(self, p, H, method):
        """
        The AMMO Surrogate Jacobian.
        Returns the exact analytical gradients of the Low-Fidelity ODE proxy 
        to guide the High-Fidelity MC optimizer.
        """
        if method != 'MC':
            return '2-point' # Fallback to finite differences for analytical methods

        def _surrogate_obj(p_tensor):
            rhos = p_tensor[:self.n_exp]
            nu_val = p_tensor[-1]
            # Map the rhos exactly to the flattened grid without heavy interpolators
            rho_mapped = rhos[self.t_indices]
            
            v_ode = self.rough_sabr_vol_ode_torch(
                self.K_flat_t, self.T_flat_t, self.alpha_flat_t, 
                rho_mapped, nu_val, H
            )
            return (v_ode - torch.tensor(self.market_vols, dtype=torch.float64)) * 10000.0

        # Compute exact Jacobian over all parameters simultaneously
        p_t = torch.tensor(p, dtype=torch.float64, requires_grad=True)
        jac = torch.autograd.functional.jacobian(_surrogate_obj, p_t)
        
        return jac.detach().cpu().numpy()
    



class CorrelationCalibrator:
    def __init__(self, atm_vol_matrix, model):
        """
        Stage 2 Calibrator (Adachi et al. 2025): Fits the N x N spatial correlation matrix
        incrementally (row-by-row) using co-terminal swaptions.
        """
        self.model = model
        self.device = model.device
        self.N = model.N
        self.grid_T = model.T.cpu().numpy()
        
        # Flatten the ATM matrix for easy filtering
        expiries, tenors, vols = [], [], []
        for exp in atm_vol_matrix.index:
            for ten in atm_vol_matrix.columns:
                vol = atm_vol_matrix.loc[exp, ten]
                if not np.isnan(vol):
                    expiries.append(float(exp))
                    tenors.append(float(ten))
                    vols.append(float(vol))
                    
        self.expiries = np.array(expiries)
        self.tenors = np.array(tenors)
        self.market_vols = np.array(vols)
        
        # Initialize Rapisarda angles matrix (N x N)
        self.omega = np.zeros((self.N, self.N))
        
        # Base constraints from Adachi 2025: d<W1, W2> = dt
        # For our 0-indexed forward rate matrix, this anchors the first two dimensions
        if self.N > 1:
            self.omega[1, 0] = 0.0 # cos(0) = 1.0 (Perfect correlation)
        if self.N > 2:
            self.omega[2:, 1] = np.pi / 2.0 # cos(pi/2) = 0.0 (Orthogonal)
   
         
    def _obj_row(self, p, row_idx, free_indices, exp_targets, ten_targets, vol_targets):
        with torch.no_grad(): # <-- Disable Autograd for SciPy loop
            # Update the free angles for this specific row
            self.omega[row_idx, free_indices] = p
            
            # Build the exact correlation matrix
            omega_tensor = torch.tensor(self.omega, device=self.device, dtype=self.model.dtype)
            Sigma_matrix = build_rapisarda_correlation_matrix(omega_tensor)
            
            # Price the target swaptions using the ultra-fast Mapped SMM
            strikes = np.zeros_like(exp_targets) # ATM = 0.0 offset
            model_vols = mapped_smm_pricer(
                self.model, Sigma_matrix, exp_targets, ten_targets, strikes, 
                dt=1.0/24.0, n_paths=4096
            )
            
            return (model_vols - vol_targets) * 10000.0


    def calibrate(self):
        print("\n" + "="*60)
        print(f"{'SPATIAL CORRELATION CALIBRATION (NxN INCREMENTAL)':^60}")
        print("="*60)
        
        start_time = time.time()
        
        # Loop through each row of the correlation matrix incrementally
        for i in range(2, self.N):
            row_start_time = time.time()
            # The forward rate R^i ends at T_{i+1}. We target swaptions co-terminal to this.
            target_maturity = self.grid_T[i+1] 
            
            # Find co-terminal swaptions: Expiry + Tenor == target_maturity
            mask = np.abs((self.expiries + self.tenors) - target_maturity) < 1e-4
            
            exp_targets = self.expiries[mask]
            ten_targets = self.tenors[mask]
            vol_targets = self.market_vols[mask]
            
            # -- LOGGING: Start of row processing --
            print(f"Row {i:2d}/{self.N-1} | Target Maturity: {target_maturity:4.1f}Y | ", end="")
            
            if len(exp_targets) == 0:
                print("Skipped (No co-terminal swaptions found in market data)")
                continue
                
            # Determine free indices for this row.
            # Adachi rule: omega[i, 1] is fixed to pi/2 for i >= 2.
            # So the free angle indices are 0 and 2...i-1
            free_indices = [0] + list(range(2, i))
            
            if not free_indices:
                print("Skipped (No free correlation angles to optimize)")
                continue
                
            # -- LOGGING: Pre-optimization details --
            print(f"Matched {len(exp_targets):2d} swaptions | Optimizing {len(free_indices):2d} angles... ", end="", flush=True)
                
            # Initial guess: angles = pi/4 (moderate positive correlation)
            guess = np.full(len(free_indices), np.pi / 4.0)
            bounds = (0.0, np.pi) 
            
            # Run the optimizer for this specific row
            res = least_squares(
                self._obj_row, guess, 
                args=(i, free_indices, exp_targets, ten_targets, vol_targets),
                bounds=bounds, method='trf', ftol=1e-5, xtol=1e-5
            )
            
            # Lock in the optimized angles for this row
            self.omega[i, free_indices] = res.x
            
            # -- LOGGING: Post-optimization results --
            rmse = np.sqrt(np.mean(res.fun**2))
            row_time = time.time() - row_start_time
            print(f"Done! RMSE: {rmse:5.2f} bps | Time: {row_time:4.2f}s")
        
        # Finalize the Sigma Matrix
        omega_tensor = torch.tensor(self.omega, device=self.device, dtype=self.model.dtype)
        Sigma_final = build_rapisarda_correlation_matrix(omega_tensor)
        
        end_time = time.time()
        print("\nStatus : SUCCESS")
        print(f"NxN Calibration Total Time: {end_time - start_time:.2f}s")
        print("="*60 + "\n")
        
        return {
            'omega_matrix': self.omega,
            'Sigma_matrix': Sigma_final.cpu().numpy()
        }