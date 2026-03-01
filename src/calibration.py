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
from src.utils import build_rapisarda_correlation_matrix, log_progress
from config import USE_TIKHONOV, LAMBDA_CURVATURE


def mc_rough_bergomi_pricer(K_flat, T_flat, alpha_flat, rho_flat, nu_in, H, n_paths=(32768/2), dt=1.0/50.0, kappa_hybrid=1, device='cpu'):
    dtype = torch.float64
    
    if not isinstance(nu_in, torch.Tensor):
        nu_tensor = torch.tensor(nu_in, device=device, dtype=dtype)
    else:
        nu_tensor = nu_in.clone().detach().to(dtype=dtype, device=device)
        
    if nu_tensor.dim() == 0:
        nu_tensor = nu_tensor.expand(len(T_flat))
    
    max_T = float(torch.max(T_flat))
    n_steps = int(math.ceil(max_T / dt))
    dt_tensor = torch.tensor(dt, dtype=dtype, device=device)
    
    H_tensor = torch.tensor(H, dtype=dtype, device=device)
    gamma_const = torch.exp(torch.lgamma(H_tensor + 0.5))
    alpha_H = H - 0.5
    diag_idx = torch.arange(n_steps, device=device)
    k_mat = torch.clamp(diag_idx[:, None] - diag_idx[None, :] + 1, min=0.0).to(dtype)
    
    exact_weights = (torch.pow(k_mat, alpha_H + 1.0) - torch.pow(torch.clamp(k_mat - 1.0, min=0.0), alpha_H + 1.0)) / (alpha_H + 1.0)
    weights = torch.where(k_mat <= kappa_hybrid, exact_weights, exact_weights) 
    kernel = weights * torch.pow(dt_tensor, alpha_H) / gamma_const
    
    t_idx, s_idx = diag_idx[:, None], diag_idx[None, :]
    kernel = torch.where(t_idx >= s_idx, kernel, torch.tensor(0.0, dtype=dtype, device=device))
    
    sobol = torch.quasirandom.SobolEngine(dimension=n_steps * 2, scramble=True, seed=42)
    u = sobol.draw(n_paths).to(device).to(dtype)
    dist = Normal(torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype))
    z = dist.icdf(torch.clamp(u, 1e-7, 1-1e-7)).view(n_paths, n_steps, 2)
    
    dz_vol = z[..., 0] * torch.sqrt(dt_tensor)
    dz_perp = z[..., 1] * torch.sqrt(dt_tensor)
    
    fbm = torch.matmul(dz_vol, kernel.T)
    
    step_indices = torch.clamp(torch.round(T_flat / dt).long() - 1, 0, n_steps - 1)
    prices = torch.zeros(len(T_flat), dtype=dtype, device=device)
    
    for i in range(len(T_flat)):
        idx = step_indices[i]
        a = alpha_flat[i]
        r = rho_flat[i]
        k = K_flat[i]
        T = T_flat[i]
        n_i = nu_tensor[i]
        
        var_comp = 0.5 * (n_i**2) * torch.sum(kernel**2, dim=1) * dt
        unit_vols = torch.exp(n_i * fbm - var_comp.unsqueeze(0))
        
        unit_vols_shifted = torch.cat([torch.ones(n_paths, 1, device=device, dtype=dtype), unit_vols[:, :-1]], dim=1)
        dz_spot = r * dz_vol[:, :idx+1] + torch.sqrt(1.0 - r**2) * dz_perp[:, :idx+1]
        
        vol_path = unit_vols_shifted[:, :idx+1]
        dS = a * vol_path * dz_spot
        S_T = torch.sum(dS, dim=1)
        
        dS_cv = a * dz_spot
        S_T_cv = torch.sum(dS_cv, dim=1)
        
        payoff_rb = torch.clamp(S_T - k, min=0.0)
        payoff_cv = torch.clamp(S_T_cv - k, min=0.0)
        
        std = a * torch.sqrt(T)
        d = -k / std if std > 1e-12 else torch.tensor(0.0, device=device, dtype=dtype)
        true_cv_price = std * (dist.cdf(d) * d + torch.exp(dist.log_prob(d)))
        
        cov = torch.mean((payoff_rb - torch.mean(payoff_rb)) * (payoff_cv - torch.mean(payoff_cv)))
        var = torch.var(payoff_cv)
        beta = cov / var if var > 1e-12 else 0.0
        
        mc_price = torch.mean(payoff_rb - beta * (payoff_cv - true_cv_price))
        prices[i] = mc_price
        
    return prices


def bachelier_iv_newton(target_prices, K, T, initial_guess_vol, max_iter=20, tol=1e-8):
    k = np.array(K)
    intrinsic = np.maximum(-k, 0.0)
    P = np.maximum(np.array(target_prices), intrinsic + 1e-12)
    t = np.array(T)
    sigma = np.array(initial_guess_vol)
    
    sqrt_t = np.sqrt(t)
    
    for _ in range(max_iter):
        std = sigma * sqrt_t
        std = np.where(std < 1e-12, 1e-12, std)
        d = -k / std
        
        pdf = norm.pdf(d)
        cdf = norm.cdf(d)
        price_current = std * (d * cdf + pdf)
        
        vega = sqrt_t * pdf
        vega = np.where(vega < 1e-12, 1e-12, vega)
        
        diff = price_current - P
        step = diff / vega
        sigma = sigma - step
        
        if np.max(np.abs(diff)) < tol:
            break
            
    return np.abs(sigma)


class RoughSABRCalibrator:
    def __init__(self, vol_matrix):
        self.vol_matrix = vol_matrix
        self.expiries = self.vol_matrix.index.values.astype(float)
        self.strike_offsets = self.vol_matrix.columns.values.astype(float)
        self.n_exp = len(self.expiries)
        
        atm_idx = np.argmin(np.abs(self.strike_offsets))
        self.alpha_ts = PchipInterpolator(self.expiries, self.vol_matrix.iloc[:, atm_idx].values, extrapolate=True)
        
        self.market_vols = self.vol_matrix.values.flatten()
        T_grid, K_grid = np.meshgrid(self.expiries, self.strike_offsets, indexing='ij')
        self.T_flat, self.K_flat = T_grid.flatten(), K_grid.flatten()
        
        valid = ~np.isnan(self.market_vols)
        self.market_vols = self.market_vols[valid]
        self.T_flat = self.T_flat[valid]
        self.K_flat = self.K_flat[valid]


    def exact_atm_alpha(self, T, target_atm_vol, rho, nu, H):
        c_H = 1.0 / (np.sqrt(2.0 * H) * math.gamma(H + 0.5))
        nu_eff = nu * c_H
        drift = (2.0 - 3.0*rho**2) / 24.0 * nu_eff**2 * (T**(2.0*H))
        return target_atm_vol / (1.0 + drift)


    def rough_sabr_vol_ode(self, k, T, alpha, rho, nu, H):
        c_H = 1.0 / (np.sqrt(2.0 * H) * math.gamma(H + 0.5))
        nu_eff = nu * c_H
        
        A = (rho * nu_eff) / (H + 0.5) * (T**(H - 0.5))
        B = (2.0 - 3.0 * rho**2) * (nu_eff**2) * (T**(2.0*H - 1.0))
        
        nu_cl = np.sqrt(np.maximum((B + 3.0 * A**2) / 2.0, 1e-12))
        rho_cl = np.clip(A / nu_cl, -0.9999, 0.9999)
        
        z_cl = (nu_cl / alpha) * k
        sq_arg = np.maximum(1.0 - 2.0 * rho_cl * z_cl + z_cl**2, 1e-10)
        inner = np.sqrt(sq_arg) + z_cl - rho_cl
        
        log_arg = np.maximum(inner / (1.0 - rho_cl), 1e-10)
        x_z = np.log(log_arg)
        
        safe_x_z = np.where(np.abs(x_z) < 1e-8, 1e-8, x_z)
        ratio = np.where(np.abs(z_cl) < 1e-8, 1.0, z_cl / safe_x_z)
        drift = (2.0 - 3.0 * rho**2) / 24.0 * (nu_eff**2) * (T**(2.0*H))
        
        return alpha * ratio * (1.0 + drift)


    def rough_sabr_vol_mc(self, k, T, alpha, rho, nu, H):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        K_t = torch.tensor(k, device=device, dtype=torch.float64)
        T_t = torch.tensor(T, device=device, dtype=torch.float64)
        alpha_t = torch.tensor(alpha, device=device, dtype=torch.float64)
        rho_t = torch.tensor(rho, device=device, dtype=torch.float64)
        nu_t = torch.tensor(nu, device=device, dtype=torch.float64)
        
        mc_prices_t = mc_rough_bergomi_pricer(
            K_t, T_t, alpha_t, rho_t, nu_t, H, 
            n_paths=16384, 
            dt=1.0/24.0, 
            kappa_hybrid=1, 
            device=device
        )
        
        mc_prices = mc_prices_t.cpu().numpy()
        ivs = bachelier_iv_newton(mc_prices, k, T, initial_guess_vol=alpha)
        
        return ivs
    

    def calibrate(self, method='PURE_MC', H_grid=np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])):
        import time
        from scipy.interpolate import PchipInterpolator
        from scipy.optimize import least_squares
        
        print("\n" + "="*60)
        print(f"{f'ROUGH SABR 1D CALIBRATION | MODE: {method.upper()}':^60}")
        print("="*60)
        
        best_rmse = np.inf
        best_H = None
        best_alphas = None
        best_rhos = None
        best_nu = None
        
        atm_idx = np.argmin(np.abs(self.strike_offsets))
        base_market_alphas = self.vol_matrix.iloc[:, atm_idx].values.copy()
        
        start_time_total = time.time()
        
        for i, H in enumerate(H_grid):
            step_start_time = time.time()
            log_progress("Stage 1", f"Grid {i+1:2d}/{len(H_grid)} | Testing Hurst (H) = {H:.3f}", level=0)
            
            # ==========================================================
            # 1. THE "DIRTY" LOCAL SEEDING FIT (Using ODE just to find the valley)
            # ==========================================================
            log_progress("Pre-Opt", "Running independent slice fits for smart seeding...", level=1)
            local_nus = np.zeros(self.n_exp)
            local_rhos = np.zeros(self.n_exp)
            
            for j, exp_t in enumerate(self.expiries):
                mask = np.abs(self.T_flat - exp_t) < 1e-6
                k_targets = self.K_flat[mask]
                v_targets = self.market_vols[mask]
                base_a = base_market_alphas[j]
                
                def slice_obj(p):
                    rho, nu = p[0], p[1]
                    alpha = self.exact_atm_alpha(exp_t, base_a, rho, nu, H)
                    v = self.rough_sabr_vol_ode(k_targets, np.full_like(k_targets, exp_t), alpha, rho, nu, H)
                    return (v - v_targets) * 10000.0
                    
                res_local = least_squares(slice_obj, [-0.1, 0.4], bounds=([-0.999, 0.001], [0.999, 10.0]), method='trf')
                local_rhos[j] = res_local.x[0]
                local_nus[j] = res_local.x[1]
            
            smart_nu_guess = np.clip(np.mean(local_nus), 0.05, 5.0)
            log_progress("Pre-Opt", f"Smart Global Nu Seed calculated: {smart_nu_guess:.4f}", level=1)

            # ==========================================================
            # 2a. THE "AMMO ODE" METHOD (Fast but mathematically flawed for Stage 1)
            # ==========================================================
            if method == 'AMMO_ODE':
                guess_global = np.concatenate(([smart_nu_guess], local_rhos))
                lower_bounds = np.concatenate(([0.001], np.full(self.n_exp, -0.999)))
                upper_bounds = np.concatenate(([10.0], np.full(self.n_exp, 0.999)))
                
                log_progress("AMMO-Opt", "Running fast ODE surrogate optimization...", level=1)
                def global_obj(p):
                    nu = p[0]
                    rhos = p[1:]
                    r_ts = PchipInterpolator(self.expiries, rhos, extrapolate=True)
                    r_flat = r_ts(self.T_flat)
                    
                    alphas = self.exact_atm_alpha(self.expiries, base_market_alphas, rhos, nu, H)
                    a_ts = PchipInterpolator(self.expiries, alphas, extrapolate=True)
                    a_flat = a_ts(self.T_flat)
                    
                    v_surr = self.rough_sabr_vol_ode(self.K_flat, self.T_flat, a_flat, r_flat, nu, H)
                    return (v_surr - self.market_vols) * 10000.0
                    
                res = least_squares(global_obj, guess_global, bounds=(lower_bounds, upper_bounds), method='trf')
                current_nu = res.x[0]
                current_rhos = res.x[1:]
                current_alphas = self.exact_atm_alpha(self.expiries, base_market_alphas, current_rhos, current_nu, H)
                
                a_ts = PchipInterpolator(self.expiries, current_alphas, extrapolate=True)
                r_ts = PchipInterpolator(self.expiries, current_rhos, extrapolate=True)
                v_final = self.rough_sabr_vol_ode(self.K_flat, self.T_flat, a_ts(self.T_flat), r_ts(self.T_flat), current_nu, H)
                rmse = np.sqrt(np.mean(((v_final - self.market_vols)*10000.0)**2))

            # ==========================================================
            # 2b. THE "TRUE PURE MC" METHOD (No ODE interference)
            # ==========================================================
            elif method in ['PURE_MC', 'MC']:
                log_progress("MC-Opt", "Starting True Pure Monte Carlo Global Optimization...", level=1)
                
                # Guess is now: [Nu, Alpha_1...Alpha_N, Rho_1...Rho_N]
                guess_global = np.concatenate(([smart_nu_guess], base_market_alphas, local_rhos))
                
                # Bounds for Nu, Alphas (must be >0), and Rhos
                lower_bounds = np.concatenate(([0.001], np.full(self.n_exp, 0.0001), np.full(self.n_exp, -0.999)))
                upper_bounds = np.concatenate(([10.0], np.full(self.n_exp, 1.0), np.full(self.n_exp, 0.999)))
                
                eval_counter = [0]
                
                def mc_global_obj(p):
                    eval_counter[0] += 1
                    
                    nu = p[0]
                    alphas = p[1:self.n_exp+1]
                    rhos = p[self.n_exp+1:]
                    
                    if eval_counter[0] % 10 == 0:
                        print(f"   [MC-Opt] Eval: {eval_counter[0]:>3} | Nu: {nu:.4f} | Mean Alpha: {np.mean(alphas)*10000:.0f}bps | Mean Rho: {np.mean(rhos):.4f}", flush=True)
                    
                    r_ts = PchipInterpolator(self.expiries, rhos, extrapolate=True)
                    r_flat = r_ts(self.T_flat)
                    
                    a_ts = PchipInterpolator(self.expiries, alphas, extrapolate=True)
                    a_flat = a_ts(self.T_flat)
                    
                    # PURE MC EVALUATION (No exact_atm_alpha scaling!)
                    v_mc = self.rough_sabr_vol_mc(self.K_flat, self.T_flat, a_flat, r_flat, nu, H)
                    return (v_mc - self.market_vols) * 10000.0
                
                log_progress("MC-Opt", "Handing over to scipy.least_squares (verbose=2 for iteration logs)...", level=1)
                
                res_mc = least_squares(mc_global_obj, guess_global, bounds=(lower_bounds, upper_bounds), 
                                       method='trf', diff_step=1e-3, ftol=1e-4, xtol=1e-4, verbose=2)
                
                log_progress("MC-Opt", f"Optimization converged after {res_mc.nfev} MC evaluations.", level=1)
                
                current_nu = res_mc.x[0]
                current_alphas = res_mc.x[1:self.n_exp+1]
                current_rhos = res_mc.x[self.n_exp+1:]
                
                a_ts = PchipInterpolator(self.expiries, current_alphas, extrapolate=True)
                r_ts = PchipInterpolator(self.expiries, current_rhos, extrapolate=True)
                v_final = self.rough_sabr_vol_mc(self.K_flat, self.T_flat, a_ts(self.T_flat), r_ts(self.T_flat), current_nu, H)
                rmse = np.sqrt(np.mean(((v_final - self.market_vols)*10000.0)**2))

            # ==========================================================
            # 3. END OF LOOP RECORDING
            # ==========================================================
            step_time = time.time() - step_start_time
            log_progress("Stage 1", f"Done! RMSE: {rmse:6.2f} bps | Nu: {current_nu:.4f} | Time: {step_time:5.2f}s\n", level=0)
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_H = H
                best_alphas = current_alphas.copy()
                best_rhos = current_rhos.copy()
                best_nu = current_nu
                
        total_time = time.time() - start_time_total
        print(f"\nStatus : SUCCESS")
        print(f"Global Hurst (H): {best_H:.6f}")
        print(f"Global Nu       : {best_nu:.4f}")
        print(f"Best RMSE       : {best_rmse:.4f} bps")
        print(f"1D Calibration Total Time: {total_time:.2f}s")
        print("="*60)
        
        nu_func = lambda t: np.full_like(t, best_nu, dtype=float) if isinstance(t, np.ndarray) else float(best_nu)
        
        return {
            'alpha_func': PchipInterpolator(self.expiries, best_alphas, extrapolate=True), 
            'H': best_H, 
            'rmse_bps': best_rmse,
            'rho_func': PchipInterpolator(self.expiries, best_rhos, extrapolate=True),
            'nu_func': nu_func
        }
        

class CorrelationCalibrator:
    def __init__(self, atm_vol_matrix, model):
        self.model = model
        self.device = model.device
        self.N = model.N
        self.grid_T = model.T.cpu().numpy()
        
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
        
        self.omega = np.zeros((self.N + 1, self.N + 1))
        
        rhos_np = np.clip(self.model.rhos.detach().cpu().numpy(), -0.9999, 0.9999)
        self.omega[1:, 0] = np.arccos(rhos_np)
        
        if self.N > 1:
            self.omega[2, 1] = 0.0 
        if self.N > 2:
            self.omega[3:, 2] = np.pi / 2.0 

    def calibrate(self):
        import time
        from src.pricers import mapped_smm_pricer, mapped_smm_ode
        
        print("\n" + "="*60)
        print(f"{'SPATIAL CORRELATION CALIBRATION (MATRIX AMMO)':^60}")
        print("="*60)
        
        start_time = time.time()
        
        for i in range(2, self.N):
            row_start_time = time.time()
            target_maturity = self.grid_T[i+1] 
            
            mask = np.abs((self.expiries + self.tenors) - target_maturity) < 1e-4
            exp_targets = self.expiries[mask]
            ten_targets = self.tenors[mask]
            vol_targets = self.market_vols[mask]
            
            print(f"Row {i:2d}/{self.N-1} | Target Maturity: {target_maturity:4.1f}Y | ", end="")
            
            if len(exp_targets) == 0:
                print("Skipped (No co-terminal swaptions found)")
                continue

            idx = i + 1 
            free_indices = [1] + list(range(3, idx))
            if not free_indices:
                print("Skipped (No free correlation angles)")
                continue
                
            print(f"Matched {len(exp_targets):2d} swaptions | Optimizing {len(free_indices):2d} angles... ", end="", flush=True)
                
            bounds = (0.0, np.pi) 
            current_p = np.full(len(free_indices), np.pi / 4.0)
            strikes = np.zeros_like(exp_targets)
            
            # best_mc_rmse = np.inf
            # best_p = current_p.copy()
            
            for ammo_iter in range(2):
                self.omega[idx, free_indices] = current_p
                omega_tensor = torch.tensor(self.omega, device=self.device, dtype=self.model.dtype)
                Sigma_matrix = build_rapisarda_correlation_matrix(omega_tensor)
                
                v_mc = mapped_smm_pricer(self.model, Sigma_matrix, exp_targets, ten_targets, strikes, dt=1.0/24.0, n_paths=4096)
                v_ode = mapped_smm_ode(self.model, Sigma_matrix, exp_targets, ten_targets, strikes)
                v_ode_np = v_ode.detach().cpu().numpy()
                
                delta_k = v_mc - v_ode_np  
                
                # JUST SAVE IT FOR THE PRINT STATEMENT
                row_rmse = np.sqrt(np.mean(((v_mc - vol_targets)*10000.0)**2))
                
                # v_mc = mapped_smm_pricer(self.model, Sigma_matrix, exp_targets, ten_targets, strikes, dt=1.0/24.0, n_paths=4096)
                # v_ode = mapped_smm_ode(self.model, Sigma_matrix, exp_targets, ten_targets, strikes)
                # v_ode_np = v_ode.detach().cpu().numpy()
                
                # mc_rmse = np.sqrt(np.mean(((v_mc - vol_targets)*10000.0)**2))
                # if mc_rmse < best_mc_rmse:
                #     best_mc_rmse = mc_rmse
                #     best_p = current_p.copy()
                    
                # delta_k = v_mc - v_ode_np
                
                def ammo_surrogate(p_inner):
                    self.omega[idx, free_indices] = p_inner
                    o_t = torch.tensor(self.omega, device=self.device, dtype=self.model.dtype)
                    S_mat = build_rapisarda_correlation_matrix(o_t)
                    v_surr = mapped_smm_ode(self.model, S_mat, exp_targets, ten_targets, strikes)
                    
                    residuals = (v_surr.detach().cpu().numpy() + delta_k - vol_targets) * 10000.0
                    
                    if USE_TIKHONOV:
                        lam = LAMBDA_CURVATURE
                        S_mat_np = S_mat.detach().cpu().numpy()
                        row_corr = S_mat_np[idx, 1:idx] 
                        
                        # A. Horizontal Curvature (Shape of current row)
                        if len(row_corr) >= 3:
                            pen_h = row_corr[2:] - 2.0 * row_corr[1:-1] + row_corr[:-2]
                        elif len(row_corr) == 2:
                            pen_h = row_corr[1:] - row_corr[:-1]
                        else:
                            pen_h = np.zeros(0)
                            
                        # B. Vertical Anchor (Closes the parallel shift loophole!)
                        if idx > 2:
                            prev_corr = S_mat_np[idx-1, 1:idx-1]
                            pen_v = row_corr[:-1] - prev_corr
                        else:
                            pen_v = np.zeros(0)
                            
                        # Combine 2D penalties
                        if len(pen_h) > 0 and len(pen_v) > 0:
                            pen = np.concatenate([pen_h, pen_v])
                        elif len(pen_h) > 0:
                            pen = pen_h
                        elif len(pen_v) > 0:
                            pen = pen_v
                        else:
                            pen = np.zeros(1)
                            
                        pen_res = np.sqrt(2.0 * lam) * pen
                        residuals = np.concatenate([residuals, pen_res])
                        
                    return residuals
                    
                def ammo_jacobian(p_inner):
                    def _diff_obj(p_tensor):
                        o_mod = torch.tensor(self.omega, device=self.device, dtype=self.model.dtype).clone()
                        o_mod[idx, free_indices] = p_tensor  
                        
                        S_mat = build_rapisarda_correlation_matrix(o_mod)
                        v_surr = mapped_smm_ode(self.model, S_mat, exp_targets, ten_targets, strikes)
                        res = (v_surr + torch.tensor(delta_k, device=self.device) - torch.tensor(vol_targets, device=self.device)) * 10000.0
                        
                        if USE_TIKHONOV:
                            lam = LAMBDA_CURVATURE
                            row_corr = S_mat[idx, 1:idx]
                            
                            if len(row_corr) >= 3:
                                pen_h = row_corr[2:] - 2.0 * row_corr[1:-1] + row_corr[:-2]
                            elif len(row_corr) == 2:
                                pen_h = row_corr[1:] - row_corr[:-1]
                            else:
                                pen_h = torch.zeros(0, device=self.device, dtype=self.model.dtype)
                                
                            if idx > 2:
                                prev_corr = S_mat[idx-1, 1:idx-1]
                                pen_v = row_corr[:-1] - prev_corr
                            else:
                                pen_v = torch.zeros(0, device=self.device, dtype=self.model.dtype)
                                
                            if len(pen_h) > 0 and len(pen_v) > 0:
                                pen = torch.cat([pen_h, pen_v])
                            elif len(pen_h) > 0:
                                pen = pen_h
                            elif len(pen_v) > 0:
                                pen = pen_v
                            else:
                                pen = torch.zeros(1, device=self.device, dtype=self.model.dtype)
                                
                            pen_res = math.sqrt(2.0 * lam) * pen
                            res = torch.cat([res, pen_res])
                            
                        return res
                        
                    p_t = torch.tensor(p_inner, device=self.device, dtype=self.model.dtype, requires_grad=True)
                    jac = torch.autograd.functional.jacobian(_diff_obj, p_t)
                    
                    jac_np = jac.detach().cpu().numpy()
                    if not np.isfinite(jac_np).all():
                        jac_np = np.nan_to_num(jac_np, nan=0.0, posinf=0.0, neginf=0.0)
                    return jac_np
                                       
                res = least_squares(
                    ammo_surrogate, current_p, bounds=bounds, 
                    jac=ammo_jacobian, method='trf', 
                    ftol=1e-5, xtol=1e-5, max_nfev=25
                )
                current_p = res.x
                
            # self.omega[idx, free_indices] = best_p
            self.omega[idx, free_indices] = current_p
            
            row_time = time.time() - row_start_time
            print(f"Done! RMSE: {row_rmse:5.2f} bps | Time: {row_time:4.2f}s")
        
        omega_tensor = torch.tensor(self.omega, device=self.device, dtype=self.model.dtype)
        Sigma_final = build_rapisarda_correlation_matrix(omega_tensor)
        
        end_time = time.time()
        print("\nStatus : SUCCESS")
        print(f"NxN Matrix AMMO Total Time: {end_time - start_time:.2f}s")
        print("="*60 + "\n")
        
        return {
            'omega_matrix': self.omega,
            'Sigma_matrix': Sigma_final.cpu().numpy()
        }