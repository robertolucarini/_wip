import torch
from src.utils import log_progress
import numpy as np


def torch_bachelier(F, K, T, vol):
    """ Differentiable Bachelier price for validation and Control Variates. """
    from torch.distributions import Normal
    dist = Normal(torch.tensor(0.0, device=F.device, dtype=F.dtype), 
                  torch.tensor(1.0, device=F.device, dtype=F.dtype))
    
    std_dev = vol * torch.sqrt(T) + 1e-12
    d = (F - K) / std_dev
    
    return (F - K) * dist.cdf(d) + std_dev * torch.exp(dist.log_prob(d))


def torch_bermudan_pricer(model, trade_specs, n_paths, time_grid, use_checkpoint=False):
    """
    High-performance AAD Bermudan Pricer.
    Updated to support the use_checkpoint flag for speed/memory control
    and fixed the LSM loop to perfectly preserve the Autograd computational graph.
    """
    log_progress("Action", "Starting Accelerated Bermudan Pricing...", 0)
    
    # 1. Simulation Phase
    log_progress("Simulation", f"Generating {n_paths} FMM paths...", 1)
    F_paths = model.simulate_forward_curve(n_paths, time_grid, freeze_drift=True, use_checkpoint=use_checkpoint)

    # 2. Setup Exercise Logic
    strike = torch.tensor(trade_specs['Strike'], device=model.device, dtype=torch.float64)
    ex_dates = torch.tensor(trade_specs['Ex_Dates'], device=model.device, dtype=torch.float64)
    
    ex_steps = [torch.argmin(torch.abs(time_grid - d)).item() for d in ex_dates]
    n_ex = len(ex_steps)
    deflated_cf = torch.zeros(n_paths, device=model.device, dtype=torch.float64)
    
    log_progress("LSM", f"Starting Backward Induction ({n_ex} exercise dates)...", 1)

    # 3. Backward Induction Loop (LSM) - AUTOGRAD SAFE
    for i in range(n_ex - 1, -1, -1):
        step = ex_steps[i]
        t_now = time_grid[step].item()
        log_progress("LSM", f"Processing Expiry T={t_now:.2f}Y ({n_ex - i}/{n_ex})", 2)
        
        k_idx = torch.sum(model.T[:-1] <= t_now).int().item()
        F_t = F_paths[:, step, k_idx:]
        taus = model.tau[k_idx:]
        
        dfs = torch.cumprod(1.0 / (1.0 + taus * F_t), dim=1)
        p_t_Tn = dfs[:, -1] 
        swap_val = torch.sum(dfs * (F_t - strike) * taus, dim=1)
        intrinsic_deflated = torch.clamp(swap_val, min=0.0) / p_t_Tn
        
        if i == n_ex - 1:
            # Direct assignment creates a new reference in the graph (Safe)
            deflated_cf = intrinsic_deflated
        else:
            annuity = torch.sum(dfs * taus, dim=1)
            par_rate = torch.sum(dfs * F_t * taus, dim=1) / annuity
            
            X = torch.stack([
                torch.ones(n_paths, device=model.device, dtype=torch.float64), 
                par_rate, 
                par_rate**2
            ], dim=1)
            
            itm = intrinsic_deflated > 0
            if itm.sum() > 50:
                # Glasserman AAD standard: Detach inputs for the regression weights
                sol = torch.linalg.lstsq(X[itm].detach(), deflated_cf[itm].detach()).solution
                
                # Evaluate continuation value for ALL paths
                continuation = X @ sol
                
                # Global exercise mask: ITM AND payoff is strictly greater than continuation
                do_ex = itm & (intrinsic_deflated > continuation)
                
            elif itm.sum() > 0:
                # FALLBACK: Too few paths for stable regression.
                # We strictly compare intrinsic value against the realized deflated future cash flow.
                # (Perfect foresight fallback, standard when regression fails).
                do_ex = itm & (intrinsic_deflated > deflated_cf)
                
            else:
                # No paths are ITM
                do_ex = torch.zeros_like(itm, dtype=torch.bool)
                
            # AUTOGRAD FIX: Out-of-place global update.
            # Safely routes gradients through the correct exercise boundary without breaking the graph.
            deflated_cf = torch.where(do_ex, intrinsic_deflated, deflated_cf)

    log_progress("AAD", "Computing Greeks via backward pass...", 0)
    p0_Tn = model.get_terminal_bond()
    return p0_Tn * torch.mean(deflated_cf)


def mapped_smm_pricer(model, Sigma_matrix, expiries, tenors, strike_offsets, dt=1.0/24.0, n_paths=4096):
    """
    Prices a set of swaptions using the Mapped SMM (Adachi et al. 2025) approximation.
    Maps the NxN FMM into 1D Rough Bergomi parameters per swaption, then leverages the fast 1D MC engine.
    """
    # Local import to prevent circular dependencies
    from src.calibration import mc_rough_bergomi_pricer, bachelier_iv_newton
    
    device = model.device
    dtype = model.dtype
    n_options = len(expiries)
    
    # 1. Base yield curve setup
    tau = model.tau
    F0 = model.F0
    P0 = torch.cumprod(torch.cat([torch.tensor([1.0], device=device, dtype=dtype), 1.0 / (1.0 + tau * F0)]), dim=0)
    
    alpha_smm = torch.zeros(n_options, device=device, dtype=dtype)
    rho_smm = torch.zeros(n_options, device=device, dtype=dtype)
    nu_smm = torch.zeros(n_options, device=device, dtype=dtype)
    
    # Calculate true Normal volatility of forward rates at t=0
    # Use absorbing boundary to match the core FMM dynamics
    eta_F0 = torch.pow(torch.clamp(F0 + model.shift, min=0.0), model.beta_sabr)
    alpha_normal = model.alphas * eta_F0 
    
    for i in range(n_options):
        T_exp = expiries[i]
        T_und = tenors[i]
        
        start_idx = torch.argmin(torch.abs(model.T - T_exp)).item()
        end_idx = torch.argmin(torch.abs(model.T - (T_exp + T_und))).item()
        
        if end_idx <= start_idx:
            continue
            
        # Swap Annuity and Par Rate
        P_I = P0[start_idx]
        P_J = P0[end_idx]
        A0 = torch.sum(tau[start_idx:end_idx] * P0[start_idx+1:end_idx+1])
        S0 = (P_I - P_J) / A0
        
        # Compute Pi weights (Adachi Eq 6: Sensitivity of Swap Rate to Forward Rates)
        pi_weights = torch.zeros(end_idx - start_idx, device=device, dtype=dtype)
        for j_local in range(end_idx - start_idx):
            j_global = start_idx + j_local
            
            # sum_{k=j}^J tau_k P(T_k)
            sum_P = torch.sum(tau[j_global:end_idx] * P0[j_global+1:end_idx+1])
            Pi_j = (tau[j_global] * P0[j_global+1]) / (A0 * P0[j_global]) * (P_J + S0 * sum_P)
            
            # pi_j maps the normal volatility of the forward rate to the swap rate
            pi_weights[j_local] = Pi_j * alpha_normal[j_global]
            
        # Mapped Normal Variance
        Sigma_slice = Sigma_matrix[start_idx+1 : end_idx+1, start_idx+1 : end_idx+1]
        v_0 = torch.matmul(pi_weights.unsqueeze(0), torch.matmul(Sigma_slice, pi_weights.unsqueeze(1))).squeeze()
        
        a_smm = torch.sqrt(torch.clamp(v_0, min=1e-14))
        alpha_smm[i] = a_smm
        
        # Mapped Vol-of-Vol Correlation
        rho_slice = Sigma_matrix[start_idx+1 : end_idx+1, 0]
        rho_mapped = torch.sum(rho_slice * pi_weights) / a_smm
        rho_smm[i] = torch.clamp(rho_mapped, -0.999, 0.999)

        # Mapped Volatility-of-Volatility (Pi-weighted average)
        nu_mapped = torch.sum(model.nus[start_idx:end_idx] * pi_weights) / (torch.sum(pi_weights) + 1e-12)
        nu_smm[i] = nu_mapped


    # 2. Price using the ultra-fast 1D Rough Bergomi MC Engine
    K_flat = torch.tensor(strike_offsets, device=device, dtype=dtype)
    T_flat = torch.tensor(expiries, device=device, dtype=dtype)
    
    # We evaluate the MC engine just once for all swaptions simultaneously!
    mc_prices = mc_rough_bergomi_pricer(
        K_flat, T_flat, alpha_smm, rho_smm, 
        nu_smm, model.H.item(), 
        n_paths=n_paths, dt=dt, kappa_hybrid=1, device=device
    )
    
    # 3. Invert the 1D model prices back to Implied Volatility
    prices_np = mc_prices.detach().cpu().numpy()
    alpha_np = alpha_smm.detach().cpu().numpy()
    ivs = bachelier_iv_newton(prices_np, strike_offsets, expiries, initial_guess_vol=alpha_np)    
    return ivs


def mapped_smm_ode(model, Sigma_matrix, expiries, tenors, strike_offsets):
    """
    Low-Fidelity Surrogate for Stage 2 (AMMO Framework).
    Uses the Effective Classical Normal SABR Mapping. 
    Now strictly armored against PyTorch Autograd NaN/Inf gradient explosions.
    """
    device = model.device
    dtype = model.dtype
    n_options = len(expiries)
    
    tau = model.tau
    F0 = model.F0
    P0 = torch.cumprod(torch.cat([torch.tensor([1.0], device=device, dtype=dtype), 1.0 / (1.0 + tau * F0)]), dim=0)
    
    # 1. FIX: Absorbing Boundary
    eta_F0 = torch.pow(torch.clamp(F0 + model.shift, min=0.0), model.beta_sabr)
    alpha_normal = model.alphas * eta_F0 
    
    alpha_smm_list, rho_smm_list, nu_smm_list = [], [], []
    
    for i in range(n_options):
        T_exp = expiries[i]
        T_und = tenors[i]
        
        start_idx = torch.argmin(torch.abs(model.T - T_exp)).item()
        end_idx = torch.argmin(torch.abs(model.T - (T_exp + T_und))).item()
        
        if end_idx <= start_idx:
            alpha_smm_list.append(torch.tensor(1e-5, device=device, dtype=dtype))
            rho_smm_list.append(torch.tensor(0.0, device=device, dtype=dtype))
            # 2. FIX: Prevent out-of-bounds index on terminal tenors
            safe_idx = min(start_idx, len(model.nus) - 1)
            nu_smm_list.append(torch.tensor(model.nus[safe_idx].item(), device=device, dtype=dtype))
            continue

            
        P_I = P0[start_idx]
        P_J = P0[end_idx]
        A0 = torch.sum(tau[start_idx:end_idx] * P0[start_idx+1:end_idx+1])
        S0 = (P_I - P_J) / A0
        
        pi_weights = torch.zeros(end_idx - start_idx, device=device, dtype=dtype)
        for j_local in range(end_idx - start_idx):
            j_global = start_idx + j_local
            sum_P = torch.sum(tau[j_global:end_idx] * P0[j_global+1:end_idx+1])
            Pi_j = (tau[j_global] * P0[j_global+1]) / (A0 * P0[j_global]) * (P_J + S0 * sum_P)
            pi_weights[j_local] = Pi_j * alpha_normal[j_global]
            
        Sigma_slice = Sigma_matrix[start_idx+1 : end_idx+1, start_idx+1 : end_idx+1]
        v_0 = torch.matmul(pi_weights.unsqueeze(0), torch.matmul(Sigma_slice, pi_weights.unsqueeze(1))).squeeze()
        a_smm = torch.sqrt(torch.clamp(v_0, min=1e-14))
        alpha_smm_list.append(a_smm)
        
        rho_slice = Sigma_matrix[start_idx+1 : end_idx+1, 0]
        rho_mapped = torch.sum(rho_slice * pi_weights) / a_smm
        rho_smm_list.append(torch.clamp(rho_mapped, -0.999, 0.999))

        nu_mapped = torch.sum(model.nus[start_idx:end_idx] * pi_weights) / (torch.sum(pi_weights) + 1e-12)
        nu_smm_list.append(nu_mapped)
              
    alpha_smm = torch.stack(alpha_smm_list)
    rho_smm = torch.stack(rho_smm_list)
    nu_smm = torch.stack(nu_smm_list)
    
    k_t = torch.tensor(strike_offsets, device=device, dtype=dtype)
    # 3. FIX: Soft clamp on Time to prevent T^(negative) infinity explosion
    T_t = torch.clamp(torch.tensor(expiries, device=device, dtype=dtype), min=1e-4)
    H = model.H
    
    c_H = 1.0 / (torch.sqrt(2.0 * H) * torch.exp(torch.lgamma(H + 0.5)))
    nu_eff = nu_smm * c_H
    
    A = (rho_smm * nu_eff) / (H + 0.5) * (T_t**(H - 0.5))
    B = (2.0 - 3.0 * rho_smm**2) * (nu_eff**2) * (T_t**(2.0*H - 1.0))
    
    nu_cl = torch.sqrt(torch.clamp((B + 3.0 * A**2) / 2.0, min=1e-12))
    rho_cl = torch.clamp(A / nu_cl, min=-0.9999, max=0.9999)
    
    # 3. GLOBALLY STABLE HAGAN NORMAL BACKBONE (Autograd Safe)
    z_cl = (nu_cl / alpha_smm) * k_t
    
    # Clamp square root to prevent infinite gradients
    sq_arg = torch.clamp(1.0 - 2.0 * rho_cl * z_cl + z_cl**2, min=1e-10)
    inner = torch.sqrt(sq_arg) + z_cl - rho_cl
    
    # Clamp log argument to prevent -Inf / NaN gradients
    log_arg = torch.clamp(inner / (1.0 - rho_cl), min=1e-10)
    x_z = torch.log(log_arg)
    
    # Safe mask to avoid 0/0 division during backward pass
    safe_x_z = torch.where(torch.abs(x_z) < 1e-8, torch.tensor(1e-8, dtype=dtype, device=device), x_z)
    ratio = torch.where(torch.abs(z_cl) < 1e-8, torch.tensor(1.0, dtype=dtype, device=device), z_cl / safe_x_z)
    
    # 4. EXACT NORMAL TIME-DRIFT
    drift = (2.0 - 3.0 * rho_smm**2) / 24.0 * (nu_eff**2) * (T_t**(2.0*H))
    
    return alpha_smm * ratio * (1.0 + drift)