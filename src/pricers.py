import torch
from src.utils import log_progress

def torch_bachelier(F, K, T, vol):
    """
    Differentiable Bachelier price for validation and Control Variates.
    Ensures the analytical signal is captured in the PyTorch Autograd graph.
    """
    from torch.distributions import Normal
    # Use the same device and dtype as the input F
    dist = Normal(torch.tensor(0.0, device=F.device, dtype=F.dtype), 
                  torch.tensor(1.0, device=F.device, dtype=F.dtype))
    
    # FIX: Add 1e-12 to avoid division by zero for T=0 or Vol=0
    std_dev = vol * torch.sqrt(T) + 1e-12
    d = (F - K) / std_dev
    
    return (F - K) * dist.cdf(d) + std_dev * torch.exp(dist.log_prob(d))

def torch_bermudan_pricer(model, trade_specs, n_paths, time_grid):
    """
    High-performance AAD Bermudan Pricer.
    Uses pre-simulated, Arbitrage-Free FMM paths under the Terminal Measure.
    """
    log_progress("Action", "Starting Accelerated Bermudan Pricing...", 0)
    
    # 1. Simulation Phase
    log_progress("Simulation", f"Generating {n_paths} FMM paths...", 1)
    # Generate the full arbitrage-free curve (Shape: n_paths, n_steps, n_tenors)
    F_paths = model.simulate_forward_curve(n_paths, time_grid, freeze_drift=True)

    # 2. Setup Exercise Logic
    strike = torch.tensor(trade_specs['Strike'], device=model.device, dtype=torch.float64)
    ex_dates = torch.tensor(trade_specs['Ex_Dates'], device=model.device, dtype=torch.float64)
    
    # Map dates to grid indices
    ex_steps = [torch.argmin(torch.abs(time_grid - d)).item() for d in ex_dates]
    n_ex = len(ex_steps)

    # Cashflow container (deflated by the terminal numeraire)
    deflated_cf = torch.zeros(n_paths, device=model.device, dtype=torch.float64)
    
    log_progress("LSM", f"Starting Backward Induction ({n_ex} exercise dates)...", 1)

    # 3. Backward Induction Loop (LSM)
    for i in range(n_ex - 1, -1, -1):
        step = ex_steps[i]
        t_now = time_grid[step].item()
        log_progress("LSM", f"Processing Expiry T={t_now:.2f}Y ({n_ex - i}/{n_ex})", 2)
        
        # Determine the index of the first active tenor in the forward market model
        k_idx = torch.sum(model.T[:-1] <= t_now).int().item()
        
        # EXTRACT: Grab the active forward curve directly from the simulated paths
        F_t = F_paths[:, step, k_idx:]
        
        # Local discount factors and payoff
        taus = model.tau[k_idx:]
        dfs = torch.cumprod(1.0 / (1.0 + taus * F_t), dim=1)
        p_t_Tn = dfs[:, -1] # Realized Bond to the terminal date P(t, T_N)
        
        # Swap payoff evaluated at time t
        swap_val = torch.sum(dfs * (F_t - strike) * taus, dim=1)
        
        # Deflate the intrinsic value to the terminal numeraire
        intrinsic_deflated = torch.clamp(swap_val, min=0.0) / p_t_Tn
        
        if i == n_ex - 1:
            # At terminal date, continuation value is zero
            deflated_cf = intrinsic_deflated
        else:
            # STATE VARIABLES for regression: Par rate proxy
            annuity = torch.sum(dfs * taus, dim=1)
            par_rate = torch.sum(dfs * F_t * taus, dim=1) / annuity
            
            # Regression basis: [1, x, x^2]
            X = torch.stack([
                torch.ones(n_paths, device=model.device, dtype=torch.float64), 
                par_rate, 
                par_rate**2
            ], dim=1)
            
            # Longstaff-Schwartz regression on In-The-Money paths
            itm = intrinsic_deflated > 0
            if itm.sum() > 50:
                # Solve regression: detach to prevent AAD from tracking the solver itself
                sol = torch.linalg.lstsq(X[itm].detach(), deflated_cf[itm].detach()).solution
                continuation = X[itm] @ sol
                
                # Compare exercise vs continuation
                do_ex = intrinsic_deflated[itm] > continuation
                deflated_cf[itm] = torch.where(do_ex, intrinsic_deflated[itm], deflated_cf[itm])

    log_progress("AAD", "Computing Greeks via backward pass...", 0)
    
    # 4. Final Numeraire Adjustment
    # Price = P(0, Tn) * Mean(Deflated_Cashflows)
    p0_Tn = model.get_terminal_bond()
    return p0_Tn * torch.mean(deflated_cf)