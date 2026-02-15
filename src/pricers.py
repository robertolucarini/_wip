import torch
from src.utils import log_progress

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
    Updated to support the use_checkpoint flag for speed/memory control.
    """
    log_progress("Action", "Starting Accelerated Bermudan Pricing...", 0)
    
    # 1. Simulation Phase - Pass the use_checkpoint flag here
    log_progress("Simulation", f"Generating {n_paths} FMM paths...", 1)
    F_paths = model.simulate_forward_curve(n_paths, time_grid, freeze_drift=True, use_checkpoint=use_checkpoint)

    # 2. Setup Exercise Logic
    strike = torch.tensor(trade_specs['Strike'], device=model.device, dtype=torch.float64)
    ex_dates = torch.tensor(trade_specs['Ex_Dates'], device=model.device, dtype=torch.float64)
    
    ex_steps = [torch.argmin(torch.abs(time_grid - d)).item() for d in ex_dates]
    n_ex = len(ex_steps)
    deflated_cf = torch.zeros(n_paths, device=model.device, dtype=torch.float64)
    
    log_progress("LSM", f"Starting Backward Induction ({n_ex} exercise dates)...", 1)

    # 3. Backward Induction Loop (LSM)
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
                sol = torch.linalg.lstsq(X[itm].detach(), deflated_cf[itm].detach()).solution
                continuation = X[itm] @ sol
                do_ex = intrinsic_deflated[itm] > continuation
                deflated_cf[itm] = torch.where(do_ex, intrinsic_deflated[itm], deflated_cf[itm])

    log_progress("AAD", "Computing Greeks via backward pass...", 0)
    p0_Tn = model.get_terminal_bond()
    return p0_Tn * torch.mean(deflated_cf)