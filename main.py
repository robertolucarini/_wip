import torch
import os
import warnings
import numpy as np
import time

# --- OPTIMIZATION & WARNING SUPPRESSION ---
torch.set_num_threads(os.cpu_count()) 
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")

from src.utils import print_summary_table, print_greek_ladder, load_discount_curve, bootstrap_forward_rates, load_swaption_vol_surface
from src.calibration import RoughSABRCalibrator
from src.torch_model import TorchRoughSABR_FMM
from src.pricers import torch_bermudan_pricer, torch_bachelier
from config import CHECK_MC, CHECK_DRIFT

if __name__ == "__main__":
    t_init = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. DATA LOADING & CALIBRATION ---
    ois_func = load_discount_curve("data/estr_disc.csv")
    grid_T, F0_rates = bootstrap_forward_rates(ois_func)
    vol_matrix = load_swaption_vol_surface("data/estr_vol_full_strikes.csv", 1.0)
    
    calibrator = RoughSABRCalibrator(vol_matrix)
    calib = calibrator.calibrate()

    print_summary_table("ROUGH SABR CALIBRATION", {
        "Global Hurst (H)": calib['H'],
        "RMSE (bps)": calib['rmse_bps'],
        "Status": "SUCCESS"
    })

    # --- 2. MODEL SETUP ---
    model = TorchRoughSABR_FMM(grid_T, F0_rates, calib['alpha_func'], 
                               calib['rho_func'], calib['nu_func'], calib['H'], device=device)

    # --- 3. SIMULATION FOR DIAGNOSTICS ---
    # Common settings for diagnostic checks
    n_paths_test = 4096 
    n_steps_per_year = 25 
    time_grid = torch.linspace(0.0, 5.0, 5 * n_steps_per_year + 1, dtype=torch.float64, device=device)

    # Scoping Fix: Ensure F_paths_frozen is defined if either check is enabled
    F_paths_frozen = None
    if CHECK_DRIFT or CHECK_MC:
        print("\n" + "="*60)
        print(f"{'FMM DIAGNOSTIC SIMULATION':^60}")
        print("="*60)
        
        with torch.no_grad():
            t0_frozen = time.time()
            # Generate frozen paths once for use in both Drift and MC validation
            F_paths_frozen = model.simulate_forward_curve(n_paths_test, time_grid, freeze_drift=True)
            time_frozen = time.time() - t0_frozen

    # --- 3a. DRIFT COMPARISON: FROZEN VS EXACT ---
    if CHECK_DRIFT:
        with torch.no_grad():
            # Simulate Exact Drift (Unfrozen Stochastic Euler-Maruyama)
            t0_unfrozen = time.time()
            F_paths_unfrozen = model.simulate_forward_curve(n_paths_test, time_grid, freeze_drift=False)
            time_unfrozen = time.time() - t0_unfrozen

            # Analyze Results (10Y Forward at T=5)
            idx_10y = torch.argmin(torch.abs(torch.tensor(grid_T) - 10.0)).item()
            mean_fwd_frozen = F_paths_frozen[:, -1, idx_10y].mean().item()
            mean_fwd_unfrozen = F_paths_unfrozen[:, -1, idx_10y].mean().item()
            fwd_0 = model.F0[idx_10y].item()

        print_summary_table("FMM PERFORMANCE & DRIFT COMPARISON", {
            "Paths Simulated": n_paths_test,
            "Time Steps": len(time_grid) - 1,
            "Frozen Drift Time (s)": time_frozen,
            "Exact Drift Time (s)": time_unfrozen,
            "Performance Penalty": f"{time_unfrozen / time_frozen:.2f}x slower" if time_frozen > 0 else "N/A",
            "Initial F0 (bps)": fwd_0 * 10000,
            "Frozen Mean (bps)": mean_fwd_frozen * 10000,
            "Exact Mean (bps)": mean_fwd_unfrozen * 10000,
            "Drift Difference (bps)": abs(mean_fwd_unfrozen - mean_fwd_frozen) * 10000
        })

    # --- 4. EUROPEAN VALIDATION (1Y ATM) ---
    if CHECK_MC:
        with torch.no_grad():
            strike_atm = model.F0[1]
            vol_atm = model.alphas[1]
            analytical_euro = torch_bachelier(model.F0[1], strike_atm, torch.tensor(1.0, device=device), vol_atm)
            
            idx_1y = torch.argmin(torch.abs(torch.tensor(grid_T) - 1.0)).item()
            step_1y = torch.argmin(torch.abs(time_grid - 1.0)).item()
            
            # Raw undiscounted payoff
            F_1y_at_1y = F_paths_frozen[:, step_1y, idx_1y]
            raw_payoff = torch.clamp(F_1y_at_1y - strike_atm, min=0.0)
            
            # Measure Change: Terminal T_N -> Forward T_1
            dfs_at_1y = 1.0 / (1.0 + model.tau[idx_1y:] * F_paths_frozen[:, step_1y, idx_1y:])
            P_T1_Tn = torch.prod(dfs_at_1y, dim=1)
            
            P_0_Tn = model.get_terminal_bond()
            P_0_T1 = torch.prod(1.0 / (1.0 + model.tau[:idx_1y] * model.F0[:idx_1y]))
            
            deflated_expectation = torch.mean(raw_payoff / P_T1_Tn)
            mc_euro = (P_0_Tn / P_0_T1) * deflated_expectation

        print_summary_table("1Y EUROPEAN VALIDATION", {
            "Analytical 1Y Euro (bps)": analytical_euro.item() * 10000,
            "Monte Carlo 1Y Euro (bps)": mc_euro.item() * 10000,
            "Pricing Error (bps)": abs(mc_euro - analytical_euro).item() * 10000,
        })

    # --- 5. BERMUDAN PRICING & AAD GREEKS ---
    specs = {'Strike': F0_rates[1], 'Ex_Dates': [1.0, 2.0, 3.0, 4.0, 5.0]}
    n_paths_pricer = 16384
    
    # Run high-performance pricing
    price = torch_bermudan_pricer(model, specs, n_paths_pricer, time_grid, use_checkpoint=False)
    price.backward()

    # --- 6. FINAL REPORTING ---
    print_summary_table("BERMUDAN PRICING", {
        "Bermudan Price (bps)": price.item() * 10000,
        "Pricing Runtime (s)": time.time() - t_init
    })

    # Print the full AAD ladder
    print_greek_ladder(
        model.F0.grad.cpu().numpy() * 10000 * 0.0001, 
        model.alphas.grad.cpu().numpy() * 10000 * 0.0001, 
        grid_T[:-1]
    )

    print(f"\nTotal Session Runtime: {time.time() - t_init:.4f} seconds")