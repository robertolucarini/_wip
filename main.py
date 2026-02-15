import torch
import os
torch.set_num_threads(os.cpu_count()) 
# Enable MKL/OpenMP optimizations
torch.backends.cudnn.benchmark = True
import warnings
# Suppress the specific PyTorch checkpointing FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
import numpy as np
import time
from src.utils import print_summary_table, print_greek_ladder, load_discount_curve, bootstrap_forward_rates, load_swaption_vol_surface
from src.calibration import RoughSABRCalibrator
from src.torch_model import TorchRoughSABR_FMM
from src.pricers import torch_bermudan_pricer, torch_bachelier
from config import CHECK_MC, CHECK_DRIFT
# Optimize for your local CPU


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

    # --- 3. DRIFT COMPARISON: FROZEN VS EXACT ---
    print("\n" + "="*60)
    print(f"{'FMM DRIFT SIMULATION (FROZEN VS EXACT)':^60}")
    print("="*60)
    
    n_paths = 4096 #16384
    # 5-Year Simulation with Daily Steps
    n_steps_per_year = 25 
    time_grid = torch.linspace(0.0, 5.0, 5 * n_steps_per_year + 1, dtype=torch.float64, device=device)

    n_paths_test = 4096 
    
    if CHECK_DRIFT:
        with torch.no_grad():
            F_paths_frozen = model.simulate_forward_curve(n_paths_test, time_grid, freeze_drift=True)

            # Simulate 1: Frozen Drift (Deterministic Weights)
            t0_frozen = time.time()
            F_paths_frozen = model.simulate_forward_curve(n_paths, time_grid, freeze_drift=True)
            time_frozen = time.time() - t0_frozen
            
            # Simulate 2: Unfrozen Drift (Exact Stochastic Euler-Maruyama)
            t0_unfrozen = time.time()
            F_paths_unfrozen = model.simulate_forward_curve(n_paths, time_grid, freeze_drift=False)
            time_unfrozen = time.time() - t0_unfrozen

            # Analyze Results (10Y Forward at T=5)
            idx_10y = torch.argmin(torch.abs(torch.tensor(grid_T) - 10.0)).item()
            mean_fwd_frozen = F_paths_frozen[:, -1, idx_10y].mean().item()
            mean_fwd_unfrozen = F_paths_unfrozen[:, -1, idx_10y].mean().item()
            fwd_0 = model.F0[idx_10y].item()

        print_summary_table("FMM PERFORMANCE & DRIFT COMPARISON", {
            "Paths Simulated": n_paths,
            "Time Steps": len(time_grid) - 1,
            "--- RUNTIME ---": "",
            "Frozen Drift Time (s)": time_frozen,
            "Exact Drift Time (s)": time_unfrozen,
            "Performance Penalty": f"{time_unfrozen / time_frozen:.2f}x slower" if time_frozen > 0 else "N/A",
            "--- TERMINAL MEAN (10Y FWD @ 5Y) ---": "",
            "Initial F0 (bps)": fwd_0 * 10000,
            "Frozen Mean (bps)": mean_fwd_frozen * 10000,
            "Exact Mean (bps)": mean_fwd_unfrozen * 10000,
            "Drift Difference (bps)": abs(mean_fwd_unfrozen - mean_fwd_frozen) * 10000
        })

    if CHECK_MC:
        # --- 4. EUROPEAN VALIDATION (1Y ATM) ---
        with torch.no_grad():
            strike_atm = model.F0[1]
            vol_atm = model.alphas[1]
            analytical_euro = torch_bachelier(model.F0[1], strike_atm, torch.tensor(1.0, device=device), vol_atm)
            
            idx_1y = torch.argmin(torch.abs(torch.tensor(grid_T) - 1.0)).item()
            step_1y = torch.argmin(torch.abs(time_grid - 1.0)).item()
            
            # Raw undiscounted payoff
            F_1y_at_1y = F_paths_frozen[:, step_1y, idx_1y]
            raw_payoff = torch.clamp(F_1y_at_1y - strike_atm, min=0.0)
            
            # THE FIX: MEASURE CHANGE (Terminal T_N -> Forward T_1)
            # 1. Calculate the realized Terminal Bond P(T_1, T_N) on every path
            dfs_at_1y = 1.0 / (1.0 + model.tau[idx_1y:] * F_paths_frozen[:, step_1y, idx_1y:])
            P_T1_Tn = torch.prod(dfs_at_1y, dim=1)
            
            # 2. Extract initial bonds P(0, T_N) and P(0, T_1)
            P_0_Tn = model.get_terminal_bond()
            P_0_T1 = torch.prod(1.0 / (1.0 + model.tau[:idx_1y] * model.F0[:idx_1y]))
            
            # 3. Deflate by the Numeraire and convert to undiscounted T_1 expectation
            deflated_expectation = torch.mean(raw_payoff / P_T1_Tn)
            mc_euro = (P_0_Tn / P_0_T1) * deflated_expectation

        print_summary_table("1Y EUROPEAN VALIDATION", {
            "Analytical 1Y Euro (bps)": analytical_euro.item() * 10000,
            "Monte Carlo 1Y Euro (bps)": mc_euro.item() * 10000,
            "Pricing Error (bps)": abs(mc_euro - analytical_euro).item() * 10000,
        })

    # --- 5. BERMUDAN PRICING & AAD GREEKS ---
    # Underlying: 1Y into 29Y swap (Physical Settlement)
    specs = {'Strike': F0_rates[1], 'Ex_Dates': [1.0, 2.0, 3.0, 4.0, 5.0]}

    n_paths_pricer = 16384
    price = torch_bermudan_pricer(model, specs, n_paths_pricer, time_grid, use_checkpoint=False)
    price.backward()

    
    # --- 6. FINAL REPORTING ---
    print_summary_table("BERMUDAN PRICING", {
        "Bermudan Price (bps)": price.item() * 10000,
        "Total Runtime (s)": time.time() - t_init
    })

    # Print the full AAD ladder
    print_greek_ladder(
        model.F0.grad.cpu().numpy() * 10000 * 0.0001, 
        model.alphas.grad.cpu().numpy() * 10000 * 0.0001, 
        grid_T[:-1]
    )

    print(f"Total Runtime: {t_init - time.time()}")
