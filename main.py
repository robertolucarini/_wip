import torch
import os
import warnings
import numpy as np
import time
torch.set_num_threads(os.cpu_count()) 
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
from src.utils import print_summary_table, print_greek_ladder, load_discount_curve, bootstrap_forward_rates, load_swaption_vol_surface
from src.calibration import RoughSABRCalibrator
from src.torch_model import TorchRoughSABR_FMM
from src.pricers import torch_bermudan_pricer, torch_bachelier
from config import CHECK_MC, CHECK_DRIFT
import pandas as pd
from src.calibration import CorrelationCalibrator
from config import CALI_MODE, CORR_MODE, BETA_SABR, SHIFT_SABR, CHECK_LIMIT, H_GRID


def load_atm_matrix(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    df.columns = [str(c).strip().upper() for c in df.columns]
    
    c_exp = next(c for c in df.columns if 'EXPIRY' in c)
    c_ten = next(c for c in df.columns if 'UNDERLYING' in c or 'TENOR' in c)
    c_str = next(c for c in df.columns if 'STRIKE' in c)
    c_val = next(c for c in df.columns if 'VOL' in c or 'VALUE' in c)
    
    # 1. Filter for strictly ATM strings
    df = df[df[c_str].astype(str).str.strip().str.upper() == 'ATM'].copy()
    
    # 2. Parse Tenors (Months to Years, Years to float)
    def parse_tenor(x):
        x = str(x).strip().upper()
        if 'M' in x: return float(x.replace('M', '')) / 12.0
        if 'Y' in x: return float(x.replace('Y', ''))
        return float(x)
        
    df[c_exp] = df[c_exp].apply(parse_tenor)
    df[c_ten] = df[c_ten].apply(parse_tenor)
    
    return df.pivot_table(values=c_val, index=c_exp, columns=c_ten, aggfunc='first')


if __name__ == "__main__":
    t_init = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. STAGE 1: 1D MARGINAL CALIBRATION ---
    ois_func = load_discount_curve("data/estr_disc.csv")
    grid_T, F0_rates = bootstrap_forward_rates(ois_func)
    vol_matrix_1y = load_swaption_vol_surface("data/estr_vol_full_strikes.csv", 1.0)
    
    # Run the fast ODE grid search to find the global Hurst (H) and Nu
    calibrator = RoughSABRCalibrator(vol_matrix_1y)
    calib = calibrator.calibrate(method=CALI_MODE, H_grid=H_GRID)

    print_summary_table("ROUGH SABR 1D CALIBRATION", {
        "Global Hurst (H)": calib['H'],
        "RMSE (bps)": calib['rmse_bps'],
        "Status": "SUCCESS"
    })

    if CORR_MODE == 'full':
        # --- 2. STAGE 2: SPATIAL CORRELATION CALIBRATION ---
        # Instantiate a temporary model to pass the 1D dynamics into the calibrator
        model_base = TorchRoughSABR_FMM(grid_T, F0_rates, calib['alpha_func'], 
                                        calib['rho_func'], calib['nu_func'], calib['H'], 
                                        beta_sabr=BETA_SABR, shift=SHIFT_SABR, correlation_mode=CORR_MODE, device=device)
        
        # Load the full Expiry x Tenor ATM Swaption Matrix
        atm_matrix = load_atm_matrix("data/estr_vol_full_strikes.csv")
        
        # Calibrate the NxN Angles row-by-row
        corr_calibrator = CorrelationCalibrator(atm_matrix, model_base)
        corr_res = corr_calibrator.calibrate()

        # --- 3. FINAL PRODUCTION MODEL ---
        # Instantiate the final model utilizing the full rank and the calibrated Rapisarda angles!
        model = TorchRoughSABR_FMM(grid_T, F0_rates, calib['alpha_func'], 
                                   calib['rho_func'], calib['nu_func'], calib['H'], 
                                   beta_sabr=BETA_SABR, shift=SHIFT_SABR, 
                                   correlation_mode=CORR_MODE, omega_matrix=corr_res['omega_matrix'], 
                                   device=device)
    else:
        # --- 3. FINAL PRODUCTION MODEL (PCA Fast-Path) ---
        print("\nSkipping NxN Calibration (PCA Mode Selected)...")
        model = TorchRoughSABR_FMM(grid_T, F0_rates, calib['alpha_func'], 
                                   calib['rho_func'], calib['nu_func'], calib['H'], 
                                   beta_sabr=BETA_SABR, shift=SHIFT_SABR, 
                                   correlation_mode=CORR_MODE, 
                                   device=device)
        


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
    
    
    # --- 4b. ROUGHNESS SENSITIVITY CHECK (Convergence to H=0.5) ---
    # This proves that the gap is due to roughness, not a bug.
    # Toggle this for one-off validation
    if CHECK_LIMIT:
        print("\n" + "="*60)
        print(f"{'ROUGHNESS LIMIT CHECK (H -> 0.5)':^60}")
        print("="*60)
        with torch.no_grad():
            # 1. Temporarily force H to 0.5 (Classical SABR regime)
            H_orig = model.H.item()
            model.H.fill_(0.5) 
            
            # 2. Re-simulate 1Y European at H=0.5
            F_paths_limit = model.simulate_forward_curve(n_paths_test, time_grid, freeze_drift=True)
            
            # 3. Recalculate benchmarks
            idx_1y = torch.argmin(torch.abs(torch.tensor(grid_T) - 1.0)).item()
            step_1y = torch.argmin(torch.abs(time_grid - 1.0)).item()
            
            # --- DIAGNOSTIC ADJUSTMENT FOR CEV / LOCAL VOL ---
            # Translate the base alpha into an effective Normal alpha at ATM
            eta_F0 = torch.pow(torch.abs(model.F0[idx_1y] + model.shift), model.beta_sabr).item()
            effective_alpha = model.alphas[idx_1y].item() * eta_F0
            
            v_classical = calibrator.rough_sabr_vol(
                k=0.0, T=1.0, 
                alpha=effective_alpha, 
                rho=model.rhos[idx_1y].item(), 
                nu=model.nus[idx_1y].item(), 
                H=0.5 # Force H=0.5 in analytical too
            )
            analytical_limit = torch_bachelier(model.F0[idx_1y], model.F0[idx_1y], 
                                               torch.tensor(1.0, device=device), torch.tensor(v_classical, device=device))
            
            
            # MC Price at H=0.5
            F_1y_at_1y = F_paths_limit[:, step_1y, idx_1y]
            raw_payoff = torch.clamp(F_1y_at_1y - model.F0[idx_1y], min=0.0)
            dfs_at_1y = 1.0 / (1.0 + model.tau[idx_1y:] * F_paths_limit[:, step_1y, idx_1y:])
            mc_limit = (model.get_terminal_bond() / torch.prod(1.0 / (1.0 + model.tau[:idx_1y] * model.F0[:idx_1y]))) * torch.mean(raw_payoff / torch.prod(dfs_at_1y, dim=1))

            # 4. Restore original H for the Bermudan Pricing
            model.H.fill_(H_orig)

        print_summary_table("LIMIT CONVERGENCE RESULTS", {
            "Analytical (Classical SABR)": analytical_limit.item() * 10000,
            "Monte Carlo (FMM @ H=0.5)": mc_limit.item() * 10000,
            "Classical Gap (bps)": abs(mc_limit - analytical_limit).item() * 10000,
            "Original Rough Gap (bps)": 2.1927, # Hardcoded from your output for comparison
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


    # ============================================================
    #       MC CALIBRATION ENGINE DIAGNOSTIC TEST
    # ============================================================
    print("\n" + "="*60)
    print(f"{'CALIBRATION ENGINE COMPARISON (POLYNOMIAL vs ODE vs MC)':^60}")
    print("="*60)
    
    # 1. Define a test case (e.g., 5-Year Expiry)
    test_T = 5.0
    test_alpha = 0.0150  # 150 bps base vol
    test_rho = -0.40     # Downward skew
    test_nu = 0.60       # Vol-of-vol
    test_H = 0.15        # Rough Hurst
    
    # Strikes: ATM, +/- 50 bps, +/- 100 bps
    test_K = np.array([-0.01, -0.005, 0.0, 0.005, 0.01]) 
    test_T_arr = np.full_like(test_K, test_T)
    test_alpha_arr = np.full_like(test_K, test_alpha)
    test_rho_arr = np.full_like(test_K, test_rho)

    # 2. Evaluate Polynomial Formula
    vol_poly = calibrator.rough_sabr_vol(
        test_K, test_T_arr, test_alpha_arr, test_rho_arr, test_nu, test_H
    )

    # 3. Evaluate ODE Formula
    vol_ode = calibrator.rough_sabr_vol_ode(
        test_K, test_T_arr, test_alpha_arr, test_rho_arr, test_nu, test_H
    )

    # 4. Evaluate MC Engine
    # Note: We use 65536 paths here for a highly precise diagnostic check
    import torch
    device = 'cpu'
    mc_prices_t = calibrator.rough_sabr_vol_mc.__globals__['mc_rough_bergomi_pricer'](
        torch.tensor(test_K, device=device, dtype=torch.float64), 
        torch.tensor(test_T_arr, device=device, dtype=torch.float64), 
        torch.tensor(test_alpha_arr, device=device, dtype=torch.float64), 
        torch.tensor(test_rho_arr, device=device, dtype=torch.float64), 
        test_nu, test_H, 
        n_paths=65536, dt=1.0/24.0, kappa_hybrid=1, device=device
    )
    
    mc_prices = mc_prices_t.cpu().numpy()
    vol_mc = calibrator.rough_sabr_vol_mc.__globals__['bachelier_iv_newton'](
        mc_prices, test_K, test_T_arr, initial_guess_vol=test_alpha_arr
    )

    # 5. Print the Smile Comparison
    print(f"Parameters: T={test_T}Y, Alpha={test_alpha*10000:.0f}bps, Rho={test_rho}, Nu={test_nu}, H={test_H}")
    print("-" * 60)
    print(f"{'Strike (bps)':>12} | {'Poly IV (bps)':>13} | {'ODE IV (bps)':>12} | {'MC IV (bps)':>11}")
    print("-" * 60)
    
    for i in range(len(test_K)):
        strike_bps = test_K[i] * 10000
        print(f"{strike_bps:>12.0f} | {vol_poly[i]*10000:>13.2f} | {vol_ode[i]*10000:>12.2f} | {vol_mc[i]*10000:>11.2f}")
    print("=" * 60)


    # # ============================================================
    # #       SUB-STEP 2.1 DIAGNOSTIC: ODE SURROGATE CHECK
    # # ============================================================
    # from src.pricers import mapped_smm_pricer, mapped_smm_ode
    # import time
    
    # print("\n" + "="*60)
    # print(f"{'MATRIX AMMO SURROGATE TEST (MC vs ODE)':^60}")
    # print("="*60)
    
    # # Create a test batch of swaptions (e.g., 5Y expiry, Tenors 1Y to 10Y)
    # test_expiries = np.full(10, 5.0)
    # test_tenors = np.linspace(1.0, 10.0, 10)
    # test_strikes = np.zeros(10) # ATM
    
    # # Reconstruct the Sigma matrix from the PCA loadings!
    # test_Sigma = torch.matmul(model.loadings, model.loadings.T)
    
    # # 1. High-Fidelity Test (Monte Carlo)
    # t0 = time.time()
    # v_mc = mapped_smm_pricer(model, test_Sigma, test_expiries, test_tenors, test_strikes, n_paths=4096)
    # time_mc = time.time() - t0
    
    # # 2. Low-Fidelity Test (Analytical ODE)
    # t0 = time.time()
    # v_ode = mapped_smm_ode(model, test_Sigma, test_expiries, test_tenors, test_strikes)
    # time_ode = time.time() - t0
    # v_ode_np = v_ode.detach().cpu().numpy()
    
    # print(f"High-Fidelity MC Time  : {time_mc:.4f} seconds")
    # print(f"Low-Fidelity ODE Time  : {time_ode:.4f} seconds")
    # print(f"Speedup Multiplier     : {time_mc / max(time_ode, 1e-6):,.0f}x faster")
    # print("-" * 60)
    # print(f"{'Tenor':>5} | {'MC Vol (bps)':>15} | {'ODE Vol (bps)':>15} | {'Gap (bps)':>10}")
    # print("-" * 60)
    # for i in range(10):
    #     print(f"{test_tenors[i]:5.1f} | {v_mc[i]*10000:>15.2f} | {v_ode_np[i]*10000:>15.2f} | {(v_mc[i] - v_ode_np[i])*10000:>10.2f}")
    # print("="*60)