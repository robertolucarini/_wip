import torch
import numpy as np
import time
from src.utils import print_summary_table, print_greek_ladder, load_discount_curve, bootstrap_forward_rates, load_swaption_vol_surface
from src.calibration import RoughSABRCalibrator
from src.torch_model import TorchRoughSABR_FMM
from src.pricers import torch_bermudan_pricer, torch_bachelier

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

    # --- 3. PRICING & AAD GREEKS ---
    # Underlying: 1Y into 29Y swap (Physical Settlement)
    specs = {'Strike': F0_rates[1], 'Ex_Dates': [1.0, 2.0, 3.0, 4.0, 5.0]}
    time_grid = torch.linspace(0.0, 5.0, 5*252, dtype=torch.float64, device=device)
    
    # Run AAD Pricer
    price = torch_bermudan_pricer(model, specs, 4096, time_grid)
    price.backward()
    
    # --- 4. MODEL VALIDATION CALL ---
    # Compare 1Y ATM European price (Analytical vs Monte Carlo)
    with torch.no_grad():
        # Analytical Bachelier for 1Y Tenor
        strike_atm = model.F0[1]
        vol_atm = model.alphas[1]
        analytical_euro = torch_bachelier(model.F0[1], strike_atm, torch.tensor(1.0, device=device), vol_atm)
        
        # Monte Carlo 1Y European (using a sub-slice of our Bermudan simulation)
        # We simulate 1 date only for validation
        val_grid = torch.linspace(0.0, 1.0, 252, dtype=torch.float64, device=device)
        rough_driver = model.generate_rough_shocks(16384, val_grid) # Higher paths for validation accuracy
        F_T = model.F0[1] + model.alphas[1] * rough_driver[:, -1]
        mc_euro = model.get_terminal_bond() * torch.mean(torch.clamp(F_T - strike_atm, min=0.0))

    # --- 5. FINAL REPORTING ---
    print_greek_ladder(
        model.F0.grad.cpu().numpy() * 10000 * 0.0001, 
        model.alphas.grad.cpu().numpy() * 10000 * 0.0001, 
        grid_T[:-1]
    )

    print_summary_table("MODEL VALIDATION & PRICING", {
        "Bermudan Price (bps)": price.item() * 10000,
        "Analytical 1Y Euro (bps)": analytical_euro.item() * 10000,
        "Monte Carlo 1Y Euro (bps)": mc_euro.item() * 10000,
        "Pricing Error (bps)": abs(mc_euro - analytical_euro).item() * 10000,
        "Total Runtime (s)": time.time() - t_init
    })