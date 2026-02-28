import os
import time
import numpy as np
import pandas as pd
import torch

from src.utils import load_swaption_vol_surface
from src.calibration import RoughSABRCalibrator
from config import H_GRID

def generate_stage1_results(methods=['polynomial', 'AMMO_ODE', 'PURE_MC']):
    """
    Runs Stage 1 calibration across multiple methods and stores the resulting 
    parameters and estimated volatility surfaces for later analysis.
    """
    # Create results directory to store outputs
    os.makedirs('results', exist_ok=True)
    
    # 1. Load Data (Targeting the 1Y underlying slice for Stage 1)
    print("Loading market data...")
    vol_matrix_1y = load_swaption_vol_surface("data/estr_vol_full_strikes.csv", 1.0)
    calibrator = RoughSABRCalibrator(vol_matrix_1y)
    
    expiries = calibrator.expiries
    strikes = calibrator.strike_offsets
    
    # Save the market surface for reference
    market_path = "results/stage1_surface_MARKET.csv"
    vol_matrix_1y.to_csv(market_path)
    print(f"Saved market surface to {market_path}\n")
    
    param_records = []
    
    for method in methods:
        print(f"{'='*60}")
        print(f"Running Calibration: {method.upper()}")
        print(f"{'='*60}")
        
        try:
            # 2. Run Calibration using your existing class
            calib = calibrator.calibrate(method=method, H_grid=H_GRID)
            
            # 3. Extract Global and Local Parameters
            H = calib['H']
            rmse = calib['rmse_bps']
            
            alphas = calib['alpha_func'](expiries)
            rhos = calib['rho_func'](expiries)
            
            # nu_func might return a float or an array depending on your setup
            nus = calib['nu_func'](expiries)
            if isinstance(nus, (float, int)):
                nus = np.full_like(expiries, nus)
            
            # Store parameters for table generation
            for i, T in enumerate(expiries):
                param_records.append({
                    'Method': method,
                    'Expiry': T,
                    'Alpha': alphas[i],
                    'Rho': rhos[i],
                    'Nu': nus[i],
                    'H': H,
                    'RMSE_bps': rmse
                })
            
            # 4. Generate Model Surface
            T_grid, K_grid = np.meshgrid(expiries, strikes, indexing='ij')
            T_flat = T_grid.flatten()
            K_flat = K_grid.flatten()
            
            a_flat = calib['alpha_func'](T_flat)
            r_flat = calib['rho_func'](T_flat)
            n_flat = calib['nu_func'](T_flat)
            
            # Evaluate the surface using the appropriate evaluator
            if method.upper() in ['PURE_MC', 'MC']:
                vols_flat = calibrator.rough_sabr_vol_mc(K_flat, T_flat, a_flat, r_flat, n_flat, H)
            elif method.upper() == 'POLYNOMIAL':
                vols_flat = calibrator.rough_sabr_vol(K_flat, T_flat, a_flat, r_flat, n_flat, H)
            else:
                vols_flat = calibrator.rough_sabr_vol_ode(K_flat, T_flat, a_flat, r_flat, n_flat, H)
            
            # Reshape and save to CSV
            vols_matrix = vols_flat.reshape(len(expiries), len(strikes))
            df_surface = pd.DataFrame(vols_matrix, index=expiries, columns=strikes)
            
            surface_path = f"results/stage1_surface_{method.upper()}.csv"
            df_surface.to_csv(surface_path)
            print(f"-> Saved fitted surface to {surface_path}\n")
            
        except Exception as e:
            print(f"-> FAILED to run {method}: {e}\n")
    
    # 5. Save all aggregated parameters to a single CSV
    if param_records:
        df_params = pd.DataFrame(param_records)
        params_path = "results/stage1_parameters.csv"
        df_params.to_csv(params_path, index=False)
        print(f"Saved all parameters to {params_path}")
        print("Done! You are ready to generate tables.")

import pandas as pd
import numpy as np

def print_full_latex_longtable(csv_path="results/stage1_parameters.csv"):
    df = pd.read_csv(csv_path)
    
    # Get all unique expiries sorted automatically!
    all_expiries = sorted(df['Expiry'].unique())
    
    latex = []
    
    # We use longtable so it breaks beautifully across pages in the PDF
    latex.append(r"\begin{center}")
    latex.append(r"\small")
    latex.append(r"\begin{longtable}{l l c c c c c}")
    latex.append(r"    \caption{Comparison of calibrated marginal parameters (Stage 1) across the full yield curve. PURE\_MC successfully untraps the volatility-of-volatility parameter ($\nu$) from the asymptotic ODE breakdown.} \label{tab:stage1_full_comparison} \\")
    latex.append(r"    \toprule")
    latex.append(r"    \textbf{Expiry} & \textbf{Method} & \textbf{$\alpha$ (bps)} & \textbf{$\rho$} & \textbf{$\nu$} & \textbf{Global $H$} & \textbf{RMSE (bps)} \\")
    latex.append(r"    \midrule")
    latex.append(r"    \endfirsthead")
    latex.append(r"")
    latex.append(r"    % Header for subsequent pages")
    latex.append(r"    \multicolumn{7}{c}%")
    latex.append(r"    {{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\")
    latex.append(r"    \toprule")
    latex.append(r"    \textbf{Expiry} & \textbf{Method} & \textbf{$\alpha$ (bps)} & \textbf{$\rho$} & \textbf{$\nu$} & \textbf{Global $H$} & \textbf{RMSE (bps)} \\")
    latex.append(r"    \midrule")
    latex.append(r"    \endhead")
    latex.append(r"")
    latex.append(r"    \midrule")
    latex.append(r"    \multicolumn{7}{r}{{Continued on next page}} \\")
    latex.append(r"    \endfoot")
    latex.append(r"")
    latex.append(r"    \bottomrule")
    latex.append(r"    \endlastfoot")
    latex.append(r"")
    
    for exp in all_expiries:
        subset = df[np.isclose(df['Expiry'], exp)]
        if subset.empty:
            continue
            
        first = True
        for _, row in subset.iterrows():
            # Only print the expiry on the first row of the group
            exp_str = f"{exp:.1f}Y" if first else ""
            method_str = str(row['Method']).replace('_', '\\_')
            
            # Format the numbers perfectly
            alpha_bps = row['Alpha'] * 10000.0
            alpha_str = f"{alpha_bps:.1f}"
            rho_str = f"{row['Rho']:.3f}"
            nu_str = f"{row['Nu']:.4f}"
            h_str = f"{row['H']:.3f}"
            rmse_str = f"{row['RMSE_bps']:.2f}"
            
            # Highlight your PURE_MC breakthrough in bold
            if row['Method'] == 'PURE_MC':
                method_str = f"\textbf{{{method_str}}}"
                nu_str = f"\textbf{{{nu_str}}}"
                
            latex.append(f"    {exp_str} & {method_str} & {alpha_str} & {rho_str} & {nu_str} & {h_str} & {rmse_str} \\\\")
            first = False
            
        latex.append(r"    \midrule")
        
    # Remove the last midrule to make it look clean against the bottomrule
    if latex[-1].strip() == r"\midrule":
        latex.pop()
        
    latex.append(r"\end{longtable}")
    latex.append(r"\end{center}")
    
    print("\n".join(latex))



if __name__ == '__main__':
    generate_stage1_results()
    # print_full_latex_longtable()
