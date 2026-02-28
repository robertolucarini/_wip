import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.utils import load_swaption_vol_surface
from src.calibration import RoughSABRCalibrator
from config import H_GRID
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.interpolate import PchipInterpolator
from src.utils import load_discount_curve, bootstrap_forward_rates
from src.torch_model import TorchRoughSABR_FMM
from src.calibration import CorrelationCalibrator
from main import load_atm_matrix
from config import BETA_SABR, SHIFT_SABR


# ========================================================================
# Calibration Stage 1
# ========================================================================
# 1. RESULTS
def generate_stage1_results(methods=['polynomial', 'AMMO_ODE', 'PURE_MC']):
    """
    Runs Stage 1 calibration across multiple methods and stores the resulting 
    parameters and estimated volatility surfaces for later analysis.

    Guarantees full storage per (method, H, expiry): Alpha, Rho, Nu,
    expiry-level RMSE in bps, and fixed-H global RMSE in bps.
    """
    os.makedirs('results', exist_ok=True)

    print("Loading market data...")
    vol_matrix_1y = load_swaption_vol_surface("data/estr_vol_full_strikes.csv", 1.0)
    calibrator = RoughSABRCalibrator(vol_matrix_1y)

    expiries = calibrator.expiries
    strikes = calibrator.strike_offsets
    h_grid = np.array(H_GRID, dtype=float)

    market_path = "results/stage1_surface_MARKET.csv"
    vol_matrix_1y.to_csv(market_path)
    print(f"Saved market surface to {market_path}\n")

    market_matrix = vol_matrix_1y.loc[expiries, strikes].values
    expected_rows_per_method = len(h_grid) * len(expiries)

    param_records = []

    for method in methods:
        print(f"{'='*60}")
        print(f"Running Calibration: {method.upper()}")
        print(f"{'='*60}")

        method_surfaces = []
        method_records = []
        completed_h = []

        for H in h_grid:
            print(f"-> Calibrating {method.upper()} at fixed H={H:.3f}")
            try:
                calib = calibrator.calibrate(method=method, H_grid=np.array([H], dtype=float))
            except Exception as e:
                print(f"   !! FAILED at H={H:.3f} for {method.upper()}: {e}")
                continue

            solved_H = float(calib['H'])
            rmse_global = float(calib['rmse_bps'])

            T_grid, K_grid = np.meshgrid(expiries, strikes, indexing='ij')
            T_flat = T_grid.flatten()
            K_flat = K_grid.flatten()

            a_flat = calib['alpha_func'](T_flat)
            r_flat = calib['rho_func'](T_flat)
            n_flat = calib['nu_func'](T_flat)

            if method.upper() in ['PURE_MC', 'MC']:
                vols_flat = calibrator.rough_sabr_vol_mc(K_flat, T_flat, a_flat, r_flat, n_flat, solved_H)
            elif method.upper() == 'POLYNOMIAL':
                vols_flat = calibrator.rough_sabr_vol(K_flat, T_flat, a_flat, r_flat, n_flat, solved_H)
            else:
                vols_flat = calibrator.rough_sabr_vol_ode(K_flat, T_flat, a_flat, r_flat, n_flat, solved_H)

            vols_matrix = vols_flat.reshape(len(expiries), len(strikes))
            df_surface_h = pd.DataFrame(vols_matrix, index=expiries, columns=strikes)
            df_surface_h['H'] = solved_H
            method_surfaces.append(df_surface_h.reset_index().rename(columns={'index': 'Expiry'}))

            alphas = calib['alpha_func'](expiries)
            rhos = calib['rho_func'](expiries)
            nus = calib['nu_func'](expiries)
            if isinstance(nus, (float, int)):
                nus = np.full_like(expiries, nus)

            expiry_rmse_bps = np.sqrt(np.mean(((vols_matrix - market_matrix) * 10000.0) ** 2, axis=1))

            for i, T in enumerate(expiries):
                method_records.append({
                    'Method': method,
                    'H': solved_H,
                    'Expiry': T,
                    'Alpha': alphas[i],
                    'Rho': rhos[i],
                    'Nu': nus[i],
                    'RMSE_bps': expiry_rmse_bps[i],
                    'RMSE_global_bps': rmse_global
                })

            completed_h.append(solved_H)

        if method_surfaces:
            df_surface = pd.concat(method_surfaces, ignore_index=True)
            surface_path = f"results/stage1_surface_{method.upper()}.csv"
            df_surface.to_csv(surface_path, index=False)
            print(f"-> Saved fitted surfaces (all completed H) to {surface_path}")

        if method_records:
            param_records.extend(method_records)

        # strict completeness check (important for long expensive runs)
        unique_h = np.unique(np.round(completed_h, 6))
        expected_h = np.unique(np.round(h_grid, 6))
        missing_h = sorted(set(expected_h.tolist()) - set(unique_h.tolist()))

        if len(method_records) != expected_rows_per_method or missing_h:
            raise RuntimeError(
                f"Stage1 results are incomplete for {method.upper()}: "
                f"expected {expected_rows_per_method} rows, got {len(method_records)}; "
                f"missing H values={missing_h}"
            )

        print(f"-> Completeness check passed for {method.upper()}: {len(method_records)} rows ({len(unique_h)} H x {len(expiries)} expiries).\n")

    if param_records:
        df_params = pd.DataFrame(param_records)
        params_path = "results/stage1_parameters.csv"
        df_params.to_csv(params_path, index=False)
        print(f"Saved all parameters to {params_path}")
        print("Done! You are ready to generate tables.")


# 2. LATEX TABLE
def print_full_latex_longtable(csv_path="results/stage1_parameters.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # THE FIX: Round the floats to eliminate precision mismatch!
    df['H'] = df['H'].round(3)
    df['Expiry'] = df['Expiry'].round(3)
    
    # Get all unique H and expiries, sorted
    all_Hs = sorted(df['H'].unique())
    all_expiries = sorted(df['Expiry'].unique())
    
    latex = []
    
    # We use longtable so it breaks beautifully across pages in the PDF
    latex.append(r"\begin{center}")
    latex.append(r"\small")
    latex.append(r"\begin{longtable}{c c c c c c c}")
    latex.append(r"    \caption{Comparison of calibrated marginal parameters (Stage 1). PURE\_MC successfully untraps the volatility-of-volatility parameter ($\nu$) from the asymptotic ODE breakdown, resulting in significantly lower RMSE.} \label{tab:stage1_full_comparison} \\")
    latex.append(r"    \toprule")
    latex.append(r"    & \multicolumn{2}{c}{\textbf{$\alpha$ (bps)}} & \multicolumn{2}{c}{\textbf{$\rho$}} & \multicolumn{2}{c}{\textbf{RMSE (bps)}} \\")
    latex.append(r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    latex.append(r"    \textbf{Expiry} & \textbf{MC} & \textbf{ODE} & \textbf{MC} & \textbf{ODE} & \textbf{MC} & \textbf{ODE} \\")
    latex.append(r"    \midrule")
    latex.append(r"    \endfirsthead")
    latex.append(r"")
    latex.append(r"    % Header for subsequent pages")
    latex.append(r"    \multicolumn{7}{c}%")
    latex.append(r"    {{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\")
    latex.append(r"    \toprule")
    latex.append(r"    & \multicolumn{2}{c}{\textbf{$\alpha$ (bps)}} & \multicolumn{2}{c}{\textbf{$\rho$}} & \multicolumn{2}{c}{\textbf{RMSE (bps)}} \\")
    latex.append(r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    latex.append(r"    \textbf{Expiry} & \textbf{MC} & \textbf{ODE} & \textbf{MC} & \textbf{ODE} & \textbf{MC} & \textbf{ODE} \\")
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
    
    for h in all_Hs:
        # Now we can safely use direct equality
        subset_h = df[df['H'] == h]
        if subset_h.empty:
            continue
            
        # Extract global Nu for the subheader
        mc_nu_row = subset_h[subset_h['Method'] == 'PURE_MC']
        ode_nu_row = subset_h[subset_h['Method'] == 'AMMO_ODE']
        
        mc_nu = mc_nu_row['Nu'].iloc[0] if not mc_nu_row.empty else np.nan
        ode_nu = ode_nu_row['Nu'].iloc[0] if not ode_nu_row.empty else np.nan
        
        mc_nu_str = f"{mc_nu:.4f}" if not np.isnan(mc_nu) else "N/A"
        ode_nu_str = f"{ode_nu:.4f}" if not np.isnan(ode_nu) else "N/A"
        
        # Add the H subheader with the global Nu values
        latex.append(f"    \\multicolumn{{7}}{{l}}{{\\textbf{{$H = {h:.2f}$}} \\quad (Global $\\nu$: MC = {mc_nu_str}, ODE = {ode_nu_str})}} \\\\")
        latex.append(r"    \midrule")
        
        first = True
        for exp in all_expiries:
            subset_exp = subset_h[subset_h['Expiry'] == exp]
            if subset_exp.empty:
                continue
                
            mc_row = subset_exp[subset_exp['Method'] == 'PURE_MC']
            ode_row = subset_exp[subset_exp['Method'] == 'AMMO_ODE']
            
            # --- Format MC Values ---
            if not mc_row.empty:
                mc_alpha = f"{mc_row['Alpha'].values[0] * 10000.0:.1f}"
                mc_rho = f"{mc_row['Rho'].values[0]:.3f}"
                mc_rmse_val = mc_row['RMSE_bps'].values[0]
                mc_rmse = f"{mc_rmse_val:.2f}"
            else:
                mc_alpha, mc_rho, mc_rmse_val, mc_rmse = "-", "-", np.inf, "-"
                
            # --- Format ODE Values ---
            if not ode_row.empty:
                ode_alpha = f"{ode_row['Alpha'].values[0] * 10000.0:.1f}"
                ode_rho = f"{ode_row['Rho'].values[0]:.3f}"
                ode_rmse_val = ode_row['RMSE_bps'].values[0]
                ode_rmse = f"{ode_rmse_val:.2f}"
            else:
                ode_alpha, ode_rho, ode_rmse_val, ode_rmse = "-", "-", np.inf, "-"
                
            # --- Bold the winning RMSE ---
            if not mc_row.empty and not ode_row.empty:
                if mc_rmse_val < ode_rmse_val:
                    mc_rmse = f"\\textbf{{{mc_rmse}}}"
                elif ode_rmse_val < mc_rmse_val:
                    ode_rmse = f"\\textbf{{{ode_rmse}}}"
            
            exp_str = f"{exp:.1f}Y"
            
            # Print RMSE only on the first row of the block to keep the table clean
            if first:
                latex.append(f"    {exp_str} & {mc_alpha} & {ode_alpha} & {mc_rho} & {ode_rho} & {mc_rmse} & {ode_rmse} \\\\")
                first = False
            else:
                latex.append(f"    {exp_str} & {mc_alpha} & {ode_alpha} & {mc_rho} & {ode_rho} & & \\\\")
                
        latex.append(r"    \midrule")
        
    # Remove the last midrule to make it look clean against the bottomrule
    if latex[-1].strip() == r"\midrule":
        latex.pop()
        
    latex.append(r"\end{longtable}")
    latex.append(r"\end{center}")
    
    print("\n".join(latex))


# 3. PARAMETERS CHARTS
def plot_parameter_grid(csv_path="results/stage1_parameters.csv", save_path="results/stage1_parameters_grid.png"):
    # 1. Load the data
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Wait for the calibration to finish!")
        return
        
    df = pd.read_csv(csv_path)
    
    # 2. Setup the 2x2 grid with shared Y-axes per row for direct visual comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey='row')
    
    # Isolate the H values to create a consistent color map
    h_values = sorted(df['H'].unique())
    colors = cm.viridis(np.linspace(0.1, 0.9, len(h_values)))
    
    # 3. Helper function to plot a specific parameter and method
    def plot_panel(ax, method, param, ylabel, multiplier=1.0):
        df_method = df[df['Method'] == method]
        
        for idx, h in enumerate(h_values):
            subset = df_method[df_method['H'] == h].sort_values('Expiry')
            if not subset.empty:
                # ADDED .to_numpy() TO FIX LOCAL PANDAS ERROR
                x_data = subset['Expiry'].to_numpy()
                y_data = (subset[param] * multiplier).to_numpy()
                
                ax.plot(x_data, y_data, 
                        marker='o', markersize=4, lw=1.5, 
                        color=colors[idx], label=f"H = {h:.2f}")
                
        ax.set_title(f"{param} Term Structure ({'True MC' if method == 'PURE_MC' else 'ODE'})", fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        if param == 'Rho':
            ax.axhline(0, color='black', lw=1, ls='--') # Add zero line for correlations
            
    # 4. Populate the 4 panels
    # Row 0: Alpha (Left: MC, Right: ODE) - multiplied by 10000 for bps
    plot_panel(axes[0, 0], method='PURE_MC', param='Alpha', ylabel=r'Base Volatility $\alpha$ (bps)', multiplier=10000.0)
    plot_panel(axes[0, 1], method='AMMO_ODE', param='Alpha', ylabel='', multiplier=10000.0)

    # Row 1: Rho (Left: MC, Right: ODE)
    plot_panel(axes[1, 0], method='PURE_MC', param='Rho', ylabel=r'Spot-Vol Correlation $\rho$')
    plot_panel(axes[1, 1], method='AMMO_ODE', param='Rho', ylabel='')
    
    # Set X-axis labels for the bottom row
    axes[1, 0].set_xlabel("Expiry (Years)", fontsize=11)
    axes[1, 1].set_xlabel("Expiry (Years)", fontsize=11)
    
    # Add a single unified legend outside the plots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5), title="Hurst Exponent", title_fontsize='11', fontsize='10')
    
    # Adjust layout to make room for the legend and title
    plt.tight_layout(rect=[0, 0, 0.98, 0.96])
    fig.suptitle("Marginal Parameter Stability: True Monte Carlo vs. ODE Asymptotic Approximation", fontsize=16, fontweight='bold')
    
    # 5. Save and display
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved high-resolution grid to {save_path}")


# 4. VOL SURFACE CHARTS
def plot_true_3d_surfaces(method='PURE_MC', results_dir='results'):
    # 1. Define file paths
    market_file = os.path.join(results_dir, 'stage1_surface_MARKET.csv')
    model_file = os.path.join(results_dir, f'stage1_surface_{method.upper()}.csv')
    
    if not os.path.exists(market_file) or not os.path.exists(model_file):
        print(f"Error: Could not find CSVs for {method}. Make sure they are in {results_dir}/")
        return

    # 2. Load the data and convert to basis points (bps)
    df_market = pd.read_csv(market_file, index_col=0) * 10000.0
    df_model = pd.read_csv(model_file, index_col=0) * 10000.0
    
    expiries = df_market.index.values.astype(float)
    strikes = df_market.columns.values.astype(float) * 10000.0 # strikes in bps
    
    # Calculate Residuals (Model - Market)
    df_res = df_model - df_market
    rmse = np.sqrt(np.nanmean(df_res.values**2))
    
    # Shared limits and Meshgrid for all 3D plots
    vmin = min(np.nanmin(df_market.values), np.nanmin(df_model.values))
    vmax = max(np.nanmax(df_market.values), np.nanmax(df_model.values))
    X, Y = np.meshgrid(strikes, expiries)
    
    # 3. Set up the Figure (wider to accommodate three 3D axes gracefully)
    fig = plt.figure(figsize=(24, 7))
    
    # Define a shared viewing angle for consistency
    view_elev = 25
    view_azim = -125
    
    # --- PANEL 1: MARKET SURFACE (True 3D Surface) ---
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, df_market.values, cmap='viridis', 
                             vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.85)
    ax1.set_title("Market Volatility Surface", fontsize=14, fontweight='bold')
    ax1.set_xlabel("\nStrike Offset (bps)", fontsize=11)
    ax1.set_ylabel("\nExpiry (Years)", fontsize=11)
    ax1.set_zlabel("\nNormal Vol (bps)", fontsize=11)
    ax1.view_init(elev=view_elev, azim=view_azim)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='Normal Volatility (bps)')

    # --- PANEL 2: MODEL SURFACE (True 3D Surface) ---
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, df_model.values, cmap='viridis', 
                             vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.85)
    ax2.set_title(f"Model Surface ({method})", fontsize=14, fontweight='bold')
    ax2.set_xlabel("\nStrike Offset (bps)", fontsize=11)
    ax2.set_ylabel("\nExpiry (Years)", fontsize=11)
    ax2.set_zlabel("\nNormal Vol (bps)", fontsize=11)
    ax2.view_init(elev=view_elev, azim=view_azim)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='Normal Volatility (bps)')

    # --- PANEL 3: RESIDUALS (3D Bar Chart) ---
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    # Prepare the 3D grid flattening for the bar chart
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = np.zeros_like(X_flat)
    dZ_flat = df_res.values.flatten()
    
    # Filter out NaNs for the 3D plot
    valid_mask = ~np.isnan(dZ_flat)
    X_valid = X_flat[valid_mask]
    Y_valid = Y_flat[valid_mask]
    Z_valid = Z_flat[valid_mask]
    dZ_valid = dZ_flat[valid_mask]
    
    # Create a custom colormap for the bars (Red for positive error, Blue for negative)
    norm = mcolors.TwoSlopeNorm(vmin=min(dZ_valid.min(), -1), vcenter=0, vmax=max(dZ_valid.max(), 1))
    colors = cm.RdBu_r(norm(dZ_valid))
    
    # Dimensions of the bars
    dx = np.full_like(X_valid, (strikes[-1] - strikes[0]) / (len(strikes) * 1.5))
    dy = np.full_like(Y_valid, (expiries[-1] - expiries[0]) / (len(expiries) * 1.5))
    
    # Plot the 3D bars
    ax3.bar3d(X_valid, Y_valid, Z_valid, dx, dy, dZ_valid, color=colors, shade=True, alpha=0.9)
    
    ax3.set_title(f"Residuals (RMSE: {rmse:.2f} bps)", fontsize=14, fontweight='bold')
    ax3.set_xlabel("\nStrike Offset (bps)", fontsize=11)
    ax3.set_ylabel("\nExpiry (Years)", fontsize=11)
    ax3.set_zlabel("\nError (bps)", fontsize=11)
    ax3.view_init(elev=view_elev, azim=view_azim)
    
    # Final layout adjustments
    plt.tight_layout()
    save_path = os.path.join(results_dir, f'vol_surface_3d_comparison_{method}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Awesome! Saved fully 3D plot to {save_path}")


# 5. SURFACES OVERLEAF



# ========================================================================
# Calibration Stage 2
# ========================================================================
# DATA PREP
def setup_stage2_calibrator_for_H(h_target, param_csv="results/stage1_parameters.csv"):
    """
    Subtask 1: Data Ingestion & Model Initialization.
    Reconstructs the PyTorch model with FLAT EXTRAPOLATION to prevent 
    short-end polynomial blowups on the uncalibrated spot rate.
    """
    if not os.path.exists(param_csv):
        raise FileNotFoundError(f"Cannot find {param_csv}. Run Stage 1 first.")
        
    df = pd.read_csv(param_csv)
    subset = df[(np.isclose(df['H'], h_target)) & (df['Method'] == 'PURE_MC')].sort_values('Expiry')
    
    if subset.empty:
        raise ValueError(f"No PURE_MC parameters found for H={h_target} in {param_csv}")
        
    expiries = subset['Expiry'].values
    alphas = subset['Alpha'].values
    rhos = subset['Rho'].values
    nus = subset['Nu'].values
    
    # --- THE FIX: FLAT EXTRAPOLATION WRAPPER ---
    min_T, max_T = expiries.min(), expiries.max()
    
    base_alpha = PchipInterpolator(expiries, alphas)
    base_rho = PchipInterpolator(expiries, rhos)
    base_nu = PchipInterpolator(expiries, nus)
    
    # By clipping the input 't' to the valid domain, we force the function 
    # to flat-line outside the bounds instead of shooting to infinity.
    alpha_func = lambda t: base_alpha(np.clip(t, min_T, max_T))
    rho_func = lambda t: base_rho(np.clip(t, min_T, max_T))
    nu_func = lambda t: base_nu(np.clip(t, min_T, max_T))
    # -------------------------------------------
    
    ois_func = load_discount_curve("data/estr_disc.csv")
    grid_T, F0_rates = bootstrap_forward_rates(ois_func)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_base = TorchRoughSABR_FMM(
        grid_T, F0_rates, alpha_func, rho_func, nu_func, h_target, 
        beta_sabr=BETA_SABR, shift=SHIFT_SABR, correlation_mode='full', device=device
    )
    
    atm_matrix = load_atm_matrix("data/estr_vol_full_strikes.csv")
    corr_calibrator = CorrelationCalibrator(atm_matrix, model_base)
    
    return corr_calibrator


# STAGE 2 RESULTS
def generate_stage2_results(param_csv="results/stage1_parameters.csv"):
    """
    Subtask 2: Matrix AMMO Execution Loop.
    Iterates through all optimal H values found in Stage 1, runs the AAD-based 
    correlation calibration, and saves the final matrices and timing benchmarks.
    """
    print("\n" + "="*60)
    print(f"{'STAGE 2: SPATIAL CORRELATION CALIBRATION (MATRIX AMMO)':^60}")
    print("="*60)
    
    if not os.path.exists(param_csv):
        print(f"Error: {param_csv} not found. Please run Stage 1 first.")
        return
        
    df = pd.read_csv(param_csv)
    
    # Isolate the Pure MC results and find all unique H values
    mc_df = df[df['Method'] == 'PURE_MC']
    if mc_df.empty:
        print("No PURE_MC results found in Stage 1 parameters. Check Stage 1 output.")
        return
        
    unique_Hs = sorted(mc_df['H'].unique())
    print(f"Found {len(unique_Hs)} Hurst exponent(s) to process: {unique_Hs}\n")
    
    performance_records = []
    
    for h in unique_Hs:
        print(f">>> Starting Matrix AMMO for H = {h:.3f} <<<")
        try:
            # 1. Reconstruct the model and calibrator for this specific H
            calibrator = setup_stage2_calibrator_for_H(h, param_csv)
            
            # 2. Run the AMMO Calibration and strictly time it
            t0 = time.time()
            corr_res = calibrator.calibrate()
            ammo_time = time.time() - t0
            
            # 3. Extract the final calibrated correlation matrix (Sigma)
            sigma_matrix = corr_res['Sigma_matrix']
            
            # 4. Save the matrix to CSV
            # Shape is (N+1) x (N+1), where index 0 is the Volatility driver Z(t), 
            # and indices 1 to N are the forward rates F_i(t)
            df_sigma = pd.DataFrame(sigma_matrix)
            sigma_path = f"results/stage2_correlation_H_{h:.3f}.csv"
            df_sigma.to_csv(sigma_path, index=False, header=False)
            
            print(f"-> Saved Calibrated Correlation Matrix to {sigma_path}")
            
            # 5. Record performance for the benchmark table
            performance_records.append({
                'H': h,
                'AMMO_Time_Seconds': ammo_time,
                'Matrix_Size': f"{sigma_matrix.shape[0]}x{sigma_matrix.shape[1]}"
            })
            
        except Exception as e:
            print(f"-> FAILED to process H={h}: {e}")
            
    # 6. Save the master performance benchmark table
    if performance_records:
        df_perf = pd.DataFrame(performance_records)
        perf_path = "results/stage2_performance.csv"
        df_perf.to_csv(perf_path, index=False)
        print(f"\nSaved AMMO performance benchmarks to {perf_path}")
        
    print("\nStage 2 Results Generation Complete!")


# HEATMAPS
def plot_stage2_correlation(h_target, results_dir='results'):
    """
    Subtask 3a: Correlation Matrix Visualization.
    Plots the anatomical breakdown of the AMMO-calibrated Sigma Matrix.
    """
    sigma_path = os.path.join(results_dir, f"stage2_correlation_H_{h_target:.3f}.csv")
    if not os.path.exists(sigma_path):
        print(f"Error: {sigma_path} not found.")
        return

    # 1. Load the calibrated full (N+1) x (N+1) matrix
    sigma = pd.read_csv(sigma_path, header=None).values

    # 2. Get the tenors for axis labeling
    from src.utils import load_discount_curve, bootstrap_forward_rates
    ois_func = load_discount_curve("data/estr_disc.csv")
    grid_T, _ = bootstrap_forward_rates(ois_func)
    forward_tenors = grid_T[:-1]  # The N forward rate tenors

    # 3. Setup the 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    
    # --- PANEL 1: Forward-Forward Spatial Correlation Heatmap ---
    ff_corr = sigma[1:, 1:] # Isolate the N x N forward rate block
    im = axes[0].imshow(ff_corr, cmap='viridis', vmin=0.0, vmax=1.0, origin='upper', aspect='auto')
    axes[0].set_title(f"Calibrated Spatial Correlation $\Sigma$ (H={h_target:.2f})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Forward Rate Index $T_j$", fontsize=12)
    axes[0].set_ylabel("Forward Rate Index $T_i$", fontsize=12)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="Correlation")

    # --- PANEL 2: Correlation Decay Profile ---
    # Plot how the 1Y Forward (Index 0) correlates with all subsequent forwards
    axes[1].plot(forward_tenors, ff_corr[0, :], marker='o', lw=2, markersize=5, color='darkblue')
    axes[1].fill_between(forward_tenors, ff_corr[0, :], 0, color='darkblue', alpha=0.1)
    axes[1].set_title("Spatial Decay Profile (Base: 1Y Forward)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Maturity (Years)", fontsize=12)
    axes[1].set_ylabel("Correlation with 1Y Fwd", fontsize=12)
    axes[1].grid(True, alpha=0.4)
    axes[1].set_ylim(0, 1.05)

    # --- PANEL 3: Spot-Vol Leverage Check ---
    # The 0th row/col of the full Sigma is the Volatility Driver Z(t)
    spot_vol_corr = sigma[0, 1:]
    axes[2].plot(forward_tenors, spot_vol_corr, marker='s', lw=2, markersize=5, color='darkred')
    axes[2].set_title(r"Preserved Spot-Vol Skew $\rho(T_i)$", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Maturity (Years)", fontsize=12)
    axes[2].set_ylabel(r"Correlation $\rho$", fontsize=12)
    axes[2].grid(True, alpha=0.4)
    axes[2].set_ylim(-1.0, 1.0)
    axes[2].axhline(0, color='black', lw=1, ls='--')

    plt.tight_layout()
    save_path = os.path.join(results_dir, f"stage2_visuals_H_{h_target:.3f}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Awesome! Saved Stage 2 Matrix Visuals to {save_path}")


def print_stage2_latex_performance(perf_csv="results/stage2_performance.csv"):
    """
    Subtask 3b: LaTeX Benchmark Table Generation.
    Prints the AAD execution speed table to copy directly into the paper.
    """
    if not os.path.exists(perf_csv):
        return
        
    df = pd.read_csv(perf_csv)
    
    latex = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Matrix AMMO Calibration Performance. Adjoint Algorithmic Differentiation computes the dense spatial Jacobian in a single backward pass, eliminating the $O(N^2)$ scaling penalty of traditional finite differences.}",
        r"    \label{tab:ammo_performance}",
        r"    \begin{tabular}{l c c c}",
        r"        \toprule",
        r"        \textbf{Hurst ($H$)} & \textbf{Matrix Size} & \textbf{Calibrated Angles} & \textbf{AMMO Execution Time (s)} \\",
        r"        \midrule"
    ]
    
    for _, row in df.iterrows():
        size_str = row['Matrix_Size']
        n_dim = int(size_str.split('x')[0])
        # The number of free Rapisarda angles calibrated is roughly (N-2)(N-1)/2
        n_angles = int((n_dim - 2) * (n_dim - 1) / 2)
        
        latex.append(f"        {row['H']:.3f} & {size_str} & {n_angles} & {row['AMMO_Time_Seconds']:.2f} \\\\")
        
    latex.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}"
    ])
    
    print("\n" + "="*60)
    print("STAGE 2 LATEX BENCHMARK TABLE")
    print("="*60)
    print("\n".join(latex))
    print("="*60 + "\n")


# ========================================================================
# Execution
# ========================================================================
if __name__ == '__main__':
    # STAGE 1
    generate_stage1_results()
    print_full_latex_longtable()
    plot_parameter_grid()
    plot_true_3d_surfaces(method='PURE_MC')

    # STAGE 2
    # Test Subtask 1 on the H you already have
    # test_H = 0.05 
    # print(f"Testing Model Reconstruction for H={test_H}...")
    # calibrator = setup_stage2_calibrator_for_H(test_H)
    # print("Successfully instantiated CorrelationCalibrator!")
    # print(f"Target ATM Matrix shape: {calibrator.market_vols.shape}")
    
    # generate_stage2_results()

    # test_h = 0.05 
    # # plot_stage2_correlation(h_target=test_h)
    # print_stage2_latex_performance()
    