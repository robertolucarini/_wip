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



# ========================================================================
# Calibration Stage 1
# ========================================================================
# 1. RESULTS
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


# 2. LATEX TABLE
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




# ========================================================================
# Execution
# ========================================================================
if __name__ == '__main__':
    generate_stage1_results()
    # print_full_latex_longtable()
    # plot_parameter_grid()
    # plot_true_3d_surfaces(method='PURE_MC')
    # export_latex_surface_data(method='PURE_MC')