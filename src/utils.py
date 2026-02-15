import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import os

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

def print_header(title):
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)

def log_progress(category, message, level=0):
    indent = "   " * level
    if level == 2:
        print(f"{indent}> {message}")
    else:
        print(f"{indent}[{category}] {message}")

def print_summary_table(title, data_dict):
    print_header(title)
    for key, value in data_dict.items():
        if isinstance(value, float):
            print(f"{key:<30}: {value:.6f}")
        else:
            print(f"{key:<30}: {value}")

def print_greek_ladder(deltas, vegas, tenors):
    print("\n" + "="*50)
    print(f"{'GREEK LADDERS (PV01/Vega in bps)':^50}")
    print("-" * 50)
    print(f"{'Tenor':<10} | {'Delta (bps)':>15} | {'Vega (bps)':>15}")
    print("-" * 50)
    for i, t in enumerate(tenors):
        d, v = deltas[i], vegas[i]
        # Only print non-negligible risk
        if abs(d) > 1e-7 or abs(v) > 1e-7:
            print(f"{t:4.1f}Y      | {d:15.4f} | {v:15.4f}")
    print("-" * 50)

# Standard Loading Functions
def load_discount_curve(path):
    df = pd.read_csv(path)
    return PchipInterpolator(df.iloc[:,0], np.log(df.iloc[:,1]), extrapolate=True)

def bootstrap_forward_rates(interp_func, grid_tau=1.0, max_t=30.0):
    grid = np.arange(0, max_t + grid_tau, grid_tau)
    dfs = np.exp(interp_func(grid))
    fwds = (dfs[:-1] / dfs[1:] - 1) / grid_tau
    return grid, fwds

def load_swaption_vol_surface(path, tenor):
    df = pd.read_csv(path, index_col=0)
    return df # Assumes pre-formatted CSV

# ==========================================
# PARSERS
# ==========================================

def parse_strike(strike_str):
    """
    Converts a strike string (e.g., 'ATM', '-50BP') into a float rate offset.
    Returns the value in absolute decimals (e.g., 50 bps = 0.0050).
    """
    s = str(strike_str).upper().strip()
    if s == 'ATM':
        return 0.0
    if s.endswith('BP'):
        try:
            return float(s[:-2]) / 10000.0
        except ValueError:
            return 0.0
    return 0.0

# ==========================================
# CURVE BOOTSTRAPPING & FORWARD RATES
# ==========================================
def parse_tenor(tenor_str):
    """
    Converts a tenor string (e.g., '1Y', '6M', '1W') into a float year fraction.
    Returns np.nan for unrecognized formats so they can be dropped.
    """
    s = str(tenor_str).upper().strip()
    # Handle fractions or composite tickers if they exist
    if '/' in s: 
        s = s.split('/')[-1]
        
    try:
        if s.endswith('Y'):
            return float(s[:-1])
        if s.endswith('M'):
            return float(s[:-1]) / 12.0
        if s.endswith('W'):
            return float(s[:-1]) / 52.0
        if s.endswith('D'):
            return float(s[:-1]) / 365.25
    except ValueError:
        pass
        
    return np.nan # Use NaN instead of 0.0 to prevent duplicate T=0 rows

def load_discount_curve(csv_path):
    """
    Loads discount factors from a CSV and builds a continuous PCHIP interpolator.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Curve file not found: {csv_path}")
        
    df = pd.read_csv(csv_path, sep=None, engine='python') # Auto-detect separator
    df.columns = [str(c).strip().upper() for c in df.columns]
    
    # Identify columns
    col_t = next((c for c in df.columns if 'TICKER' in c or 'TENOR' in c or c == 'T'), None)
    col_v = next((c for c in df.columns if 'DF' in c or 'VALUE' in c or 'RATE' in c), None)
    
    if not col_t or not col_v:
        raise ValueError("Missing time or value columns in the curve CSV.")
        
    # Parse tenors and values
    df['T'] = df[col_t].apply(parse_tenor)
    df['DF'] = pd.to_numeric(df[col_v], errors='coerce')
    
    # Drop NaNs (unrecognized tenors) and sort
    df = df.dropna(subset=['T', 'DF']).sort_values('T')
    
    # CRITICAL FIX: Drop any duplicate tenors that might still exist 
    # keeping the last one (usually the cleanest quote)
    df = df.drop_duplicates(subset=['T'], keep='last')
    
    # Ensure T=0 exists for P(0,0) = 1.0
    if df['T'].min() > 0.001:
        row_0 = pd.DataFrame({'T': [0.0], 'DF': [1.0]})
        df = pd.concat([row_0, df[['T', 'DF']]], ignore_index=True)
        
    # Interpolate log discount factors to guarantee positive forward rates
    log_df_interpolator = PchipInterpolator(df['T'].values, np.log(df['DF'].values), extrapolate=True)
    
    # Return a lambda that exponentiates the interpolated log values
    return lambda t: np.exp(log_df_interpolator(t))

def bootstrap_forward_rates(discount_func, grid_tau=1.0, max_maturity=30.0):
    """
    Extracts discrete forward overnight rates from a continuous discount function.
    F(t, T_1, T_2) = (1 / tau) * (P(t, T_1) / P(t, T_2) - 1)
    
    Returns:
    tuple: (time_grid, forward_rates)
    """
    time_grid = np.arange(0, max_maturity + grid_tau + 0.001, grid_tau)
    
    # Calculate discount factors at grid points
    dfs = discount_func(time_grid)
    
    # Calculate simply compounded forward rates
    forward_rates = (dfs[:-1] / dfs[1:] - 1) / grid_tau
    
    return time_grid, forward_rates

# ==========================================
# VOLATILITY SURFACE LOADER
# ==========================================

def load_swaption_vol_surface(csv_path, target_underlying_tenor):
    """
    Loads normal swaption volatility data and extracts the surface for a specific 
    underlying swap tenor. Pivots the data for easy SABR calibration.
    
    Returns:
    pandas.DataFrame: A pivot table with Expiries as the index, Strike Offsets as columns, 
                      and Normal Volatility as values.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Vol file not found: {csv_path}")
        
    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = [str(c).strip().upper() for c in df.columns]
    
    # Identify standard column names
    c_exp = next((c for c in df.columns if 'EXPIRY' in c), None)
    c_ten = next((c for c in df.columns if 'UNDERLYING' in c or 'TENOR' in c), None)
    c_str = next((c for c in df.columns if 'STRIKE' in c), None)
    c_val = next((c for c in df.columns if ('VALUE' in c or 'VOL' in c) and 'ID' not in c), None)
    
    if not all([c_exp, c_ten, c_str, c_val]):
        raise ValueError("Missing required columns in volatility CSV.")
        
    # Parse standard formats
    df['Expiry'] = df[c_exp].apply(parse_tenor)
    df['Underlying_Tenor'] = df[c_ten].apply(parse_tenor)
    df['Strike_Offset'] = df[c_str].apply(parse_strike)
    df['Vol'] = pd.to_numeric(df[c_val], errors='coerce')
    
    # Drop NAs
    df = df.dropna(subset=['Expiry', 'Underlying_Tenor', 'Strike_Offset', 'Vol'])
    
    # Filter for the specific swap tenor (e.g., 1Y, 5Y swaps)
    # Using a small tolerance (0.05) to avoid floating point mismatch
    slice_df = df[np.abs(df['Underlying_Tenor'] - target_underlying_tenor) < 0.05].copy()
    
    if slice_df.empty:
        raise ValueError(f"No volatility data found for target underlying tenor: {target_underlying_tenor}Y")
        
    # Pivot the data into a clean matrix: Index=Expiry, Columns=Strike, Values=Vol
    vol_matrix = slice_df.pivot_table(
        values='Vol', 
        index='Expiry', 
        columns='Strike_Offset', 
        aggfunc='first'
    )
    
    # Sort index and columns sequentially
    vol_matrix = vol_matrix.sort_index().sort_index(axis=1)
    
    return vol_matrix