import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator

class RoughSABRCalibrator:
    def __init__(self, vol_matrix):
        self.vol_matrix = vol_matrix
        # Ensure expiries and strikes are floats for math operations
        self.expiries = self.vol_matrix.index.values.astype(float)
        self.strike_offsets = self.vol_matrix.columns.values.astype(float)
        self.n_exp = len(self.expiries)
        
        # Pre-calculate ATM Term Structure: alpha(T)
        atm_idx = np.argmin(np.abs(self.strike_offsets))
        self.alpha_ts = PchipInterpolator(self.expiries, self.vol_matrix.iloc[:, atm_idx].values, extrapolate=True)
        
        # Flatten market data for the global optimizer
        self.market_vols = self.vol_matrix.values.flatten()
        T_grid, K_grid = np.meshgrid(self.expiries, self.strike_offsets, indexing='ij')
        self.T_flat, self.K_flat = T_grid.flatten(), K_grid.flatten()
        
        # Clean out NaNs if any exist in the market data
        valid = ~np.isnan(self.market_vols)
        self.market_vols = self.market_vols[valid]
        self.T_flat = self.T_flat[valid]
        self.K_flat = self.K_flat[valid]

    def rough_sabr_vol(self, k, T, alpha, rho, nu, H):
        """ Core Rough SABR asymptotic expansion formula """
        # Skew term scales with T^(H-0.5)
        skew = (rho * nu) / (2.0 * alpha * (H + 0.5)) * k * (T**(H - 0.5))
        # Smile term scales with T^(2H-1)
        smile = (2.0 - 3.0*rho**2)/24.0 * (nu/alpha)**2 * (k**2) * (T**(2.0*H - 1.0))
        # ATM Drift scales with T^(2H)
        drift = (2.0 - 3.0*rho**2)/24.0 * nu**2 * (T**(2.0*H))
        
        return alpha * (1.0 + skew + smile + drift)

    def _obj(self, p):
        # Unpack params: [rho_1...rho_n, nu_1...nu_n, H]
        rhos, nus, H = p[:self.n_exp], p[self.n_exp:2*self.n_exp], p[-1]
        
        # Interpolate rho(T) and nu(T) term structures
        r_ts = PchipInterpolator(self.expiries, rhos, extrapolate=True)
        n_ts = PchipInterpolator(self.expiries, nus, extrapolate=True)
        
        # Evaluate model vols across the grid
        v = self.rough_sabr_vol(
            self.K_flat, self.T_flat, 
            self.alpha_ts(self.T_flat), 
            r_ts(self.T_flat), 
            n_ts(self.T_flat), 
            H
        )
        return (v - self.market_vols) * 10000.0

    def calibrate(self):
        print("\n" + "="*60)
        print(f"{'ROUGH SABR CALIBRATION (TERM STRUCTURE)':^60}")
        print("="*60)
        
        # Initial guess: [Rhos, Nus, H]
        guess = np.concatenate([np.full(self.n_exp, -0.1), np.full(self.n_exp, 0.4), [0.1]])
        
        # PINNING H: Tighten the Hurst bound to stay in the highly rough regime (H <= 0.25)
        low_bounds = np.concatenate([np.full(self.n_exp, -0.99), np.full(self.n_exp, 0.001), [0.01]])
        high_bounds = np.concatenate([np.full(self.n_exp, 0.99), np.full(self.n_exp, 5.0), [0.25]])
        
        res = least_squares(
            self._obj, guess, 
            bounds=(low_bounds, high_bounds), 
            ftol=1e-10, xtol=1e-10
        )
        
        rmse = np.sqrt(np.mean(res.fun**2))
        H_opt = res.x[-1]
        
        print(f"Status : SUCCESS")
        print(f"Global Hurst (H): {H_opt:.6f}")
        print(f"RMSE           : {rmse:.4f} bps")
        
        # Fixed: Returning 'rmse_bps' to prevent KeyError in main.py
        return {
            'alpha_func': self.alpha_ts, 
            'H': H_opt, 
            'rmse_bps': rmse,
            'rho_func': PchipInterpolator(self.expiries, res.x[:self.n_exp], extrapolate=True),
            'nu_func': PchipInterpolator(self.expiries, res.x[self.n_exp:2*self.n_exp], extrapolate=True)
        }