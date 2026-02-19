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


    def rough_sabr_vol_ode(self, k, T, alpha, rho, nu, H):
        """ 
        Advanced Rough SABR closed-form approximation using the ODE solution 
        from Fukasawa & Gatheral (2022).
        """
        # Calculate the scaled moneyness y
        y = (nu * (T**(H - 0.5)) * k) / alpha
        
        # Clip rho for numerical stability in log and arctan operations
        rho_safe = np.clip(rho, -0.9999, 0.9999)
        
        def G_half(z):
            inner = np.sqrt(1.0 + rho_safe * z + z**2 / 4.0) - rho_safe - z / 2.0
            return 4.0 * (np.log(inner / (1.0 - rho_safe)))**2
            
        def G_zero(z):
            term1 = np.log(1.0 + 2.0 * rho_safe * z + z**2)
            denom = np.sqrt(1.0 - rho_safe**2)
            term2 = (2.0 * rho_safe / denom) * (np.arctan(rho_safe / denom) - np.arctan((z + rho_safe) / denom))
            return term1 + term2

        # Avoid division by zero at exactly ATM (k=0)
        safe_y = np.where(np.abs(y) < 1e-12, 1e-12, y)
        
        z0 = safe_y / (2.0 * H + 1.0)
        z_half = 2.0 * safe_y / (2.0 * H + 1.0)
        
        w0 = 3.0 * (1.0 - 2.0 * H) / (2.0 * H + 3.0)
        w_half = 2.0 * H / (2.0 * H + 3.0)
        
        # Interpolate between H=0 and H=1/2 extreme solutions
        G_A = ((2.0 * H + 1.0)**2) * (w0 * G_zero(z0) + w_half * G_half(z_half))
        
        # Prevent floating-point negatives and division-by-zero
        G_A_safe = np.clip(G_A, a_min=1e-14, a_max=None)
        
        # The limit of |y|/sqrt(G_A) as y -> 0 is 1.0
        ratio = np.where(np.abs(y) < 1e-12, 1.0, np.abs(safe_y) / np.sqrt(G_A_safe))

        
        return alpha * ratio


    def _obj(self, p, H, method):
        # Unpack params: [rho_1...rho_n, nu_global]
        rhos, nu_global = p[:self.n_exp], p[-1]
        
        # Interpolate rho(T) term structure
        r_ts = PchipInterpolator(self.expiries, rhos, extrapolate=True)
        
        # Select the pricing method
        if method == 'polynomial':
            v = self.rough_sabr_vol(
                self.K_flat, self.T_flat, 
                self.alpha_ts(self.T_flat), 
                r_ts(self.T_flat), 
                nu_global, 
                H
            )
        elif method == 'ODE':
            v = self.rough_sabr_vol_ode(
                self.K_flat, self.T_flat, 
                self.alpha_ts(self.T_flat), 
                r_ts(self.T_flat), 
                nu_global, 
                H
            )
        else:
            raise ValueError(f"Unknown calibration method: {method}")
            
        return (v - self.market_vols) * 10000.0

    def calibrate(self, method='ODE', H_grid=np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])):
        print("\n" + "="*60)
        print(f"{f'ROUGH SABR CALIBRATION (GLOBAL NU | {method.upper()})':^60}")
        print("="*60)
        
        best_rmse = np.inf
        best_H = None
        best_res = None
        
        # Initial guess: [Rhos, Nu_global]
        guess = np.concatenate([np.full(self.n_exp, -0.1), [0.4]])
        low_bounds = np.concatenate([np.full(self.n_exp, -0.99), [0.001]])
        high_bounds = np.concatenate([np.full(self.n_exp, 0.99), [5.0]])
        
        # Discrete Grid Search over Hurst exponent
        for H in H_grid:
            res = least_squares(
                self._obj, guess, args=(H, method),
                bounds=(low_bounds, high_bounds), 
                ftol=1e-10, xtol=1e-10
            )
            rmse = np.sqrt(np.mean(res.fun**2))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_H = H
                best_res = res
                
        print(f"Status : SUCCESS")
        print(f"Global Hurst (H): {best_H:.6f}")
        print(f"Global Nu       : {best_res.x[-1]:.4f}")
        print(f"RMSE            : {best_rmse:.4f} bps")
        
        best_rhos = best_res.x[:self.n_exp]
        best_nu = best_res.x[-1]
        
        return {
            'alpha_func': self.alpha_ts, 
            'H': best_H, 
            'rmse_bps': best_rmse,
            'rho_func': PchipInterpolator(self.expiries, best_rhos, extrapolate=True),
            # Returning constant interpolator for nu to keep downstream FMM compatibility 
            'nu_func': PchipInterpolator(self.expiries, np.full(self.n_exp, best_nu), extrapolate=True)
        }