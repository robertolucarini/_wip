import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

# Ensure the output directory exists
os.makedirs("pics", exist_ok=True)

def bachelier_price(F, K, T, vol, option_type='payer'):
    if T <= 0: return np.maximum(F - K if option_type == 'payer' else K - F, 0)
    d = (F - K) / (vol * np.sqrt(T))
    if option_type == 'payer':
        return (F - K) * norm.cdf(d) + vol * np.sqrt(T) * norm.pdf(d)
    else:
        return (K - F) * norm.cdf(-d) + vol * np.sqrt(T) * norm.pdf(d)

class ModelValidator:
    def __init__(self, calibrator, params):
        self.calibrator = calibrator
        self.params = params
        self.expiries = calibrator.expiries
        self.strikes = calibrator.strike_offsets

    def check_1_oos_residual_analysis(self):
        print("\n[Validation 1/3] Generating Residual Heatmap...")
        
        # NEW LOGIC: Pre-calculate parameter grids to avoid broadcasting errors
        a_grid = self.params['alpha_func'](self.calibrator.T_grid)
        r_grid = self.params['rho_func'](self.calibrator.T_grid)
        n_grid = self.params['nu_func'](self.calibrator.T_grid)
        
        model_vols = self.calibrator.rough_sabr_implied_vol(
            self.calibrator.K_grid, 
            self.calibrator.T_grid, 
            a_grid, r_grid, n_grid,
            self.params['H']
        )
        
        residuals = (model_vols - self.calibrator.vol_matrix.values) * 10000.0
        
        plt.figure(figsize=(10, 6))
        im = plt.imshow(residuals, aspect='auto', cmap='RdBu_r', 
                        extent=[self.strikes[0], self.strikes[-1], self.expiries[-1], self.expiries[0]])
        plt.colorbar(im, label='Residual Error (bps)')
        plt.title(f"Rough SABR Residual Map (RMSE: {self.params['rmse_bps']:.2f} bps)")
        plt.xlabel("Strike Offset (bps)")
        plt.ylabel("Expiry (Years)")
        
        plt.savefig("pics/val_1_residuals.png")
        plt.close()
        print("-> Saved: pics/val_1_residuals.png")

    def check_2_pdf_positivity(self, test_expiry=1.0):
        print(f"\n[Validation 2/3] Checking PDF Positivity for T={test_expiry}...")
        k_dense = np.linspace(self.strikes[0]-0.005, self.strikes[-1]+0.005, 300)
        dk = k_dense[1] - k_dense[0]
        
        a = self.params['alpha_func'](test_expiry)
        r = self.params['rho_func'](test_expiry)
        n = self.params['nu_func'](test_expiry)
        h = self.params['H']
        
        vols = self.calibrator.rough_sabr_implied_vol(k_dense, test_expiry, a, r, n, h)
        prices = [bachelier_price(0.04, 0.04 + k, test_expiry, v) for k, v in zip(k_dense, vols)]
        
        pdf = np.diff(prices, n=2) / (dk**2)
        is_arbitrage_free = np.all(pdf >= -1e-6)
        
        plt.figure(figsize=(8, 5))
        plt.plot(k_dense[1:-1], pdf, color='forestgreen')
        plt.axhline(0, color='black', lw=0.5, ls='--')
        plt.title(f"Risk-Neutral Density Check (T={test_expiry}Y)")
        plt.xlabel("Strike Offset")
        plt.ylabel("Density")
        
        status = "PASSED" if is_arbitrage_free else "FAILED"
        plt.savefig(f"pics/val_2_pdf_check_{test_expiry}y.png")
        plt.close()
        print(f"-> Saved: pics/val_2_pdf_check_{test_expiry}y.png | Status: {status}")

    def check_3_parameter_stability(self):
        print("\n[Validation 3/3] Running Parameter Stability (Hurst Consistency)...")
        h_samples = []
        # Run 3 trials with slight random variations to check if H is a stable global minimum
        for i in range(3):
            # Add 1bp of noise to the market data
            noise = np.random.normal(0, 0.0001, self.calibrator.market_vols.shape)
            temp_market = self.calibrator.market_vols + noise
            
            # Simple re-calibration of H
            def mini_obj(h):
                model = self.calibrator.rough_sabr_implied_vol(
                    self.calibrator.K_flat, self.calibrator.T_flat,
                    self.params['alpha_func'](self.calibrator.T_flat),
                    self.params['rho_func'](self.calibrator.T_flat),
                    self.params['nu_func'](self.calibrator.T_flat), h
                )
                return np.sum((model - temp_market)**2)
            
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(mini_obj, bounds=(0.01, 0.49), method='bounded')
            h_samples.append(res.x)
            print(f"   Stability Trial {i+1}: H={res.x:.4f}")
            
        print(f"-> Hurst StdDev: {np.std(h_samples):.6f}")