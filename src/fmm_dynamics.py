import numpy as np

class RoughSABRFMM:
    def __init__(self, time_grid, F0, beta, rho):
        """
        Initializes the Rough SABR Forward Market Model dynamics.
        
        Parameters:
        time_grid (numpy.ndarray): 1D array of time steps (e.g., [0.01, 0.02, ... T]).
        F0 (float or numpy.ndarray): Initial forward overnight rate(s) at t=0.
        beta (float): CEV elasticity parameter (0 <= beta <= 1). 
                      beta=1 is lognormal, beta=0 is normal.
        rho (float): Correlation between the forward rate and volatility (-1 <= rho <= 1).
        """
        self.time_grid = np.asarray(time_grid)
        self.dt = np.diff(self.time_grid, prepend=0.0) # Time step sizes
        
        self.F0 = F0
        self.beta = beta
        self.rho = rho

    def simulate_rates(self, rough_vol_paths, shock_vol, shock_rate_independent):
        """
        Simulates the forward rates using the Euler-Maruyama scheme with frozen drift.
        
        Parameters:
        rough_vol_paths (numpy.ndarray): The V_t paths from our RoughVolterraKernel.
        shock_vol (numpy.ndarray): The standard Brownian increments (dW_V) driving volatility.
        shock_rate_independent (numpy.ndarray): Independent Brownian increments (dZ) 
                                                for the rate process.
                                                
        Returns:
        numpy.ndarray: Simulated forward rate paths of shape (n_paths, n_steps).
        """
        n_paths, n_steps = rough_vol_paths.shape
        
        # 1. Correlate the Brownian motions
        # dW_F = rho * dW_V + sqrt(1 - rho^2) * dZ
        dW_F = self.rho * shock_vol + np.sqrt(1.0 - self.rho**2) * shock_rate_independent
        
        # 2. Initialize the forward rate paths array
        # We use np.full to handle both scalar F0 and array F0 effortlessly
        F_t = np.full((n_paths, n_steps), self.F0, dtype=np.float64)
        
        # 3. Euler-Maruyama Simulation
        # Since the drift is "frozen" (approximated as deterministic or zero under the 
        # forward swap measure), the primary driver is the stochastic diffusion term:
        # dF = V_t * (F_t ^ beta) * dW_F
        
        for i in range(1, n_steps):
            # Extract previous step values
            F_prev = F_t[:, i-1]
            V_prev = rough_vol_paths[:, i-1]
            
            # Prevent negative rates from causing complex numbers if beta < 1
            # by applying an absorbing boundary at a tiny positive number
            F_prev_safe = np.maximum(F_prev, 1e-8)
            
            # Calculate the diffusion term
            diffusion = V_prev * (F_prev_safe ** self.beta) * dW_F[:, i]
            
            # Step forward
            F_t[:, i] = F_prev + diffusion
            
        return F_t

# --- Example Usage ---
# time_grid = np.linspace(1/252, 1.0, 252)
# F0 = 0.04  # 4% initial forward rate
# beta = 0.5 # Square-root (CIR-like) elasticity
# rho = -0.6 # Negative correlation (typical for rates/equities to create skew)
# 
# fmm_engine = RoughSABRFMM(time_grid, F0, beta, rho)
# 
# # Assuming 'rough_vols', 'dW_V', and 'dZ' are generated from previous modules
# # simulated_rates = fmm_engine.simulate_rates(rough_vols, dW_V, dZ)