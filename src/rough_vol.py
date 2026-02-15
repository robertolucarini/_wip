import numpy as np
import scipy.special as sp

class RoughVolterraKernel:
    def __init__(self, time_grid, H):
        """
        Initializes the fractional Volterra integral kernel.
        
        Parameters:
        time_grid (numpy.ndarray): 1D array of time steps (e.g., [0.01, 0.02, ... T]).
        H (float): Hurst parameter (0 < H < 0.5 for rough volatility).
        """
        self.time_grid = np.asarray(time_grid)
        self.H = H
        self.n_steps = len(self.time_grid)
        
        # Pre-compute the integration matrix upon initialization to save time
        # during Monte Carlo loops
        self._kernel_matrix = self._build_kernel_matrix()

    def _build_kernel_matrix(self):
        """
        Builds the Riemann-Liouville fractional integration kernel matrix.
        Vectorized entirely using numpy broadcasting.
        """
        # 1. Create a 2D grid of t_i (evaluation time) and t_j (historical time)
        t_i = self.time_grid[:, None]
        t_j = self.time_grid[None, :]
        
        # 2. The Volterra kernel has a singularity when t_i = t_j (division by zero).
        # We use a tiny epsilon (1e-12) to prevent NaN errors while preserving accuracy.
        dt = np.maximum(t_i - t_j, 1e-12)
        
        # 3. Apply the power-law kernel: (t - s)^(H - 1/2) / Gamma(H + 1/2)
        gamma_factor = sp.gamma(self.H + 0.5)
        kernel = (dt ** (self.H - 0.5)) / gamma_factor
        
        # 4. Zero out the upper triangle. 
        # Future increments (t_j >= t_i) cannot affect the past.
        kernel = np.tril(kernel, k=-1)
        
        return kernel

    def generate_rough_process(self, standard_paths):
        """
        Transforms standard Brownian paths into a rough fractional process.
        
        Parameters:
        standard_paths (numpy.ndarray): Matrix of shape (n_paths, n_steps) 
                                        from our PCAPathConstructor.
                                        
        Returns:
        numpy.ndarray: Rough paths of shape (n_paths, n_steps).
        """
        # Step 1: Extract the increments (dW) from the standard cumulative paths.
        # We prepend a column of zeros to maintain the array shape.
        zero_col = np.zeros((standard_paths.shape[0], 1))
        dW = np.diff(standard_paths, axis=1)
        dW = np.hstack((zero_col, dW))
        
        # Step 2: Discrete Volterra Integration via Matrix Multiplication
        # Instead of a slow for-loop, we do: rough_path_i = Sum_j (K_{i,j} * dW_j)
        # Shape math: (n_paths, n_steps) @ (n_steps, n_steps).T -> (n_paths, n_steps)
        rough_paths = dW @ self._kernel_matrix.T
        
        return rough_paths

# --- Example Usage ---
# time_grid = np.linspace(1/252, 1.0, 252)
# H = 0.1  # Highly rough volatility
# vol_engine = RoughVolterraKernel(time_grid, H)
#
# # 'W_t' is the output from PCAPathConstructor
# # rough_W_t = vol_engine.generate_rough_process(W_t)