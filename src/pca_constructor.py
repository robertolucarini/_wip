import numpy as np
import scipy.linalg as la

class PCAPathConstructor:
    def __init__(self, time_grid):
        """
        Initializes the PCA constructor for standard Brownian Motion.
        
        Parameters:
        time_grid (numpy.ndarray): 1D array of time steps (e.g., [0.01, 0.02, ... T]).
        """
        self.time_grid = np.asarray(time_grid)
        self.n_steps = len(self.time_grid)
        
        # Pre-compute the decomposition upon initialization
        self._eigenvalues, self._eigenvectors = self._decompose_covariance()

    def _decompose_covariance(self):
        """
        Builds the covariance matrix for standard Brownian motion and 
        performs the eigenvalue decomposition.
        """
        # 1. Build the covariance matrix: C[i, j] = min(t_i, t_j)
        # We use numpy broadcasting to do this instantly without loops
        t_matrix_i = self.time_grid[:, None]
        t_matrix_j = self.time_grid[None, :]
        cov_matrix = np.minimum(t_matrix_i, t_matrix_j)
        
        # 2. Eigenvalue Decomposition (eigh is optimized for symmetric matrices)
        eigenvalues, eigenvectors = la.eigh(cov_matrix)
        
        # 3. Sort in descending order (largest eigenvalues first)
        # This is the crucial PCA step: pairing the biggest variance 
        # with the first (best) Sobol dimensions.
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Filter out any tiny negative eigenvalues caused by floating point limits
        eigenvalues = np.maximum(eigenvalues, 0.0)
        
        return eigenvalues, eigenvectors

    def construct_paths(self, shocks):
        """
        Transforms independent standard normal shocks into PCA-constructed 
        Brownian paths.
        
        Parameters:
        shocks (numpy.ndarray): Matrix of shape (n_paths, n_steps) from our VarianceReductionEngine.
        
        Returns:
        numpy.ndarray: Simulated Brownian paths of shape (n_paths, n_steps).
        """
        # The PCA transformation: Paths = Z * sqrt(Lambda) * V^T
        # Z: shocks, Lambda: eigenvalues, V: eigenvectors
        
        # Scale eigenvectors by the square root of the eigenvalues
        scaled_eigenvectors = self._eigenvectors * np.sqrt(self._eigenvalues)
        
        # Matrix multiplication to construct all paths simultaneously
        # Shape: (n_paths, n_steps) @ (n_steps, n_steps) -> (n_paths, n_steps)
        brownian_paths = shocks @ scaled_eigenvectors.T
        
        return brownian_paths

# --- Example Usage ---
# time_grid = np.linspace(1/252, 1.0, 252) # 1 year of daily steps
# pca_engine = PCAPathConstructor(time_grid)
# 
# # Assuming 'shocks' is the output from our VarianceReductionEngine
# # W_t = pca_engine.construct_paths(shocks)