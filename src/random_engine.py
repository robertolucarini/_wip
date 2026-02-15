import numpy as np
from scipy.stats import qmc, norm
import warnings

class VarianceReductionEngine:
    def __init__(self, n_paths, dimensions, seed=None):
        """
        Initializes the QMC Variance Reduction Engine.
        
        Parameters:
        n_paths (int): Number of base paths to generate. For optimal Sobol 
                       properties, this should be a power of 2.
        dimensions (int): Total number of random variables needed per path 
                          (e.g., time steps * driving factors).
        seed (int): Optional seed for reproducibility.
        """
        self.n_paths = n_paths
        self.dimensions = dimensions
        self.seed = seed
        
        # Sobol sequences are mathematically optimized for powers of 2
        if not (n_paths != 0 and ((n_paths & (n_paths - 1)) == 0)):
            warnings.warn("For optimal QMC properties, n_paths should be a power of 2 "
                          "(e.g., 1024, 2048, 4096).")

    def generate_shocks(self):
        """
        Generates Sobol sequences, applies the inverse normal transform, 
        and appends antithetic variates.
        
        Returns:
        numpy.ndarray: Matrix of shape (2 * n_paths, dimensions) containing
                       optimized standard normal shocks.
        """
        # 1. Initialize the Sobol sequence generator
        sampler = qmc.Sobol(d=self.dimensions, seed=self.seed)
        
        # 2. Generate uniform Sobol points [0, 1]
        m = int(np.log2(self.n_paths))
        if 2**m == self.n_paths:
            # Faster, exact base-2 generation if n_paths is a power of 2
            uniform_points = sampler.random_base2(m=m)
        else:
            uniform_points = sampler.random(n=self.n_paths)
        
        # Clip to prevent exactly 0 or 1, which would result in infinity 
        # during the inverse normal transformation
        uniform_points = np.clip(uniform_points, 1e-10, 1.0 - 1e-10)
        
        # 3. Apply the Inverse Normal Transform (Percent Point Function)
        standard_normal_shocks = norm.ppf(uniform_points)
        
        # 4. Generate and Append Antithetic Variates
        # For every path Z, we append -Z to perfectly balance the mean to 0
        antithetic_shocks = -standard_normal_shocks
        
        # Stack vertically: top half is original, bottom half is mirrored
        optimized_shocks = np.vstack((standard_normal_shocks, antithetic_shocks))
        
        return optimized_shocks

# --- Example Usage ---
# engine = VarianceReductionEngine(n_paths=1024, dimensions=250)
# shocks = engine.generate_shocks()
# print(f"Shape of shocks matrix: {shocks.shape}") # Expected: (2048, 250)