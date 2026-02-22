import numpy as np



BETA_SABR = 0.5
SHIFT_SABR = 0.0
H_GRID = np.array([0.15])   # 0.05, 0.10, , 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50


CALI_MODE = "MC"   # polynomial, ODE, MC  
CORR_MODE = "full"  # pca, full

if CORR_MODE == "full":
    CALI_MODE = "MC"


CHECK_MC = True
CHECK_DRIFT = False
CHECK_LIMIT = True

