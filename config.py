import numpy as np



BETA_SABR = 0.0
SHIFT_SABR = 0.0
H_GRID = np.array([0.05, 0.15])#, 0.20, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5])#, 0.20, 0.25, 0.30])   # 0.05, 0.10, , 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50


CALI_MODE = "PURE_MC"   # polynomial, ODE, AMMO_ODE, PURE_MC
CORR_MODE = "full"  # pca, full

if CORR_MODE == "full":
    CALI_MODE = "PURE_MC"


CHECK_MC = True
CHECK_DRIFT = False
CHECK_LIMIT = False

# --- STAGE 2 CALIBRATION REGULARIZATION ---
# Replaces the statistical data-smoothing heuristic with Lesniewski's 
# Second-Order Tikhonov (Mean Curvature) Regularization.
USE_TIKHONOV = True
LAMBDA_CURVATURE = 30.0  # Tuning parameter for spatial smoothness. (e.g., 1.0 to 100.0)