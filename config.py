

CHECK_MC = True
CHECK_DRIFT = False
CHECK_LIMIT = True

BETA_SABR = 0.5
SHIFT_SABR = 0.0


CALI_MODE = "pca"   # polynomial, ODE, MC  
CORR_MODE = "full"  # pca, full

if CORR_MODE == "full":
    CALI_MODE = "MC"
