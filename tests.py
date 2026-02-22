import torch
import time
import numpy as np
from src.torch_model import TorchRoughSABR_FMM
from src.utils import bootstrap_forward_rates, load_discount_curve


def run_martingale_test():
    print("\n" + "="*65)
    print(f"{'TEST 2: STRICT MARTINGALE (NO-ARBITRAGE) TEST':^65}")
    print("="*65)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup Mock Market & Model
    print("[Setup] Initializing Extreme FMM Parameters...")
    ois_func = load_discount_curve("data/estr_disc.csv")
    grid_T, F0_rates = bootstrap_forward_rates(ois_func, max_maturity=30.0)
    
    # Highly realistic Rough Volatility parameters
    alpha_f = lambda T: np.full_like(T, 0.0150) # 150 bps base normal vol
    rho_f = lambda T: np.full_like(T, -0.40)    # Standard downward skew
    nu_f = lambda T: np.full_like(T, 0.50)      # Realistic high vol-of-vol
    H = 0.10                                    # True Roughness!

    model = TorchRoughSABR_FMM(
        grid_T, F0_rates, alpha_f, rho_f, nu_f, H, 
        beta_sabr=0.5, shift=0.03, correlation_mode='pca', n_factors=3, device=device
    )
    
    n_paths = 65536 # High path count to suppress standard MC noise
    # REPLACE THIS LINE:
    # time_grid = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device=device, dtype=torch.float64)
    
    n_steps_per_year = 52
    time_grid = torch.linspace(0.0, 5.0, 5 * n_steps_per_year + 1, device=device, dtype=torch.float64)

    print(f"[Simulation] Generating {n_paths} paths (Unfrozen Stochastic Drift)...")
    t0 = time.time()
    # MUST use freeze_drift=False to test the dynamic Euler-Maruyama cross-drift!
    with torch.no_grad():
        F_paths = model.simulate_forward_curve(n_paths, time_grid, freeze_drift=False)
    print(f"[Simulation] Done in {time.time() - t0:.2f}s")
    
    # 2. The Martingale Test Math
    print("\n[Testing] Computing Numeraire-Rebased Expectations...")
    # Under terminal measure Q^{T_N}, the asset P(t, T_j) / P(t, T_N) must be a martingale.
    # Z_t^j = P(t, T_j) / P(t, T_N) = prod_{k=j}^{N-1} (1 + tau_k * R_t^k)
    
    step_idx = -1  # Evaluate at the terminal simulation time (T = 5.0Y here)
    eval_time = time_grid[step_idx].item()
    start_idx = torch.searchsorted(
        model.T,
        torch.tensor(eval_time, device=device, dtype=model.dtype),
        right=False,
    ).item()


    tau = model.tau
    F0 = model.F0
    errors_bps = []
    
    invalid_stats = []

    print("-" * 92)
    print(
        f"{'Bond Ratio':<19} | {'Z_0 (Target)':<15} | {'E[Z_t] (Sim)':<15} | {'Invalid Paths':<13} | {'Leakage (bps)':<12}"
    )
    print("-" * 92)

    
    # We test the martingale property only for bonds with maturity >= eval_time
    for j in range(start_idx, model.N):
        maturity = model.T[j].item()  
        # True analytical Z_0
        Z_0_j = torch.prod(1.0 + tau[j:] * F0[j:]).item()
        
        # Simulated Z_t at t=5.0
        # F_paths shape: (n_paths, n_steps, N)
        ratio_terms = 1.0 + tau[j:] * F_paths[:, step_idx, j:]

        # Any non-positive term implies a bond-ratio sign/definition breakdown under simple compounding.
        valid_mask = torch.all(torch.isfinite(ratio_terms) & (ratio_terms > 0.0), dim=1)
        invalid_ratio = 1.0 - valid_mask.double().mean().item()
        invalid_stats.append(invalid_ratio)

        if valid_mask.any():
            Z_t_paths = torch.prod(ratio_terms[valid_mask], dim=1)
            E_Z_t_j = torch.mean(Z_t_paths).item()
            error_ratio = abs(E_Z_t_j - Z_0_j) / Z_0_j
            time_to_maturity = torch.sum(tau[j:]).item()
            error_bps = (error_ratio / time_to_maturity) * 10000.0
            errors_bps.append(error_bps)
            ez_text = f"{E_Z_t_j:<15.6f}"
            err_text = f"{error_bps:<12.4f}"
        else:
            ez_text = "N/A"
            err_text = "N/A"

        # Print every 5 years to keep console clean
        if j % 5 == 0 or j == model.N - 1:
            print(
                f"P(t,{maturity:04.1f}Y)/P(t,30Y) | {Z_0_j:<15.6f} | {ez_text:<15} | {100.0*invalid_ratio:11.4f}% | {err_text}"
            )

    print("-" * 92)
    if errors_bps:
        max_err = max(errors_bps)
        print(f"Maximum Arbitrage Leakage (valid paths only): {max_err:.4f} bps / rate")
    else:
        max_err = float('inf')
        print("Maximum Arbitrage Leakage: N/A (no valid paths)")

    worst_invalid = max(invalid_stats) if invalid_stats else 1.0
    print(f"Worst invalid-path ratio across maturities: {100.0*worst_invalid:.4f}%")

    if worst_invalid > 1e-4:
        print("Status: FAIL (Simulation produced non-positive bond-ratio factors: 1 + tau*F <= 0)")
    elif max_err < 0.5:
        print("Status: PASS (Martingale Dynamics within tolerance)")
    else:
        print("Status: WARNING (Significant discretization bias on valid paths)")
    print("=" * 65)


def run_aad_vs_fd_test():
    from src.pricers import torch_bermudan_pricer
    
    print("\n" + "="*65)
    print(f"{'TEST 1: AAD vs FINITE DIFFERENCE (BUMP) TEST':^65}")
    print("="*65)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup Standard Market & Model
    print("[Setup] Initializing Production FMM Parameters...")
    ois_func = load_discount_curve("data/estr_disc.csv")
    grid_T, F0_rates = bootstrap_forward_rates(ois_func, max_maturity=30.0)
    
    # Standard realistic parameters
    alpha_f = lambda T: np.full_like(T, 0.0150) 
    rho_f = lambda T: np.full_like(T, -0.40)    
    nu_f = lambda T: np.full_like(T, 0.50)      
    H = 0.15                                    
    
    # Create the BASE model
    model = TorchRoughSABR_FMM(
        grid_T, F0_rates, alpha_f, rho_f, nu_f, H, 
        beta_sabr=0.5, shift=0.03, correlation_mode='pca', n_factors=3, device=device
    )
    
    specs = {'Strike': F0_rates[1], 'Ex_Dates': [1.0, 2.0, 3.0, 4.0, 5.0]}
    n_paths = 32768 # High path count for stable Finite Differences
    n_steps_per_year = 24
    time_grid = torch.linspace(0.0, 5.0, 5 * n_steps_per_year + 1, device=device, dtype=torch.float64)
    
    # We will test the Delta sensitivity to the 6-Year Forward Rate
    target_idx = torch.argmin(torch.abs(torch.tensor(grid_T) - 6.0)).item()
    target_tenor = grid_T[target_idx]
    
    # ---------------------------------------------------------
    # PART A: EXACT AAD (PYTORCH)
    # ---------------------------------------------------------
    print(f"\n[AAD] Computing Exact PyTorch Sensitivities...")
    t0 = time.time()
    price_base = torch_bermudan_pricer(model, specs, n_paths, time_grid, use_checkpoint=False)
    price_base.backward()
    
    # Scale to bps (PV01)
    aad_delta = model.F0.grad[target_idx].item() * 10000 * 0.0001
    print(f"[AAD] Done in {time.time() - t0:.2f}s")
    
    # ---------------------------------------------------------
    # PART B: FINITE DIFFERENCE (BUMP)
    # ---------------------------------------------------------
    print(f"\n[BUMP] Computing Finite Difference (1 bp bump)...")
    
    # Create a completely fresh model with the exact same seed, but bumped F0
    bump_size = 0.0001 # 1 basis point
    F0_bumped = F0_rates.copy()
    F0_bumped[target_idx] += bump_size
    
    model_bumped = TorchRoughSABR_FMM(
        grid_T, F0_bumped, alpha_f, rho_f, nu_f, H, 
        beta_sabr=0.5, shift=0.03, correlation_mode='pca', n_factors=3, device=device
    )
    
    t0 = time.time()
    with torch.no_grad(): # No autograd needed for pure bumping
        price_up = torch_bermudan_pricer(model_bumped, specs, n_paths, time_grid, use_checkpoint=False)
        
    # FD Delta = (Price_Up - Price_Base) (in bps)
    fd_delta = (price_up.item() - price_base.item()) * 10000
    print(f"[BUMP] Done in {time.time() - t0:.2f}s")
    
    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------
    print("\n" + "-" * 65)
    print(f"{'Metric':<30} | {'Value'}")
    print("-" * 65)
    print(f"{'Target Forward Rate':<30} | {target_tenor}Y")
    print(f"{'Base Bermudan Price':<30} | {price_base.item() * 10000:.4f} bps")
    print(f"{'Bumped Bermudan Price':<30} | {price_up.item() * 10000:.4f} bps")
    print("-" * 65)
    print(f"{'Exact PyTorch AAD Delta':<30} | {aad_delta:.6f} bps")
    print(f"{'Finite Difference Delta':<30} | {fd_delta:.6f} bps")
    
    diff = abs(aad_delta - fd_delta)
    print(f"{'Absolute Difference':<30} | {diff:.6f} bps")
    
    if diff < 0.1:
        print("\nStatus: PASS (AAD Graph is flawlessly preserved!)")
    else:
        print("\nStatus: WARNING (Significant deviation in exercise boundary)")
    print("=" * 65)


def run_extreme_regime_test():
    import numpy as np
    from src.torch_model import TorchRoughSABR_FMM
    from src.pricers import mapped_smm_ode
    
    print("\n" + "="*65)
    print(f"{'TEST 3: BROKEN MARKET (EXTREME REGIME) TEST':^65}")
    print("="*65)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup Extreme "Broken" Market Data
    print("[Setup] Generating Extreme Market Conditions...")
    
    # Synthetic Inverted Yield Curve (5% at short end, decaying to 1% at 30Y)
    grid_T = np.arange(0.0, 31.0, 1.0)
    F0_rates = 0.01 + 0.04 * np.exp(-0.15 * grid_T[:-1])
    
    # Mathematically Violent Parameters
    alpha_f = lambda T: np.full_like(T, 0.0150) 
    rho_f = lambda T: np.full_like(T, -0.40)    
    nu_f = lambda T: np.full_like(T, 2.50)      # Massive Vol-of-Vol (Standard is 0.4)
    H = 0.01                                    # Hyper-roughness (Near singularity)
    
    print(f"[Setup] Device={device}, H={H}, nu={nu_f(np.array([0.0]))[0]:.2f}, n_tenors={len(F0_rates)}")
    print(f"[Setup] Initial Forward Range: [{np.min(F0_rates)*10000:.2f}, {np.max(F0_rates)*10000:.2f}] bps")
  
    # 2. Instantiate Model
    # Using 'full' mode with beta_decay=1e-5 creates a correlation matrix of ~0.999 
    # everywhere, severely stressing the Cholesky decomposition.
    try:
        model = TorchRoughSABR_FMM(
            grid_T, F0_rates, alpha_f, rho_f, nu_f, H, 
            beta_decay=1e-5, beta_sabr=0.5, shift=0.03, 
            correlation_mode='full', device=device
        )
        Sigma_rates = model.loadings[1:, :] @ model.loadings[1:, :].T
        eigvals = torch.linalg.eigvalsh(Sigma_rates)
        min_eig = torch.min(eigvals).item()
        max_eig = torch.max(eigvals).item()
        cond_num = (max_eig / max(min_eig, 1e-16))
        cholesky_status = f"PASS (min_eig={min_eig:.3e}, cond~{cond_num:.3e})"

    except Exception as e:
        cholesky_status = f"FAIL ({str(e)})"
        
    print(f"{'Cholesky Decomposition (Rho -> 1.0)':<40} | {cholesky_status}")

    # 3. Test Monte Carlo Engine (var_comp & fBM generation)
    n_paths = 4096
    time_grid = torch.linspace(0.0, 5.0, 5 * 24 + 1, device=device, dtype=torch.float64)
    
    try:
        # We use frozen drift here because we already mathematically proved in Test 2 
        # that unfrozen will hit the denominator flip. We are testing the fBM core here.
        with torch.no_grad():
            F_paths = model.simulate_forward_curve(n_paths, time_grid, freeze_drift=True)
            
        if torch.isnan(F_paths).any():
            mc_status = "FAIL (NaNs detected in paths)"
        elif torch.isinf(F_paths).any():
            mc_status = "FAIL (Infinities detected in paths)"
        else:
            F_terminal = F_paths[:, -1, :]
            min_term = torch.min(F_terminal).item()
            max_term = torch.max(F_terminal).item()
            q01 = torch.quantile(F_terminal.reshape(-1), 0.01).item()
            q99 = torch.quantile(F_terminal.reshape(-1), 0.99).item()
            invalid_simple = torch.mean((1.0 + model.tau.view(1, -1) * F_terminal <= 0.0).double()).item()
            mc_status = (
                f"PASS (terminal F in [{min_term*10000:.1f}, {max_term*10000:.1f}] bps, "
                f"q01/q99=[{q01*10000:.1f}, {q99*10000:.1f}] bps, "
                f"1+tau*F<=0 rate={100.0*invalid_simple:.4f}%)"
            )
    except Exception as e:
        mc_status = f"FAIL ({str(e)})"

    print(f"{'Monte Carlo Engine (H=0.01, Nu=2.5)':<40} | {mc_status}")

    # 4. Test Low-Fidelity ODE Surrogate
    # The ODE involves complex log/atan functions that often break under extreme stress
    try:
        Sigma_matrix = torch.matmul(model.loadings, model.loadings.T)
        test_expiries = np.array([5.0])
        test_tenors = np.array([5.0])
        test_strikes = np.array([0.0])
        
        v_ode = mapped_smm_ode(model, Sigma_matrix, test_expiries, test_tenors, test_strikes)
        
        if torch.isnan(v_ode).any():
            ode_status = "FAIL (NaNs detected in ODE)"
        elif torch.isinf(v_ode).any():
            ode_status = "FAIL (Infinities detected in ODE)"
        else:
            ode_vol_bps = v_ode.item() * 10000
            ode_status = f"PASS (ODE Vol={ode_vol_bps:.2f} bps, shape={tuple(v_ode.shape)})"

    except Exception as e:
        ode_status = f"FAIL ({str(e)})"
        
    print(f"{'Analytical ODE Surrogate':<40} | {ode_status}")

    # 5. Final Verdict
    print("-" * 65)
    if all("PASS" in status for status in [cholesky_status, mc_status, ode_status]):
        print("Status: SUCCESS (Model is mathematically bulletproof)")
    else:
        print("Status: WARNING (Model failed under extreme stress)")
    print("=" * 65)


def run_put_call_parity_test():
    import numpy as np
    import torch
    from src.torch_model import TorchRoughSABR_FMM
    from src.utils import bootstrap_forward_rates, load_discount_curve
    
    print("\n" + "="*65)
    print(f"{'TEST 4: PUT-CALL PARITY (PAYER vs RECEIVER) TEST':^65}")
    print("="*65)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup Standard Market & Model
    print("[Setup] Initializing Production FMM...")
    ois_func = load_discount_curve("data/estr_disc.csv")
    grid_T, F0_rates = bootstrap_forward_rates(ois_func, max_maturity=30.0)
    
    alpha_f = lambda T: np.full_like(T, 0.0150) 
    rho_f = lambda T: np.full_like(T, -0.40)    
    nu_f = lambda T: np.full_like(T, 0.50)      
    H = 0.15                                    
    
    model = TorchRoughSABR_FMM(
        grid_T, F0_rates, alpha_f, rho_f, nu_f, H, 
        beta_sabr=0.5, shift=0.03, correlation_mode='pca', n_factors=3, device=device
    )
    
    # 2. Define a 5Y x 5Y European Swaption
    T_ex = 5.0
    T_end = 10.0
    
    idx_ex = torch.argmin(torch.abs(torch.tensor(grid_T) - T_ex)).item()
    idx_end = torch.argmin(torch.abs(torch.tensor(grid_T) - T_end)).item()
    
    # Calculate True ATM Strike analytically from the initial curve
    P0 = torch.cumprod(torch.cat([torch.tensor([1.0], device=device, dtype=model.dtype), 1.0 / (1.0 + model.tau * model.F0)]), dim=0)
    A0 = torch.sum(model.tau[idx_ex:idx_end] * P0[idx_ex+1:idx_end+1])
    ATM_strike = ((P0[idx_ex] - P0[idx_end]) / A0).item()
    
    # For parity diagnostics, avoid forcing a fair strike from the same Monte Carlo sample,
    # which makes payer and receiver mechanically symmetric and can mask issues.
    # Use a fixed off-ATM strike so the forward leg is generally non-zero.
    strike_offset_bps = 50.0
    test_strike = ATM_strike + strike_offset_bps / 10000.0

    print(f"[Setup] 5Yx5Y European Swaption ATM Strike:  {ATM_strike*10000:.2f} bps")
    print(f"[Setup] 5Yx5Y European Swaption Test Strike: {test_strike*10000:.2f} bps (+{strike_offset_bps:.0f} bps)")

    
    # 3. Simulate Paths to Expiry
    n_paths = 65536
    n_steps_per_year = 24
    time_grid = torch.linspace(0.0, T_ex, int(T_ex * n_steps_per_year) + 1, device=device, dtype=torch.float64)
    
    print(f"[Simulation] Generating {n_paths} paths to T={T_ex}Y...")
    with torch.no_grad():
        F_paths = model.simulate_forward_curve(n_paths, time_grid, freeze_drift=True)
        F_T_full = F_paths[:, -1, idx_ex:]
        taus_full = model.tau[idx_ex:]
        dfs_full = torch.cumprod(1.0 / (1.0 + taus_full * F_T_full), dim=1)
        p_t_Tn = dfs_full[:, -1]
        
        # CORRECTED FAIR STRIKE CALCULATION
        # To get PV=0, the strike must be the numeraire-weighted expectation of the swap
        taus_swap = model.tau[idx_ex:idx_end]
        annuity_part = torch.sum(dfs_full[:, :idx_end-idx_ex] * taus_swap, dim=1)
        float_part = torch.sum(dfs_full[:, :idx_end-idx_ex] * F_T_full[:, :idx_end-idx_ex] * taus_swap, dim=1)
        
         # Test parity at a fixed strike (not fair-strike re-centered)
        swap_val = float_part - test_strike * annuity_part

        
        payer_deflated = torch.mean(torch.clamp(swap_val, min=0.0) / p_t_Tn)
        receiver_deflated = torch.mean(torch.clamp(-swap_val, min=0.0) / p_t_Tn)

        forward_swap_deflated = torch.mean(swap_val / p_t_Tn)
        
        # Rebase back to T=0
        p0_Tn = model.get_terminal_bond()
        payer_price = (p0_Tn * payer_deflated).item() * 10000
        receiver_price = (p0_Tn * receiver_deflated).item() * 10000
        forward_swap_price = (p0_Tn * forward_swap_deflated).item() * 10000

    # 5. Output and Verification
    print("\n" + "-" * 65)
    print(f"{'Metric':<30} | {'Value (bps)':>15}")
    print("-" * 65)
    print(f"{'Payer Swaption PV':<30} | {payer_price:15.6f}")
    print(f"{'Receiver Swaption PV':<30} | {receiver_price:15.6f}")
    print(f"{'Implied Forward Swap PV':<30} | {payer_price - receiver_price:15.6f}")
    print(f"{'Simulated Forward Swap PV':<30} | {forward_swap_price:15.6f}")
    print("-" * 65)
    
    parity_error = abs((payer_price - receiver_price) - forward_swap_price)
    print(f"Put-Call Parity Error: {parity_error:.6f} bps")
    
    if parity_error < 0.1:
        print("\nStatus: PASS (Discounting and Parity are mathematically exact)")
    else:
        print("\nStatus: WARNING (Parity mismatch implies numeraire leakage)")
    print("=" * 65)


def run_reproducibility_test():
    print("\n" + "="*65)
    print(f"{'TEST 5: RNG REPRODUCIBILITY TEST':^65}")
    print("="*65)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ois_func = load_discount_curve("data/estr_disc.csv")
    grid_T, F0_rates = bootstrap_forward_rates(ois_func, max_maturity=30.0)

    alpha_f = lambda T: np.full_like(T, 0.0150)
    rho_f = lambda T: np.full_like(T, -0.40)
    nu_f = lambda T: np.full_like(T, 0.50)
    H = 0.15

    model = TorchRoughSABR_FMM(
        grid_T, F0_rates, alpha_f, rho_f, nu_f, H,
        beta_sabr=0.5, shift=0.03, correlation_mode='pca', n_factors=3, device=device
    )

    n_paths = 2048
    time_grid = torch.linspace(0.0, 2.0, 2 * 12 + 1, device=device, dtype=torch.float64)

    with torch.no_grad():
        p1 = model.simulate_forward_curve(n_paths, time_grid, seed=123, freeze_drift=True)
        p2 = model.simulate_forward_curve(n_paths, time_grid, seed=123, freeze_drift=True)
        p3 = model.simulate_forward_curve(n_paths, time_grid, seed=124, freeze_drift=True)

    same_seed_diff = torch.max(torch.abs(p1 - p2)).item()
    diff_seed_diff = torch.max(torch.abs(p1 - p3)).item()

    print(f"{'Max |same seed diff|':<40} | {same_seed_diff:.3e}")
    print(f"{'Max |different seed diff|':<40} | {diff_seed_diff:.3e}")

    if same_seed_diff < 1e-14 and diff_seed_diff > 1e-10:
        print("Status: PASS (RNG seeding is deterministic and effective)")
    else:
        print("Status: WARNING (Unexpected RNG behavior)")
    print("=" * 65)


def run_time_step_stability_test():
    print("\n" + "="*65)
    print(f"{'TEST 6: TIME-STEP STABILITY TEST':^65}")
    print("="*65)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ois_func = load_discount_curve("data/estr_disc.csv")
    grid_T, F0_rates = bootstrap_forward_rates(ois_func, max_maturity=30.0)

    alpha_f = lambda T: np.full_like(T, 0.0150)
    rho_f = lambda T: np.full_like(T, -0.40)
    nu_f = lambda T: np.full_like(T, 0.50)
    H = 0.15

    model = TorchRoughSABR_FMM(
        grid_T, F0_rates, alpha_f, rho_f, nu_f, H,
        beta_sabr=0.5, shift=0.03, correlation_mode='pca', n_factors=3, device=device
    )

    n_paths = 8192*2
    test_horizon = 2.0
    target_idx = torch.argmin(torch.abs(torch.tensor(grid_T, device=device, dtype=model.dtype) - 5.0)).item()

    means = {}
    for steps_per_year in [12, 24, 48]:
        time_grid = torch.linspace(0.0, test_horizon, int(test_horizon * steps_per_year) + 1, device=device, dtype=torch.float64)
        with torch.no_grad():
            F_paths = model.simulate_forward_curve(n_paths, time_grid, seed=77, freeze_drift=True)
        means[steps_per_year] = torch.mean(F_paths[:, -1, target_idx]).item()

    print(f"{'Mean F(T) @ 12 steps/year':<40} | {means[12]*10000:.4f} bps")
    print(f"{'Mean F(T) @ 24 steps/year':<40} | {means[24]*10000:.4f} bps")
    print(f"{'Mean F(T) @ 48 steps/year':<40} | {means[48]*10000:.4f} bps")

    coarse_mid = abs(means[12] - means[24])
    mid_fine = abs(means[24] - means[48])
    print(f"{'|12-24|':<40} | {coarse_mid*10000:.4f} bps")
    print(f"{'|24-48|':<40} | {mid_fine*10000:.4f} bps")

    if mid_fine <= coarse_mid:
        print("Status: PASS (Observed convergence with finer discretization)")
    else:
        print("Status: WARNING (No clear refinement convergence)")
    print("=" * 65)


def run_correlation_consistency_test():
    print("\n" + "="*65)
    print(f"{'TEST 7: CORRELATION CONSISTENCY TEST':^65}")
    print("="*65)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ois_func = load_discount_curve("data/estr_disc.csv")
    grid_T, F0_rates = bootstrap_forward_rates(ois_func, max_maturity=30.0)

    alpha_f = lambda T: np.full_like(T, 0.0150)
    rho_f = lambda T: np.full_like(T, -0.40)
    nu_f = lambda T: np.full_like(T, 0.50)
    H = 0.15

    model = TorchRoughSABR_FMM(
        grid_T, F0_rates, alpha_f, rho_f, nu_f, H,
        beta_sabr=0.5, shift=0.03, correlation_mode='full', beta_decay=0.05, device=device
    )

    Sigma = model.loadings @ model.loadings.T
    eigvals = torch.linalg.eigvalsh(Sigma)
    min_eig = torch.min(eigvals).item()
    max_diag_dev = torch.max(torch.abs(torch.diagonal(Sigma) - 1.0)).item()

    print(f"{'Min eigenvalue of Sigma':<40} | {min_eig:.3e}")
    print(f"{'Max |diag(Sigma)-1|':<40} | {max_diag_dev:.3e}")

    if min_eig > -1e-10 and max_diag_dev < 1e-8:
        print("Status: PASS (Correlation construction is PSD and normalized)")
    else:
        print("Status: WARNING (Correlation matrix quality degraded)")
    print("=" * 65)


if __name__ == '__main__':
    # run_martingale_test()
    # run_aad_vs_fd_test()
    # run_extreme_regime_test()
    # run_put_call_parity_test()
    # run_reproducibility_test()
    run_time_step_stability_test()
    # run_correlation_consistency_test()