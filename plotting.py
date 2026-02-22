import argparse
import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import CALI_MODE, H_GRID, CORR_MODE, BETA_SABR, SHIFT_SABR
from src.calibration import RoughSABRCalibrator, CorrelationCalibrator
from src.pricers import mapped_smm_ode, mapped_smm_pricer
from src.torch_model import TorchRoughSABR_FMM
from src.utils import load_swaption_vol_surface, parse_tenor, load_discount_curve, bootstrap_forward_rates  
from config import CALI_MODE, H_GRID
from src.calibration import RoughSABRCalibrator
from src.utils import load_swaption_vol_surface, parse_tenor
import torch


def _available_underlyings(csv_path: str):
    df = pd.read_csv(csv_path, sep=None, engine='python')
    cols = [str(c).strip().upper() for c in df.columns]
    col_ten = next((c for c in cols if 'UNDERLYING' in c or 'TENOR' in c), None)
    if col_ten is None:
        raise ValueError("Could not detect underlying tenor column in volatility CSV.")

    raw_col = df.columns[cols.index(col_ten)]
    tenors = df[raw_col].apply(parse_tenor)
    uniq = np.array(sorted(t for t in tenors.dropna().unique() if t > 0.0), dtype=float)
    return uniq


def _select_underlying_tenor(csv_path: str, underlying_tenor: float, underlying_index: Optional[int]):
    available = _available_underlyings(csv_path)
    if underlying_index is None:
        return underlying_tenor, available

    if underlying_index < 0 or underlying_index >= len(available):
        raise IndexError(
            f"underlying_index={underlying_index} out of range [0, {len(available)-1}]"
        )
    return float(available[underlying_index]), available


def _model_surface(calibrator: RoughSABRCalibrator, calib: dict, method: str):
    T_grid, K_grid = np.meshgrid(calibrator.expiries, calibrator.strike_offsets, indexing='ij')
    T_flat = T_grid.flatten()
    K_flat = K_grid.flatten()

    alpha = calib['alpha_func'](T_flat)
    rho = calib['rho_func'](T_flat)
    nu = calib['nu_func'](T_flat)
    H = calib['H']

    method_l = method.lower()
    if method_l == 'ode':
        vols = calibrator.rough_sabr_vol_ode(K_flat, T_flat, alpha, rho, nu, H)
    elif method_l == 'polynomial':
        vols = calibrator.rough_sabr_vol(K_flat, T_flat, alpha, rho, nu, H)
    elif method_l == 'mc':
        vols = calibrator.rough_sabr_vol_mc(K_flat, T_flat, alpha, rho, nu, H)
    else:
        raise ValueError(f"Unknown model surface method: {method}")

    return vols.reshape(len(calibrator.expiries), len(calibrator.strike_offsets))


def _calibrate_for_underlying(vol_csv: str, underlying_tenor: float, underlying_index: Optional[int], calibration_method: str):
    tenor, available = _select_underlying_tenor(vol_csv, underlying_tenor, underlying_index)
    market_vol_matrix = load_swaption_vol_surface(vol_csv, tenor)

    calibrator = RoughSABRCalibrator(market_vol_matrix)
    calib = calibrator.calibrate(method=calibration_method, H_grid=H_GRID)
    return tenor, available, market_vol_matrix, calibrator, calib


def make_market_vs_model_surface_plot(
    vol_csv: str,
    underlying_tenor: float = 1.0,
    underlying_index: Optional[int] = None,
    calibration_method: str = CALI_MODE,
    surface_method: str = 'ode',
    output_path: str = 'pics/market_vs_model_surface.png',
):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    tenor, available, market_vol_matrix, calibrator, calib = _calibrate_for_underlying(
        vol_csv, underlying_tenor, underlying_index, calibration_method
    )

    market_vols_bps = market_vol_matrix.values * 10000.0
    
    # Stage-aware model surface:
    # - If CORR_MODE=full, run stage-2 correlation calibration and use mapped SMM pricing.
    # - Otherwise, fallback to stage-1 marginal surface.
    if CORR_MODE.lower() == 'full':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_final = _build_stage2_model(vol_csv, calib, device=device)
        Sigma = model_final.loadings @ model_final.loadings.T

        expiries = np.repeat(calibrator.expiries, len(calibrator.strike_offsets))
        tenors = np.full_like(expiries, tenor)
        strikes = np.tile(calibrator.strike_offsets, len(calibrator.expiries))

        if surface_method.lower() == 'mc':
            model_flat = mapped_smm_pricer(model_final, Sigma, expiries, tenors, strikes)
        else:
            model_flat = mapped_smm_ode(model_final, Sigma, expiries, tenors, strikes).detach().cpu().numpy()

        model_vols_bps = model_flat.reshape(len(calibrator.expiries), len(calibrator.strike_offsets)) * 10000.0
        stage_label = 'Stage-2 (Mapped SMM)'
    else:
        model_vols_bps = _model_surface(calibrator, calib, method=surface_method) * 10000.0
        stage_label = 'Stage-1 (Marginal)'

    
    
    residuals_bps = model_vols_bps - market_vols_bps

    strikes_bps = calibrator.strike_offsets * 10000.0
    expiries = calibrator.expiries
    extent = [strikes_bps[0], strikes_bps[-1], expiries[-1], expiries[0]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    vmin = min(np.nanmin(market_vols_bps), np.nanmin(model_vols_bps))
    vmax = max(np.nanmax(market_vols_bps), np.nanmax(model_vols_bps))

    im0 = axes[0].imshow(market_vols_bps, aspect='auto', cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Market Vol Surface ({tenor:.1f}Y underlying)")
    axes[0].set_xlabel('Strike Offset (bps)')
    axes[0].set_ylabel('Expiry (Years)')
    fig.colorbar(im0, ax=axes[0], label='Normal Vol (bps)')

    im1 = axes[1].imshow(model_vols_bps, aspect='auto', cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Model Vol Surface ({surface_method.upper()} | {stage_label})")
    axes[1].set_xlabel('Strike Offset (bps)')
    axes[1].set_ylabel('Expiry (Years)')
    fig.colorbar(im1, ax=axes[1], label='Normal Vol (bps)')

    rlim = max(abs(np.nanmin(residuals_bps)), abs(np.nanmax(residuals_bps)))
    im2 = axes[2].imshow(residuals_bps, aspect='auto', cmap='RdBu_r', extent=extent, vmin=-rlim, vmax=rlim)
    axes[2].set_title(f"Residuals (Model - Market), RMSE={calib['rmse_bps']:.2f} bps")
    axes[2].set_xlabel('Strike Offset (bps)')
    axes[2].set_ylabel('Expiry (Years)')
    fig.colorbar(im2, ax=axes[2], label='Residual (bps)')

    fig.suptitle(
        f"Market vs Model Vol Surface | Underlying={tenor:.1f}Y | Calib={calibration_method.upper()} | {stage_label} | H={calib['H']:.3f}",
        fontsize=12,
    )
    
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Available underlying tenors: {np.array2string(available, precision=2)}")


def make_optimal_parameter_term_structure_plot(
    vol_csv: str,
    underlying_tenor: float = 1.0,
    underlying_index: Optional[int] = None,
    calibration_method: str = CALI_MODE,
    output_path: str = 'pics/optimal_parameter_term_structure.png',
):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    tenor, available, _, calibrator, calib = _calibrate_for_underlying(
        vol_csv, underlying_tenor, underlying_index, calibration_method
    )

    T = calibrator.expiries
    alpha_stage1 = calib['alpha_func'](T) * 10000.0
    rho_stage1 = calib['rho_func'](T)
    nu_stage1 = calib['nu_func'](T)
    H = calib['H']

    if CORR_MODE.lower() == 'full':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_final = _build_stage2_model(vol_csv, calib, device=device)
        Sigma = model_final.loadings @ model_final.loadings.T
        alpha_map, rho_map = _mapped_smm_term_structures(model_final, Sigma, T, tenor)

        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)

        axes[0].plot(T, alpha_stage1, marker='o', color='tab:blue', label='Stage-1 alpha(T)')
        axes[0].plot(T, alpha_map * 10000.0, marker='x', color='tab:purple', label='Stage-2 mapped alpha_smm(T)')
        axes[0].set_ylabel('alpha [bps]')
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc='best')

        axes[1].plot(T, rho_stage1, marker='o', color='tab:orange', label='Stage-1 rho(T)')
        axes[1].plot(T, rho_map, marker='x', color='tab:red', label='Stage-2 mapped rho_smm(T)')
        axes[1].axhline(0.0, color='black', lw=0.8, ls='--')
        axes[1].set_ylabel('rho')
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc='best')

        axes[2].plot(T, nu_stage1, marker='o', color='tab:green', label='Stage-1 nu(T)')
        axes[2].set_xlabel('Expiry (Years)')
        axes[2].set_ylabel('nu')
        axes[2].grid(alpha=0.3)
        axes[2].legend(loc='best')

        stage_label = 'Stage-2 mapped overlays (full mode)'
    else:
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)

    axes[0].plot(T, alpha_stage1, marker='o', color='tab:blue')
    axes[0].set_ylabel('alpha(T) [bps]')
    axes[0].grid(alpha=0.3)

    axes[1].plot(T, rho_stage1, marker='o', color='tab:orange')
    axes[1].axhline(0.0, color='black', lw=0.8, ls='--')
    axes[1].set_ylabel('rho(T)')
    axes[1].grid(alpha=0.3)

    axes[2].plot(T, nu_stage1, marker='o', color='tab:green')
    axes[2].set_xlabel('Expiry (Years)')
    axes[2].set_ylabel('nu(T)')
    axes[2].grid(alpha=0.3)

    stage_label = 'Stage-1 term-structures (non-full mode)'

    fig.suptitle(
        f"Optimal Parameter Term Structures | Underlying={tenor:.1f}Y | "
        f"Calib={calibration_method.upper()} | {stage_label} | H={H:.3f} | RMSE={calib['rmse_bps']:.2f} bps",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Available underlying tenors: {np.array2string(available, precision=2)}")


def _build_stage2_model(vol_csv: str, stage1_calib: dict, device: str):
    ois_func = load_discount_curve("data/estr_disc.csv")
    grid_T, F0_rates = bootstrap_forward_rates(ois_func)

    model_base = TorchRoughSABR_FMM(
        grid_T, F0_rates,
        stage1_calib['alpha_func'], stage1_calib['rho_func'], stage1_calib['nu_func'], stage1_calib['H'],
        beta_sabr=BETA_SABR, shift=SHIFT_SABR, correlation_mode='full', device=device
    )

    atm_matrix = _load_atm_matrix(vol_csv)
    corr_calibrator = CorrelationCalibrator(atm_matrix, model_base)
    corr_res = corr_calibrator.calibrate()

    model_final = TorchRoughSABR_FMM(
        grid_T, F0_rates,
        stage1_calib['alpha_func'], stage1_calib['rho_func'], stage1_calib['nu_func'], stage1_calib['H'],
        beta_sabr=BETA_SABR, shift=SHIFT_SABR, correlation_mode='full',
        omega_matrix=corr_res['omega_matrix'], device=device
    )
    return model_final


def _load_atm_matrix(csv_path: str):
    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = [str(c).strip().upper() for c in df.columns]

    c_exp = next(c for c in df.columns if 'EXPIRY' in c)
    c_ten = next(c for c in df.columns if 'UNDERLYING' in c or 'TENOR' in c)
    c_str = next(c for c in df.columns if 'STRIKE' in c)
    c_val = next(c for c in df.columns if 'VOL' in c or 'VALUE' in c)

    df = df[df[c_str].astype(str).str.strip().str.upper() == 'ATM'].copy()
    df[c_exp] = df[c_exp].apply(parse_tenor)
    df[c_ten] = df[c_ten].apply(parse_tenor)
    df[c_val] = pd.to_numeric(df[c_val], errors='coerce')
    df = df.dropna(subset=[c_exp, c_ten, c_val])

    return df.pivot_table(values=c_val, index=c_exp, columns=c_ten, aggfunc='first')


def _mapped_smm_term_structures(model, Sigma_matrix, expiries, tenor):
    """Compute Stage-2 mapped (effective) alpha/rho term-structures for a fixed underlying tenor."""
    dtype = model.dtype
    device = model.device

    tau = model.tau
    F0 = model.F0
    P0 = torch.cumprod(
        torch.cat([torch.tensor([1.0], device=device, dtype=dtype), 1.0 / (1.0 + tau * F0)]),
        dim=0,
    )

    eta_F0 = torch.pow(torch.abs(F0 + model.shift), model.beta_sabr)
    alpha_normal = model.alphas * eta_F0

    alpha_smm = []
    rho_smm = []
    for T_exp in expiries:
        start_idx = torch.argmin(torch.abs(model.T - float(T_exp))).item()
        end_idx = torch.argmin(torch.abs(model.T - float(T_exp + tenor))).item()

        if end_idx <= start_idx:
            alpha_smm.append(1e-12)
            rho_smm.append(0.0)
            continue

        P_I = P0[start_idx]
        P_J = P0[end_idx]
        A0 = torch.sum(tau[start_idx:end_idx] * P0[start_idx + 1 : end_idx + 1])
        S0 = (P_I - P_J) / A0

        pi_weights = torch.zeros(end_idx - start_idx, device=device, dtype=dtype)
        for j_local in range(end_idx - start_idx):
            j_global = start_idx + j_local
            sum_P = torch.sum(tau[j_global:end_idx] * P0[j_global + 1 : end_idx + 1])
            Pi_j = (tau[j_global] * P0[j_global + 1]) / (A0 * P0[j_global]) * (P_J + S0 * sum_P)
            pi_weights[j_local] = Pi_j * alpha_normal[j_global]

        Sigma_slice = Sigma_matrix[start_idx + 1 : end_idx + 1, start_idx + 1 : end_idx + 1]
        v_0 = torch.matmul(pi_weights.unsqueeze(0), torch.matmul(Sigma_slice, pi_weights.unsqueeze(1))).squeeze()
        a_smm = torch.sqrt(torch.clamp(v_0, min=1e-14))

        rho_slice = Sigma_matrix[start_idx + 1 : end_idx + 1, 0]
        rho_map = torch.sum(rho_slice * pi_weights) / a_smm

        alpha_smm.append(a_smm.item())
        rho_smm.append(torch.clamp(rho_map, -0.999, 0.999).item())

    return np.array(alpha_smm), np.array(rho_smm)


def main():
    parser = argparse.ArgumentParser(description='Plot market vs model volatility surface diagnostics.')
    parser.add_argument('--vol-csv', default='data/estr_vol_full_strikes.csv')
    parser.add_argument('--underlying-tenor', type=float, default=1.0)
    parser.add_argument('--underlying-index', type=int, default=None)
    parser.add_argument('--calibration-method', choices=['polynomial', 'ODE', 'MC'], default=CALI_MODE)
    parser.add_argument('--surface-method', choices=['polynomial', 'ode', 'mc'], default='mc')
    parser.add_argument('--output-surface', default='pics/market_vs_model_surface.png')
    parser.add_argument('--output-params', default='pics/optimal_parameter_term_structure.png')
    args = parser.parse_args()

    make_market_vs_model_surface_plot(
        vol_csv=args.vol_csv,
        underlying_tenor=args.underlying_tenor,
        underlying_index=args.underlying_index,
        calibration_method=args.calibration_method,
        surface_method=args.surface_method,
        output_path=args.output_surface,
    )

    # make_optimal_parameter_term_structure_plot(
    #     vol_csv=args.vol_csv,
    #     underlying_tenor=args.underlying_tenor,
    #     underlying_index=args.underlying_index,
    #     calibration_method=args.calibration_method,
    #     output_path=args.output_params,
    # )


if __name__ == '__main__':
    main()