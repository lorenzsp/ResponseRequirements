# ==================== Imports ====================
import sys
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import differential_evolution

from segwo_utils import InterpolatedOrbits, compute_covariance

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path.cwd() / "tests"))
sys.path.insert(0, str(Path.cwd() / "src"))

from boosted_jaxgb import JaxGBFull

print(f"JAX version: {jax.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Available devices: {jax.devices()}")

# ==================== Observation Configuration ====================

T_OBS_DAYS  = 365.0
TMAX        = T_OBS_DAYS * 24 * 3600
N_FREQ_BINS = 128
T0          = 0.0
DF          = 1.0 / TMAX

print(f"Observation time: {T_OBS_DAYS:.1f} days ({TMAX:.2e} seconds)")
print(f"Frequency resolution: {DF:.2e} Hz")

# ==================== LISA Orbits Setup ====================

with h5py.File("data/processed_trajectories.h5", "r") as ds:
    t_orb_dataset = ds["t_interp"][()]
    x_orb_dataset = ds["spacecraft_positions"][()]
    v_orb_dataset = ds["spacecraft_velocities"][()]
    ltts_dataset  = ds["owlt_12_23_31_13_32_21"][()]

    mask_time     = t_orb_dataset < TMAX * 1.1
    t_orb_dataset = t_orb_dataset[mask_time]
    x_orb_dataset = x_orb_dataset[:, mask_time, :]
    v_orb_dataset = v_orb_dataset[:, mask_time, :]
    ltts_dataset  = ltts_dataset[:, mask_time]

t_orb       = t_orb_dataset
x_orb       = np.median(x_orb_dataset, axis=0)
v_orb       = np.median(v_orb_dataset, axis=0)
ltts_median = np.median(ltts_dataset, axis=0)

print(f"Number of realizations: {x_orb_dataset.shape[0]}")

orbits = InterpolatedOrbits(t_orb, x_orb,
                            spacecraft_velocities=v_orb,
                            ltts=ltts_median,
                            interp_order=3)

orbits_v0 = InterpolatedOrbits(t_orb, x_orb,
                               spacecraft_velocities=v_orb * 0.0,
                               ltts=ltts_median,
                               interp_order=3)

# Injection model: relativistic; template model: non-relativistic
rel_model   = JaxGBFull(orbits=orbits,    t_obs=TMAX, t0=T0, n=N_FREQ_BINS)
nonrel_model = JaxGBFull(orbits=orbits_v0, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)

# ==================== Sky grid ====================

nside        = 6
npix         = hp.nside2npix(nside)
thetas, phis = hp.pix2ang(nside, np.arange(npix))
betas, lambs = np.pi / 2 - thetas, phis

# ==================== Load mismatch map ====================

MISMATCH_FILE = "gb_mismatch_maxphi_results_15.0days.h5"
MISMATCH_KEY  = "mismatch_nonrel_vs_rel_with_nominal"

with h5py.File(MISMATCH_FILE, "r") as f:
    f0_vec = f["f0_vec"][()]
    temp   = f[MISMATCH_KEY][()]          # shape: (n_freq, n_pix)

# For each frequency, pick the sky pixel with the maximum mismatch
sky_inds = np.argmax(temp, axis=1)        # shape: (n_freq,)

print(f"\nFrequencies: {f0_vec}")
print(f"Worst-case sky pixels: {sky_inds}")
print(f"Max mismatch per frequency: {temp[np.arange(len(f0_vec)), sky_inds]}")

# ==================== DE / likelihood helpers ====================

NTEMPS, NWALKERS, NLEAVES_MAX = 32, 32, 1
BATCH = NTEMPS * NWALKERS

_temp_params_buf = np.zeros((BATCH, 8))


def _likelihood(params, template_generator, inv_cov, d_inj_tot, kmin, kmax):
    """
    Vectorised log-likelihood.

    params : ndarray, shape (n_samples, 8)  or (8,) for a single set
    Returns ndarray of shape (n_samples,)
    """
    n = len(params)
    _temp_params_buf[:n] = params
    return np.asarray(
        template_generator.log_likelihood(
            jnp.asarray(_temp_params_buf), d_inj_tot, inv_cov,
            kmin, kmax,
            max_batch_size=BATCH,
            tdi_generation=2.0, tdi_combination="AET",
        )
    )[:n]


PARAM_NAMES = ["f0", "fdot", "A", "beta", "lambda", "psi", "iota", "phi0"]

# ==================== Loop over frequencies ====================

n_freq = len(f0_vec)

results = dict(
    f0_vec             = f0_vec,
    sky_inds           = sky_inds,
    mismatch_at_true   = np.full(n_freq, np.nan),
    mismatch_at_bestfit= np.full(n_freq, np.nan),
    snr                = np.full(n_freq, np.nan),
    true_params        = np.full((n_freq, 8), np.nan),
    best_fit_params    = np.full((n_freq, 8), np.nan),
    bias               = np.full((n_freq, 8), np.nan),   # best_fit - true (absolute)
    relative_bias      = np.full((n_freq, 8), np.nan),   # (best_fit - true) / |true|, 0-safe
    de_success         = np.zeros(n_freq, dtype=bool),
)

for i_f, f0 in enumerate(f0_vec):
    sky_ind  = sky_inds[i_f]
    beta_inj = betas[sky_ind]
    lamb_inj = lambs[sky_ind]

    print(f"\n{'='*65}")
    print(f"[{i_f+1}/{n_freq}]  f0 = {f0:.4e} Hz  |  sky_ind = {sky_ind}"
          f"  |  beta = {beta_inj:.3f} rad  |  lambda = {lamb_inj:.3f} rad")
    print(f"  Mismatch from map: {temp[i_f, sky_ind]:.6f}")

    true_params = np.array([
        f0,        # f0 (Hz)
        0.0,       # fdot (Hz/s)
        0.31e-18,  # amplitude (strain)
        beta_inj,  # ecliptic latitude (rad)
        lamb_inj,  # ecliptic longitude (rad)
        0.0,       # polarization (rad)
        0.0,       # inclination (rad)
        0.0,       # initial phase (rad)
    ])
    results["true_params"][i_f] = true_params

    # ---------- frequency-bin setup ----------
    kmin  = int(np.array(rel_model.get_kmin(f0, 0.0)))
    kmax  = kmin + N_FREQ_BINS
    freqs = DF * (np.arange(N_FREQ_BINS) + kmin)

    cov_AET     = compute_covariance(freqs, ltts_median).mean(axis=0)
    inv_cov_AET = np.linalg.inv(cov_AET)

    # ---------- inject with relativistic model ----------
    A_inj, E_inj, T_inj = rel_model.get_tdi(
        jnp.array(true_params), tdi_generation=2.0, tdi_combination="AET"
    )
    d_inj     = np.stack([A_inj, E_inj, T_inj], axis=0)          # (3, N_FREQ_BINS)
    d_inj_tot = jnp.repeat(jnp.asarray(d_inj)[None, :, :], BATCH, axis=0)

    snr = float(rel_model.inner_product(d_inj[None], d_inj[None], inv_cov_AET)[0]) ** 0.5
    print(f"  Injected SNR: {snr:.2f}")
    results["snr"][i_f] = snr

    # ---------- mismatch at true parameters (nonrel template vs rel injection) ----------
    A_t, E_t, T_t = nonrel_model.get_tdi(
        jnp.array(true_params), tdi_generation=2.0, tdi_combination="AET"
    )
    d_true_tmpl = np.stack([A_t, E_t, T_t], axis=0)
    mm_true = float(nonrel_model.mismatch(d_true_tmpl[None], d_inj[None], inv_cov_AET)[0])
    print(f"  Mismatch at true params:   {mm_true:.6f}")
    results["mismatch_at_true"][i_f] = mm_true

    # ---------- bounds for DE ----------
    f0_width = min(1e-6, f0 * 0.01)
    bounds = [
        (max(f0 - f0_width, 1e-5), f0 + f0_width),  # f0
        (0.0,      1e-18),                             # fdot
        (1e-24,    1e-15),                             # A
        (-np.pi/2, np.pi/2),                           # beta
        (0.0,      2 * np.pi),                         # lambda
        (0.0,      2 * np.pi),                         # psi
        (0.0,      np.pi),                             # iota
        (0.0,      2 * np.pi),                         # phi0
    ]

    # Initial population: uniform draw within bounds, always include truth
    rng      = np.random.default_rng(seed=i_f)
    lows     = np.array([lo for lo, hi in bounds])
    highs    = np.array([hi for lo, hi in bounds])
    init_pop = rng.uniform(lows, highs, size=(BATCH, 8))
    init_pop[0] = true_params

    # ---------- maximize likelihood via DE (nonrel template) ----------
    def neg_ll(x, _d=d_inj_tot, _inv=inv_cov_AET, _km=kmin, _kM=kmax):
        return -_likelihood(x.T, nonrel_model, _inv, _d, _km, _kM)

    out_de    = differential_evolution(
        neg_ll, bounds=bounds, init=init_pop,
        vectorized=True, maxiter=1000, polish=True, disp=True,
    )
    best_fit = out_de.x
    results["de_success"][i_f] = out_de.success

    # ---------- mismatch at best fit ----------
    A_bf, E_bf, T_bf = nonrel_model.get_tdi(
        jnp.array(best_fit), tdi_generation=2.0, tdi_combination="AET"
    )
    d_bf_tmpl  = np.stack([A_bf, E_bf, T_bf], axis=0)
    mm_bestfit = float(nonrel_model.mismatch(d_bf_tmpl[None], d_inj[None], inv_cov_AET)[0])
    print(f"  Mismatch at best fit:      {mm_bestfit:.6f}")
    results["mismatch_at_bestfit"][i_f] = mm_bestfit

    # ---------- bias ----------
    bias         = best_fit - true_params
    denom        = np.where(np.abs(true_params) > 0, np.abs(true_params), 1.0)
    relative_bias = bias / denom

    results["best_fit_params"][i_f] = best_fit
    results["bias"][i_f]            = bias
    results["relative_bias"][i_f]   = relative_bias

    print(f"  DE success: {out_de.success}  |  {out_de.message}")
    print(f"  Absolute bias:  { dict(zip(PARAM_NAMES, bias)) }")
    print(f"  Relative bias:  { dict(zip(PARAM_NAMES, relative_bias)) }")

# ==================== Save results ====================

out_file = f"data/gb_bias_results_{T_OBS_DAYS:.1f}days.h5"
with h5py.File(out_file, "w") as f:
    for key, val in results.items():
        f.create_dataset(key, data=val)
    f.attrs["T_OBS_DAYS"]    = T_OBS_DAYS
    f.attrs["N_FREQ_BINS"]   = N_FREQ_BINS
    f.attrs["nside"]         = nside
    f.attrs["mismatch_key"]  = MISMATCH_KEY
    f.attrs["injection_model"] = "rel_nominal"
    f.attrs["template_model"]  = "nonrel_nominal"
    f.attrs["param_names"]     = PARAM_NAMES

print(f"\nResults saved to {out_file}")

# ==================== Summary ====================

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
header = f"{'f0 (Hz)':>12}  {'sky_ind':>7}  {'SNR':>6}  {'mm_true':>10}  {'mm_bestfit':>10}  {'DE ok':>5}"
print(header)
print("-" * len(header))
for i_f in range(n_freq):
    print(
        f"{f0_vec[i_f]:>12.4e}  {sky_inds[i_f]:>7d}  "
        f"{results['snr'][i_f]:>6.1f}  "
        f"{results['mismatch_at_true'][i_f]:>10.6f}  "
        f"{results['mismatch_at_bestfit'][i_f]:>10.6f}  "
        f"{'yes' if results['de_success'][i_f] else 'no':>5}"
    )
