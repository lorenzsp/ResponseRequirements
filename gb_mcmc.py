# ==================== Imports ====================
import os
import sys
import tempfile
from pathlib import Path
import corner
import healpy as hp
import eryn

# Eryn imports for transdimensional MCMC
from eryn.moves import MHMove, StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.ensemble import EnsembleSampler
from eryn.state import State


import h5py
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from segwo_utils import (InterpolatedOrbits, compute_strain2x, compute_covariance, compute_violation_ratios)
from scipy.optimize import differential_evolution

# Configure JAX for 64-bit precision (required for GW phase accuracy)
jax.config.update("jax_enable_x64", True)

# Add local paths for JaxGB
sys.path.insert(0, str(Path.cwd() / "tests"))
sys.path.insert(0, str(Path.cwd() / "src"))

from jaxgb.jaxgb import JaxGB
from boosted_jaxgb import JaxGBFull

print(f"JAX version: {jax.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Available devices: {jax.devices()}")

# ==================== Observation Configuration ====================

# Observation parameters
T_OBS_DAYS = 365.                      # Observation time in days
TMAX = T_OBS_DAYS * 24 * 3600          # Observation time in seconds
N_FREQ_BINS = 128                       # Number of frequency bins for heterodyned response
T0 = 0.0                                  # Start time (seconds)
DF = 1.0 / TMAX                         # Frequency resolution (Hz)

print(f"Observation time: {T_OBS_DAYS:.1f} days ({TMAX:.2e} seconds)")
print(f"Frequency resolution: {DF:.2e} Hz")
# ==================== LISA Orbits Setup ====================
with h5py.File("data/processed_trajectories.h5", "r") as ds:
    t_orb_dataset   = ds["t_interp"][()]
    x_orb_dataset   = ds["spacecraft_positions"][()]
    v_orb_dataset   = ds["spacecraft_velocities"][()]
    ltts_dataset    = ds['owlt_12_23_31_13_32_21'][()]
    
    mask_time = t_orb_dataset < TMAX*1.1
    t_orb_dataset = t_orb_dataset[mask_time]
    x_orb_dataset = x_orb_dataset[:, mask_time, :]
    v_orb_dataset = v_orb_dataset[:, mask_time, :]
    ltts_dataset = ltts_dataset[:, mask_time]


t_orb        = t_orb_dataset
x_orb        = np.median(x_orb_dataset, axis=0)
v_orb        = np.median(v_orb_dataset, axis=0)
ltts_median  = np.median(ltts_dataset,  axis=0)
realizations = x_orb_dataset.shape[0]
N            = realizations
print(f"Number of realizations: {realizations}")
orbits = InterpolatedOrbits(t_orb, x_orb,
                            spacecraft_velocities=v_orb,
                            ltts=ltts_median,
                            interp_order=3)

orbits_v0 = InterpolatedOrbits(t_orb, x_orb,
                            spacecraft_velocities=v_orb*0.0,
                            ltts=ltts_median,
                            interp_order=3)

i = 0
perturbed_orbits = InterpolatedOrbits(
                t_orb_dataset,
                x_orb_dataset[i],
                spacecraft_velocities=v_orb_dataset[i],
                ltts=ltts_dataset[i],
                interp_order=3,
            )

# non relativistic model
test_nominal_orbit = JaxGB(orbits=orbits_v0, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)
nonrel_nominal_orbit = JaxGBFull(orbits=orbits_v0, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)

# relativistic model
rel_nominal_orbit = JaxGBFull(orbits=orbits, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)
rel_perturbed_orbit = JaxGBFull(orbits=perturbed_orbits, t_obs=TMAX, t0=T0, n=N_FREQ_BINS)

# test
# b'SDSSJ1337' SNR 12.407709826733793 Frequency 0.0003365330899526027 Frequency derivative 1.856006058763817e-20

output_file = f"data/gb_mismatch_results_{T_OBS_DAYS:.1f}days.h5"
mismatch = {}
with h5py.File(output_file, "r") as f:
    f0_vec = f["f0_vec"][()]
    for key in f.keys():
        if key.startswith("mismatch_"):
            mismatch[key] = f[key][()]

nside        = 6
npix         = hp.nside2npix(nside)
thetas, phis = hp.pix2ang(nside, np.arange(npix))
betas, lambs = np.pi / 2 - thetas, phis
f0_vec = np.logspace(-4,0.0, num=10)

temp = mismatch["mismatch_nonrel_vs_rel_with_nominal"]
# Lowest frequency, worst sky location at that frequency
i_f = 0
sky_ind = np.argmax(temp, axis=1)[i_f]
ind = (i_f, sky_ind)
print(f"Injection: f0_vec[{i_f}] = {f0_vec[i_f]:.4e} Hz, sky_ind = {sky_ind}, mismatch = {temp[i_f, sky_ind]:.6f}")

injection_source_params = np.array([
            f0_vec[ind[0]],                          # f0 (Hz)
            0.0,                           # fdot (Hz/s) - no evolution
            1.5267589596714682e-18,                         # amplitude (strain)
            betas[ind[1]],                           # ecliptic latitude (rad)
            lambs[ind[1]],                           # ecliptic longitude (rad)
            np.pi/3,                           # polarization (rad)
            np.pi/3,                       # inclination (rad)
            np.pi/3,                           # initial phase (rad)
])

print("\nInjection source parameters:")
for i, param in enumerate(injection_source_params):
    print(f"  {i}: {param:.4e}")

A_nom, E_nom, T_nom = rel_nominal_orbit.get_tdi(jnp.array(injection_source_params), tdi_generation=2.0, tdi_combination="AET")

kmin = int(np.array(rel_nominal_orbit.get_kmin(injection_source_params[0],injection_source_params[1])))
kmax = kmin + rel_nominal_orbit.n
freqs = DF * (np.arange(N_FREQ_BINS) + kmin)
cov_AET = compute_covariance(freqs, ltts_median).mean(axis=0)
inv_cov_AET = np.linalg.inv(cov_AET)
d_inj = np.stack([A_nom, E_nom, T_nom], axis=0)

def inner_product(d1, d2, inv_cov):
    # d1, d2: (3, N_freq) — expand to (1, 3, N_freq) for the batched method
    return float(rel_nominal_orbit.inner_product(d1[None], d2[None], inv_cov)[0])

# snr_fixed = 50.0
# SNR_injection = inner_product(d_inj, d_inj, inv_cov_AET)**0.5
# d_inj = snr_fixed * d_inj / SNR_injection
SNR_injection = inner_product(d_inj, d_inj, inv_cov_AET)**0.5
print("Injected SNR", SNR_injection)
# breakpoint()

# injection
NTEMPS, NWALKERS, NLEAVES_MAX = 8, 64, 1
d_inj_tot = jnp.repeat(jnp.asarray(d_inj)[None,:,:],NWALKERS*NTEMPS,axis=0)

test_params = np.repeat(jnp.asarray(injection_source_params)[None,:],NWALKERS*NTEMPS,axis=0)
dminush = rel_nominal_orbit.get_data_minus_template(jnp.asarray(test_params), d_inj_tot, kmin, kmax, tdi_generation=2.0, tdi_combination="AET")

temp_params = np.zeros((NWALKERS*NTEMPS,injection_source_params.shape[0]))
def likelihood(params, template_generator, inv_cov, temp_params):
    temp_params[:len(params)] = params
    return np.asarray(template_generator.log_likelihood(
        jnp.asarray(temp_params), d_inj_tot, inv_cov, kmin, kmax,
        max_batch_size=NTEMPS * NWALKERS,
        tdi_generation=2.0, tdi_combination="AET",
    ))[:len(params)]

def setup_priors(f0_center, f0_width, A_center=1e-18):
    """Set up prior distributions for galactic binary parameters."""
    
    # Amplitude bounds (based on SNR considerations)
    min_A = A_center / 2  # Very faint
    max_A = A_center * 2  # Very bright
    
    # Frequency derivative bounds
    fdot_max = 1e-18  # Hz/s
    fdot_min = 0.0    # Hz/s
    
    # Frequency bounds
    f_min = f0_center - f0_width
    f_max = f0_center + f0_width
    
    # Prior on sampling parameters (log amplitude for efficiency)
    priors = {
        "gb": ProbDistContainer({
            0: uniform_dist(f_min, f_max),                       # f0
            1: uniform_dist(fdot_min, fdot_max),                 # fdot
            2: uniform_dist(min_A, max_A),  # log10(amplitude)
            3: uniform_dist(-np.pi/2, np.pi/2),                  # beta
            4: uniform_dist(0.0, 2 * np.pi),                     # lambda
            5: uniform_dist(0.0, 2 * np.pi),                     # psi
            6: uniform_dist(0.0, np.pi),                         # iota
            7: uniform_dist(0.0, 2 * np.pi),                     # phi0
        })
    }
    
    bounds = [(f_min, f_max),
              (fdot_min, fdot_max),
              (min_A, max_A),
              (-np.pi/2, np.pi/2),
              (0.0, 2 * np.pi),
              (0.0, 2 * np.pi),
              (0.0, np.pi),
              (0.0, 2 * np.pi),
              ]
    # Periodic parameters
    periodic = {
        "gb": {
            4: 2 * np.pi,  # lambda
            5: 2 * np.pi,  # psi  
            7: 2 * np.pi,  # phi0
        }
    }
    
    ndims = 8  # Number of parameters per source
    
    print("\nPrior bounds:")
    print(f"  log10(A): [{np.log10(min_A):.1f}, {np.log10(max_A):.1f}]")
    print(f"  f0: [{f_min:.6e}, {f_max:.6e}] Hz")
    print(f"  fdot: [{fdot_min:.2e}, {fdot_max:.2e}] Hz/s")
    
    return priors, periodic, ndims, bounds

priors, periodic, ndims, bounds = setup_priors(injection_source_params[0], 1e-6, A_center=injection_source_params[2])

################################# Mismatch check at true params ##########################################################
print("\n" + "=" * 60)
print("Mismatch check: nonrel template vs rel injection at true params")
A_check, E_check, T_check = nonrel_nominal_orbit.get_tdi(jnp.array(injection_source_params), tdi_generation=2.0, tdi_combination="AET")
d_check = np.stack([A_check, E_check, T_check], axis=0)
print("Mismatch (nonrel template, true params):", float(nonrel_nominal_orbit.mismatch(d_check[None], d_inj[None], inv_cov_AET)[0]))
ll_inj  = likelihood(jnp.atleast_2d(injection_source_params), rel_nominal_orbit,   inv_cov_AET, temp_params)
ll_nonrel = likelihood(jnp.atleast_2d(injection_source_params), nonrel_nominal_orbit, inv_cov_AET, temp_params)
print(f"Loglike (rel template, true params):   {ll_inj[0]:.4f}")
print(f"Loglike (nonrel template, true params): {ll_nonrel[0]:.4f}")

################################# MCMC ##########################################################
# injection: relativistic model

# Template: non-relativistic model; 
template_generator = nonrel_nominal_orbit
backend_name = f"data/template_nonrel_{T_OBS_DAYS}days_SNR{SNR_injection:.2f}_f{injection_source_params[0]:.2e}.h5"

# Template: relativistic model; 
template_generator = rel_nominal_orbit
backend_name = f"data/template_rel_{T_OBS_DAYS}days_SNR{SNR_injection:.2f}_f{injection_source_params[0]:.2e}.h5"

# Template: perturbed relativistic model; 
template_generator = rel_perturbed_orbit
backend_name = f"data/template_rel_perturbed_{T_OBS_DAYS}days_SNR{SNR_injection:.2f}_f{injection_source_params[0]:.2e}.h5"

list_templates = [
    nonrel_nominal_orbit,
    rel_nominal_orbit,
    rel_perturbed_orbit
]
list_backend_names = [
    f"data/template_nonrel_{T_OBS_DAYS}days_SNR{SNR_injection:.2f}_f{injection_source_params[0]:.2e}.h5",
    f"data/template_rel_{T_OBS_DAYS}days_SNR{SNR_injection:.2f}_f{injection_source_params[0]:.2e}.h5",
    f"data/template_rel_perturbed_{T_OBS_DAYS}days_SNR{SNR_injection:.2f}_f{injection_source_params[0]:.2e}.h5",
]

for template_generator, backend_name in zip(list_templates, list_backend_names):
    print("\n" + "=" * 60)
    print(f"Running MCMC with backend {backend_name}")
    print("=" * 60)
    # Create sampler
    sampler = EnsembleSampler(
        NWALKERS,
        ndims,
        likelihood,
        priors,
        branch_names=["gb"],
        args=(template_generator, inv_cov_AET, temp_params),
        tempering_kwargs=dict(ntemps=NTEMPS),
        periodic=periodic,
        vectorize=True,
        backend=backend_name,
    )

    # Initialize state
    print("\nInitializing sampler state...")

    # Draw initial positions from prior
    start_points = {
        "gb": priors["gb"].rvs(size=(NTEMPS, NWALKERS, NLEAVES_MAX))
    }

    start_inds = {
        "gb": np.zeros((NTEMPS, NWALKERS, NLEAVES_MAX), dtype=bool)
    }
    start_inds["gb"][:, :, 0] = True  # One source active per walker
    # set the coldest chain to start at the injection parameters, others are randomly perturbed around it
    for par_i in range(len(injection_source_params)):
        if par_i != 1:  # Amplitude: sample in log space
            start_points["gb"][0, :, 0, par_i] = np.random.normal(loc=injection_source_params[par_i], scale=1e-6*injection_source_params[par_i], size=(NWALKERS))  # Perturb parameters for testing

    # set the other temperatures to be consistent in frequency and amplitude, but random in the other parameters to allow multimodality exploration
    for par_i in [0,2]:  # f0 and amplitude: sample in log space
        start_points["gb"][1:, :, 0, par_i] = np.random.normal(loc=injection_source_params[par_i], scale=1e-6*injection_source_params[par_i], size=(NTEMPS-1, NWALKERS))  # Perturb parameters for testing

    # set one walker in the coldest chain to start at the injection parameters
    start_points["gb"][0, 0, 0] = injection_source_params.copy()  # Start at true params for testing

    start_state = State(start_points)

    # Compute initial prior and likelihood
    lp = sampler.compute_log_prior(start_state.branches_coords)
    start_state.log_prior = lp

    ll = sampler.compute_log_like(start_state.branches_coords, logp=lp, inds=start_inds)
    start_state.log_like = ll[0]

    print(f"Initial log-likelihood range: [{ll[0].min():.2f}, {ll[0].max():.2f}]")

    # Time the likelihood
    import time
    tic = time.time()
    _ = sampler.compute_log_like(start_state.branches_coords, logp=lp)
    toc = time.time()
    print(f"Likelihood evaluation time: {(toc - tic) * 1000:.2f} ms")

    # # Run MCMC
    # print("\n" + "=" * 60)
    # print("Running MCMC  [injection: rel_nominal | template: nonrel_nominal]")
    # print("=" * 60)

    n_iterations = 5000
    print(f"Running {n_iterations} iterations...")

    sampler.run_mcmc(start_state, n_iterations, progress=True)

    print("\n" + "=" * 60)
    print("MCMC Complete!")
    print("=" * 60)

