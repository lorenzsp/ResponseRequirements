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
T_OBS_DAYS = 15.0 * 1.0                      # Observation time in days
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

output_file = "gb_mismatch_maxphi_results_15.0days.h5"
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
# temp = mismatch["mismatch_perturbed_vs_nominal_with_rel"]
ind = np.unravel_index(np.argmax(temp, axis=None), temp.shape)

injection_source_params = np.array([
            f0_vec[ind[0]],                          # f0 (Hz)
            0.0,                           # fdot (Hz/s) - no evolution
            0.31e-18,                         # amplitude (strain)
            betas[ind[1]],                           # ecliptic latitude (rad)
            lambs[ind[1]],                           # ecliptic longitude (rad)
            0.0,                           # polarization (rad)
            0.0,                           # inclination (rad)
            0.0,                           # initial phase (rad)
])

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

# injection
NTEMPS, NWALKERS, NLEAVES_MAX = 8, 32, 1
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

def setup_priors(f0_center, f0_width):
    """Set up prior distributions for galactic binary parameters."""
    
    # Amplitude bounds (based on SNR considerations)
    min_A = 1e-24  # Very faint
    max_A = 1e-15  # Very bright
    
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

priors, periodic, ndims, bounds = setup_priors(injection_source_params[0], 1e-6)

################################# Mismatch ##########################################################
mismatch_check = True
if mismatch_check:
    runs = zip([rel_nominal_orbit, nonrel_nominal_orbit, rel_perturbed_orbit], ['rel', 'nonrel', 'perturbed'])

    for template_generator, name in runs:
        print("===============================")
        print(f"\nRunning analysis for {name} template...")
        # generate template and check mismatch
        A, E, T = template_generator.get_tdi(jnp.array(injection_source_params), tdi_generation=2.0, tdi_combination="AET")
        d_template = np.stack([A, E, T], axis=0)
        print("Mismatch at true parameters:", float(template_generator.mismatch(d_template[None], d_inj[None], inv_cov_AET)[0]))

        # check loglike on injection
        ll_true = likelihood(jnp.atleast_2d(injection_source_params), rel_nominal_orbit, inv_cov_AET, temp_params)
        ll_pert = likelihood(jnp.atleast_2d(injection_source_params), template_generator, inv_cov_AET, temp_params)
        print("Loglike true:", ll_true, "loglike pert: ", ll_pert)

        # minimize with differential evolution
        def neg_ll(x, *args, **kwargs):
            return - likelihood(x.T, template_generator, inv_cov_AET, temp_params)

        init = priors["gb"].rvs(size=(NTEMPS*NWALKERS*NLEAVES_MAX))
        init[0] = injection_source_params
        print(priors["gb"].logpdf(init)[0])
        out_de = differential_evolution(neg_ll, init = init, bounds=bounds, disp=False, vectorized=True, maxiter=500, polish=True)
        best_fit = out_de.x
        A, E, T = template_generator.get_tdi(jnp.array(best_fit), tdi_generation=2.0, tdi_combination="AET")
        d_template = np.stack([A, E, T], axis=0)

        print("Mismatch at best fit:", float(template_generator.mismatch(d_template[None], d_inj[None], inv_cov_AET)[0]))
        
        denom = injection_source_params.copy()
        denom[denom == 0] = 1.0
        print("Run name:", name, " Relative difference from true:",np.abs(best_fit-injection_source_params)/denom)

################################# MCMC ##########################################################
template_generator = nonrel_nominal_orbit
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
    # update_fn=update_fn,
    # update_iterations=50,
    backend=f"backend_run.h5",
)

# Initialize state
print("\nInitializing sampler state...")

# Draw initial positions from prior
start_points = {
    "gb": priors["gb"].rvs(size=(NTEMPS, NWALKERS, NLEAVES_MAX))
}

# mean = injection_source_params.copy()
# mean[1] = 1e-20
# sigma = 1e-20 * mean
# start_points["gb"] = out_de.population.reshape((NTEMPS, NWALKERS, NLEAVES_MAX, ndims)) # np.random.multivariate_normal(injection_source_params, np.eye(ndims)*sigma, size=(NTEMPS, NWALKERS, NLEAVES_MAX))
start_points["gb"][:,:,:,1] = priors["gb"].rvs(size=(NTEMPS, NWALKERS, NLEAVES_MAX))[:,:,:,1]
# start_points["gb"][:,:NWALKERS//2] = priors["gb"].rvs(size=(NTEMPS, NWALKERS//2, NLEAVES_MAX))
# Initialize with one source active per walker (start conservative)
start_inds = {
    "gb": np.zeros((NTEMPS, NWALKERS, NLEAVES_MAX), dtype=bool)
}
start_inds["gb"][:, :, 0] = True  # One source active

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

# Run MCMC
print("\n" + "=" * 60)
print("Running MCMC")
print("=" * 60)

n_iterations = 5000
print(f"Running {n_iterations} iterations...")

sampler.run_mcmc(start_state, n_iterations, progress=True)

print("\n" + "=" * 60)
print("MCMC Complete!")
print("=" * 60)

