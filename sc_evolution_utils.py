"""
two_body_mc_analysis.py

Monte Carlo verification of STM (State Transition Matrix) propagation
vs. full nonlinear integration for a spacecraft orbiting the Sun.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time as pytime

# Constants
MU_SUN = 1.32712440018e20  # m^3/s^2
AU = 1.495978707e11        # m
DAY = 86400.0              # s


# =========================
# Dynamics
# =========================
def acceleration(r):
    rn = np.linalg.norm(r)
    return -MU_SUN * r / rn**3

def two_body_ode(t, y):
    r = y[0:3]
    v = y[3:6]
    a = acceleration(r)
    dydt = np.zeros(6)
    dydt[0:3] = v
    dydt[3:6] = a
    return dydt

def jacobian_r(r):
    rn = np.linalg.norm(r)
    I3 = np.eye(3)
    return MU_SUN * (3.0 * np.outer(r, r) / rn**5 - I3 / rn**3)

def two_body_with_variational(t, y):
    r = y[0:3]
    v = y[3:6]
    a = acceleration(r)
    A11 = np.zeros((3,3))
    A12 = np.eye(3)
    A21 = jacobian_r(r)
    A22 = np.zeros((3,3))
    A = np.block([[A11, A12],
                  [A21, A22]])
    phi = y[6:].reshape((6,6))
    dphi = A.dot(phi)
    dydt = np.zeros_like(y)
    dydt[0:3] = v
    dydt[3:6] = a
    dydt[6:] = dphi.flatten()
    return dydt


# =========================
# Simulation Functions
# =========================
def integrate_nominal(r0, v0, t_span, t_eval):
    sol_nom = solve_ivp(two_body_ode, t_span, np.concatenate([r0, v0]),
                        t_eval=t_eval, rtol=1e-10, atol=1e-13)
    return sol_nom.y[0:3, :].T, sol_nom.y[3:6, :].T  # shape (n_out, 3)

def integrate_with_stm(r0, v0, t_span, t_eval):
    phi0 = np.eye(6).flatten()
    y0 = np.concatenate([r0, v0, phi0])
    sol_var = solve_ivp(two_body_with_variational, t_span, y0,
                        t_eval=t_eval, rtol=1e-10, atol=1e-13)
    phis_mat = sol_var.y[6:, :].T.reshape((-1, 6, 6))
    return phis_mat

def monte_carlo_stm(mc_samples, phis_mat):
    n_out = phis_mat.shape[0]
    Nmc = mc_samples.shape[0]
    mc_pos_errors = np.zeros((n_out, Nmc))
    for i in range(n_out):
        Phi = phis_mat[i]
        propagated = Phi @ mc_samples.T
        pos_errs = np.linalg.norm(propagated[0:3, :], axis=0)
        mc_pos_errors[i, :] = pos_errs
    return mc_pos_errors

def monte_carlo_nonlinear(r0, v0, mc_samples, rs_nom, t_span, t_eval):
    Nmc = mc_samples.shape[0]
    n_out = t_eval.size
    mc_pos_errors = np.zeros((n_out, Nmc))
    mc_rs_samples = np.zeros((Nmc, n_out, 3))
    mc_vs_samples = np.zeros((Nmc, n_out, 3))
    t_start = pytime.time()
    for j in range(Nmc):
        dx0 = mc_samples[j]
        y0_sample = np.concatenate([r0 + dx0[0:3], v0 + dx0[3:6]])
        sol_sample = solve_ivp(two_body_ode, t_span, y0_sample,
                               t_eval=t_eval, rtol=1e-10, atol=1e-13)
        rs_sample = sol_sample.y[0:3, :].T
        mc_rs_samples[j] = rs_sample
        mc_vs_samples[j] = sol_sample.y[3:6, :].T
        pos_diff = rs_sample - rs_nom
        mc_pos_errors[:, j] = np.linalg.norm(pos_diff, axis=1)
    t_end = pytime.time()
    print(f"Nonlinear MC integrations (N={Nmc}) done in {t_end - t_start:.2f} s")
    return mc_pos_errors, mc_rs_samples, mc_vs_samples


# =========================
# Plotting Functions
# =========================
def plot_3d_trajectories(rs_nom, mc_rs_samples):
    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection='3d')
    ax.plot3D(rs_nom[:,0]/AU, rs_nom[:,1]/AU, rs_nom[:,2]/AU,
              label='Nominal trajectory', linewidth=2)
    for traj in mc_rs_samples:
        ax.plot3D(traj[:,0]/AU, traj[:,1]/AU, traj[:,2]/AU,
                  color='gray', alpha=0.2, linewidth=0.5)
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    ax.set_title('Nominal Orbit & Monte Carlo Realizations (3D)')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_envelopes(time_days, mc_pos_errors_stm, mc_pos_errors_nl):
    pctiles = [2.5, 50, 97.5]
    stm_p = np.percentile(mc_pos_errors_stm, pctiles, axis=1)
    nl_p = np.percentile(mc_pos_errors_nl, pctiles, axis=1)

    plt.figure(figsize=(9,5))
    plt.fill_between(time_days, stm_p[0,:]/1e3, stm_p[2,:]/1e3,
                     alpha=0.25, label='STM 5-95%')
    plt.plot(time_days, stm_p[1,:]/1e3, '-', label='STM median')
    plt.fill_between(time_days, nl_p[0,:]/1e3, nl_p[2,:]/1e3,
                     alpha=0.25, label='Nonlinear 5-95%')
    plt.plot(time_days, nl_p[1,:]/1e3, '--', label='Nonlinear median')
    plt.xlabel('Time [days]')
    plt.ylabel('Position error norm [km]')
    plt.title('STM vs Nonlinear Monte Carlo (median & 5–95% envelope)')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return stm_p, nl_p

def plot_relative_difference(time_days, stm_p, nl_p):
    median_rel_diff = (stm_p[1,:] - nl_p[1,:]) / np.maximum(nl_p[1,:], 1e-12)
    plt.figure(figsize=(9,4))
    plt.plot(time_days, median_rel_diff)
    plt.xlabel('Time [days]')
    plt.ylabel('Relative difference')
    plt.title('Relative difference between STM median and Nonlinear median')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_component_ratios(times, r_nominal, v_nominal, r_all, v_all):
    """
    Plots the ratio of each component (x,y,z) of position and velocity
    for each realization compared to the nominal trajectory.
    
    Parameters
    ----------
    times : array
        Time array [s].
    r_nominal : array, shape (N, 3)
        Nominal position vector over time.
    v_nominal : array, shape (N, 3)
        Nominal velocity vector over time.
    r_all : array, shape (n_realizations, N, 3)
        All realization position vectors over time.
    v_all : array, shape (n_realizations, N, 3)
        All realization velocity vectors over time.
    """
    
    # --- POSITION RATIOS ---
    fig_r, axes_r = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    comp_labels = ['x', 'y', 'z']
    
    for i in range(3):
        for r in r_all:
            ratio = r[:, i] / r_nominal[:, i]
            axes_r[i].plot(times / 86400, ratio, alpha=0.5)  # time in days
        axes_r[i].axhline(1.0, color='k', linestyle='--', lw=1)
        axes_r[i].set_ylabel(f"r_{comp_labels[i]} ratio")
    
    axes_r[-1].set_xlabel("Time [days]")
    fig_r.suptitle("Position Component Ratios (realization / nominal)", fontsize=14)
    fig_r.tight_layout()
    
    # --- VELOCITY RATIOS ---
    fig_v, axes_v = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    for i in range(3):
        for v in v_all:
            ratio = v[:, i] / v_nominal[:, i]
            axes_v[i].plot(times / 86400, ratio, alpha=0.5)  # time in days
        axes_v[i].axhline(1.0, color='k', linestyle='--', lw=1)
        axes_v[i].set_ylabel(f"v_{comp_labels[i]} ratio")
    
    axes_v[-1].set_xlabel("Time [days]")
    fig_v.suptitle("Velocity Component Ratios (realization / nominal)", fontsize=14)
    fig_v.tight_layout()
    
    plt.show()

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Initial conditions (example)
    r0 = np.array([1.48876183e+11, 0.0, -1.25e9])
    v0 = np.array([0.0, 29928.37870446, 0.0])

    # Time settings
    days_total = 30
    t0 = 0.0
    tf = days_total * DAY
    n_out = 201
    t_eval = np.linspace(t0, tf, n_out)
    time_days = t_eval / DAY

    # Nominal trajectory
    rs_nom, vs_nom = integrate_nominal(r0, v0, (t0, tf), t_eval)

    # STM integration
    phis_mat = integrate_with_stm(r0, v0, (t0, tf), t_eval)

    # Monte Carlo samples
    Nmc = 200
    sigma_pos = 60e3 / 3   # m
    sigma_vel = 0.1 / 3    # m/s
    rng = np.random.default_rng(6789)
    # draw each component indipendent
    mc_samples = rng.normal(size=(Nmc, 6))
    mc_samples[:, 0:3] *= sigma_pos
    mc_samples[:, 3:6] *= sigma_vel
    # Draw a random normalized vector (example usage)
    mc_samples = rng.normal(size=(Nmc, 6))
    rand_vec = rng.normal(size=3)
    rand_vec /= np.linalg.norm(rand_vec)
    mc_samples[0, 0:3] += rand_vec * sigma_pos
    rand_vec = rng.normal(size=3)
    rand_vec /= np.linalg.norm(rand_vec)
    mc_samples[0, 3:6] += rand_vec * sigma_vel
    # MC: STM-based
    mc_pos_errors_stm = monte_carlo_stm(mc_samples, phis_mat)

    # MC: Nonlinear
    mc_pos_errors_nl, mc_rs_samples, mc_vs_samples = monte_carlo_nonlinear(
        r0, v0, mc_samples, rs_nom, (t0, tf), t_eval
    )

    plot_component_ratios(time_days, rs_nom, vs_nom, mc_rs_samples, mc_vs_samples)
    # Plots
    # plot_3d_trajectories(rs_nom, mc_rs_samples)
    stm_p, nl_p = plot_envelopes(time_days, mc_pos_errors_stm, mc_pos_errors_nl)
    # plot_relative_difference(time_days, stm_p, nl_p)
    
    # Summary
    final_stm_median = stm_p[1,-1]
    final_nl_median = nl_p[1,-1]
    final_stm_p95 = stm_p[2,-1]
    final_nl_p95 = nl_p[2,-1]

    print(f"Summary at final time {days_total} days:")
    print(f"  STM median final pos error = {final_stm_median:.3e} m")
    print(f"  Nonlinear median final pos error = {final_nl_median:.3e} m")
    print(f"  STM 95th pct final pos error = {final_stm_p95:.3e} m")
    print(f"  Nonlinear 95th pct final pos error = {final_nl_p95:.3e} m")
    print(f"  Relative median difference (STM vs NL) = "
          f"{(final_stm_median - final_nl_median)/final_nl_median:.3%}")
