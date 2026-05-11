"""
Plotting utilities for the SEGWO response analysis.

All functions here operate on pre-computed numpy arrays. They are deliberately
kept separate from computation code so that figures can be regenerated from
saved HDF5 data without repeating expensive computations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
import healpy as hp
from matplotlib.colors import LogNorm

from lisaconstants.indexing import LINKS
from lisaconstants import C as C_SI



def plot_orbit_3d(orbital_info, T, figsize=(8, 4.5), Nshow=10, lam=None, beta=None, output_file="3d_orbit_around_sun.png", scatter_points=None, include_sun=True):
    """
    Plot the 3D orbit of the spacecraft around the Sun.

    Args:
        orbital_info: dict containing orbital information.
        T: Time duration in years.
        Nshow: Number of points to show.
        lam: Longitude of the source in radians.
        beta: Latitude of the source in radians.
        output_file: Output file name for the plot.
        scatter_points: List of tuples [(radius, lam, beta, color_function), ...] for additional 3D scatter points.
        include_sun: Whether to include the Sun in the plot.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    max_r = 0.0

    orbital_info["t"]

    mask = (orbital_info["t"] - orbital_info["t"].min() < 86400 * 365 * T)
    Nmax = len(orbital_info["x"][mask, 0])
    ind = np.linspace(0, Nmax, num=Nshow, dtype=int)
    sc1 = orbital_info["x"][ind, 0] / C_SI / 60
    sc2 = orbital_info["x"][ind, 1] / C_SI / 60
    sc3 = orbital_info["x"][ind, 2] / C_SI / 60
    const_center = (sc1 + sc2 + sc3) / 3

    armlength = np.linalg.norm((sc1 - sc2), axis=1)[0]
    sc1 = sc1 - const_center * 0.9
    sc2 = sc2 - const_center * 0.9
    sc3 = sc3 - const_center * 0.9
    
    ax.plot(*sc1.T, color='tab:blue', linewidth=1, alpha=0.5, linestyle='-')
    ax.plot(*sc2.T, color='tab:orange', linewidth=1, alpha=0.5, linestyle='-.')
    ax.plot(*sc3.T, color='tab:green', linewidth=1, alpha=0.5, linestyle='--')
    
    # plot laser
    for i in range(len(sc1)):
        ax.plot([sc1[i, 0], sc2[i, 0]], [sc1[i, 1], sc2[i, 1]], [sc1[i, 2], sc2[i, 2]], color='r', linewidth=2)
        ax.plot([sc1[i, 0], sc3[i, 0]], [sc1[i, 1], sc3[i, 1]], [sc1[i, 2], sc3[i, 2]], color='r', linewidth=2)
        ax.plot([sc2[i, 0], sc3[i, 0]], [sc2[i, 1], sc3[i, 1]], [sc2[i, 2], sc3[i, 2]], color='r', linewidth=2)
    
    ax.plot(*sc1.T, 'o', linewidth=5)
    ax.plot(*sc2.T, 'o', linewidth=5)
    ax.plot(*sc3.T, 'o', linewidth=5)
    np.linalg.norm(sc1, axis=1)[0]
    
    # # Plot the plane
    # ax.plot_surface(xx, yy, zz, alpha=0.5, color='cyan', edgecolor='none')
    max_r = max(max_r, np.max(np.linalg.norm(sc1, axis=1)))

    # ax.set_xlabel('X Position (min)')
    # ax.set_ylabel('Y Position (min)')
    # ax.set_zlabel('Z Position (min)')
    # ax.set_title("3D Plot of Orbital Positions")
    if include_sun:
        ax.scatter(0.0, 0.0, 0.0, color='orange', marker='o', label="Sun", s=500)

    if (lam is not None) and (beta is not None):
        ax.quiver(0, 0, 0, np.cos(beta) * np.cos(lam), np.cos(beta) * np.sin(lam), np.sin(beta), color='k', label="Source Location")

    if scatter_points:
        for radius, lam_s, beta_s, color_function in scatter_points:
            radius = np.linalg.norm(sc1, axis=1)[0] * 1.2
            x = radius * np.cos(beta_s) * np.cos(lam_s)
            y = radius * np.cos(beta_s) * np.sin(lam_s)
            z = radius * np.sin(beta_s)
            # x += sc1[0,0]
            # y += sc1[0,1]
            # z += sc1[0,2]
            ax.scatter(x, y, z, color=color_function, marker='o', s=100, alpha=0.5)#, label=f"Scatter Point (r={radius}, λ={lam_s}, β={beta_s})")
        max_r = radius

    # ax.legend()
    # Adjust the inclination of the 3D plot view
    ax.view_init(elev=20, azim=45)  # Set elevation and azimuthal angle
    
    # Add a background of stars
    num_stars = 1000
    star_x = np.random.uniform(-max_r * 1.5, max_r * 1.5, num_stars)
    star_y = np.random.uniform(-max_r * 1.5, max_r * 1.5, num_stars)
    star_z = np.random.uniform(-max_r * 1.5, max_r * 1.5, num_stars)
    # ax.scatter(star_x, star_y, star_z, color='white', s=1, alpha=0.1, label="Stars")
    # Set black background for the plot
    # ax.set_facecolor('black')
    # fig.patch.set_facecolor('black')
    ax.set_axis_off()
    ax.grid(False)

    # Remove axis ticks, labels, and background
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # ax.set_axis_off()
    # remove axes from 
    # plt.colorbar()
    
    ax.set_xlim([-max_r, max_r])
    ax.set_ylim([-max_r, max_r])
    ax.set_zlim([-max_r, max_r])
    # plt.show()
    plt.tight_layout()
    # plt.subplots_adjust(left=-0.4, right=1.4, top=1.4, bottom=-0.4)
    plt.savefig(output_file,dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"3D orbit plot saved to {output_file}")


def plot_response(f, npix, strain2x_abs, pols=('h+', 'hx'), folder="",
                  output_file="strain2x.png", metric="min"):
    """Plot the nominal TDI response amplitude vs frequency and as sky maps.

    Parameters
    ----------
    f : (F,) array
        Frequency array [Hz].
    npix : int
        Number of HEALPix pixels.
    strain2x_abs : (F, P, 3, 2) array
        Absolute value of the strain-to-TDI mixing matrix.
    pols : sequence of str
        Polarization labels, length 2.
    folder : str
        Output directory (created if it does not exist).
    output_file : str
        Filename for the frequency-domain plot.
    metric : str
        Sky aggregation metric: "max", "mean", or "min".
    """
    if metric == "max":
        metric_func = np.max
    elif metric == "mean":
        metric_func = np.mean
    elif metric == "min":
        metric_func = np.min
    else:
        raise ValueError("Invalid metric. Use 'max', 'mean', or 'min'.")

    if folder:
        os.makedirs(folder, exist_ok=True)

    # --- Frequency-domain amplitude plot ---
    fig, ax = plt.subplots(1, 1)
    for i in range(3):
        for j in range(2):
            ax.loglog(f, metric_func(strain2x_abs[:, :, i, j], axis=1),
                      label=f'{metric_func.__name__} TDI {"AET"[i]}, {pols[j]}')
    ax.set_ylim(1e-15, 100)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(folder, output_file) if folder else output_file
    plt.savefig(out, dpi=300)
    plt.close()

    # --- Sky maps ---
    for link in range(3):
        for pol in range(2):
            gw_response_map = np.zeros(npix)
            for pix in range(npix):
                gw_response_map[pix] = metric_func(strain2x_abs[:, pix, link, pol], axis=0)
            plt.figure()
            hp.mollview(
                gw_response_map,
                title=f"GW Response: {pols[pol]}, TDI {'AET'[link]}",
                rot=[0, 0],
            )
            hp.graticule()
            out = os.path.join(folder, f"gw_response_map_{pols[pol]}_link{link}.png") \
                if folder else f"gw_response_map_{pols[pol]}_link{link}.png"
            plt.savefig(out, dpi=300)
            plt.close()


def plot_strain_errors(f, strain2x_abs_error, strain2x_angle_error,
                       pols=('h+', 'hx'), output_file="strain2x_errors.png",
                       metric="max"):
    """Plot relative amplitude error and phase mismatch vs frequency.

    Parameters
    ----------
    f : (F,) array
        Frequency array [Hz].
    strain2x_abs_error : (F, P, 3, 2) array
        Relative amplitude error (max over realisations).
    strain2x_angle_error : (F, P, 3, 2) array
        Phase mismatch |1 - exp(i Δφ)| (max over realisations).
    pols : sequence of str
        Polarization labels.
    output_file : str
        Full path for the saved figure.
    metric : str
        Sky aggregation metric: "max" or "mean".
    """
    if metric == "max":
        metric_func = np.max
    elif metric == "mean":
        metric_func = np.mean
    else:
        raise ValueError("Invalid metric. Use 'max' or 'mean'.")

    fig, axs = plt.subplots(2, 1, figsize=(6, 7))

    for i in range(3):
        for j in range(2):
            axs[0].loglog(f, metric_func(strain2x_abs_error[:, :, i, j], axis=1),
                          label=f'{metric_func.__name__} TDI {"AET"[i]}, {pols[j]}')
    axs[0].set_xlabel("Frequency [Hz]")
    axs[0].set_ylabel("Relative error in Amplitude")
    axs[0].legend()

    for i in range(3):
        for j in range(2):
            axs[1].loglog(f, metric_func(strain2x_angle_error[:, :, i, j], axis=1),
                          label=f'{metric_func.__name__} TDI {"AET"[i]}, {pols[j]}')
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("Phase mismatch |1 - e^{iΔφ}|")
    axs[1].legend()

    plt.tight_layout()
    outdir = os.path.dirname(output_file)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_gw_response_maps(strain2x_error, f, npix, pols=('h+', 'hx'),
                          folder="", metric="max"):
    """Plot sky maps of the response error (amplitude or phase).

    Parameters
    ----------
    strain2x_error : (F, P, 3, 2) array
        Error quantity (amplitude or phase) per frequency/pixel/channel/pol.
    f : (F,) array
        Frequency array [Hz] (not directly plotted, kept for API consistency).
    npix : int
        Number of HEALPix pixels.
    pols : sequence of str
        Polarization labels.
    folder : str
        Output directory (created if it does not exist).
    metric : str
        Frequency aggregation metric: "max" or "mean".
    """
    if metric == "max":
        metric_func = np.max
    elif metric == "mean":
        metric_func = np.mean
    else:
        raise ValueError("Invalid metric. Use 'max' or 'mean'.")

    if folder:
        os.makedirs(folder, exist_ok=True)

    for link in range(3):
        for pol in range(2):
            gw_response_map = np.zeros(npix)
            for pix in range(npix):
                gw_response_map[pix] = metric_func(strain2x_error[:, pix, link, pol], axis=0)
            plt.figure()
            hp.mollview(
                gw_response_map,
                title=f"Response Error: {pols[pol]}, TDI {'AET'[link]}",
                rot=[0, 0],
            )
            hp.graticule()
            out = os.path.join(folder, f"gw_response_map_{pols[pol]}_link{link}.png") \
                if folder else f"gw_response_map_{pols[pol]}_link{link}.png"
            plt.savefig(out, dpi=300)
            plt.close()
            
            ##############################################
            z = strain2x_error[:, :, link, pol]
            vmin = np.max([1e-12,z.min()])
            vmax = np.min([1.0,z.max()])
            z[z<vmin] = vmin
            z[z>vmax] = vmax
            
            dmin = np.floor(np.log10(vmin)) # down to lower decade
            dmax = np.ceil(np.log10(vmax)) # up to upper decade

            levels = np.logspace(dmin, dmax, num=int(dmax - dmin + 1))
            plt.figure(figsize=(6, 4))
            cf = plt.contourf(np.arange(z.shape[1]), f, z, levels=levels, norm=LogNorm(vmin=vmin, vmax=vmax),cmap="viridis")
            plt.yscale("log")  # same as semilogy for axis scaling
            cbar = plt.colorbar(cf)
            cbar.set_label("max mismatch")
            plt.xlabel("Sky pixel")
            plt.ylabel("Frequency (Hz)")
            out = os.path.join(folder, f"f_sky_response_{pols[pol]}_link{link}.png") \
                if folder else f"gw_response_map_{pols[pol]}_link{link}.png"
            plt.savefig(out, dpi=300)



def plot_ltt_residuals_histogram(ltt_residuals, output_file, figsize=(3.5, 3)):
    """Histogram of light-travel-time residuals for the first 3 links.

    Parameters
    ----------
    ltt_residuals : (N, 6) array
        LTT residuals for N realisations and 6 links [s].
    output_file : str
        Full output path.
    figsize : tuple
        Figure size in inches.
    """
    linestyle = ['-', '--', '-.']
    plt.figure(figsize=figsize)
    for i in range(3):
        plt.hist(ltt_residuals[:, i], bins=20, alpha=0.5,
                 label=rf"$ij=${LINKS[i]}", density=False,
                 histtype='step', linestyle=linestyle[i], linewidth=2.5)
    plt.xlabel(r"$\Delta L_{ij}$ [s]")
    plt.ylabel("Counts")
    plt.legend(loc='upper left')
    plt.tight_layout()
    outdir = os.path.dirname(output_file)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_position_residuals_histogram(position_residuals, output_file, figsize=(3.5, 3)):
    """Histogram of spacecraft position residuals (SC1 components) in km.

    Parameters
    ----------
    position_residuals : (N, 3, 3) array
        Position residuals for N realisations, 3 spacecraft, 3 coordinates [m].
    output_file : str
        Full output path.
    figsize : tuple
        Figure size in inches.
    """
    linestyle = ['-', '--', '-.']
    plt.figure(figsize=figsize)
    plt.hist(position_residuals[:, 0, 0] / 1e3, bins=20, alpha=0.5,
             label="$x$", density=False,
             histtype='step', linestyle=linestyle[0], linewidth=2.5)
    plt.hist(position_residuals[:, 0, 1] / 1e3, bins=20, alpha=0.5,
             label="$y$", density=False,
             histtype='step', linestyle=linestyle[1], linewidth=2.5)
    plt.hist(position_residuals[:, 0, 2] / 1e3, bins=20, alpha=0.5,
             label="$z$", density=False,
             histtype='step', linestyle=linestyle[2], linewidth=2.5)
    plt.xlabel(r"Position $\Delta x$ [km]")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    outdir = os.path.dirname(output_file)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_angle_histogram(angle, output_file, figsize=(3.5, 3), expected_sigma=None):
    plt.figure(figsize=figsize)
    plt.hist(angle, bins=20, alpha=0.5, density=False, histtype='step', linewidth=2.5)
    if expected_sigma is not None:
        plt.axvline(expected_sigma, color='r')
    plt.xlabel(r"Angle $\delta \phi$")
    plt.ylabel("Counts")
    plt.tight_layout()
    outdir = os.path.dirname(output_file)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()


    
def plot_mismatch(f, npix, mismatch, output_file, figsize=(6, 4), pols=('h+', 'hx'), folder="", metric="max"):
    if metric == "max":
        metric_func = np.max
    elif metric == "mean":
        metric_func = np.mean
    else:
        raise ValueError("Invalid metric. Use 'max' or 'mean'.")

    if folder:
        os.makedirs(folder, exist_ok=True)
    
    # mismatch over sky max across frequencies and realizations
    response_map = metric_func(mismatch,axis=(0,1))
    for pol in range(2):
        plt.figure()
        hp.mollview(response_map,title=f"Mismatch",rot=[0, 0])
        hp.graticule()
        out = os.path.join(folder, f"mismatch_sky_map_{pols[pol]}.png") \
            if folder else f"mismatch_sky_map_{pols[pol]}.png"
        plt.savefig(out, dpi=300)
        plt.close()
    
    frquency_map = metric_func(mismatch,axis=(0,2))
    plt.figure()
    for j in range(2):
        plt.loglog(f, frquency_map, label=f'{pols[j]}')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Mismatch")
    plt.legend()

    plt.tight_layout()
    outdir = os.path.dirname(output_file)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()
