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

from lisaconstants.indexing import LINKS


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
