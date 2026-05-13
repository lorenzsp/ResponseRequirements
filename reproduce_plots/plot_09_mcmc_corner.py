"""
Plot 09 — MCMC corner overlay (Non-Boosted vs Boosted posteriors).

Requires:
  <workspace_root>/data/mcmc_chains_processed.h5   (already lightweight)

Output:
  reproduce_plots/figures/mcmc_corner_overlay.pdf
"""

from pathlib import Path
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import apply_style, DATA_DIR, OUTPUT_DIR, COL1

apply_style()

# ── Parameter labels and selection ───────────────────────────────────────────
labels_full = np.array([
    r"$f_0$ [Hz]",
    r"$\dot f_0$ [Hz/s]",
    r"$A$",
    r"$\beta$",
    r"$\lambda$",
    r"$\psi$",
    r"$\iota$",
    r"$\phi_0$",
    r"$\log L$",
])
index_to_plot = np.asarray([0, 2], dtype=int)

# ── Load chains ───────────────────────────────────────────────────────────────
with h5py.File(DATA_DIR / "mcmc_chains_processed.h5", "r") as hf:
    injection_params = hf["injection_params"][()]
    samples_list     = []

    for model in ["non_boosted", "boosted"]:
        grp          = hf[model]
        cold_samples = grp["cold_samples"][()]
        cold_logl    = grp["cold_logl"][()]
        samp_log     = np.hstack((cold_samples, cold_logl[:, np.newaxis]))
        samples_list.append(samp_log[:, index_to_plot])

sample_labels = ["Non-Boosted", "Boosted"]

# ── Corner kwargs ─────────────────────────────────────────────────────────────
CORNER_KWARGS = dict(
    labels=labels_full[index_to_plot].tolist(),
    bins=40,
    truths=injection_params[index_to_plot],
    label_kwargs=dict(fontsize=12),
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=False,
    show_titles=False,
    max_n_ticks=4,
    truth_color="k",
    labelpad=0.3,
)


def _normalisation_weight(n, n_ref):
    return np.ones(n) * (n_ref / n)


def overlaid_corner(samples_list, sample_labels, name_save=None,
                    corn_kw=None, title=None):
    n         = len(samples_list)
    _, ndim   = samples_list[0].shape
    max_len   = max(len(s) for s in samples_list)
    colors    = ["C0", "C1"]

    plot_range = []
    for d in range(ndim):
        lo = min(np.min(samples_list[i][:, d]) for i in range(n))
        hi = max(np.max(samples_list[i][:, d]) for i in range(n))
        plot_range.append([lo, hi])

    corn_kw = {} if corn_kw is None else dict(corn_kw)
    corn_kw.update(range=plot_range)

    weights = [_normalisation_weight(len(samples_list[i]), max_len)
               for i in range(n)]

    fig = plt.figure(figsize=(COL1, COL1))
    fig = corner.corner(samples_list[0], fig=fig, color=colors[0],
                        weights=weights[0], **corn_kw)
    axes    = np.array(fig.axes).reshape((ndim, ndim))
    maxy_all = [[axes[i, i].get_ybound()[-1] for i in range(ndim)]]

    for i in range(1, n):
        fig = corner.corner(samples_list[i], fig=fig, color=colors[i],
                            weights=weights[i], **corn_kw)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        maxy_all.append([axes[j, j].get_ybound()[-1] for j in range(ndim)])

    maxy_all = np.asarray(maxy_all)
    axes     = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        axes[i, i].set_ylim(0.0, np.max(maxy_all[:, i]))

    axes[0, -1].legend(
        handles=[mlines.Line2D([], [], color=colors[i],
                               label=sample_labels[i]) for i in range(n)],
        frameon=False,
        loc="upper right",
        title=title,
        fontsize=8,
        title_fontsize=8,
    )
    plt.subplots_adjust(hspace=0.15, wspace=0.15)

    if name_save is not None:
        plt.savefig(str(name_save) + ".pdf", pad_inches=0.2,
                    bbox_inches="tight")
        print(f"✓ {Path(name_save).name}.pdf")
    else:
        plt.show()


overlaid_corner(
    samples_list=samples_list,
    sample_labels=sample_labels,
    name_save=str(OUTPUT_DIR / "mcmc_corner_overlay"),
    corn_kw=CORNER_KWARGS,
    title="Galactic Binary Posterior",
)
