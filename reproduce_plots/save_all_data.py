"""
Extract and save minimal data from heavy segwo_results/ directories into
lightweight HDF5 files in reproduce_plots/data/.

Run this script once from the workspace root or from reproduce_plots/:

    python reproduce_plots/save_all_data.py

Outputs (all in reproduce_plots/data/):
  perturbation_data.h5       — LTT / position / angle distributions
  mismatch_frequency_data.h5 — pre-aggregated mismatch vs frequency
  amplitude_phase_errors.h5  — pre-aggregated amplitude & phase errors
  gb_segwo_mismatch.h5       — realistic-case mismatch means for GB plot
  orbit_data.h5              — spacecraft positions/velocities/LTTs
"""

from pathlib import Path
import sys
import os
import numpy as np
import h5py

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(exist_ok=True)

CASES = {
    "Arm-length (no boost)":  "segwo_results/static/arm1_rot0.0_trans0.0_boost0.0/",
    "Arm-length (boost)":     "segwo_results/static/arm1_rot0.0_trans0.0_boost1.0/",
    "Rotation (no boost)":    "segwo_results/static/arm0.0_rot50000.0_trans0.0_boost0.0/",
    "Rotation (boost)":       "segwo_results/static/arm0.0_rot50000.0_trans0.0_boost1.0/",
    "Translation (no boost)": "segwo_results/static/arm0.0_rot0.0_trans50000.0_boost0.0/",
    "Translation (boost)":    "segwo_results/static/arm0.0_rot0.0_trans50000.0_boost1.0/",
    "Realistic (no boost)":   "segwo_results/15.0days_evolving_boost0.0/",
    "Realistic (boost)":      "segwo_results/15.0days_evolving_boost1.0/",
}


def load_results(results_dir):
    hdf5_path = ROOT / results_dir / "results.h5"
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as hf:
        data = {
            "f":                     hf["metadata/frequencies"][()],
            "npix":                  int(hf["metadata"].attrs["npix"]),
            "nside":                 int(hf["metadata"].attrs["nside"]),
            "mismatch":              hf["errors/mismatch"][()],
            "strain2x_abs_error":    hf["errors/strain2x_abs_error"][()],
            "strain2x_angle_error":  hf["errors/strain2x_angle_error"][()],
            "amp_violation_ratio":   float(hf["errors"].attrs["amp_violation_ratio"]),
            "phase_violation_ratio": float(hf["errors"].attrs["phase_violation_ratio"]),
        }
        for key, path in [
            ("angle",              "errors/angle"),
            ("ltt_residuals",      "perturbed/ltt_residuals"),
            ("position_residuals", "perturbed/position_residuals"),
            ("mismatch_boost",     "nominal/mismatch_boost"),
        ]:
            if path in hf:
                data[key] = hf[path][()]
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 1. orbit_data.h5  (minimal subset of processed_trajectories.h5)
# ─────────────────────────────────────────────────────────────────────────────
src = ROOT / "data" / "processed_trajectories.h5"
dst = OUT_DIR / "orbit_data.h5"
if src.exists():
    with h5py.File(src, "r") as s, h5py.File(dst, "w") as d:
        d.create_dataset("t_interp",              data=s["t_interp"][()])
        # Store the median over realisations to keep file small
        pos = s["spacecraft_positions"][()]
        vel = s["spacecraft_velocities"][()]
        ltt = s["owlt_12_23_31_13_32_21"][()]
        d.create_dataset("x_median",  data=np.median(pos, axis=0))
        d.create_dataset("v_median",  data=np.median(vel, axis=0))
        d.create_dataset("ltt_median", data=np.median(ltt, axis=0))
    print(f"✓ orbit_data.h5")
else:
    print(f"✗ {src} not found — skipping orbit_data.h5")

# ─────────────────────────────────────────────────────────────────────────────
# 2. perturbation_data.h5
# ─────────────────────────────────────────────────────────────────────────────
perturb_cases = {
    "arm_boost":         "Arm-length (boost)",
    "translation_boost": "Translation (boost)",
    "rotation_boost":    "Rotation (boost)",
    "realistic_boost":   "Realistic (boost)",
}

dst = OUT_DIR / "perturbation_data.h5"
with h5py.File(dst, "w") as d:
    for grp_name, case_label in perturb_cases.items():
        path = CASES[case_label]
        try:
            dat = load_results(path)
            grp = d.create_group(grp_name)
            if "ltt_residuals" in dat:
                grp.create_dataset("ltt_residuals",      data=dat["ltt_residuals"])
            if "position_residuals" in dat:
                grp.create_dataset("position_residuals", data=dat["position_residuals"])
            if "angle" in dat:
                grp.create_dataset("angle",              data=dat["angle"])
            print(f"  ✓ {grp_name}")
        except FileNotFoundError as e:
            print(f"  ✗ {grp_name}: {e}")
print(f"✓ perturbation_data.h5")

# ─────────────────────────────────────────────────────────────────────────────
# 3. mismatch_frequency_data.h5  (pre-aggregated over N_real and sky)
# ─────────────────────────────────────────────────────────────────────────────
mismatch_cases = [
    "Arm-length (boost)",
    "Rotation (boost)",
    "Translation (boost)",
    "Realistic (boost)",
    "Realistic (no boost)",
]

dst = OUT_DIR / "mismatch_frequency_data.h5"
f_written = False
with h5py.File(dst, "w") as d:
    for case_label in mismatch_cases:
        path = CASES[case_label]
        try:
            dat = load_results(path)
            mm = np.nan_to_num(dat["mismatch"])   # (N_real, F, npix, 2)
            grp_name = case_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            grp = d.create_group(grp_name)
            grp.create_dataset("mismatch_mean", data=mm.mean(axis=(0, 2)))   # (F, 2)
            grp.create_dataset("mismatch_min",  data=mm.min(axis=(0, 2)))
            grp.create_dataset("mismatch_max",  data=mm.max(axis=(0, 2)))
            if not f_written:
                d.create_dataset("frequencies", data=dat["f"])
                f_written = True
            print(f"  ✓ {case_label}")
        except FileNotFoundError as e:
            print(f"  ✗ {case_label}: {e}")
print(f"✓ mismatch_frequency_data.h5")

# ─────────────────────────────────────────────────────────────────────────────
# 4. amplitude_phase_errors.h5  (pre-aggregated over N_real)
# ─────────────────────────────────────────────────────────────────────────────
error_cases = ["Realistic (no boost)", "Realistic (boost)"]

dst = OUT_DIR / "amplitude_phase_errors.h5"
f_written = False
with h5py.File(dst, "w") as d:
    for case_label in error_cases:
        path = CASES[case_label]
        try:
            dat = load_results(path)
            amp  = dat["strain2x_abs_error"]    # (F, N_real, 3, 2)
            ang  = dat["strain2x_angle_error"]  # (F, N_real, 3, 2)
            grp_name = case_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            grp = d.create_group(grp_name)
            grp.create_dataset("amp_mean",   data=amp.mean(axis=1))   # (F, 3, 2)
            grp.create_dataset("amp_min",    data=amp.min(axis=1))
            grp.create_dataset("amp_max",    data=amp.max(axis=1))
            grp.create_dataset("angle_mean", data=ang.mean(axis=1))
            grp.create_dataset("angle_min",  data=ang.min(axis=1))
            grp.create_dataset("angle_max",  data=ang.max(axis=1))
            if not f_written:
                d.create_dataset("frequencies", data=dat["f"])
                f_written = True
            print(f"  ✓ {case_label}")
        except FileNotFoundError as e:
            print(f"  ✗ {case_label}: {e}")
print(f"✓ amplitude_phase_errors.h5")

# ─────────────────────────────────────────────────────────────────────────────
# 5. gb_segwo_mismatch.h5  (realistic mismatch means needed for GB plot)
# ─────────────────────────────────────────────────────────────────────────────
dst = OUT_DIR / "gb_segwo_mismatch.h5"
with h5py.File(dst, "w") as d:
    for case_label, grp_name in [
        ("Realistic (no boost)", "realistic_no_boost"),
        ("Realistic (boost)",    "realistic_boost"),
    ]:
        path = CASES[case_label]
        try:
            dat = load_results(path)
            mm = np.nan_to_num(dat["mismatch"])   # (N_real, F, npix, 2)
            grp = d.create_group(grp_name)
            grp.create_dataset("mismatch_mean_sky_pol0",
                               data=mm.mean(axis=(0, 2))[:, 0])  # (F,)
            print(f"  ✓ {case_label}")
        except FileNotFoundError as e:
            print(f"  ✗ {case_label}: {e}")
    # Store frequencies from last loaded case
    try:
        d.create_dataset("frequencies", data=dat["f"])
    except Exception:
        pass
print(f"✓ gb_segwo_mismatch.h5")

print("\nAll done. Lightweight data saved to:", OUT_DIR)
