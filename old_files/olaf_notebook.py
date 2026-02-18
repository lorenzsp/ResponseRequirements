import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from lisaorbits import StaticConstellation
from lisaconstants import c
import segwo
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA
np.random.seed(42)
# Constants
L = 2.5e9  # Arm length in meters
f = np.logspace(-4, np.log10(2e-2), 100)  # Frequency axis
nside = 12  # Healpy resolution
npix = hp.nside2npix(nside)
thetas, phis = hp.pix2ang(nside, np.arange(npix))
betas, lambs = np.pi / 2 - thetas, phis
pols = ['h+', 'hx']
A = (Z2_ETA - X2_ETA) / np.sqrt(2)
E = (X2_ETA - 2 * Y2_ETA + Z2_ETA) / np.sqrt(6)
T = (X2_ETA + Y2_ETA + Z2_ETA) / np.sqrt(3)

# Helper Functions
def compute_static_orbits():
    """Compute static constellation orbits."""
    orbits = StaticConstellation.from_armlengths(L, L, L)
    ltts = orbits.compute_ltt(t=[0.0])
    positions = orbits.compute_position(t=[0.0])
    return orbits, ltts, positions

def compute_response(orbits, ltts, positions):
    """Compute strain-to-link response."""
    return segwo.response.compute_strain2link(f, betas, lambs, ltts, positions)

def compute_tdi_response(strain2link, ltts):
    """Compute strain-to-TDI response."""
    eta_list = [f"eta_{mosa}" for mosa in orbits.LINKS]
    link2x = segwo.cov.construct_mixing_from_pytdi(f, eta_list, [A, E, T], ltts)
    link2x = link2x[:, :, np.newaxis, :, :]
    return segwo.cov.compose_mixings([strain2link, link2x])

def perturbed_static_orbits(arm_lengths, armlength_error, rotation_error, translation_error, rot_fac=2.127):
    """Generate perturbed static orbits."""
    perturbed_ltt_orbits = StaticConstellation.from_armlengths(
        arm_lengths[0] + np.random.normal(0, armlength_error),
        arm_lengths[1] + np.random.normal(0, armlength_error),
        arm_lengths[2] + np.random.normal(0, armlength_error)
    )
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    avg_distance = np.mean(np.linalg.norm(perturbed_ltt_orbits.sc_positions, axis=1))
    angle = rot_fac * rotation_error / avg_distance
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.outer(axis, axis)
    rotated_positions = np.dot(R, perturbed_ltt_orbits.sc_positions.T).T
    translation = np.random.normal(0, translation_error, size=(3,))
    perturbed_positions = rotated_positions + translation
    return StaticConstellation(perturbed_positions[0], perturbed_positions[1], perturbed_positions[2])

def compute_perturbed_responses(N, arm_lengths, armlength_error, rotation_error, translation_error):
    """Compute perturbed responses for multiple realizations."""
    perturbed_ltts = np.zeros((N, 6))
    perturbed_positions = np.zeros((N, 3, 3))
    for i in range(N):
        perturbed_orbit = perturbed_static_orbits(arm_lengths, armlength_error, rotation_error, translation_error)
        perturbed_ltts[i] = perturbed_orbit.compute_ltt(t=[0.0])
        perturbed_positions[i] = perturbed_orbit.compute_position(t=[0.0])
    return perturbed_ltts, perturbed_positions

def compute_relative_errors(a, b):
    """Compute relative errors."""
    return np.abs(a - b) / np.average(np.abs(b), axis=2)[:, :, np.newaxis, :, :]

def compute_absolute_errors(a, b):
    """Compute absolute errors."""
    return np.abs(a - b)

def analyze_violations(errors, threshold, strain2x):
    """Analyze violations of a given threshold."""
    # Data points which violate requirement
    violation_points = np.sum((errors > threshold) & (strain2x != 0))
    total_points = np.sum(errors**0)
    return violation_points / total_points

def plot_response_map(response_map, title):
    """Plot a sky map of the response."""
    hp.mollview(response_map, title=title, rot=[0, 0])
    hp.graticule()
    plt.show()

# Main Analysis
orbits, ltts, positions = compute_static_orbits()
strain2link = compute_response(orbits, ltts, positions)
strain2x = compute_tdi_response(strain2link, ltts)

# Perturbed Analysis
N = 10
arm_lengths = [L, L, L]
armlength_error = 1.
rotation_error = 50e3
translation_error = 50e3
print("Performing analysis on static orbits with perturbations")
print("Arm length error:", armlength_error)
print("Rotation error:", rotation_error)
print("Translation error:", translation_error)
perturbed_ltts, perturbed_positions = compute_perturbed_responses(
    N, arm_lengths, armlength_error, rotation_error, translation_error
)
print("Perturbed LTTs and positions computed.")
# Compute perturbed strain2link and strain2x
strain2link_perturbed = segwo.response.compute_strain2link(
    f, betas, lambs, perturbed_ltts, perturbed_positions
)
print("Perturbed strain2link computed.")

strain2x_perturbed = compute_tdi_response(strain2link_perturbed, ltts)
print("Perturbed strain2link and strain2x computed.", strain2x_perturbed.shape, strain2x.shape)

# plot strain
index_realization = 0
index_freq = 10
index_pix = 100
pol = 0 
tdi = 0
plt.figure()
for index_realization in range(N):
    plt.loglog(f, np.abs(strain2x_perturbed[index_realization, :, index_pix, tdi, pol] - strain2x[0, :, index_pix, tdi, pol]))
plt.show()
# breakpoint()
# Compute Errors
rel_abs_error = compute_relative_errors(np.abs(strain2x_perturbed), np.abs(strain2x))
abs_ang_error = compute_absolute_errors(np.angle(strain2x_perturbed), np.angle(strain2x))
strain2x_abs_error = np.std(rel_abs_error, axis=0)
strain2x_angle_error = np.std(abs_ang_error, axis=0)

# Result heavily depends on pixel
# pix = npix // 2
# pix = 1005
# print(betas[pix], lambs[pix])

# for i in range(3):
#     for j in range(1):
#         plt.loglog(f, strain2x_abs_error[:, pix, i, j], label=f'TDI {'AET'[i]}, {pols[j]}')
#         plt.xlabel("Frequency [Hz]")
#         plt.ylabel("Relative error in |R|")
# plt.legend()
# plt.show()

# Analyze Violations
amp_req = 1e-4
phase_req = 1e-2
amp_violation_ratio = analyze_violations(strain2x_abs_error, amp_req, strain2x_perturbed)
phase_violation_ratio = analyze_violations(strain2x_angle_error, phase_req, strain2x_perturbed)


# histogram of abs errors per frequency
# frequency_bins = np.arange(0, len(f), 5)
# frequency = f[frequency_bins]

# plt.figure()
# for fbin in frequency_bins:
#     plt.hist(np.log10(strain2x_abs_error[fbin].flatten()), bins=30, alpha=0.5, label=f"f={f[fbin]:.2f} Hz", density=True)
# plt.xlabel("log10 Amplitude Error")
# plt.axvline(x=np.log10(amp_req), color='r', linestyle='--', label='Amplitude Requirement')
# plt.title("Amplitude Error Histogram")
# plt.show()

# plt.figure()
# for fbin in frequency_bins:
#     plt.hist(np.log10(strain2x_angle_error[fbin].flatten()), bins=30, alpha=0.5, label=f"f={f[fbin]:.2f} Hz", density=True)
# plt.xlabel("log10 Phase Error")
# plt.axvline(x=np.log10(phase_req), color='r', linestyle='--', label='Phase Requirement')
# plt.title("Phase Error Histogram")
# plt.show()

print(f"Amplitude Violation Ratio: {amp_violation_ratio}")
print(f"Phase Violation Ratio: {phase_violation_ratio}")

# Plot Example Response Maps
# f_eval = 50
# for link in range(3):
#     for pol in range(2):
#         gw_response_map = strain2x_angle_error[f_eval, :, link, pol]
#         plot_response_map(
#             gw_response_map,
#             title=f"Amplitude Error Map for {pols[pol]}, TDI {'AET'[link]}, f={f[f_eval]:.2f} Hz"
#         )
