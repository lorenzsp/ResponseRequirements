import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from lisaorbits import StaticConstellation
from lisaconstants import c
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA
from segwo.response import compute_strain2link
from segwo.cov import construct_mixing_from_pytdi, compose_mixings


def relative_errors_sky(a, b):
    """ Compute the relative errors between two arrays, 
        wrt. the direction of maximum amplitude
    """
    return np.abs(a - b) / np.abs(b)#np.average(np.abs(b), axis=2)[:, :, np.newaxis, :, :]

def absolute_errors(a, b):
    """ Compute the relative errors between two arrays
    """
    return np.abs(a - b)

def perturbed_static_orbits(arm_lengths, armlength_error, rotation_error, translation_error, rot_fac = 2.127):
    """ Apply perturbations to the static orbits 
    
    We want to create a situation where the armlengths are known very well, but the
    absolute positions of the spacecraft are not known very well.

    This is achieved by applying a small perturbation to the armlengths
    followed by a rotation and translation to the spacecraft 
    positions (around random axis and directions).
    
    Parameters
    ----------
    arm_lengths : np.array
        Nominal armlengths of the constellation.
    armlength_error : float
        Standard deviation of the perturbation to apply to the armlengths, in meters.
    rotation_error : float
        Standard deviation of the rotation to apply to the spacecraft positions, in equivalent meters of displacement.
    translation_error : float
        Standard deviation of the translation to apply to the spacecraft positions, in meters.

    """
    # Create new orbits with perturbed armlengths
    perturbed_ltt_orbits = StaticConstellation.from_armlengths(arm_lengths[0] + np.random.normal(0, armlength_error), 
                                                       arm_lengths[1] + np.random.normal(0, armlength_error), 
                                                       arm_lengths[2] + np.random.normal(0, armlength_error))
    
    # Apply rotation by an angle phi along a random axis in 3d space

    # Generate a random rotation matrix
    # Random axis of rotation
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)  # Normalize the axis

    # Average distance of the spacecraft from the center of mass
    avg_distance = np.mean(np.linalg.norm(perturbed_ltt_orbits.sc_positions, axis=1))

    # In the small angle approximation, the rotation by an angle phi causes a displacement
    # of the spacecraft positions by a distance d = r * phi. Solving for phi and using d = error_magnitude gives
    # phi = d / r
    # TODO: improve on the math here; not all rotations affect all S/C, so this is
    # off by a factor of 2 or so; for now just fitted by hand
    angle = rot_fac * rotation_error / avg_distance

    # Rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.outer(axis, axis)

    # Apply the rotation to the spacecraft positions
    rotated_positions = np.dot(R, perturbed_ltt_orbits.sc_positions.T).T

    # Apply translation to the spacecraft positions
    translation = np.random.normal(0, translation_error, size=(3,))
    perturbed_positions = rotated_positions + translation
    
    # Create a new StaticConstellation object with the perturbed positions
    perturbed_orbits = StaticConstellation(perturbed_positions[0], perturbed_positions[1], perturbed_positions[2])
    return perturbed_orbits

def generate_realizations(arm_lengths, armlength_error, rotation_error, translation_error, N):
    """
    Generate realizations of perturbed light travel times and spacecraft positions.

    Parameters
    ----------
    arm_lengths : list
        Nominal armlengths of the constellation.
    armlength_error : float
        Standard deviation of the perturbation to apply to the armlengths, in meters.
    rotation_error : float
        Standard deviation of the rotation to apply to the spacecraft positions, in equivalent meters of displacement.
    translation_error : float
        Standard deviation of the translation to apply to the spacecraft positions, in meters.
    N : int
        Number of realizations to generate.

    Returns
    -------
    perturbed_ltt : np.array
        Array of perturbed light travel times for each realization.
    perturbed_positions : np.array
        Array of perturbed spacecraft positions for each realization.
    """
    perturbed_ltt = np.zeros((N, 6))
    perturbed_positions = np.zeros((N, 3, 3))

    for i in range(N):
        perturbed_orbit = perturbed_static_orbits(
            arm_lengths=arm_lengths, 
            armlength_error=armlength_error, 
            rotation_error=rotation_error, 
            translation_error=translation_error
        )

        perturbed_ltt[i] = perturbed_orbit.compute_ltt(t=[0.0])
        perturbed_positions[i] = perturbed_orbit.compute_position(t=[0.0])

    return perturbed_ltt, perturbed_positions


def compute_strain2x(frequencies, betas, lambs, ltts, positions, orbits, A, E, T):
    """
    Compute the strain2x matrix for given frequencies, sky locations, and orbits.

    Parameters
    ----------
    frequencies : np.array
        Array of frequency values.
    betas : np.array
        Array of beta (latitude) values for sky locations.
    lambs : np.array
        Array of lambda (longitude) values for sky locations.
    ltts : np.array
        Light travel times for the constellation.
    positions : np.array
        Spacecraft positions for the constellation.
    orbits : StaticConstellation
        StaticConstellation object representing the constellation.
    A, E, T : np.array
        TDI combinations.

    Returns
    -------
    strain2x : np.array
        The composed mixing matrix to go directly from strain to TDI variables.
    """
    # Compute the complex signal response for the given frequencies and sky locations
    strain2link = compute_strain2link(frequencies, betas, lambs, ltts, positions)
    # Shape: times, frequencies, pixels, links, polarizations
    # Ie., it's the mixing matrix to go from h+/hx in terms of strain to single link
    # response in terms of fractional frequency deviation. See segwo documentation for details.
    strain2link.shape

    # Define eta variables for the links
    eta_list = [f"eta_{mosa}" for mosa in orbits.LINKS]

    # Construct the mixing matrix from pytdi
    link2x = construct_mixing_from_pytdi(frequencies, eta_list, [A, E, T], ltts)

    # Add a new axis for each pixel
    link2x = link2x[:, :, np.newaxis, :, :]

    # Compose the mixing matrices to go directly from strain to TDI variables
    strain2x = compose_mixings([strain2link, link2x])

    return strain2x



def plot_strain_errors(f, strain2x_abs_error, strain2x_angle_error, pols = ['h+', 'hx'], output_file="strain2x_errors.png", metric="max"):
    """
    Plots the relative errors in |R| and angle for strain2x and saves the figure.

    Parameters:
        f (array): Frequency array.
        strain2x_abs_error (array): Relative error in |R| array.
        strain2x_angle_error (array): Relative error in angle array.
        pols (list): List of polarization labels.
        output_file (str): File name to save the plot.
        metric (str): Metric to use for error calculation ("max", "mean").
    """
    if metric == "max":
        metric_func = np.max
    elif metric == "mean":
        metric_func = np.mean
    else:
        raise ValueError("Invalid metric. Use 'max' or 'mean'.")
    
    fig, axs = plt.subplots(2, 1)

    # Plot relative error in |R|
    for i in range(3):
        for j in range(2):
            axs[0].loglog(f, metric_func(strain2x_abs_error[:, :, i, j], axis=1), label=f'{metric_func.__name__} TDI {"AET"[i]}, {pols[j]}')
    axs[0].set_xlabel("Frequency [Hz]")
    axs[0].set_ylabel("Relative error in Amplitude")
    axs[0].legend()

    # Plot relative error in angle
    for i in range(3):
        for j in range(2):
            axs[1].loglog(f, metric_func(strain2x_angle_error[:, :, i, j], axis=1), label=f'{metric_func.__name__} TDI {"AET"[i]}, {pols[j]}')
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("Absolute error in Phase")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_gw_response_maps(strain2x_abs_error, f, npix, pols=['h+', 'hx'], folder="", metric="max"):
    """
    Plots gravitational wave response sky maps for each polarization and TDI link.

    Parameters:
        strain2x_abs_error (array): Relative error in |R| array.
        f (array): Frequency array.
        npix (int): Number of pixels in the sky map.
        pols (list): List of polarization labels.
        f_eval (int): Index of the frequency to evaluate.
        metric (str): Metric to use for error calculation ("max", "mean").
    """
    if metric == "max":
        metric_func = np.max
    elif metric == "mean":
        metric_func = np.mean
    else:
        raise ValueError("Invalid metric. Use 'max' or 'mean'.")
    
    for link in range(3):
        for pol in range(2):
            plt.figure()
            gw_response_map = np.zeros(npix)

            # Populate the map with the GW response for each pixel
            for pix in range(npix):
                max_sky = metric_func(strain2x_abs_error[:, pix, link, pol], axis=0)
                gw_response_map[pix] = max_sky

            # Plotting the map
            hp.mollview(
                gw_response_map,
                title=f"Gravitational Wave Response Sky Map for {pols[pol]}, TDI {'AET'[link]}",
                rot=[0, 0]
            )
            hp.graticule()
            plt.savefig(folder + f"gw_response_map_{pols[pol]}_link{link}.png", dpi=300)
            plt.close()

def compute_violation_ratios(strain2x_abs_error, strain2x_angle_error, amp_req=1e-4, phase_req=1e-3):
    """
    Compute the amplitude and phase violation ratios.

    Parameters:
        strain2x_abs_error (ndarray): Relative amplitude error in strain amplitude.
        strain2x_angle_error (ndarray): Absolute error in strain phase.
        strain2x (ndarray): Original strain values.
        amp_req (float): Amplitude requirement threshold.
        phase_req (float): Phase requirement threshold.

    Returns:
        tuple: Amplitude violation ratio, Phase violation ratio.
    """
    # Amplitude violation
    amp_violation_points = np.sum((strain2x_abs_error > amp_req))
    total_points_amp = np.sum(strain2x_abs_error**0)
    amp_violation_ratio = amp_violation_points / total_points_amp
    print("Amplitude violation ratio:", strain2x_abs_error[(strain2x_abs_error > amp_req)])
    # Phase violation
    phase_violation_points = np.sum((strain2x_angle_error > phase_req))
    total_points_phase = np.sum(strain2x_angle_error**0)
    phase_violation_ratio = phase_violation_points / total_points_phase
    print("Phase violation ratio:", strain2x_angle_error[(strain2x_angle_error > phase_req)])
    # print values of violations

    return amp_violation_ratio, phase_violation_ratio

if __name__ == "__main__":
    # check we get the correct error statistics
    # Now put it together, both rotation and translation error, with the scaling applied
    # check we get the correct error statistics
    # Only armlengths + rotation error
    # Define static combination with equal arms
    orbits = StaticConstellation.from_armlengths(2.5e9, 2.5e9, 2.5e9)

    # Compute the positions of the spacecraft and the light travel times at t=0.0
    ltts = orbits.compute_ltt(t=[0.0])
    positions = orbits.compute_position(t=[0.0])

    N = 10
    perturbed_ltt, perturbed_positions =  generate_realizations(
        arm_lengths=[2.5e9, 2.5e9, 2.5e9],
        armlength_error=1,
        rotation_error=50e3,
        translation_error=50e3,
        N=N)
    ltt_residuals = np.zeros((N, 6))
    position_residuals = np.zeros((N, 3, 3))

    for i in range(N):
        ltt_residuals[i, :] = perturbed_ltt[i] - ltts
        position_residuals[i, :] = perturbed_positions[i] - positions

    for i in range(6):
        print(f"ltt std for link {orbits.LINKS[i]}", np.std(ltt_residuals[:, i])*c, "meters")

    for i in range(3):
        print(f"Position std for sc {i}:", np.std(position_residuals[:,i]), "meters")

    print("should be consistent with", 50e3 * np.sqrt(2))

