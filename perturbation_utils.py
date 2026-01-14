import numpy as np
from scipy.signal import sawtooth
from orbits_utils import ESAOrbits, EqualArmlengthOrbits
from lisaorbits import StaticConstellation
from lisatools.utils.constants import *
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

try:
    import cupy as xp
except ImportError:
    import numpy as xp

def get_Fourier_design_matrix(t, f, phase):
    # compute arguments for sines and cosines
    argument = 2. * np.pi * np.outer(t, f) + phase[None,:]
    # stack into F matrix
    return np.concatenate([np.sin(argument), np.cos(argument)], axis=1)


def plot_orbit_3d(orbit, T, Nshow=10, lam=None, beta=None, output_file="3d_orbit_around_sun.png", scatter_points=None):
    """
    Plot the 3D orbit of the spacecraft around the Sun.

    Args:
        orbit: Orbit Object.
        T: Time duration in years.
        Nshow: Number of points to show.
        lam: Longitude of the source in radians.
        beta: Latitude of the source in radians.
        output_file: Output file name for the plot.
        scatter_points: List of tuples [(radius, lam, beta, color_function), ...] for additional 3D scatter points.
    """
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    max_r = 0.0

    orbital_info = {which: getattr(orbit, which + "_base") for which in ["ltt", "x", "n", "v"]}
    orbital_info["t"] = orbit.t_base

    mask = (orbit.t_base - orbit.t_base.min() < 86400 * 365 * T)
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
    
    # # Calculate the normal vector of the plane passing through the three spacecraft points
    # vec1 = sc2[0] - sc1[0]
    # vec2 = sc3[0] - sc1[0]
    # normal = np.cross(vec1, vec2)

    # # Define a grid of points for the plane
    # d = -np.dot(normal, sc1[0])  # Plane equation: ax + by + cz + d = 0
    # xx, yy = np.meshgrid(
    #     np.linspace(-max_r, max_r, 100),
    #     np.linspace(-max_r, max_r, 100)
    # )
    # zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]

    # # Plot the plane
    # ax.plot_surface(xx, yy, zz, alpha=0.5, color='cyan', edgecolor='none')
    max_r = max(max_r, np.max(np.linalg.norm(sc1, axis=1)))

    # ax.set_xlabel('X Position (min)')
    # ax.set_ylabel('Y Position (min)')
    # ax.set_zlabel('Z Position (min)')
    ax.set_title("3D Plot of Orbital Positions")
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
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
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
    plt.savefig("figures_perturbation/"+output_file,dpi=300)
    print(f"3D orbit plot saved to {output_file}")


def random_vectors_on_sphere(size):
    """
    Generate an array of random 3D unit vectors uniformly distributed on a sphere.

    Args:
        size (int): Number of unit vectors to generate.

    Returns:
        np.ndarray: An array of shape (size, 3) containing 3D unit vectors.
    """
    phi = np.random.uniform(0, 2 * np.pi, size)  # Random azimuthal angles
    cos_theta = np.random.uniform(-1, 1, size)  # Random cosines of polar angles
    theta = np.arccos(cos_theta)                # Polar angles

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.column_stack((x, y, z))

def get_static_variation(arm_lengths, armlength_error, rotation_error, translation_error, rot_fac = 2.127):
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
    arm_dev = np.random.normal(0, armlength_error, size=(3,))
    # print("arm_dev", arm_dev)
    perturbed_ltt_orbits = StaticConstellation.from_armlengths(arm_lengths[0] + arm_dev[0], 
                                                       arm_lengths[1] + arm_dev[1], 
                                                       arm_lengths[2] + arm_dev[2])
    
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
    # angle = rot_fac * rotation_error / avg_distance
    angle_std = rot_fac * rotation_error / avg_distance

    angle = np.random.normal(0, angle_std)

    # Rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    # Apply the rotation to the spacecraft positions
    rotated_positions = np.dot(R, perturbed_ltt_orbits.sc_positions.T).T

    # Apply translation to the spacecraft positions
    translation = np.random.normal(0, translation_error, size=(3,))
    perturbed_positions = rotated_positions + translation
    # print("translation", translation)
    # print("rotation", angle)
    # Create a new StaticConstellation object with the perturbed positions
    perturbed_orbits = StaticConstellation(perturbed_positions[0], perturbed_positions[1], perturbed_positions[2])
    return perturbed_orbits


def create_orbit_with_static_dev(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=1, rotation_error=50e3, translation_error=50e3, dt=10., T=1.0):
    """
    Creates an ESAOrbits object with static deviations in arm lengths, rotation, and translation.

    This function generates a perturbed orbit using specified arm lengths and error parameters,
    writes the orbit data to a temporary HDF5 file, and loads it into an ESAOrbits object with
    linear interpolation setup enabled.

    Args:
        arm_lengths (list of float, optional): Nominal arm lengths in meters. Defaults to [2.5e9, 2.5e9, 2.5e9].
        armlength_error (float, optional): Standard deviation of arm length error in meters. Defaults to 1.
        rotation_error (float, optional): Standard deviation of rotation error in meters. Defaults to 50e3.
        translation_error (float, optional): Standard deviation of translation error in meters. Defaults to 50e3.
        dt (float, optional): Time step in seconds for orbit sampling. Defaults to 10.
        T (float, optional): Total duration in years for the orbit. Defaults to 1.0.

    Returns:
        ESAOrbits: Configured ESAOrbits object with the perturbed orbit.

    Raises:
        Any exceptions raised by get_static_variation, porbit.write, or ESAOrbits initialization.
    """
    porbit = get_static_variation(arm_lengths=arm_lengths, armlength_error=armlength_error, rotation_error=rotation_error, translation_error=translation_error)
    porbit.write("temp_orbit.h5", dt=dt, size=int(T*YRSID_SI/dt), t0=0.0, mode="w")    
    orb = ESAOrbits("temp_orbit.h5",use_gpu=True)
    orb.configure(linear_interp_setup=True)
    return orb


def create_orbit_with_periodic_dev(fpath="new_orbits.h5", use_gpu=True, armlength_error=1, rotation_error=50e3, translation_error=50e3, period=15*86400, equal_armlength=True, **kwargs):
    """
    Creates an orbit object with periodic deviations in armlength, rotation, and translation errors.
    This function generates an orbit (either with equal armlengths or using ESA orbit data) and applies time-varying deviations
    to its position, velocity, and light travel time (ltt) based on the specified error parameters. The deviations are modeled
    as periodic functions with a given period, using cubic spline interpolation.
        fpath (str): File path for the orbit data (used if equal_armlength is False).
        armlength_error (float): Magnitude of armlength error to apply (meters).
        rotation_error (float): Magnitude of rotation error to apply (radians or meters, depending on implementation).
        translation_error (float): Magnitude of translation error to apply (meters).
        period (float): Period of the periodic deviation (seconds).
        equal_armlength (bool): If True, use EqualArmlengthOrbits; otherwise, use ESAOrbits.
        **kwargs: Additional keyword arguments passed to the orbit constructors.
        ESAOrbits or EqualArmlengthOrbits: Configured orbit object with periodic deviations applied.
    """
    # create the deviation dictionary
    if equal_armlength:
        orb_dev = EqualArmlengthOrbits(use_gpu=use_gpu)
    else:
        orb_dev = ESAOrbits(fpath, use_gpu=use_gpu)
    deviation = {which: np.zeros_like(getattr(orb_dev, which + "_base")) for which in ["ltt", "x", "n", "v"]}
    time_vec = orb_dev.t_base
    porbit = get_static_variation(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=0, rotation_error=0, translation_error=0)
    
    # creating time varying function
    t_dev = np.arange(time_vec[0], time_vec[-1], period)
    
    delta_x = []
    delta_ltt = []
    for el in range(len(t_dev)):
        porbit_dev = get_static_variation(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=armlength_error, rotation_error=rotation_error, translation_error=translation_error)
        # need only the first index because constant in time
        delta_x_temp = porbit.compute_position(orb_dev.t_base)[0] - porbit_dev.compute_position(orb_dev.t_base)[0]
        delta_x.append(delta_x_temp)
        
        ltt_temp = porbit.compute_ltt(orb_dev.t_base)[0] - porbit_dev.compute_ltt(orb_dev.t_base)[0]
        delta_ltt.append(ltt_temp)

    delta_x = np.asarray(delta_x)
    delta_ltt = np.asarray(delta_ltt)

    # # Plot distribution of delta_x (norm per spacecraft, in km)
    # plt.figure()
    # for sc in range(3):
    #     plt.hist(np.linalg.norm(delta_x[:, sc], axis=-1) / 1e3, bins=30, alpha=0.5, label=f"SC{sc}")
    # plt.xlabel("Deviation of SC position [km]")
    # plt.ylabel("Count")
    # plt.legend()
    # plt.title("Distribution of SC Position Deviations")
    # plt.tight_layout()
    # plt.savefig("figures_perturbation/"+"delta_x_distribution.png", dpi=300)
    # plt.close()

    # # Plot distribution of delta_ltt (in meters)
    # plt.figure()
    # plt.hist(delta_ltt.flatten() * C_SI, bins=30, alpha=0.7, color='g')
    # plt.xlabel("Deviation in LTT [m]")
    # plt.ylabel("Count")
    # plt.title("Distribution of Light Travel Time Deviations")
    # plt.tight_layout()
    # plt.savefig("figures_perturbation/"+"delta_ltt_distribution.png", dpi=300)

    if len(t_dev) != 1:
        # time varying
        # set the last point to the same as the beginning to ensure periodicity
        delta_x[-1] = delta_x[0]
        delta_ltt[-1] = delta_ltt[0]
        x_dev = CubicSpline(t_dev, delta_x.reshape((len(t_dev), -1)))
        deviation["x"] += x_dev(time_vec).reshape((len(time_vec), 3, 3))
        deviation["v"] += x_dev.derivative()(time_vec).reshape((len(time_vec), 3, 3))
        deviation["ltt"] += CubicSpline(t_dev, delta_ltt, bc_type='natural')(time_vec)
    else:
        # static
        deviation["x"] += delta_x[0]
        deviation["v"] += 0.0
        deviation["ltt"] += delta_ltt[0]

    orb_dev.configure(t_arr=orb_dev.t_base, linear_interp_setup=False, deviation=deviation)
    return orb_dev

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    static_orb = create_orbit_with_static_dev(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=0.0, rotation_error=0.0, translation_error=0.0, dt=1000., T=1.0)
    start = time.time()
    static_orb_dev = create_orbit_with_static_dev(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=1, rotation_error=50e3, translation_error=50e3, dt=1000., T=1.0)
    end = time.time()
    print("Time to create static orbit with deviations:", end - start)
    
    periodic_orb = create_orbit_with_periodic_dev(fpath="new_orbits.h5", use_gpu=True, armlength_error=0.0, rotation_error=0.0, translation_error=0.0)
    start = time.time()
    periodic_orb_dev = create_orbit_with_periodic_dev(fpath="new_orbits.h5", use_gpu=True, armlength_error=1, rotation_error=50e3, translation_error=50e3)
    end = time.time()
    print("Time to create periodic orbit with deviations:", end - start)
    
    # first index is time and since it does not vary we just want multiple realizations
    num_deviations = 50
    delta_x_periodic = []
    delta_ltt_periodic = []
    delta_x_static = []
    delta_ltt_static = []

    # Periodic deviations
    for dev_i in range(num_deviations):
        temp_dev = create_orbit_with_periodic_dev(arm_lengths=[2.5e9, 2.5e9, 2.5e9], 
                                                  armlength_error=1, 
                                                  rotation_error=50e3, 
                                                  translation_error=50e3, 
                                                  dt=1000., T=1.0)
        ind = np.random.randint(0, len(temp_dev.x)-1)
        delta_x_periodic.append(temp_dev.x[ind] - periodic_orb.x[ind])
        delta_ltt_periodic.append(temp_dev.ltt[ind] - periodic_orb.ltt[ind])
    delta_x_periodic = np.asarray(delta_x_periodic)
    delta_ltt_periodic = np.asarray(delta_ltt_periodic)

    # Static deviations
    for dev_i in range(num_deviations):
        temp_dev = create_orbit_with_static_dev(arm_lengths=[2.5e9, 2.5e9, 2.5e9], 
                                                armlength_error=1, 
                                                rotation_error=50e3, 
                                                translation_error=50e3, 
                                                dt=1000., T=1.0)
        ind = np.random.randint(0, len(temp_dev.x)-1)
        delta_x_static.append(temp_dev.x[ind] - static_orb.x[ind])
        delta_ltt_static.append(temp_dev.ltt[ind] - static_orb.ltt[ind])
    delta_x_static = np.asarray(delta_x_static)
    delta_ltt_static = np.asarray(delta_ltt_static)

    # check deviation of spacecraft position
    plt.figure()
    sc = 0
    plt.hist(np.linalg.norm(delta_x_periodic[:, sc], axis=-1)/1e3, bins=30, alpha=.3, label=f"SC{sc} periodic")
    plt.hist(np.linalg.norm(delta_x_static[:, sc], axis=-1)/1e3, bins=30, alpha=.3, label=f"SC{sc} static")
    plt.xlabel("Deviation of SC position [km]")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures_perturbation/"+"DeltX_deviation_x_compare.png", dpi=300)
    plt.close('all')
    mean_dx_periodic, std_dx_periodic = np.mean(np.linalg.norm(delta_x_periodic[:, sc], axis=-1)), np.std(np.linalg.norm(delta_x_periodic[:, sc], axis=-1))
    mean_dx_static, std_dx_static = np.mean(np.linalg.norm(delta_x_static[:, sc], axis=-1)), np.std(np.linalg.norm(delta_x_static[:, sc], axis=-1))

    # check deviation of LTT
    plt.figure()
    plt.hist(delta_ltt_periodic.flatten()*C_SI, bins=30, density=True, alpha=0.5, label="Periodic")
    plt.hist(delta_ltt_static.flatten()*C_SI, bins=30, density=True, alpha=0.5, label="Static")
    plt.xlabel("Deviation in LTT [m]")
    plt.ylabel("Count")
    plt.axvline(1e-9*C_SI, linestyle='--', color='k')
    # link https://www.politesi.polimi.it/retrieve/f5abe8da-2e5e-4a94-9fac-358e776eb1bb/2025_04_Marchese_Executive_Summary.pdf
    plt.axvline(-1e-9*C_SI, linestyle='--', color='k', label="3-sigma Marchese Executive Summary Fig 11")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures_perturbation/"+"LTT_typical_deviation_compare.png", dpi=300)
    plt.close('all')
    # mean_ltt, std_ltt = np.mean(delta_ltt.flatten()), np.std(delta_ltt.flatten())

    for orb_dev, orb_default, name in [(periodic_orb_dev, periodic_orb, "evolving"), (static_orb_dev, static_orb, "static")]:
        plot_orbit_3d(orb_dev, T=1.0, output_file=f"{name}_orbit_3d.png")
        ##############################################################
        # coordinates
        coord_color = [(r"$x_{\rm ref} - x$ [km]", "C0"), (r"$y_{\rm ref} - y$ [km]","C1"), (r"$z_{\rm ref}- z$ [km]","C2")]
        change = "x"  # can be "ltt", "x", "n", "v"
        fig, ax = plt.subplots(3, 1, sharex=True)
        arr = getattr(orb_dev, change)
        arr_def = getattr(orb_default, change)
        for ii in range(3):
            for sc in range(3):
                ax[ii].plot(np.arange(arr.shape[0]) * orb_default.dt / 86400, 
                            (arr_def[:, sc, ii]-arr[:, sc, ii])/1e3, 
                            label=f"SC{sc}",color=coord_color[sc][1], alpha=0.5)
            ax[ii].axhline(0.0, linestyle='--', color='k')
            ax[ii].set_ylabel(coord_color[ii][0])

        ax[2].set_xlabel("Time [days]")
        plt.tight_layout()
        plt.savefig("figures_perturbation/"+name + "_" + "delta_x_y_z.png", dpi=300)
        ##############################################################
        # L12
        sc = 0
        change = "ltt"  # can be "ltt", "x", "n", "v"
        coord_color = [("$L_{12}$ [s]", "g")]
        fig, ax = plt.subplots(3, 1, sharex=True)
        arr = getattr(orb_dev, change)
        arr_def = getattr(orb_default, change)
        ii = 0
        ax[ii].set_title("Light Travel Time")
        ax[ii].plot(np.arange(arr.shape[0]) * orb_default.dt / 86400, 
            arr[:, ii], color=coord_color[sc][1], alpha=0.9, label="default")
        
        ax[ii].plot(np.arange(arr.shape[0]) * orb_default.dt / 86400, 
            arr_def[:, ii], linestyle='--', color='k',alpha=0.9, label="deviation")
        ax[ii].set_ylabel(coord_color[ii][0])
                
        ax[1].semilogy(np.arange(arr.shape[0]) * orb_default.dt / 86400, 
            np.abs(arr[:, ii]-arr_def[:, ii]), linestyle='--', color='k',alpha=0.9)
        ax[1].set_ylabel("Difference [s]")

        ax[2].semilogy(np.arange(arr.shape[0]) * orb_default.dt / 86400, 
            np.abs(1-arr[:, ii]/arr_def[:, ii]), linestyle='--', color='k',alpha=0.9)
        ax[2].set_ylabel("Relative Difference")

        ax[2].set_xlabel("Time [days]")
        ax[0].legend()
        ax[0].set_xlim(0, 300)
        plt.tight_layout()
        plt.savefig("figures_perturbation/"+name + "_" + "delta_armlength.png", dpi=300)
        ##############################################################
        plt.figure()
        sc = 0
        for sc in range(3):
            time_vec = orb_dev.t
            deviation_lof = orb_dev.x[:, sc, :] - orb_default.x[:, sc, :]
            plt.semilogy(time_vec/86400, np.linalg.norm(deviation_lof,axis=1)/1e3 ,label=f"deviation sc{sc}", alpha=0.9)
        plt.axhline(1e3, linestyle='--', color='k', label="3 sigma Reference from ESA")
        plt.xlabel("Time [days]")
        plt.legend()
        plt.ylabel("Deviation Radius [km]")
        plt.tight_layout()
        plt.savefig("figures_perturbation/"+name + "_" + "delta_radius.png", dpi=300)