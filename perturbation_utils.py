import numpy as np
from scipy.signal import sawtooth
from orbits_utils import ESAOrbits
from lisaorbits import StaticConstellation
from lisatools.utils.constants import *
from matplotlib import pyplot as plt
try:
    import cupy as xp
except ImportError:
    import numpy as xp

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
    # plt.tight_layout()
    # plt.subplots_adjust(left=-0.4, right=1.4, top=1.4, bottom=-0.4)
    plt.savefig(output_file,dpi=300)
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

def get_periodic_variation(time_vec, t_initial=0, period=14*86400, rho=1.0):
    """
    Generate a periodic variation in the local orbital frame
    :param time_vec: time vector
    :param t_initial: initial time
    :param period: period of the variation
    :param rho: amplitude of the variation
    :return: 2D array of shape (len(time_vec), 3)
    """
    size = len(time_vec)
    periodic = sawtooth(2 * np.pi * (time_vec-t_initial)/period)
    # res =  random_vectors_on_sphere(size=size)[0] * rho * (1 + periodic[:,None])/2
    periodic = (1-np.cos(2 * np.pi * (time_vec-t_initial)/period)) / 2
    res =  random_vectors_on_sphere(size=size)[0] * rho * periodic[:,None]
    # fixed random
    res = np.ones_like(random_vectors_on_sphere(size=size)) * random_vectors_on_sphere(size=size)[0] * rho
    # random
    res = random_vectors_on_sphere(size=size) * rho
    return res

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
    porbit = get_static_variation(
        arm_lengths=arm_lengths, 
        armlength_error=armlength_error, 
        rotation_error=rotation_error, 
        translation_error=translation_error)
    porbit.write("temp_orbit.h5", dt=dt, size=int(T*YRSID_SI/dt), t0=0.0, mode="w")    
    orb = ESAOrbits("temp_orbit.h5",use_gpu=True)
    orb.configure(linear_interp_setup=True)
    return orb


def create_orbit_with_periodic_dev(delta_x, fpath, use_gpu):
    """
    Create an orb_dev object with deviations based on the given sigma.

    Parameters:
    delta_x (float): Size of the deviation.
    fpath (str): File path for the orbit data.
    use_gpu (bool): Whether to use GPU for computations.

    Returns:
    ESAOrbits: The orb_dev object with configured deviations.
    """
    # create the deviation dictionary
    orb_dev = ESAOrbits(fpath, use_gpu=use_gpu)
    deviation = {which: np.zeros_like(getattr(orb_dev, which + "_base")) for which in ["ltt", "x", "n", "v"]}
    
    # time
    time_vec = orb_dev.t_base
    xbase = orb_dev.x_base
    local_orbital_frame_pos = np.sum(xbase, axis=1) / 3
    
    # loop over spacecraft
    for sc in range(3):
        # deviation in the local orbital frame
        deviation_lof = get_periodic_variation(time_vec, t_initial=0, period=14 * 86400, rho=delta_x)
        deviation["x"][:, sc, :] += deviation_lof
        deviation["v"][:, sc, :] += np.gradient(deviation_lof, time_vec, axis=0)
    
    orb_dev.configure(linear_interp_setup=True, deviation=deviation)
    return orb_dev

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    static_orb = create_orbit_with_static_dev(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=0.0, rotation_error=0.0, translation_error=0.0, dt=10., T=1.0)
    static_orb_dev = create_orbit_with_static_dev(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=1, rotation_error=50e3, translation_error=50e3, dt=10., T=1.0)
    
    periodic_orb = create_orbit_with_periodic_dev(delta_x=0.0, fpath="new_orbits.h5", use_gpu=True)
    periodic_orb_dev = create_orbit_with_periodic_dev(delta_x=50e3, fpath="new_orbits.h5", use_gpu=True)
    
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
                            label=f"SC{sc}",color=coord_color[sc][1], alpha=0.3)
            ax[ii].axhline(0.0, linestyle='--', color='k')
            ax[ii].set_ylabel(coord_color[ii][0])

        ax[2].set_xlabel("Time [days]")
        plt.savefig(name + "_" + "delta_x_y_z.png")
        ##############################################################
        # L12
        sc = 0
        coord_color = [("$L_{12}$ [s]", "g")]
        fig, ax = plt.subplots(3, 1, sharex=True)
        arr = getattr(orb_dev, change)
        arr_def = getattr(orb_default, change)
        ii = 0
        deviation = np.linalg.norm(arr_def[:, ii]-arr_def[:, ii+1],axis=1)/C_SI
        ax[ii].plot(np.arange(arr.shape[0]) * orb_default.dt / 86400, 
                    deviation, 
                    color=coord_color[sc][1], alpha=0.3, label="default")
        
        ref = np.linalg.norm(arr[:, ii]-arr[:, ii+1],axis=1)/C_SI
        ax[ii].plot(np.arange(arr.shape[0]) * orb_default.dt / 86400, ref, linestyle='--', color='k',alpha=0.2, label="deviation")
        ax[ii].set_ylabel(coord_color[ii][0])
                
        arr = getattr(orb_dev, change)
        arr_def = getattr(orb_default, change)
        ii = 0
        deviation = np.linalg.norm(arr_def[:, ii]-arr_def[:, ii+1],axis=1) # /C_SI
        ref = np.linalg.norm(arr[:, ii]-arr[:, ii+1],axis=1) # /C_SI
        diff = np.abs(ref-deviation)
        ax[1].plot(np.arange(arr.shape[0]) * orb_default.dt / 86400, diff, linestyle='--', color='k',alpha=0.2)
        ax[1].set_ylabel("Armlength difference [m]")

        arr = getattr(orb_dev, change)
        arr_def = getattr(orb_default, change)
        ii = 0
        deviation = np.linalg.norm(arr_def[:, ii]-arr_def[:, ii+1],axis=1)/C_SI
        ref = np.linalg.norm(arr[:, ii]-arr[:, ii+1],axis=1)/C_SI
        diff = np.abs(ref-deviation)/ref
        ax[2].plot(np.arange(arr.shape[0]) * orb_default.dt / 86400, diff, linestyle='--', color='k',alpha=0.2)
        ax[2].set_ylabel("Relative")

        ax[2].set_xlabel("Time [days]")
        ax[0].legend()
        plt.savefig(name + "_" + "delta_armlength.png")
        ##############################################################
        plt.figure()
        sc = 0
        for sc in range(3):
            time_vec = orb_dev.t
            deviation_lof = orb_dev.x[:, sc, :] - orb_default.x[:, sc, :]
            plt.semilogy(time_vec/86400, np.linalg.norm(deviation_lof,axis=1)/1e3 ,label=f"deviation sc{sc}", alpha=0.5)
        plt.axhline(1e3, linestyle='--', color='k', label="3 sigma Reference from ESA")
        plt.xlabel("Time [days]")
        plt.legend()
        plt.ylabel("Deviation Radius [km]")
        plt.savefig(name + "_" + "delta_radius.png")