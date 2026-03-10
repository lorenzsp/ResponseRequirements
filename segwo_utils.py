import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from lisaorbits.utils import dot, norm, receiver, emitter, arrayindex, atleast_2d

from lisaorbits import StaticConstellation, Orbits
from lisaconstants import SUN_SCHWARZSCHILD_RADIUS, c
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA
from segwo.response import compute_strain2link
from segwo.cov import construct_mixing_from_pytdi, compose_mixings
import scipy.interpolate

#: ((3,) ndarray) Spacecraft indices.
SC = np.array([1, 2, 3])

#: ((6,) ndarray) Link (or MOSA) indices.
LINKS = np.array([12, 23, 31, 13, 32, 21])

class InterpolatedOrbits(Orbits):
    """Interpolate an array of spacecraft positions.

    Splines are used to interpolate the spacecraft positions and velocities. The
    analytic derivatives of the splines are used to compute spacecraft velocities
    and accelerations if they are not provided.

    TPS deviations are numerically integrated.

    Args:
        t_interp ((N,) array-like): interpolating TCB times (needs to be ordered) [s]
        spacecraft_positions ((N, 3, 3) array-like):
            array of spacecraft positions with dimension ``(t, sc, coordinate)`` [m]
        spacecraft_velocities ((N, 3, 3) array-like or None):
            array of spacecraft velocities with dimension ``(t, sc, coordinate)``
            [m/s], or None to compute velocities as the derivatives of the
            interpolated positions
        interp_order (int): interpolation order to be used, 3 to 5
        ext (str): extrapolation mode for elements not in the interval defined by the knot sequence
        check_input (bool):
            whether to check that input contains only finite numbers -- disabling may give
            a performance gain, but may result in problems (crashes, non-termination or invalid results)
            if input file contains infinities or ``NaNs``
        **kwargs: all other args from :class:`lisaorbits.Orbits`

    Raises:
        ValueError: if interp_order is not valid
    """

    def __init__(self,
                 t_interp,
                 spacecraft_positions,
                 spacecraft_velocities=None,
                 ltts=None,
                 interp_order=5,
                 ext='raise',
                 check_input=True,
                 **kwargs):

        super().__init__(**kwargs)

        #: array: Interpolating TCB times [s].
        self.t_interp = np.asarray(t_interp)
        #: array: Spacecraft positions [m].
        self.spacecraft_positions = np.asarray(spacecraft_positions)
        #: array: Spacecraft velocities [m/s].
        self.spacecraft_velocities = np.asarray(spacecraft_velocities) \
            if spacecraft_velocities is not None \
            else None
        #: int: Spline interpolation order.
        self.interp_order = int(interp_order)
        if self.interp_order < 3 or self.interp_order > 5:
            raise ValueError(f"Invalid value for '{self.interp_order}', must be 3, 4 or 5.")
        #: str: Extrapolation mode for elements not in the interval defined by the knot sequence.
        self.ext = str(ext)
        #: bool: Whether to check that input contains only finite numbers.
        self.check_input = bool(check_input)

        # Check t_interp, spacecraft_positions and spacecraft_velocities' shapes
        self._check_shapes()

        # pylint: disable=unnecessary-lambda-assignment
        interpolate = lambda x: scipy.interpolate.InterpolatedUnivariateSpline(
            self.t_interp, x,
            k=self.interp_order,
            ext=self.ext,
            check_finite=self.check_input
        )

        # Compute spline interpolation for positions
        self.interp_x = {sc: interpolate(self.spacecraft_positions[:, sc - 1, 0]) for sc in self.SC}
        self.interp_y = {sc: interpolate(self.spacecraft_positions[:, sc - 1, 1]) for sc in self.SC}
        self.interp_z = {sc: interpolate(self.spacecraft_positions[:, sc - 1, 2]) for sc in self.SC}

        if spacecraft_velocities is None:
            # Compute derivatives of spline objects for spacecraft velocities
            self.interp_vx = {sc: self.interp_x[sc].derivative() for sc in self.SC}
            self.interp_vy = {sc: self.interp_y[sc].derivative() for sc in self.SC}
            self.interp_vz = {sc: self.interp_z[sc].derivative() for sc in self.SC}
        else:
            # Compute spline interpolation for velocities
            self.interp_vx = {sc: interpolate(spacecraft_velocities[:, sc - 1, 0]) for sc in self.SC}
            self.interp_vy = {sc: interpolate(spacecraft_velocities[:, sc - 1, 1]) for sc in self.SC}
            self.interp_vz = {sc: interpolate(spacecraft_velocities[:, sc - 1, 2]) for sc in self.SC}

        # Compute spline interpolation for light travel times if provided
        if ltts is not None:
            print("Interpolating light travel times (ltts). Make sure that ltts are in the same order as LINKS:", self.LINKS)
            self.interp_ltt = {link: interpolate(ltts[:, ind]) for ind,link in enumerate(self.LINKS)}
        else:
            self.interp_ltt = None
        
        # Compute derivatives of spline objects for spacecraft accelerations
        self.interp_ax = {sc: self.interp_vx[sc].derivative() for sc in self.SC}
        self.interp_ay = {sc: self.interp_vy[sc].derivative() for sc in self.SC}
        self.interp_az = {sc: self.interp_vz[sc].derivative() for sc in self.SC}

        self.interp_dtau = {}
        self.interp_tau = {}
        self.tau_init = {}
        self.tau_t = {}
        for sc in self.SC:
            pos_norm = norm(self.spacecraft_positions[:, sc - 1])
            v_squared = self.interp_vx[sc](self.t_interp)**2 \
                + self.interp_vy[sc](self.t_interp)**2 \
                + self.interp_vz[sc](self.t_interp)**2
            dtau = -0.5 * (SUN_SCHWARZSCHILD_RADIUS / pos_norm + v_squared / c**2)
            self.interp_dtau[sc] = interpolate(dtau)
            # Antiderivative of dtau is integral from t_interp_0 to t, so tau(t) - tau(t_interp_0)
            # To use initial condition, we compute integral from t_init to t, which
            # is tau(t) - tau(t_init) = tau(t), but also int_{t_init}^{t_interp_0} + int_{t_interp_0}^t
            self.tau_t[sc] = self.interp_dtau[sc].antiderivative() # int_{t_interp_0}^t dtau
            self.tau_init[sc] = self.tau_t[sc](self.t_init)  # int_{t_interp_0}^{t_init} dtau
            self.interp_tau[sc] = lambda t, sc=sc: self.tau_t[sc](t) - self.tau_init[sc]

    def _write_metadata(self, hdf5):
        super()._write_metadata(hdf5)
        self._write_attr(hdf5, 'interp_order', 'ext', 'check_input')

    def _check_shapes(self):
        """Check array shapes.

        We check that ``t_interp`` is of shape (N,), and ``spacecraft_positions`` and
        ``spacecraft_velocities`` (if not None) are of shape (N, 3, 3).

        Raises:
            ValueError: if the shapes are invalid.
        """
        if len(self.t_interp.shape) != 1:
            raise ValueError(f"time array has shape {self.t_interp.shape}, must be (N).")

        size = self.t_interp.shape[0]
        if len(self.spacecraft_positions.shape) != 3 or \
           self.spacecraft_positions.shape[0] != size or \
           self.spacecraft_positions.shape[1] != 3 or \
           self.spacecraft_positions.shape[2] != 3:
            raise ValueError(
                f"spacecraft position array has shape "
                f"{self.spacecraft_positions.shape}, expected ({size}, 3, 3).")
        if self.spacecraft_velocities is not None and (
                len(self.spacecraft_velocities.shape) != 3 or \
                self.spacecraft_velocities.shape[0] != size or \
                self.spacecraft_velocities.shape[1] != 3 or \
                self.spacecraft_velocities.shape[2] != 3
            ):
            raise ValueError(
                f"spacecraft velocity array has shape "
                f"{self.spacecraft_velocities.shape}, expected ({size}, 3, 3).")

    @staticmethod
    def _broadcast(t, sc):
        """Broadcast t to have compatible shape with sc.

        Add a second axis to t if necessary, and broadcast to sc's shape.

        Args:
            t ((N,) or (N, M) array-like): TCB times [s]
            sc ((M,) array-like): spacecraft indices

        Returns:
            tuple: The broadcasted time array and length of second axis ``(t, n)``.
        """
        t = atleast_2d(t)
        broad_t, _ = np.broadcast_arrays(t, sc)
        return broad_t, broad_t.shape[1]

    def compute_position(self, t, sc=SC):
        t, n = self._broadcast(t, sc)
        sc_x = np.stack([self.interp_x[sc[i]](t[:, i]) for i in range(n)], axis=-1) # (N, M)
        sc_y = np.stack([self.interp_y[sc[i]](t[:, i]) for i in range(n)], axis=-1) # (N, M)
        sc_z = np.stack([self.interp_z[sc[i]](t[:, i]) for i in range(n)], axis=-1) # (N, M)

        return np.stack([sc_x, sc_y, sc_z], axis=-1) # (N, M, 3)

    def compute_velocity(self, t, sc=SC):
        t, n = self._broadcast(t, sc)
        sc_vx = np.stack([self.interp_vx[sc[i]](t[:, i]) for i in range(n)], axis=-1) # (N, M)
        sc_vy = np.stack([self.interp_vy[sc[i]](t[:, i]) for i in range(n)], axis=-1) # (N, M)
        sc_vz = np.stack([self.interp_vz[sc[i]](t[:, i]) for i in range(n)], axis=-1) # (N, M)

        return np.stack([sc_vx, sc_vy, sc_vz], axis=-1) # (N, M, 3)

    def compute_acceleration(self, t, sc=SC):
        t, n = self._broadcast(t, sc)
        sc_ax = np.stack([self.interp_ax[sc[i]](t[:, i]) for i in range(n)], axis=-1) # (N, M)
        sc_ay = np.stack([self.interp_ay[sc[i]](t[:, i]) for i in range(n)], axis=-1) # (N, M)
        sc_az = np.stack([self.interp_az[sc[i]](t[:, i]) for i in range(n)], axis=-1) # (N, M)

        return np.stack([sc_ax, sc_ay, sc_az], axis=-1) # (N, M, 3)

    def compute_tps_deviation(self, t, sc=SC):
        t, n = self._broadcast(t, sc)
        return np.stack([self.interp_tau[sc[i]](t[:, i]) for i in range(n)], axis=-1) # (N, M)

    def compute_tps_deviation_derivative(self, t, sc=SC):
        t, n = self._broadcast(t, sc)
        return np.stack([self.interp_dtau[sc[i]](t[:, i]) for i in range(n)], axis=-1)  # (N, M)

    def compute_ltt(self, t, link=LINKS):
        """Compute light travel times (LTTs).

        Light travel times are the differences between the TCB time of reception
        of a photon at one spacecraft, and the TCB time of emission of the same photon
        by another spacecraft.

        The default implementation calls :meth:`lisaorbits.Orbits._compute_ltt_analytic`
        or :meth:`lisaorbits.Orbits._compute_ltt_iterative` depending on the value
        of :attr:`lisaorbits.Orbits.tt_method`.

        Subclasses can override this method with custom implementations.

        Args:
            t ((N,) or (N, M) array-like): TCB times [s]
            link ((M,) array-like): link indices

        Returns:
            (N, M) ndarray: Light travel times [s].

        Raises:
            ValueError: if the computation method is invalid
        """
        if self.interp_ltt is not None:
            t, n = self._broadcast(t, link)
            return np.stack([self.interp_ltt[link[i]](t[:, i]) for i in range(n)], axis=-1) # (N, M)
        if self.tt_method == 'analytic':
            return self._compute_ltt_analytic(t, link) # (N, M)
        if self.tt_method == 'iterative':
            return self._compute_ltt_iterative(t, link) # (N, M)
        raise ValueError(f"Invalid light travel time computation method '{self.tt_method}', "
                         "must be 'analytic' or 'iterative'.")


def relative_errors_sky(a, b):
    """ Compute the relative errors between two arrays, 
        wrt. the direction of maximum amplitude
    """
    return np.abs(a - b) / np.abs(b) # np.average(np.abs(b), axis=(1,2))[:, np.newaxis, np.newaxis, :, :]

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
    # methods "baghi+23" and "hartwig+23" are available, the latter is more accurate but also more computationally expensive
    strain2link = compute_strain2link(frequencies, betas, lambs, ltts, positions, method="baghi+23")
    # Shape: times, frequencies, pixels, links, polarizations
    # Ie., it's the mixing matrix to go from h+/hx in terms of strain to single link
    # response in terms of fractional frequency deviation. See segwo documentation for details.
    strain2link.shape

    # Define eta variables for the links
    eta_list = [f"eta_{mosa}" for mosa in LINKS]

    # Construct the mixing matrix from pytdi
    link2x = construct_mixing_from_pytdi(frequencies, eta_list, [A, E, T], ltts)

    # Add a new axis for each pixel
    link2x = link2x[:, :, np.newaxis, :, :]

    # Compose the mixing matrices to go directly from strain to TDI variables
    strain2x = compose_mixings([strain2link, link2x])

    return strain2x



def plot_response(f, npix, strain2x_abs, pols = ['h+', 'hx'], folder="", output_file="strain2x.png", metric="min"):
    """
    Plots the relative errors in |R| and angle for strain2x and saves the figure.

    Parameters:
        f (array): Frequency array.
        npix (int): Number of pixels in the sky map.
        strain2x_abs (array): Absolute value of strain2x array.
        pols (list): List of polarization labels.
        output_file (str): File name to save the plot.
        metric (str): Metric to use for error calculation ("max", "mean").
    """
    if metric == "max":
        metric_func = np.max
    elif metric == "mean":
        metric_func = np.mean
    elif metric == "min":
        metric_func = np.min
    else:
        raise ValueError("Invalid metric. Use 'max', 'mean', or 'min'.")

    fig, axs = plt.subplots(1, 1)

    # Plot relative error in |R|
    for i in range(3):
        for j in range(2):
            axs.loglog(f, metric_func(strain2x_abs[:, :, i, j], axis=1), label=f'{metric_func.__name__} TDI {"AET"[i]}, {pols[j]}')
    axs.set_xlabel("Frequency [Hz]")
    axs.set_ylabel("Amplitude")
    axs.legend()

    plt.tight_layout()
    plt.savefig(folder + output_file, dpi=300)
    plt.close()
    
    
    for link in range(3):
        for pol in range(2):
            plt.figure()
            gw_response_map = np.zeros(npix)

            # Populate the map with the GW response for each pixel
            for pix in range(npix):
                max_sky = metric_func(strain2x_abs[:, pix, link, pol], axis=0)
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

