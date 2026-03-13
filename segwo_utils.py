import numpy as np
from lisaorbits.utils import dot, norm, receiver, emitter, arrayindex, atleast_2d

from lisaorbits import StaticConstellation, Orbits
from lisaconstants import SUN_SCHWARZSCHILD_RADIUS, c
from segwo.response import compute_strain2link, project_covariance
from segwo.cov import construct_mixing_from_pytdi, compose_mixings, construct_covariance_from_psds, project_covariance
import scipy.interpolate
from lisaconstants.indexing import LINKS
from lisaconstants.indexing import SPACECRAFT as SC

# Plotting functions live in plot_utils; re-exported here for backward compatibility.
from plot_utils import (  # noqa: F401
    plot_response,
    plot_strain_errors,
    plot_gw_response_maps,
    plot_ltt_residuals_histogram,
    plot_position_residuals_histogram,
)

from pytdi.core import LISATDICombination
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA

X2_ETA: LISATDICombination
Y2_ETA: LISATDICombination
Z2_ETA: LISATDICombination

# Define a dictionary of TDI combinations mapping our noises into each of the 6
# single links (aka, the 6 eta variables)
ETA_COMBS: dict[int, LISATDICombination] = {}

# We start by defining it for eta 12:
#
# A PyTDI combination is defined as a dictionary, where the keys are the names
# of the input variables (N_tm_12, N_oms_12, etc.), and the values are lists of
# tuples. Each tuple contains a scaling coefficient and a list of delays to be
# applied on the corresponding variable.
ETA_COMBS[12] = LISATDICombination(
    {
        "N_tm_21": [(1, ["D_12"])],
        "N_tm_12": [(1, [])],
        "N_oms_12": [(1, [])],
    }
)

# Use PyTDI symmetry operations to define the other links

# We can do cyclic permuations, rotating indices 1->2->3->1
ETA_COMBS[23] = ETA_COMBS[12].rotated()
ETA_COMBS[31] = ETA_COMBS[23].rotated()

# We can do reflections along the axis going through the respective spacecraft,
# ie., exchanging indices 2<->3, 3<->1, and 1<->2
ETA_COMBS[13] = ETA_COMBS[12].reflected(1)
ETA_COMBS[21] = ETA_COMBS[23].reflected(2)
ETA_COMBS[32] = ETA_COMBS[31].reflected(3)

# Define a dictionary containing the TDI combinations mapping the noises into
# the eta variables, making sure that the keys map the measurement labels used
# in `X2_ETA`, `Y2_ETA`, and `Z2_ETA` (to allow composing them)
ETA_COMBS_SET = {f"eta_{mosa}": ETA_COMBS[mosa] for mosa in LINKS}

# Define list of noise labels (our input variables)
noise_list = [f"N_oms_{mosa}" for mosa in LINKS] + [f"N_tm_{mosa}" for mosa in LINKS]

# Define TDI combinations transforming XYZ into AET
A = (Z2_ETA - X2_ETA) / np.sqrt(2)
E = (X2_ETA - 2 * Y2_ETA + Z2_ETA) / np.sqrt(6)
T = (X2_ETA + Y2_ETA + Z2_ETA) / np.sqrt(3)
# ---------------------------------------------------------------------------
# TDI combinations
# ---------------------------------------------------------------------------
# Form the dictionary mapping the TDI combinations into the eta variables
xyz_eta_dict = {"X": X2_ETA, "Y": Y2_ETA, "Z": Z2_ETA}

# Compute and simplify the AET combinations for single links
A_ETA = (A @ xyz_eta_dict).simplified()
E_ETA = (E @ xyz_eta_dict).simplified()
T_ETA = (T @ xyz_eta_dict).simplified()

# Compute and simplify the AET combinations for noises
A_NOISE = (A_ETA @ ETA_COMBS_SET).simplified()
E_NOISE = (E_ETA @ ETA_COMBS_SET).simplified()
T_NOISE = (T_ETA @ ETA_COMBS_SET).simplified()

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
        self.interp_x = {sc: interpolate(self.spacecraft_positions[:, sc - 1, 0]) for sc in SC}
        self.interp_y = {sc: interpolate(self.spacecraft_positions[:, sc - 1, 1]) for sc in SC}
        self.interp_z = {sc: interpolate(self.spacecraft_positions[:, sc - 1, 2]) for sc in SC}

        if spacecraft_velocities is None:
            # Compute derivatives of spline objects for spacecraft velocities
            self.interp_vx = {sc: self.interp_x[sc].derivative() for sc in SC}
            self.interp_vy = {sc: self.interp_y[sc].derivative() for sc in SC}
            self.interp_vz = {sc: self.interp_z[sc].derivative() for sc in SC}
        else:
            # Compute spline interpolation for velocities
            self.interp_vx = {sc: interpolate(spacecraft_velocities[:, sc - 1, 0]) for sc in SC}
            self.interp_vy = {sc: interpolate(spacecraft_velocities[:, sc - 1, 1]) for sc in SC}
            self.interp_vz = {sc: interpolate(spacecraft_velocities[:, sc - 1, 2]) for sc in SC}

        # Compute spline interpolation for light travel times if provided
        if ltts is not None:
            print("Interpolating light travel times (ltts). Make sure that ltts are in the same order as LINKS:", LINKS)
            self.interp_ltt = {link: interpolate(ltts[:, ind]) for ind,link in enumerate(LINKS)}
            self.interp_ltt_deriv = {link: interpolate(ltts[:, ind]).derivative() for ind,link in enumerate(LINKS)}
        else:
            self.interp_ltt = None
        
        # Compute derivatives of spline objects for spacecraft accelerations
        self.interp_ax = {sc: self.interp_vx[sc].derivative() for sc in SC}
        self.interp_ay = {sc: self.interp_vy[sc].derivative() for sc in SC}
        self.interp_az = {sc: self.interp_vz[sc].derivative() for sc in SC}

        self.interp_dtau = {}
        self.interp_tau = {}
        self.tau_init = {}
        self.tau_t = {}
        for sc in SC:
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
    # sky_average=np.average(np.abs(b), axis=(2))[:, :, np.newaxis, :, :]
    return np.abs(a - b) / b

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


def compute_covariance(f, ltts):
    """
    Compute the noise covariance in the TDI variables for given frequencies and mixing matrix.
    Parameters
    ----------
    f : np.array
        Array of frequency values.
    ltts : np.array
        Light travel times for the constellation.
    Returns
    -------
    noise_cov_tdi_pytdi : np.array
        The noise covariance in the TDI variables, projected using the provided mixing matrix.
    """
    

    # The OMS noise is defined in terms of displacement (meters), which we convert
    # to fractional frequency shifts
    displ_2_ffd = 2 * np.pi * f / c
    oms = (15e-12) ** 2 * displ_2_ffd**2 * (1 + ((2e-3) / f) ** 4)

    # The TM noise is defined in terms of acceleration (m/s^2), which we convert to
    # fractional frequency shifts
    acc_2_ffd = 1 / (2 * np.pi * f * c)
    tm = (3e-15) ** 2 * acc_2_ffd**2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)
    
    # Construct the overall noise covariance
    noise_cov = construct_covariance_from_psds([oms] * 6 + [tm] * 6)

    # Construct the mixing matrix for the noise covariance
    # Compute the corresponding mixing matrix
    noise2aet = construct_mixing_from_pytdi(f, noise_list, [A_NOISE, E_NOISE, T_NOISE], ltts)
    
    noise_cov_tdi_pytdi = project_covariance(noise_cov, noise2aet)

    return noise_cov_tdi_pytdi


def compute_strain2x(frequencies, betas, lambs, ltts, positions):
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

