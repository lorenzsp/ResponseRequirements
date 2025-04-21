from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
from dataclasses import dataclass
import requests
from copy import deepcopy
import h5py
try:
    import cupy as xp
except ImportError:
    import numpy as xp
from scipy import interpolate
from lisatools.utils.constants import *
from lisatools.utils.utility import get_array_module
from lisatools.detector import Orbits
import numpy as np
np.random.seed(2601)
# import for cpu/gpu
from lisatools.cutils.detector_cpu import pycppDetector as pycppDetector_cpu

try:
    import cupy as cp
    from lisatools.cutils.detector_gpu import pycppDetector as pycppDetector_gpu

except (ImportError, ModuleNotFoundError) as e:
    pycppDetector_gpu = None  # for doc string purposes


SC = [1, 2, 3]
LINKS = [12, 23, 31, 13, 32, 21]

LINEAR_INTERP_TIMESTEP = 600.00  # sec (0.25 hr)

# create function to randomly draw parameters
def draw_parameters(A=1e-22, f=1e-3, fdot=1e-17):
    if A is None:
        A = np.random.uniform(1e-24, 1e-22)
    if f is None:
        f = np.random.uniform(1e-4, 1e-2)
    if fdot is None:
        fdot = np.random.uniform(-1e-17, 1e-17)
    # draw random parameters
    cos_inc = np.random.uniform(-1, 1)  # Random cosines of polar angles
    iota = np.arccos(cos_inc) #np.random.uniform(0, np.pi)
    phi0 = np.random.uniform(0, 2 * np.pi)
    psi = np.random.uniform(0, 2 * np.pi)
    cos_lam = np.random.uniform(-1, 1)  # Random cosines of polar angles
    lam = np.arccos(cos_lam)-np.pi/2 # np.random.uniform(-np.pi / 2, np.pi / 2)
    beta = np.random.uniform(0.0, 2 * np.pi)
    return A, f, fdot, iota, phi0, psi, lam, beta


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

class GBWave:
    def __init__(self, use_gpu=False):

        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0):

        # get the t array 
        t = self.xp.arange(0.0, T * YRSID_SI, dt)
        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)

        fddot = 11.0 / 3.0 * fdot ** 2 / f

        # phi0 is phi(t = 0) not phi(t = t0)
        phase = (
            2 * np.pi * (f * t + 1.0 / 2.0 * fdot * t ** 2 + 1.0 / 6.0 * fddot * t ** 3)
            - phi0
        )

        hSp = -self.xp.cos(phase) * A * (1.0 + cosiota * cosiota)
        hSc = -self.xp.sin(phase) * 2.0 * A * cosiota

        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        return hp + 1j * hc
    
    def plot_input_hp_output_A(self, A, f, fdot, iota, phi0, psi, lam, beta, Response, T=1.0, dt=10.0):
        hp = self.__call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0)[0]
        chans = Response(A, f, fdot, iota, phi0, psi, lam, beta)
        chans[1] = hp
        fig, ax = plt.subplots(2, 1, sharex=True)
        for i, lab in enumerate(["A", "h_plus"]):
            ax[i].plot(np.arange(len(chans[0])) * dt / YRSID_SI, chans[i])
            ax[i].set_ylabel(lab)
        plt.savefig("TD_vars.png")



class MyOrbits(Orbits):
    """LISA Orbit Base Class

    Args:
        filename: File name. File should be in the style of LISAOrbits
        use_gpu: If ``True``, use a gpu.
        armlength: Armlength of detector.

    """

    def __init__(
        self,
        filename: str,
        use_gpu: bool = False,
        armlength: Optional[float] = 2.5e9,
    ) -> None:
        self.use_gpu = use_gpu
        self.filename = filename
        self.armlength = armlength
        self._setup()
        self.configured = False

    @property
    def xp(self):
        """numpy or cupy based on self.use_gpu"""
        xp = np if not self.use_gpu else cp
        return xp

    @property
    def armlength(self) -> float:
        """Armlength parameter."""
        return self._armlength

    @armlength.setter
    def armlength(self, armlength: float) -> None:
        """armlength setter."""

        if isinstance(armlength, float):
            # TODO: put error check that it is close
            self._armlength = armlength

        else:
            raise ValueError("armlength must be float.")

    @property
    def LINKS(self) -> List[int]:
        """Link order."""
        return LINKS

    @property
    def SC(self) -> List[int]:
        """Spacecraft order."""
        return SC

    @property
    def link_space_craft_r(self) -> List[int]:
        """Receiver (first) spacecraft"""
        return [int(str(link_i)[0]) for link_i in self.LINKS]

    @property
    def link_space_craft_e(self) -> List[int]:
        """Sender (second) spacecraft"""
        return [int(str(link_i)[1]) for link_i in self.LINKS]

    def _setup(self) -> None:
        """Read in orbital data from file and store."""
        with self.open() as f:
            for key in f.attrs.keys():
                setattr(self, key + "_base", f.attrs[key])

    @property
    def filename(self) -> str:
        """Orbit file name."""
        return self._filename

    @filename.setter
    def filename(self, filename: str) -> None:
        """Set file name."""

        assert isinstance(filename, str)

        if os.path.exists(filename):
            self._filename = filename

        else:
            # get path
            path_to_this_file = __file__.split("detector.py")[0]

            # make sure orbit_files directory exists in the right place
            if not os.path.exists(path_to_this_file + "orbit_files/"):
                os.mkdir(path_to_this_file + "orbit_files/")
            path_to_this_file = path_to_this_file + "orbit_files/"

            if not os.path.exists(path_to_this_file + filename):
                # download files from github if they are not there
                github_file = f"https://github.com/mikekatz04/LISAanalysistools/raw/main/lisatools/orbit_files/{filename}"
                r = requests.get(github_file)

                # if not success
                if r.status_code != 200:
                    raise ValueError(
                        f"Cannot find {filename} within default files located at github.com/mikekatz04/LISAanalysistools/lisatools/orbit_files/."
                    )
                # write the contents to a local file
                with open(path_to_this_file + filename, "wb") as f:
                    f.write(r.content)

            # store
            self._filename = path_to_this_file + filename

    def open(self) -> h5py.File:
        """Opens the h5 file in the proper mode.

        Returns:
            H5 file object: Opened file.

        Raises:
            RuntimeError: If backend is opened for writing when it is read-only.

        """
        f = h5py.File(self.filename, "r")
        return f

    @property
    def t_base(self) -> np.ndarray:
        """Time array from file."""
        with self.open() as f:
            t_base = np.arange(self.size_base) * self.dt_base
        return t_base

    @property
    def ltt_base(self) -> np.ndarray:
        """Light travel times along links from file."""
        with self.open() as f:
            ltt = f["tcb"]["ltt"][:]
        return ltt

    @property
    def n_base(self) -> np.ndarray:
        """Normal unit vectors towards receiver along links from file."""
        with self.open() as f:
            n = f["tcb"]["n"][:]
        return n

    @property
    def x_base(self) -> np.ndarray:
        """Spacecraft position from file."""
        with self.open() as f:
            x = f["tcb"]["x"][:]
        return x

    @property
    def v_base(self) -> np.ndarray:
        """Spacecraft velocities from file."""
        with self.open() as f:
            v = f["tcb"]["v"][:]
        return v

    @property
    def t(self) -> np.ndarray:
        """Configured time array."""
        self._check_configured()
        return self._t

    @t.setter
    def t(self, t: np.ndarray):
        """Set configured time array."""
        assert isinstance(t, np.ndarray) and t.ndim == 1
        self._t = t

    @property
    def ltt(self) -> np.ndarray:
        """Light travel time."""
        self._check_configured()
        return self._ltt

    @ltt.setter
    def ltt(self, ltt: np.ndarray) -> np.ndarray:
        """Set light travel time."""
        assert ltt.shape[0] == len(self.t)

    @property
    def n(self) -> np.ndarray:
        """Normal vectors along links."""
        self._check_configured()
        return self._n

    @n.setter
    def n(self, n: np.ndarray) -> np.ndarray:
        """Set Normal vectors along links."""
        return self._n

    @property
    def x(self) -> np.ndarray:
        """Spacecraft positions."""
        self._check_configured()
        return self._x

    @x.setter
    def x(self, x: np.ndarray) -> np.ndarray:
        """Set Spacecraft positions."""
        return self._x

    @property
    def v(self) -> np.ndarray:
        """Spacecraft velocities."""
        self._check_configured()
        return self._v

    @v.setter
    def v(self, v: np.ndarray) -> np.ndarray:
        """Set Spacecraft velocities."""
        return self._v

    def configure(
        self,
        t_arr: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        linear_interp_setup: Optional[bool] = False,
        deviation: Optional[dict] = None,
    ) -> None:
        """Configure the orbits to match the signal response generator time basis.

        The base orbits will be scaled up or down as needed using Cubic Spline interpolation.
        The higherarchy of consideration to each keyword argument if multiple are given:
        ``linear_interp_setup``, ``t_arr``, ``dt``.

        If nothing is provided, the base points are used.

        Args:
            t_arr: New time array.
            dt: New time step. Will take the time duration to be that of the input data.
            linear_interp_setup: If ``True``, it will create a dense grid designed for linear interpolation with a constant time step.

        """
        self.deviation = deviation
        x_orig = self.t_base

        # everything up base on input
        if linear_interp_setup:
            # setup spline
            make_cpp = True
            dt = LINEAR_INTERP_TIMESTEP
            Tobs = self.t_base[-1]
            Nobs = int(Tobs / dt)
            t_arr = np.arange(Nobs) * dt
            if t_arr[-1] < self.t_base[-1]:
                t_arr = np.concatenate([t_arr, self.t_base[-1:]])
        elif t_arr is not None:
            # check array inputs and fill dt
            assert np.all(t_arr >= self.t_base[0]) and np.all(t_arr <= self.t_base[-1])
            make_cpp = True
            dt = abs(t_arr[1] - t_arr[0])

        elif dt is not None:
            # fill array based on dt and base t
            make_cpp = True
            Tobs = self.t_base[-1]
            Nobs = int(Tobs / dt)
            t_arr = np.arange(Nobs) * dt
            if t_arr[-1] < self.t_base[-1]:
                t_arr = np.concatenate([t_arr, self.t_base[-1:]])

        else:
            make_cpp = False
            t_arr = self.t_base

        x_new = t_arr.copy()
        self.t = t_arr.copy()

        # use base quantities, and interpolate to prepare new arrays accordingly
        for which in ["ltt", "x", "n", "v"]:
            arr = getattr(self, which + "_base")
            if self.deviation is not None:
                arr += self.deviation[which]
            arr_tmp = arr.reshape(self.size_base, -1)
            arr_out_tmp = np.zeros((len(x_new), arr_tmp.shape[-1]))
            for i in range(arr_tmp.shape[-1]):
                arr_out_tmp[:, i] = interpolate.CubicSpline(x_orig, arr_tmp[:, i])(
                    x_new
                )
            arr_out = arr_out_tmp.reshape((len(x_new),) + arr.shape[1:])
            setattr(self, "_" + which, arr_out)

        # make sure base spacecraft and link inormation is ready
        lsr = np.asarray(self.link_space_craft_r).copy().astype(np.int32)
        lse = np.asarray(self.link_space_craft_e).copy().astype(np.int32)
        ll = np.asarray(self.LINKS).copy().astype(np.int32)

        # indicate this class instance has been configured
        self.configured = True

        # prepare cpp class args to load when needed
        if make_cpp:
            self.pycppdetector_args = [
                dt,
                len(self.t),
                self.xp.asarray(self.n.flatten().copy()),
                self.xp.asarray(self.ltt.flatten().copy()),
                self.xp.asarray(self.x.flatten().copy()),
                self.xp.asarray(ll),
                self.xp.asarray(lsr),
                self.xp.asarray(lse),
                self.armlength,
            ]
            self.dt = dt
        else:
            self.pycppdetector_args = None
            self.dt = dt

    @property
    def dt(self) -> float:
        """new time step if it exists"""
        if self._dt is None:
            raise ValueError("dt not available for t_arr only.")
        return self._dt

    @dt.setter
    def dt(self, dt: float) -> None:
        self._dt = dt

    @property
    def pycppdetector(self) -> pycppDetector_cpu | pycppDetector_gpu:
        """C++ class"""
        if self._pycppdetector_args is None:
            raise ValueError(
                "Asking for c++ class. Need to set linear_interp_setup = True when configuring."
            )
        pycppDetector = pycppDetector_cpu if not self.use_gpu else pycppDetector_gpu
        self._pycppdetector = pycppDetector(*self._pycppdetector_args)
        return self._pycppdetector

    @property
    def pycppdetector_args(self) -> tuple:
        """args for the c++ class."""
        return self._pycppdetector_args

    @pycppdetector_args.setter
    def pycppdetector_args(self, pycppdetector_args: tuple) -> None:
        self._pycppdetector_args = pycppdetector_args

    @property
    def size(self) -> int:
        """Number of time points."""
        self._check_configured()
        return len(self.t)

    def _check_configured(self) -> None:
        if not self.configured:
            raise ValueError(
                "Cannot request property. Need to use configure() method first."
            )

    def get_light_travel_times(
        self, t: float | np.ndarray, link: int | np.ndarray
    ) -> float | np.ndarray:
        """Compute light travel time as a function of time.

        Computes with the c++ backend.

        Args:
            t: Time array in seconds.
            link: which link. Must be ``in self.LINKS``.

        Returns:
            Light travel times.

        """
        # test and prepare inputs
        if isinstance(t, float) and isinstance(link, int):
            squeeze = True
            t = self.xp.atleast_1d(t)
            link = self.xp.atleast_1d(link).astype(np.int32)

        elif isinstance(t, self.xp.ndarray) and isinstance(link, int):
            squeeze = False
            t = self.xp.atleast_1d(t)
            link = self.xp.full_like(t, link, dtype=np.int32)

        elif isinstance(t, self.xp.ndarray) and isinstance(link, self.xp.ndarray):
            squeeze = False
            t = self.xp.asarray(t)
            link = self.xp.asarray(link).astype(np.int32)
        else:
            raise ValueError(
                "(t, link) can be (float, int), (np.ndarray, int), (np.ndarray, np.ndarray)."
            )

        # buffer array and c computation
        ltt_out = self.xp.zeros_like(t)
        self.pycppdetector.get_light_travel_time_arr_wrap(
            ltt_out, t, link, len(ltt_out)
        )

        # prepare output
        if squeeze:
            return ltt_out[0]
        return ltt_out

    def get_pos(self, t: float | np.ndarray, sc: int | np.ndarray) -> np.ndarray:
        """Compute light travel time as a function of time.

        Computes with the c++ backend.

        Args:
            t: Time array in seconds.
            sc: which spacecraft. Must be ``in self.SC``.

        Returns:
            Position of spacecraft.

        """
        # test and setup inputs accordingly
        if isinstance(t, float) and isinstance(sc, int):
            squeeze = True
            t = self.xp.atleast_1d(t)
            sc = self.xp.atleast_1d(sc).astype(np.int32)

        elif isinstance(t, self.xp.ndarray) and isinstance(sc, int):
            squeeze = False
            t = self.xp.atleast_1d(t)
            sc = self.xp.full_like(t, sc, dtype=np.int32)

        elif isinstance(t, self.xp.ndarray) and isinstance(sc, self.xp.ndarray):
            squeeze = False
            t = self.xp.asarray(t)
            sc = self.xp.asarray(sc).astype(np.int32)

        else:
            raise ValueError(
                "(t, sc) can be (float, int), (np.ndarray, int), (np.ndarray, np.ndarray). If the inputs follow this, make sure the orbits class GPU setting matches the arrays coming in (GPU or CPU)."
            )

        # buffer arrays for input into c code
        pos_x = self.xp.zeros_like(t)
        pos_y = self.xp.zeros_like(t)
        pos_z = self.xp.zeros_like(t)

        # c code computation
        self.pycppdetector.get_pos_arr_wrap(pos_x, pos_y, pos_z, t, sc, len(pos_x))

        # prepare output
        output = self.xp.array([pos_x, pos_y, pos_z]).T
        if squeeze:
            return output.squeeze()
        return output

    def get_normal_unit_vec(
        self, t: float | np.ndarray, link: int | np.ndarray
    ) -> np.ndarray:
        """Compute link normal vector as a function of time.

        Computes with the c++ backend.

        Args:
            t: Time array in seconds.
            link: which link. Must be ``in self.LINKS``.

        Returns:
            Link normal vectors.

        """
        # test and prepare inputs
        if isinstance(t, float) and isinstance(link, int):
            squeeze = True
            t = self.xp.atleast_1d(t)
            link = self.xp.atleast_1d(link).astype(np.int32)

        elif isinstance(t, self.xp.ndarray) and isinstance(link, int):
            squeeze = False
            t = self.xp.atleast_1d(t)
            link = self.xp.full_like(t, link, dtype=np.int32)

        elif isinstance(t, self.xp.ndarray) and isinstance(link, self.xp.ndarray):
            squeeze = False
            t = self.xp.asarray(t)
            link = self.xp.asarray(link).astype(np.int32)
        else:
            raise ValueError(
                "(t, link) can be (float, int), (np.ndarray, int), (np.ndarray, np.ndarray)."
            )

        # c code with buffers
        normal_unit_vec_x = self.xp.zeros_like(t)
        normal_unit_vec_y = self.xp.zeros_like(t)
        normal_unit_vec_z = self.xp.zeros_like(t)

        # c code
        self.pycppdetector.get_normal_unit_vec_arr_wrap(
            normal_unit_vec_x,
            normal_unit_vec_y,
            normal_unit_vec_z,
            t,
            link,
            len(normal_unit_vec_x),
        )

        # prep outputs
        output = self.xp.array(
            [normal_unit_vec_x, normal_unit_vec_y, normal_unit_vec_z]
        ).T
        if squeeze:
            return output.squeeze()
        return output

    @property
    def ptr(self) -> int:
        """pointer to c++ class"""
        return self.pycppdetector.ptr


class EqualArmlengthOrbits(MyOrbits):
    """Equal Armlength Orbits

    Orbit file: equalarmlength-orbits.h5

    Args:
        *args: Arguments for :class:`Orbits`.
        **kwargs: Kwargs for :class:`Orbits`.

    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__("equalarmlength-orbits.h5", *args, **kwargs)

class DefaultOrbits(EqualArmlengthOrbits):
    """Set default orbit class to Equal Armlength orbits for now."""

    pass


class ESAOrbits(MyOrbits):
    """ESA Orbits

    Orbit file: esa-trailing-orbits.h5

    Args:
        *args: Arguments for :class:`Orbits`.
        **kwargs: Kwargs for :class:`Orbits`.

    """

    def __init__(self, fpath, *args, **kwargs):
        # breakpoint()
        # super().__init__("esa-trailing-orbits.h5", *args, **kwargs)
        super().__init__(fpath, *args, **kwargs)

from matplotlib import pyplot as plt
def plot_orbit_3d(fpath, T, Nshow=10, lam=None, beta=None, output_file="3d_orbit_around_sun.png", scatter_points=None):
    """
    Plot the 3D orbit of the spacecraft around the Sun.

    Args:
        fpath: Path to the orbit file.
        T: Time duration in years.
        Nshow: Number of points to show.
        lam: Longitude of the source in radians.
        beta: Latitude of the source in radians.
        output_file: Output file name for the plot.
        scatter_points: List of tuples [(radius, lam, beta, color_function), ...] for additional 3D scatter points.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    max_r = 0.0

    orb_default = ESAOrbits(fpath)
    orb_default.configure(linear_interp_setup=True)
    orbital_info = {which: getattr(orb_default, which + "_base") for which in ["ltt", "x", "n", "v"]}
    orbital_info["t"] = orb_default.t_base

    mask = (orb_default.t_base - orb_default.t_base.min() < 86400 * 365 * T)
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
    ax.scatter(0.0, 0.0, 0.0, color='orange', marker='o', label="Sun", s=300)

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
    plt.savefig(output_file)
    print(f"3D orbit plot saved to {output_file}")

########################################
@dataclass
class LISAModelSettings:
    """Required LISA model settings:

    Args:
        Soms_d: OMS displacement noise.
        Sa_a: Acceleration noise.
        orbits: Orbital information.
        name: Name of model.

    """

    Soms_d: float
    Sa_a: float
    orbits: Orbits
    name: str


class LISAModel(LISAModelSettings, ABC):
    """Model for the LISA Constellation

    This includes sensitivity information computed in
    :py:mod:`lisatools.sensitivity` and orbital information
    contained in an :class:`Orbits` class object.

    This class is used to house high-level methods useful
    to various needed computations.

    """

    def __str__(self) -> str:
        out = "LISA Constellation Configurations Settings:\n"
        for key, item in self.__dict__.items():
            out += f"{key}: {item}\n"
        return out

    def lisanoises(
        self,
        f: float | np.ndarray,
        unit: Optional[str] = "relative_frequency",
    ) -> Tuple[float, float]:
        """Calculate both LISA noise terms based on input model.
        Args:
            f: Frequency array.
            unit: Either ``"relative_frequency"`` or ``"displacement"``.
        Returns:
            Tuple with acceleration term as first value and oms term as second value.
        """

        # TODO: fix this up
        Soms_d_in = self.Soms_d
        Sa_a_in = self.Sa_a

        frq = f
        ### Acceleration noise
        ## In acceleration
        Sa_a = Sa_a_in * (1.0 + (0.4e-3 / frq) ** 2) * (1.0 + (frq / 8e-3) ** 4)
        ## In displacement
        Sa_d = Sa_a * (2.0 * np.pi * frq) ** (-4.0)
        ## In relative frequency unit
        Sa_nu = Sa_d * (2.0 * np.pi * frq / C_SI) ** 2
        Spm = Sa_nu

        ### Optical Metrology System
        ## In displacement
        Soms_d = Soms_d_in * (1.0 + (2.0e-3 / f) ** 4)
        ## In relative frequency unit
        Soms_nu = Soms_d * (2.0 * np.pi * frq / C_SI) ** 2
        Sop = Soms_nu

        if unit == "displacement":
            return Sa_d, Soms_d
        elif unit == "relative_frequency":
            return Spm, Sop


# defaults
scirdv1 = LISAModel((15.0e-12) ** 2, (3.0e-15) ** 2, DefaultOrbits(), "scirdv1")
proposal = LISAModel((10.0e-12) ** 2, (3.0e-15) ** 2, DefaultOrbits(), "proposal")
mrdv1 = LISAModel((10.0e-12) ** 2, (2.4e-15) ** 2, DefaultOrbits(), "mrdv1")
sangria = LISAModel((7.9e-12) ** 2, (2.4e-15) ** 2, DefaultOrbits(), "sangria")

__stock_list_models__ = [scirdv1, proposal, mrdv1, sangria]
__stock_list_models_name__ = [tmp.name for tmp in __stock_list_models__]


def get_available_default_lisa_models() -> List[LISAModel]:
    """Get list of default LISA models

    Returns:
        List of LISA models.

    """
    return __stock_list_models__


def get_default_lisa_model_from_str(model: str) -> LISAModel:
    """Return a LISA model from a ``str`` input.

    Args:
        model: Model indicated with a ``str``.

    Returns:
        LISA model associated to that ``str``.

    """
    if model not in __stock_list_models_name__:
        raise ValueError(
            "Requested string model is not available. See lisatools.detector documentation."
        )
    return globals()[model]


def check_lisa_model(model: Any) -> LISAModel:
    """Check input LISA model.

    Args:
        model: LISA model to check.

    Returns:
        LISA Model checked. Adjusted from ``str`` if ``str`` input.

    """
    if isinstance(model, str):
        model = get_default_lisa_model_from_str(model)

    if not isinstance(model, LISAModel):
        raise ValueError("model argument not given correctly.")

    return model
