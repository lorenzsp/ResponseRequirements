import numpy as np
import matplotlib.pyplot as plt
try:
    import cupy as xp
except ImportError:
    import numpy as xp
from myresponse import ResponseWrapper
from lisatools.utils.constants import YRSID_SI
from lisatools.utils.constants import C_SI
# GB functions
# create function to randomly draw parameters
def draw_parameters(A=1e-22, f=1e-3, fdot=1e-17):
    """
    Draws or assigns parameters for a gravitational wave binary system.

    Parameters
    ----------
    A : float or None, optional
        Amplitude of the signal. If None, drawn uniformly from [1e-24, 1e-22]. Default is 1e-22.
    f : float or None, optional
        Frequency of the signal (Hz). If None, drawn uniformly from [1e-4, 1e-2]. Default is 1e-3.
    fdot : float or None, optional
        Frequency derivative (Hz/s). If None, drawn uniformly from [-1e-17, 1e-17]. Default is 1e-17.

    Returns
    -------
    A : float
        Amplitude of the signal.
    f : float
        Frequency of the signal (Hz).
    fdot : float
        Frequency derivative (Hz/s).
    iota : float
        Inclination angle (radians), drawn from a uniform distribution in cos(iota).
    phi0 : float
        Initial phase (radians), drawn uniformly from [0, 2π].
    psi : float
        Polarization angle (radians), drawn uniformly from [0, 2π].
    lam : float
        Ecliptic longitude (radians), drawn from a uniform distribution in cos(lam) and shifted to [-π/2, π/2].
    beta : float
        Ecliptic latitude (radians), drawn uniformly from [0, 2π].
    """
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


class GBWave:
    def __init__(self, use_gpu=False, T=1.0, dt=10.0):

        if use_gpu:
            self.xp = xp
        else:
            self.xp = np
        
        
        self.t = self.xp.arange(0.0, T * YRSID_SI, dt)
        self.window = 1.0 # self.xp.asarray(tukey(len(self.t), alpha=0.01))

    def __call__(self, A, f, fdot, iota, phi0, psi, hp_flag=1.0, hc_flag=1.0, T=1.0, dt=10.0):

        # get the t array 
        t = self.t
        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)

        fddot = 11.0 / 3.0 * fdot ** 2 / f

        # phi0 is phi(t = 0) not phi(t = t0)
        phase = (
            2 * np.pi * (f * t + 1.0 / 2.0 * fdot * t ** 2 + 1.0 / 6.0 * fddot * t ** 3)
            - phi0
        )

        hSp = -self.xp.cos(phase) * A * (1.0 + cosiota * cosiota) * hp_flag
        hSc = -self.xp.sin(phase) * 2.0 * A * cosiota * hc_flag

        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        # redefine
        # hp = self.xp.cos(phase) * hp_flag
        # hc = self.xp.cos(phase) * hc_flag

        # Apply a Tukey window to the signal with alpha=0.01
        hp *= self.window
        hc *= self.window
        
        return hp + 1j * hc
    
    def plot_input_hp_output_A(self, A, f, fdot, iota, phi0, psi, lam, beta, Response, T=1.0, dt=10.0):
        hp = self.__call__(self, A, f, fdot, iota, phi0, psi)[0]
        chans = Response(A, f, fdot, iota, phi0, psi, lam, beta)
        chans[1] = hp
        fig, ax = plt.subplots(2, 1, sharex=True)
        for i, lab in enumerate(["A", "h_plus"]):
            ax[i].plot(np.arange(len(chans[0])) * dt / YRSID_SI, chans[i])
            ax[i].set_ylabel(lab)
        plt.savefig("TD_vars.png",dpi=300)


def get_response(orbit, T=1.0, dt=10., use_gpu=True, t0 = 10000.0):
    gb = GBWave(use_gpu=use_gpu, T=T, dt=dt)
    # default settings
    # order of the langrangian interpolation
    order = 25
    # 1st or 2nd or custom (see docs for custom)
    tdi_gen = "2nd generation"
    index_lambda = 6
    index_beta = 7
    tdi_kwargs_esa = dict(order=order, tdi=tdi_gen, tdi_chan="AET",)
    return ResponseWrapper(
    gb,
    T,
    dt,
    index_lambda,
    index_beta,
    t0=t0,
    flip_hx=False,  # set to True if waveform is h+ - ihx
    use_gpu=use_gpu,
    remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
    is_ecliptic_latitude=True,  # False if using polar angle (theta)
    remove_garbage=True,  # removes the beginning of the signal that has bad information
    orbits=orbit,
    **tdi_kwargs_esa,
    )

if __name__ == "__main__":
    from orbits_utils import ESAOrbits
    # Example usage
    A, f, fdot, iota, phi0, psi, lam, beta = draw_parameters()
    param = np.asarray([A, f, fdot, iota, phi0, psi, lam, beta])
    gb = GBWave(use_gpu=True, T=1.0, dt=10.0)
    h_strain = gb(A, f, fdot, iota, phi0, psi)
    fpath = "new_orbits.h5"
    orb_default = ESAOrbits(fpath, use_gpu=True)
    deviation = {which: np.zeros_like(getattr(orb_default, which + "_base")) for which in ["ltt", "x", "n", "v"]}
    orb_default.configure(linear_interp_setup=True, deviation=deviation)
    gb_response = get_response(orb_default)
    
    # check generate waveform is consistent with the h_strain
    h_response = gb_response.generate_waveform(*param[:-2])
    assert np.sum(h_strain - h_response)==0.0
    
    # apply the response
    AET = xp.asarray(gb_response.apply_response(h_response, param[6], param[7]))
    AET_test = xp.asarray(gb_response(*param))
    assert np.sum(AET - AET_test)==0.0
    print("My response applied and checked successfully.")
    
