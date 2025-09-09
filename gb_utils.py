import numpy as np
import matplotlib.pyplot as plt
try:
    import cupy as xp
except ImportError:
    import numpy as xp
from myresponse import ResponseWrapper
from lisatools.utils.constants import YRSID_SI, C_SI
from lisatools.utils.constants import MTSUN_SI as MSUN_SEC
import time
from scipy.signal.windows import tukey

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

import numpy as np

class BHWave:
    def __init__(self, use_gpu=False, T=1.0, dt=10.0, window=True):
        if use_gpu:
            import cupy
            self.xp = cupy
        else:
            self.xp = np
        
        self.index_lambda = 7
        self.index_beta = 8
        self.PARSEC_SEC = 1.0292712503e8
        
        self.t = self.xp.arange(0.0, T * YRSID_SI, dt)
        if window:
            self.window = self.xp.asarray(tukey(len(self.t), alpha=0.01))
        else:
            self.window = 1.0

    def __call__(self, Mc, eta, iota, phic, tc, DL, psi, hp_flag=1.0, hc_flag=1.0, T=1.0, dt=10.0):
        xp = self.xp
        t = self.t
        
        # Compute total mass m from Mc and eta
        m = Mc / (eta ** (3.0 / 5.0))
        
        # Convert to geometric units (seconds)
        m_sec = m * MSUN_SEC
        eta_sec = eta
        tc_sec = tc * YRSID_SI
        dl_sec = DL * 1e9 * self.PARSEC_SEC
        
        # Theta(t)
        Theta = eta * (tc_sec - t) / (5.0 * m_sec)
        
        # Mask where Theta > 0
        mask = Theta > 0
        
        Phi = xp.zeros_like(t)
        omega = xp.zeros_like(t)
        
        if xp.any(mask):
            th = Theta[mask]
            
            # For phase
            th58 = th ** (5.0 / 8.0)
            th38 = th ** (3.0 / 8.0)
            th14 = th ** (1.0 / 4.0)
            th18 = th ** (1.0 / 8.0)
            coeff_phase1 = 3715.0 / 8064.0 + 55.0 / 96.0 * eta
            coeff_phase2 = 9275495.0 / 14450688.0 + 284875.0 / 258048.0 * eta + 1855.0 / 2048.0 * eta**2
            Phi[mask] = phic - 2.0 / eta * (th58 + coeff_phase1 * th38 - 3.0 * np.pi / 4.0 * th14 + coeff_phase2 * th18)
            
            # For omega
            th_m38 = th ** (-3.0 / 8.0)
            th_m58 = th ** (-5.0 / 8.0)
            th_m34 = th ** (-3.0 / 4.0)
            th_m78 = th ** (-7.0 / 8.0)
            coeff_om1 = 743.0 / 2688.0 + 11.0 / 32.0 * eta
            coeff_om2 = 1855099.0 / 14450688.0 + 56975.0 / 258048.0 * eta + 371.0 / 2048.0 * eta**2
            omega[mask] = 1.0 / (8.0 * m_sec) * (th_m38 + coeff_om1 * th_m58 - 3.0 * np.pi / 10.0 * th_m34 + coeff_om2 * th_m78)

        omega[omega<0.0] = 0.0
        # x(t)
        x = (m_sec * omega) ** (2.0 / 3.0)
        # close to zero 1-self.xp.gradient(Phi,self.t) / (2 * omega)
        # frequency evolution
        f_ev = omega / self.xp.pi
        # get minimum and maximum frequency
        mask_diff_zero = omega > 0.0
        self.fmin = f_ev[mask_diff_zero].min()
        self.fmax = f_ev[mask_diff_zero].max()

        # Amplitudes
        cosi = xp.cos(iota)
        A_t = 2.0 * (eta * m_sec / dl_sec) * x
        
        hSp = -xp.cos(Phi) * A_t * (1.0 + cosi**2) * hp_flag
        hSc = -xp.sin(Phi) * A_t * 2.0 * cosi * hc_flag
        
        cos2psi = xp.cos(2.0 * psi)
        sin2psi = xp.sin(2.0 * psi)
        
        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi
        
        # Apply window
        hp *= self.window
        hc *= self.window
        # # check for nans
        # if xp.any(xp.isnan(hp)) or xp.any(xp.isnan(hc)):
        #     # check for nans in x
        #     if xp.any(xp.isnan(x)):
        #         breakpoint()
        #         if xp.any(xp.isnan(omega)):
        #             raise ValueError("NaN detected in x and omega")
        #         raise ValueError("NaN detected in x")
        #     raise ValueError("NaN detected in strain")
    
        return hp + 1j * hc

class GBWave:
    def __init__(self, use_gpu=False, T=1.0, dt=10.0):

        if use_gpu:
            self.xp = xp
        else:
            self.xp = np
        
        self.index_lambda = 6
        self.index_beta = 7
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


def get_response(orbit, WaveformClass=GBWave, T=1.0, dt=10., use_gpu=True, t0 = 10000.0):
    gb = WaveformClass(use_gpu=use_gpu, T=T, dt=dt)
    # default settings
    # order of the langrangian interpolation
    order = 25
    # 1st or 2nd or custom (see docs for custom)
    tdi_gen = "2nd generation"
    tdi_kwargs_esa = dict(order=order, tdi=tdi_gen, tdi_chan="AET",)
    return ResponseWrapper(
    gb,
    T,
    dt,
    gb.index_lambda,
    gb.index_beta,
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
    f = 1e-4
    A = 1e-25
    dt = 0.5
    T = 0.1
    param = np.asarray([A, f, fdot, iota, phi0, psi, lam, beta])
    gb = GBWave(use_gpu=True, T=T, dt=dt)
    h_strain = gb(A, f, fdot, iota, phi0, psi)
    fpath = "new_orbits.h5"
    orb_default = ESAOrbits(fpath, use_gpu=True)
    deviation = {which: np.zeros_like(getattr(orb_default, which + "_base")) for which in ["ltt", "x", "n", "v"]}
    orb_default.configure(linear_interp_setup=True, deviation=deviation)
    gb_response = get_response(orb_default, T=T, dt=dt, use_gpu=True)
    
    # check generate waveform is consistent with the h_strain
    h_response = gb_response.generate_waveform(*param[:-2])
    assert np.sum(h_strain - h_response)==0.0
    
    # apply the response
    AET = xp.asarray(gb_response.apply_response(h_response, param[6], param[7]))
    AET_test = xp.asarray(gb_response(*param))
    assert np.sum(AET - AET_test)==0.0
    print("My response applied and checked successfully.")
    # plot fft of AET
    fft = xp.fft.fft(AET, axis=1).get()
    freq = xp.fft.fftfreq(AET.shape[1], d=dt).get()
    plt.figure(figsize=(10, 6))
    plt.loglog(freq, np.abs(fft[0]), label='A', color='blue')
    plt.savefig('fft_A.png', dpi=300)
    
    # Parameters
    eta = 0.25
    Mc_list = [1e5 * eta**(3/5), 1e6 * eta**(3/5), 1e7 * eta**(3/5)]
    iota = np.pi / 4.0
    phic = 0.0
    tc = 0.09
    DL = 1.0
    psi = 0.0

    waveform = BHWave(use_gpu=True, T=T, dt=dt)
    bh_response = get_response(orb_default, WaveformClass=BHWave, T=T, dt=dt, use_gpu=True)
    
    t = waveform.t.get() / YRSID_SI

    plt.figure(figsize=(12, 8))

    for Mc in Mc_list:
        
        tic = time.time()
        h_strain = waveform(Mc, eta, iota, phic, tc, DL, psi).get()
        toc = time.time()
        print(f"Waveform generation took: {toc - tic:.4f} seconds")
        hp, hc = h_strain.real, -h_strain.imag
        # plt.plot(t, hp, label=f'h+ ({label})', linestyle='-')
        # plt.plot(t, hc, label=f'h× ({label})', linestyle='--')

        param = np.asarray([Mc, eta, iota, phic, tc, DL, psi, lam, beta])
        tic = time.time()
        AET = xp.asarray(bh_response(*param))
        fft = np.abs(xp.fft.rfft(AET, axis=1).get())**2
        freq = xp.fft.rfftfreq(len(AET[0]), d=dt).get()
        toc = time.time()
        print(f"Response generation took: {toc - tic:.4f} seconds")
        
        label = f'Mc = {Mc:.0e} M⊙'

        plt.loglog(freq, fft[0], label=f'A ({label})', linestyle='-')
        # plt.loglog(freq, fft[1], label=f'E ({label})', linestyle='--')
        # plt.loglog(freq, fft[2], label=f'T ({label})', linestyle='--')

    plt.xlabel('Time (years)')
    plt.ylabel('Strain')
    plt.title('TaylorF2 Waveforms for Different Chirp Masses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('taylorf2_waveforms.png', dpi=300)
