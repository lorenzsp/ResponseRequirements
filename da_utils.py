import numpy as np
import matplotlib.pyplot as plt
try:
    import cupy as xp
except ImportError:
    import numpy as xp

from scipy.signal.windows import tukey
from scipy.interpolate import CubicSpline

psd = np.load("TDI2_AE_psd.npy")
# interpolate the psd with a cubic spline
psd_interp = CubicSpline(psd[:, 0], psd[:, 1])

def inner_product(a, b, dt=1.0):
    a_fft = xp.fft.rfft(a, axis=1) * dt
    b_fft = xp.fft.rfft(b, axis=1) * dt
    f_fft = xp.fft.rfftfreq(a.shape[1], dt)
    df = f_fft[1] - f_fft[0]
    f_fft[0] = f_fft[1]  # Avoid division by zero
    psd = xp.asarray(psd_interp(f_fft.get()))
    return 4 * xp.sum(xp.conj(a_fft) * b_fft / psd, axis=1).real * df

if __name__=="__main__":
    # Example usage
    x = np.linspace(0, 10, 100)
    y = psd_interp(x)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(psd[:, 0], psd[:, 1])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.savefig('psd_plot.png')

    from orbits_utils import ESAOrbits
    from gb_utils import GBWave, draw_parameters, get_response
    # 
    T = 0.1
    dt = 10.
    # GB parameters
    psi = 0.0 # np.random.uniform(0, 2 * np.pi)
    iota = 0.0 # np.arccos(np.random.uniform(-1, 1))
    phi0 = 0.0 # np.random.uniform(0, 2 * np.pi)
    A = 1e-21
    f = 1e-3
    fdot = 0.0
    lam, beta = 0.0, 0.0  # Ecliptic coordinates
    param = np.asarray([A, f, fdot, iota, phi0, psi, lam, beta])
    
    from perturbation_utils import create_orbit_with_periodic_dev, create_orbit_with_static_dev
    periodic_orb = create_orbit_with_periodic_dev(delta_x=0.0, fpath="new_orbits.h5", use_gpu=True)
    static_orb = create_orbit_with_static_dev(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=0.0, rotation_error=0.0, translation_error=0.0, dt=86400., T=1.0)
    gb_response = get_response(static_orb, dt=dt, T=T, use_gpu=True)
    resp = gb_response(*param)
    AET = xp.asarray(resp)
    time = np.arange(AET.shape[1]) * 10.0 / 86400  # Convert dt to days
    snr = inner_product(AET, AET, dt=dt)**0.5
    print(f"Signal-to-Noise Ratio (SNR): {snr}")
    plt.figure(figsize=(10, 6))
    plt.plot(time, AET[0].get(), label='A', color='blue')
    plt.plot(time, AET[1].get(), label='E', color='orange')
    plt.plot(time, AET[2].get(), label='T', color='green')
    plt.xlabel('Time (days)')
    plt.ylabel('TDI Channels')
    plt.title('TDI Channels over Time')
    plt.legend()
    plt.savefig('tdi_channels.png', dpi=300)
    
    import healpy as hp
    from tqdm import tqdm
    nside = 4
    npix = hp.nside2npix(nside)
    thetas, phis = hp.pix2ang(nside, np.arange(npix))
    # betas ecliptic latitude https://arxiv.org/pdf/2204.06633
    betas, lambs = np.pi / 2 - thetas, phis
    gw_response_map = np.zeros(npix)
    h_strain = gb_response.generate_waveform(*param[:-2])
    for ii in tqdm(range(len(betas)), desc="Processing sky"):
        param[6], param[7] = lambs[ii], betas[ii]
        # test with no deviation
        AET = xp.asarray(gb_response.apply_response(h_strain, param[6], param[7]))
        snr = inner_product(AET, AET, dt=dt).sum()**0.5
        gw_response_map[ii] = snr
    
    plt.figure(figsize=(10, 6))
    hp.mollview(gw_response_map, title="GW Response Map", unit="SNR", cmap="viridis", norm="log")
    hp.graticule()
    plt.savefig("gw_response_map.png", dpi=300)
