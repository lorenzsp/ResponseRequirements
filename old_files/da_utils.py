import numpy as np
import matplotlib.pyplot as plt
try:
    import cupy as xp
except ImportError:
    import numpy as xp

from scipy.signal.windows import tukey
from scipy.interpolate import CubicSpline

import healpy as hp
from tqdm import tqdm

psd = np.load("TDI2_AE_psd.npy")
# interpolate the psd with a cubic spline
psd_interp = CubicSpline(psd[:, 0], psd[:, 1])

def inner_product(a, b, dt=1.0, fmin=1e-4, fmax=1.0):
    a_fft = xp.fft.rfft(a, axis=1) * dt
    b_fft = xp.fft.rfft(b, axis=1) * dt
    f_fft = xp.fft.rfftfreq(a.shape[1], dt)
    df = f_fft[1] - f_fft[0]
    f_fft[0] = f_fft[1]  # Avoid division by zero
    psd = xp.asarray(psd_interp(f_fft.get()))
    mask = (f_fft >= fmin) & (f_fft <= fmax)
    a_fft = a_fft[:, mask]
    b_fft = b_fft[:, mask]
    psd = psd[mask]
    if xp.isnan(psd).any():
        raise ValueError("NaN values found in PSD")
    if xp.isinf(a_fft).any() or xp.isinf(b_fft).any():
        raise ValueError("Infinite values found in FFTs")
    result = 4 * xp.sum(xp.conj(a_fft) * b_fft / psd, axis=1).real * df
    if xp.isnan(result).any():
        raise ValueError("NaN values found in inner product result")
    return result

def get_sky_grid(nside):
    npix = hp.nside2npix(nside)
    thetas, phis = hp.pix2ang(nside, np.arange(npix))
    betas = np.pi / 2 - thetas  # ecliptic latitude
    lambs = phis                # ecliptic longitude
    gw_response_map = np.zeros(npix)
    return betas, lambs, gw_response_map

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
    T = 0.5
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
    static_orb = create_orbit_with_static_dev(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=0.0, rotation_error=0.0, translation_error=0.0, dt=86400., T=T*1.01)
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

    nside = 4
    betas, lambs, gw_response_map = get_sky_grid(nside)
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

    import time
    # mismatch analysis
    start = time.time()
    static_orb_deviation = create_orbit_with_static_dev(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=1, rotation_error=50e3, translation_error=50e3, dt=86400., T=T*1.01)
    gb_response_deviation = get_response(static_orb_deviation, dt=dt, T=T, use_gpu=True)
    end = time.time()
    print(f"Time taken for response with deviation: {end - start:.2f} seconds")
    start = time.time()
    AET_deviation = xp.asarray(gb_response_deviation(*param))
    max_length = min(AET.shape[1], AET_deviation.shape[1])
    AET = AET[:, :max_length]
    AET_deviation = AET_deviation[:, :max_length]
    h_hdev = inner_product(AET, AET_deviation, dt=dt)**0.5
    h_h = inner_product(AET, AET, dt=dt)**0.5
    hdev_hdev = inner_product(AET_deviation, AET_deviation, dt=dt)**0.5
    mismatch = 1 - (h_hdev / (h_h * hdev_hdev)**0.5)
    end = time.time()
    print(f"Mismatch with deviation: {mismatch}", f"Time taken: {end - start:.2f} seconds")
