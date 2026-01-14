import numpy as np
import matplotlib.pyplot as plt
try:
    import cupy as xp
except ImportError:
    import numpy as xp
from da_utils import inner_product, get_sky_grid
from gb_utils import get_response
from perturbation_utils import create_orbit_with_periodic_dev, create_orbit_with_static_dev
from tqdm import tqdm
import healpy as hp

# Global rcParams for LaTeX consistency
plt.rcParams.update({
    'text.usetex': True,  # Enable LaTeX rendering
    'font.family': 'serif',  # Match LaTeX default
    'font.serif': ['Computer Modern Roman'],  # Or 'cmr10' if needed
    'font.size': 16,  # Match paper's body text size (adjust to 11 if your doc is 11pt)
    'axes.labelsize': 16,  # Labels same as text
    'axes.titlesize': 16,  # Titles same as text
    'legend.fontsize': 14,  # Legend same as text
    'xtick.labelsize': 16,  # Ticks same as text
    'ytick.labelsize': 16,
    'figure.figsize': (6.5, 3.7),  # Width ~ \textwidth (6 inches), height for aspect ratio
    'savefig.dpi': 300,  # High resolution for PDF
    'lines.linewidth': 1.5,  # Thicker lines for visibility
    'axes.linewidth': 0.8,  # Thinner axes for clean look
})

def compute_mismatch(gb_response_dev, h_strain, param, AET, dt, fmin=1e-4, fmax=1.0):
    AET_deviation = xp.asarray(gb_response_dev.apply_response(h_strain, param[gb_response_dev.index_lambda], param[gb_response_dev.index_beta]))
    # AET_deviation_1 = xp.asarray(gb_response_dev(*param))
    # if np.sum(AET_deviation * AET_deviation_1)/np.sum(AET_deviation * AET_deviation)!=1.0:
    #     breakpoint()
    max_length = min(AET.shape[1], AET_deviation.shape[1])
    AET = AET[:, :max_length]
    AET_deviation = AET_deviation[:, :max_length]
    # check that none of the arrays are nans
    if xp.isnan(AET).any() or xp.isnan(AET_deviation).any():
        raise ValueError("NaN values found in AET or AET_deviation")
    
    h_hdev = inner_product(AET, AET_deviation, dt=dt, fmin=fmin, fmax=fmax)
    h_h = inner_product(AET, AET, dt=dt, fmin=fmin, fmax=fmax)
    hdev_hdev = inner_product(AET_deviation, AET_deviation, dt=dt, fmin=fmin, fmax=fmax)
 
    mismatch = 1 - (h_hdev / (h_h * hdev_hdev)**0.5)
    if xp.isnan(mismatch).any():
        print("Mismatch calculation resulted in NaN values.", h_hdev, h_h, hdev_hdev)
        # raise ValueError("NaN values found in mismatch calculation")
    
    return mismatch, AET_deviation

def create_plot_error(AET, AET_deviation, time, filename='response_deviation.png', fmin=1e-4, fmax=1.0):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Time domain plot
    axs[0].plot(time/86400, np.abs(AET[0].get()-AET_deviation[0].get()), label='AET', color='blue')
    # axs[0].plot(time, AET_deviation[0].get(), label='AET with Deviation', color='orange', alpha=0.7)
    axs[0].set_xlabel('Time (days)')
    axs[0].set_ylabel('Absolute Difference A')
    axs[0].set_title('Time Domain')
    axs[0].legend()
    axs[0].grid()

    # Frequency domain plot
    fft_AET = np.abs(xp.fft.rfft(AET[0]).get())
    fft_AET_dev = np.abs(xp.fft.rfft(AET_deviation[0]).get())
    freq = xp.fft.rfftfreq(len(AET[0]), d=(time[1] - time[0])).get()  # d in seconds

    freq_mask = (freq >= fmin) & (freq <= fmax)
    freq = freq[freq_mask]
    fft_AET = fft_AET[freq_mask]
    fft_AET_dev = fft_AET_dev[freq_mask]
    axs[1].loglog(freq, np.abs(fft_AET_dev-fft_AET), label='AET', color='blue')
    # axs[1].loglog(freq, fft_AET_dev, label='AET with Deviation', color='orange', alpha=0.7)
    
    # axs[1].axvline(fmin, color='red', linestyle='--', label='fmin')
    # axs[1].axvline(fmax, color='green', linestyle='--', label='fmax')
    # axs[1].set_xlim(fmin, fmax)
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Absolute Difference in FFT of A')
    axs[1].set_title('Frequency Domain')
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.savefig("figures_gb_analysis/"+filename, dpi=300)
    plt.close(fig)

if __name__=="__main__":
    
    T = 365/365
    dt = 10.0
    # GB parameters
    psi = 0.0 # np.random.uniform(0, 2 * np.pi)
    iota = 0.0 # np.arccos(np.random.uniform(-1, 1))
    phi0 = 0.0 # np.random.uniform(0, 2 * np.pi)
    A = 1e-21
    f = 1e-3
    fdot = 0.0
    default_orbit = create_orbit_with_static_dev
    # default_orbit = lambda **kwargs: create_orbit_with_periodic_dev(equal_armlength=False, **kwargs, period=2*365*86400.0)
    nside = 4
    betas, lambs, gw_response_map = get_sky_grid(nside)
    ind = 10
    lam, beta = lambs[ind], betas[ind]  # Ecliptic coordinates
    param = np.asarray([A, f, fdot, iota, phi0, psi, lam, beta])
    dt_resp = 86400.0/4  # seconds
    default = default_orbit(armlength_error=0.0, rotation_error=0.0, translation_error=0.0, dt=dt_resp, T=T*1.01)
    gb_def = get_response(default, dt=dt, T=T, use_gpu=True)
    h_strain = gb_def.generate_waveform(*param[:-2])
    AET = xp.asarray(gb_def(*param))
    time = np.arange(AET.shape[1]) * dt / 86400  # Convert dt to days
    snr = inner_product(AET, AET, dt=dt)**0.5
    print(f"Signal-to-Noise Ratio (SNR): {snr}")

    # create random deviations
    n_realizations = 1
    armlength_error = 1.0
    rotation_error = 50e3
    translation_error = 50e3

    # #################################################
    # across frequencies
    print("Starting mismatch analysis across frequencies...")
    frequency_vector = np.logspace(-4, 0.0, 50)
    mismatch_results = np.zeros((n_realizations, len(frequency_vector), 3))
    for f_i, freq in tqdm(enumerate(frequency_vector)):
        print(f"Frequency: {freq} Hz")
        param[1] = freq
        h_strain = gb_def.generate_waveform(*param[:-2])
        AET = xp.asarray(gb_def.apply_response(h_strain, param[6], param[7]))
        # AET = xp.asarray(gb_def(*param))
        for real_i in range(n_realizations):
            static_orb_deviation = default_orbit(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=armlength_error, rotation_error=rotation_error, translation_error=translation_error, dt=dt_resp, T=T*1.01)
            gb_response_deviation = get_response(static_orb_deviation, dt=dt, T=T, use_gpu=True)
            fmin = freq * 0.99
            fmax = freq * 1.01
            mismatch, _ = compute_mismatch(gb_response_deviation, h_strain, param, AET, dt, fmin=fmin, fmax=fmax)
            mismatch_results[real_i, f_i] = np.abs(mismatch.get())
            # print(f"Realization {real_i + 1}/{n_realizations}, Mismatch: {mismatch_results[real_i, f_i]}")
    
    plt.figure()
    plt.loglog(frequency_vector, mismatch_results[:,:,0].mean(axis=(0)), label='Mean Mismatch', color='blue')
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mismatch')
    plt.title('Mismatch Analysis with Deviations')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures_gb_analysis/"+'mismatch_vs_frequency.png', dpi=300)
    # #################################################
    # across the sky
    print("Starting mismatch analysis across the sky...")
    param[1] = 1e-3  # Reset frequency for sky analysis
    fmin = param[1] * 0.99
    fmax = param[1] * 1.01
    nside = 4
    betas, lambs, gw_response_map = get_sky_grid(nside)
    mismatch_results = np.zeros((n_realizations, len(betas), 3))
    h_strain = gb_def.generate_waveform(*param[:-2], hp_flag=1.0, hc_flag=0.0)
    for real_i in range(n_realizations):
        static_orb_deviation = default_orbit(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=armlength_error, rotation_error=rotation_error, translation_error=translation_error, dt=dt_resp, T=T*1.01)
        gb_response_deviation = get_response(static_orb_deviation, dt=dt, T=T, use_gpu=True)
        for s_i in tqdm(range(len(betas)), desc="Processing sky"):
            lam, beta = lambs[s_i], betas[s_i]
            param[6], param[7] = lam, beta
            print(f"Sky Position: (λ: {lam}, β: {beta})")
            AET = xp.asarray(gb_def.apply_response(h_strain, param[6], param[7]))
            # AET = xp.asarray(gb_def(*param, hp_flag=1.0, hc_flag=0.0))
            mismatch, _ = compute_mismatch(gb_response_deviation, h_strain, param, AET, dt, fmin=fmin, fmax=fmax)
            mismatch_results[real_i, s_i] = np.abs(mismatch.get())
            # print(f"Realization {real_i + 1}/{n_realizations}, Mismatch: {mismatch_results[real_i, f_i]}")
    gw_response_map = mismatch_results[:,:,0].mean(axis=(0))

    # Ensure positivity of gw_response_map
    gw_response_map = np.abs(gw_response_map)

    plt.figure()
    hp.mollview(gw_response_map, title="GW Response Map", unit="Mismatch", cmap="viridis", norm="log")
    hp.graticule()
    plt.tight_layout()
    plt.savefig("figures_gb_analysis/"+"mismatch_vs_sky_map.png", dpi=300)
    #################################################
    # as a function of different errors
    armlength_error = 1.0
    rotation_error = 50e3
    translation_error = 50e3

    error_vec = np.logspace(0, 2, 10)
    
    param = np.asarray([A, f, fdot, iota, phi0, psi, lam, beta])
    fmin = param[1] * 0.99
    fmax = param[1] * 1.01
    h_strain = gb_def.generate_waveform(*param[:-2])
    AET = xp.asarray(gb_def(*param))
    
    plt.figure()
    # create error increase for differen error types
    print("Starting mismatch analysis with different errors...")
    index_arm = np.asarray([1, 0, 0])
    index_rot = np.asarray([0, 1, 0])
    index_tran = np.asarray([0, 0, 1])
    for ind_ref, label in zip([index_arm, index_rot, index_tran],["armlength", "rotation", "translation"]):
        mismatch_results = np.zeros((n_realizations, len(error_vec), 3))
        for err_i, error in tqdm(enumerate(error_vec), desc="Processing errors"):
            ind = ind_ref * error
            for real_i in range(n_realizations):
                static_orb_deviation = default_orbit(armlength_error=armlength_error * ind[0], rotation_error=rotation_error * ind[1], translation_error=translation_error * ind[2], dt=dt_resp, T=T*1.01)
                gb_response_deviation = get_response(static_orb_deviation, dt=dt, T=T, use_gpu=True)
            
                mismatch, AET_dev = compute_mismatch(gb_response_deviation, h_strain, param, AET, dt, fmin=fmin, fmax=fmax)
                mismatch_results[real_i, err_i] = np.abs(mismatch.get())
                if err_i == len(error_vec)-1:
                    create_plot_error(AET, AET_dev, time * 86400, filename=f'response_deviation_error{error:.1f}_{label[:3]}.png', fmin=fmin, fmax=fmax)
        print(f"Realization {real_i + 1}/{n_realizations}, Mismatch: {mismatch_results[real_i, err_i]}")
        plt.loglog(error_vec, mismatch_results[:,:,0].mean(axis=(0)), ':', label=label)
    plt.xscale('log')
    plt.xlabel('Error')
    plt.ylabel('Mismatch')
    plt.title('Mismatch Analysis with Deviations')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures_gb_analysis/"+'mismatch_vs_error.png', dpi=300)
    