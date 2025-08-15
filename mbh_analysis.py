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

def compute_mismatch(gb_response_dev, h_strain, param, AET, dt):
    AET_deviation = xp.asarray(gb_response_dev.apply_response(h_strain, param[7], param[8]))
    AET_deviation_1 = xp.asarray(gb_response_dev(*param))
    if np.sum(AET_deviation - AET_deviation_1)!=0.0:
        breakpoint()
    max_length = min(AET.shape[1], AET_deviation.shape[1])
    AET = AET[:, :max_length]
    AET_deviation = AET_deviation[:, :max_length]
    # check that none of the arrays are nans
    if xp.isnan(AET).any() or xp.isnan(AET_deviation).any():
        raise ValueError("NaN values found in AET or AET_deviation")
    
    fmin = 1e-4
    fmax = 1.0
    h_hdev = inner_product(AET, AET_deviation, dt=dt, fmin=fmin, fmax=fmax)
    h_h = inner_product(AET, AET, dt=dt, fmin=fmin, fmax=fmax)
    hdev_hdev = inner_product(AET_deviation, AET_deviation, dt=dt, fmin=fmin, fmax=fmax)
 
    mismatch = 1 - (h_hdev / (h_h * hdev_hdev)**0.5)
    if xp.isnan(mismatch).any():
        print("Mismatch calculation resulted in NaN values.", h_hdev, h_h, hdev_hdev)
        # raise ValueError("NaN values found in mismatch calculation")
    
    return mismatch

if __name__=="__main__":
    from gb_utils import BHWave
    T = 14/365
    dt = 10.0
    # default_orbit = create_orbit_with_static_dev
    default_orbit = lambda **kwargs: create_orbit_with_periodic_dev(equal_armlength=False, period=5*86400.0, **kwargs)
    nside = 4
    betas, lambs, gw_response_map = get_sky_grid(nside)
    ind = 10
    lam, beta = lambs[ind], betas[ind]  # Ecliptic coordinates
    Mc, eta, iota, phic, tc, DL, psi = 1e5 * 0.25**(3/5), 0.25, 0.0, 0.0, T*1.01, 1.0, 0.0
    param = np.asarray([Mc, eta, iota, phic, tc, DL, psi, lam, beta])
    
    dt_resp = 86400.0/4  # seconds
    default = default_orbit(armlength_error=0.0, rotation_error=0.0, translation_error=0.0, dt=dt_resp, T=T*1.01)
    bh_def = get_response(default, WaveformClass=BHWave, dt=dt, T=T, use_gpu=True)
    h_strain = bh_def.generate_waveform(*param[:-2])
    AET = xp.asarray(bh_def(*param))
    AET_apply_resp = xp.asarray(bh_def.apply_response(h_strain, param[7], param[8]))
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
    M_vector = np.logspace(3, 7, 25)
    mismatch_results = np.zeros((n_realizations, len(M_vector), 3))
    for ii, mass in tqdm(enumerate(M_vector)):
        param[0] = mass * 0.25**(3/5)
        print(f"Mass: {mass} Hz")

        h_strain = bh_def.generate_waveform(*param[:-2])
        AET = xp.asarray(bh_def.apply_response(h_strain, param[7], param[8]))
        for real_i in range(n_realizations):
            static_orb_deviation = default_orbit(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=armlength_error, rotation_error=rotation_error, translation_error=translation_error, dt=dt_resp, T=T*1.01)
            bh_response_deviation = get_response(static_orb_deviation, WaveformClass=BHWave, dt=dt, T=T, use_gpu=True)
        
            mismatch = compute_mismatch(bh_response_deviation, h_strain, param, AET, dt)
            mismatch_results[real_i, ii] = np.abs(mismatch.get())
            # print(f"Realization {real_i + 1}/{n_realizations}, Mismatch: {mismatch_results[real_i, f_i]}")
    
    plt.figure(figsize=(10, 6))
    plt.loglog(M_vector, mismatch_results[:,:,0].mean(axis=(0)), label='Mean Mismatch', color='blue')
    plt.xscale('log')
    # plt.xlabel('Frequency (Hz)')
    plt.xlabel('Mass (Msun)')
    plt.ylabel('Mismatch')
    plt.title('Mismatch Analysis with Deviations')
    plt.legend()
    plt.savefig('mismatch_vs_M.png', dpi=300)
    #################################################
    # across the sky
    print("Starting mismatch analysis across the sky...")
    param[0] = 1e6 * 0.25**(3/5)  # Reset Mass
    nside = 4
    betas, lambs, gw_response_map = get_sky_grid(nside)
    mismatch_results = np.zeros((n_realizations, len(betas), 3))
    h_strain = bh_def.generate_waveform(*param[:-2])
    for real_i in range(n_realizations):
        static_orb_deviation = default_orbit(arm_lengths=[2.5e9, 2.5e9, 2.5e9], armlength_error=armlength_error, rotation_error=rotation_error, translation_error=translation_error, dt=dt_resp, T=T*1.01)
        bh_response_deviation = get_response(static_orb_deviation, WaveformClass=BHWave, dt=dt, T=T, use_gpu=True)
        for s_i in tqdm(range(len(betas)), desc="Processing sky"):
            lam, beta = lambs[s_i], betas[s_i]
            param[7], param[8] = lam, beta
            print(f"Sky Position: (λ: {lam}, β: {beta})")
            AET = xp.asarray(bh_def.apply_response(h_strain, param[7], param[8]))
            # AET = xp.asarray(bh_def(*param)
            mismatch = compute_mismatch(bh_response_deviation, h_strain, param, AET, dt)
            mismatch_results[real_i, s_i] = np.abs(mismatch.get())
            # print(f"Realization {real_i + 1}/{n_realizations}, Mismatch: {mismatch_results[real_i, f_i]}")
    gw_response_map = mismatch_results[:,:,0].mean(axis=(0))
    
    plt.figure(figsize=(10, 6))
    hp.mollview(gw_response_map, title="GW Response Map", unit="Mismatch", cmap="viridis", norm="log")
    hp.graticule()
    plt.savefig("mismatch_vs_sky_map.png", dpi=300)
    ################################################
    # as a function of different errors
    armlength_error = 1.0
    rotation_error = 50e3
    translation_error = 50e3

    error_vec = np.logspace(0, 2, 10)
    
    Mc, eta, iota, phic, tc, DL, psi = 1e5 * 0.25**(3/5), 0.25, 0.0, 0.0, T*0.99, 1.0, 0.0
    param = np.asarray([Mc, eta, iota, phic, tc, DL, psi, lam, beta])
    h_strain = bh_def.generate_waveform(*param[:-2])
    AET = xp.asarray(bh_def(*param))
    plt.figure(figsize=(10, 6))
    plt.plot(time, AET[0].get(), label='h_strain', color='blue')
    plt.savefig('h_strain.png', dpi=300)

    plt.figure(figsize=(10, 6))
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
                bh_response_deviation = get_response(static_orb_deviation, WaveformClass=BHWave, dt=dt, T=T, use_gpu=True)
            
                mismatch = compute_mismatch(bh_response_deviation, h_strain, param, AET, dt)
                mismatch_results[real_i, err_i] = np.abs(mismatch.get())
        print(f"Realization {real_i + 1}/{n_realizations}, Mismatch: {mismatch_results[real_i, err_i]}")
        plt.loglog(error_vec, mismatch_results[:,:,0].mean(axis=(0)), ':', label=label)
    plt.xscale('log')
    plt.xlabel('Error')
    plt.ylabel('Mismatch')
    plt.title('Mismatch Analysis with Deviations')
    plt.legend()
    plt.savefig('mismatch_vs_error.png', dpi=300)
    