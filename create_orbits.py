import logging

# import scientific libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import h5py

# %%
# set logging level to INFO (more verbose and informative)
logging.basicConfig(level=logging.INFO)
logging.getLogger("lisainstrument").setLevel(logging.INFO)

orbits_dt = 1000
orbits_t0 = 2173211130.0 # s
orbits_size = int(np.ceil(3600 * 24 * 365 / orbits_dt)) # a year, for illustration purposes
orbits_trim = 100

from lisaorbits import KeplerianOrbits, OEMOrbits
orbits = OEMOrbits.from_included("esa-trailing")

print("t_start:", orbits.t_start, "s")
print("t_end:", orbits.t_end, "s")
print("duration: ", (orbits.t_end - orbits.t_start) / (3600 * 24 * 365), "years")
print("size :", orbits_size, "samples")
print("dt :", orbits_dt, "s")
print("t0 :", orbits_t0, "s")
t_orbits = np.linspace(orbits.t_start + orbits_trim * orbits_dt, orbits.t_end, 1000)
orbits.plot_spacecraft(t_orbits, 1, output="3d_spacecraft.png")
orbits.plot_links(t_orbits, output="3d_links.png")

# %%
orbits.write("new_orbits.h5", dt=orbits_dt, size=orbits_size, t0=orbits_t0, mode="w")

# %%
with h5py.File("new_orbits.h5", "r") as hdf5:
    print("Groups:", list(hdf5.keys()))
    print("Attributes:", dict(hdf5.attrs))
    print("Datasets in 'tcb':", list(hdf5["tcb"].keys()))
    print("TCB LTTs:", hdf5["tcb"]["ltt"][:])
