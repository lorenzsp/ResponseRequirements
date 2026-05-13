"""JaxGBFull — galactic binary response with the full velocity-corrected kernel.

Replaces the Cornish–Rubbo sinc approximation in :class:`JaxGB` with the
exact first-order-in-v/c single-link transfer function (Eq. A17 of
arXiv:2509.10038V).

Background
----------
The parent :class:`JaxGB._construct_slow_part` implements the CR kernel:

    gs^CR = A_ij / (2*(1 + k̂·n̂_ij))
            × [ exp(-i φ_tx)  −  exp(-i (Ω_tx + φ_rx)) ]

This follows by algebraic expansion of the sinc formula used in the parent:

    0.5j (f/f★) sinc(arg/π) exp(-i arg)
      = (1 − exp(-2i arg)) / (2*(1 + k̂·n̂_ij))

with arg = ½ Ω_tx (1 + k̂·n̂_ij), φ_sc = ω₀ k̂·x_sc/c − arg_s the
heterodyned position phase, and Ω_tx = ω_gw(ξ_tx) L/c the arm-length
phase at the transmitter retarded time.

The full kernel (Eq. A17) replaces the unit velocity brackets with first-
order corrections:

    gs^full = A_ij / (2*(1 + k̂·n̂_ij))
              × [ B_tx · exp(-i φ_tx)  −  B_rx · exp(-i (Ω_tx + φ_rx)) ]

    B_rx = 1 − k̂·v_j  +  n̂_ij·v_i        (receiver bracket)
    B_tx = 1 − k̂·v_i  +  n̂_ij·(v_j−2v_i) (transmitter bracket)

where v_i, v_j are the transmitter / receiver velocities in units of c
(dimensionless).  Setting v = 0 recovers gs^CR exactly.

Sign conventions (matching the parent throughout)
    • k̂  points FROM source TOWARD detector (wave propagation direction).
    • n̂_ij = (x_rx − x_tx)/L  points from transmitter to receiver.
    • Denominator reads  (1 + k̂·n̂_ij).  This is equivalent to
      (1 − k̂_paper·n̂_paper) in arXiv:2509.10038V, which uses the
      opposite conventions for both vectors (k̂_paper = −k̂, and n̂_paper
      from rx to tx).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from lisaconstants import c  # pylint: disable=no-name-in-module

if TYPE_CHECKING:
    from lisaorbits import Orbits  # noqa: F401

# Adjust to your project layout, e.g. from lisagwresponse.jaxgb import JaxGB
from jaxgb.jaxgb import JaxGB
from jaxgb.tdi import to_tdi_combination, to_tdi_generation

def _add_to(tdi_one, tdi_sum, kmin_arr, kmax, n):
    def _update_fpoint(idx, val):
        _mask = ((idx + kmin_arr) >= 0) & ((idx + kmin_arr) < kmax)
        return jax.lax.dynamic_update_index_in_dim(
            val,
            val[:, idx + kmin_arr] + _mask * tdi_sum[:, idx],
            idx + kmin_arr,
            axis=1,
        )


    return jax.lax.fori_loop(0, n, _update_fpoint, tdi_one)

vadd = jax.vmap(_add_to, in_axes=(0,0,0,None,None))

class JaxGBFull(JaxGB):
    """Galactic binary response with the velocity-corrected two-exponential kernel.

    Drop-in replacement for :class:`JaxGB`.  Only
    :meth:`_construct_slow_part` is overridden; all public methods
    (:meth:`get_link_responses`, :meth:`get_tdi`, …) are inherited
    unchanged.

    Parameters
    ----------
    orbits
        Orbits instance (lisaorbits).
    t_obs
        Observation duration (s).
    t0
        Start time (s).
    n
        Number of slow-response evaluation points (even integer).
    """

    def __init__(
        self,
        orbits: "Orbits",
        *,
        t_obs: float = 6.2914560e7,
        t0: float = 0.0,
        n: int = 128,
        old_implementation: bool = False,  # for testing only
    ) -> None:
        super().__init__(orbits, t_obs=t_obs, t0=t0, n=n)
        self.old_implementation = old_implementation

        self.velocity = (
            self.orbits.compute_velocity(self.t0 + self.tm, [1, 2, 3])
            .swapaxes(0, 1)
            .swapaxes(1, 2)
        ) / c  # (3, 3, n)

        self.ltts = self.orbits.compute_ltt(self.t0 + self.tm)
        ltt = jnp.mean(self.orbits.compute_ltt(self.t0 + self.tm).T, axis=0)  # (6, n) -> (n,)
        self.arm_length = ltt * c
        self.position = (
            self.orbits.compute_position(self.t0 + self.tm, [1, 2, 3])
            .swapaxes(0, 1)
            .swapaxes(1, 2)
        )  # (3, 3, n)
        self.fstar = c / (self.arm_length * 2 * np.pi)
        
        if self.old_implementation:
            print("WARNING: Using old implementation of JaxGBFull with no velocity corrections. This is for testing purposes only and should not be used for production runs.")
            ltt = self._get_ltt_simple()
            self.arm_length = ltt * c
            self.fstar = c / (self.arm_length * 2 * np.pi)

    # ------------------------------------------------------------------

    def _construct_slow_part(
        self,
        f0_fdot_phi0: jax.Array,
        k: jax.Array,
        dpdc: tuple[jax.Array, jax.Array],
        pol_tensors: tuple[jax.Array, jax.Array],
    ) -> jax.Array:
        """Heterodyned slow response using the full velocity-corrected kernel.

        Parameters
        ----------
        f0_fdot_phi0 : jax.Array, shape (nsrc, 3)
            Per-source ``[f0, fdot, phi0]``.
        k : jax.Array, shape (3, nsrc)
            GW propagation unit vectors.
        dpdc : tuple of complex jax.Array, each (nsrc,)
            Complex strain amplitudes (D+, D×).
        pol_tensors : tuple of jax.Array, each (nsrc, 3, 3)
            Polarisation tensors (e+, e×).

        Returns
        -------
        jax.Array, shape (nsrc, 6, n)
            Slow heterodyned link responses.
            Link order: ``[12, 23, 31, 13, 32, 21]``.
        """
        nsrc = len(f0_fdot_phi0)

        # ── Arm unit vectors ──────────────────────────────────────────────────
        if self.old_implementation:
            # r convention: [r_12, r_13, r_23, r_31]  (identical to parent)
            r = jnp.zeros((4, 3, self.n))
            r = r.at[0].set(self.position[1] - self.position[0])
            r = r.at[1].set(self.position[2] - self.position[0])
            r = r.at[2].set(self.position[2] - self.position[1])
            r = r.at[3].set(-r[1])
            r /= self.arm_length
        
        # # option 1
        # # MOSAS = np.array([12, 23, 31, 13, 32, 21])
        # r = jnp.zeros((4, 3, self.n))
        # r = r.at[0].set((self.position[1] - self.position[0])/(self.ltts[:,0] * c))
        # r = r.at[1].set((self.position[2] - self.position[0])/(self.ltts[:,3] * c))
        # r = r.at[2].set((self.position[2] - self.position[1])/(self.ltts[:,1] * c))
        # r = r.at[3].set(-r[1]/(self.ltts[:,2] * c))
        
        # New implementation that normalizes the unit vectors
        else:
            r = jnp.zeros((4, 3, self.n))
            r = r.at[0].set((self.position[1] - self.position[0])/np.linalg.norm(self.position[1] - self.position[0], axis=0))
            r = r.at[1].set((self.position[2] - self.position[0])/np.linalg.norm(self.position[2] - self.position[0], axis=0))
            r = r.at[2].set((self.position[2] - self.position[1])/np.linalg.norm(self.position[2] - self.position[1], axis=0))
            r = r.at[3].set(-r[1])

        # print("Mean relative difference", np.mean(self.arm_length/(self.ltts[:,0] * c)-1))

        # ── k̂·n̂ for all 6 directed links ─────────────────────────────────────
        # kdotr convention: [12, 21, 13, 31, 23, 32]  (identical to parent)
        kdotr = jnp.zeros((nsrc, 6, self.n))
        kdotr = kdotr.at[:, ::2].set(jnp.dot(k.T, r[:3]))   # k·n̂_12, k·n̂_13, k·n̂_23
        kdotr = kdotr.at[:, 1::2].set(-kdotr[:, ::2])        # k·n̂_21, k·n̂_31, k·n̂_32

        # ── Position phases, retarded times, instantaneous frequency ──────────
        kdotp_raw = jnp.dot(k.T, self.position) / c  # k̂·x_sc/c, (nsrc, 3, n)

        f0, fdot, phi0 = f0_fdot_phi0[:, 0], f0_fdot_phi0[:, 1], f0_fdot_phi0[:, 2]
        xi    = self.tm - kdotp_raw                                # retarded time
        fi    = f0[:, None, None] + fdot[:, None, None] * xi      # inst. GW freq.
        fonfs = fi / self.fstar                                    # Ω_gw·L/c
        
        # ── Strain coupling A_ij ──────────────────────────────────────────────
        # (identical to parent)  idx selects r_12, r_23, r_31 from r
        idx   = jnp.array([0, 2, 3])
        eplus, ecross = pol_tensors
        aij   = (
            jnp.swapaxes(jnp.dot(eplus,  r[idx]), 1, 2) * r[idx]
            * dpdc[0][:, None, None, None]
            + jnp.swapaxes(jnp.dot(ecross, r[idx]), 1, 2) * r[idx]
            * dpdc[1][:, None, None, None]
        )
        asum = aij.sum(axis=2)  # A_ij for links [12, 23, 31], shape (nsrc, 3, n)

        # ── Heterodyned position phases φ_sc = ω₀ k̂·x_sc/c − arg_s ──────────
        # (identical to parent)
        q     = jnp.rint(f0 * self.t_obs)
        df    = 2.0 * np.pi * (q / self.t_obs)
        om    = 2.0 * np.pi * f0

        arg_s = (
            (phi0[:, None] + (om - df)[:, None] * self.tm[None, :])[:, None, :]
            + np.pi * fdot[:, None, None] * xi**2
        )  # (nsrc, 3, n)
        kdotp = om[:, None, None] * kdotp_raw - arg_s  # (nsrc, 3, n)

        # ── Link index arrays ─────────────────────────────────────────────────
        # gs convention (before final reorder): [12, 23, 31, 21, 32, 13]
        tx_sc    = jnp.array([0, 1, 2, 1, 2, 0])   # transmitter spacecraft (0-idx)
        rx_sc    = jnp.array([1, 2, 0, 0, 1, 2])   # receiver   spacecraft (0-idx)
        link_ord = jnp.array([0, 4, 3, 1, 5, 2])   # gs-order → kdotr-order

        # ── n̂_ij for all 6 links, shape (6, 3, n) ────────────────────────────
        n_all = jnp.stack([r[0], r[2], r[3], -r[0], -r[2], r[1]])

        # ── Velocity projections ──────────────────────────────────────────────
        vel = jnp.asarray(self.velocity)  # (3_sc, 3_xyz, n), v/c dimensionless

        # k̂·v for each spacecraft → (nsrc, 3_sc, n)
        k_dot_v    = jnp.einsum("sx,yxn->syn", k.T, vel)
        k_dot_v_tx = k_dot_v[:, tx_sc, :]    # (nsrc, 6, n)
        k_dot_v_rx = k_dot_v[:, rx_sc, :]    # (nsrc, 6, n)

        # n̂_ij · v for tx and rx spacecraft → (6, n)
        n_dot_v_tx = (n_all * vel[tx_sc]).sum(axis=1)  # n̂_ij · v_i
        n_dot_v_rx = (n_all * vel[rx_sc]).sum(axis=1)  # n̂_ij · v_j

        # ── Velocity brackets ─────────────────────────────────────────────────
        # B_rx = 1 − k̂·v_j + n̂_ij·v_i
        B_rx = 1.0 - k_dot_v_rx + n_dot_v_tx[None, :, :]          # (nsrc, 6, n)
        # B_tx = 1 − k̂·v_i + n̂_ij·(v_j − 2 v_i)
        B_tx = (
            1.0 - k_dot_v_tx
            + (n_dot_v_rx - 2.0 * n_dot_v_tx)[None, :, :]
        )                                                            # (nsrc, 6, n)

        # ── Phase terms per link ──────────────────────────────────────────────
        phi_tx  = kdotp[:, tx_sc, :]   # φ_tx,          (nsrc, 6, n)
        phi_rx  = kdotp[:, rx_sc, :]   # φ_rx,          (nsrc, 6, n)
        omega_L = fonfs[:, tx_sc, :]   # Ω_gw(ξ_tx)·L/c (nsrc, 6, n)
        # # new implementation
        # fstar_new = 1 / (self.ltts[:,tx_sc] * 2 * np.pi)
        # omega_L = (fonfs[:, tx_sc, :] * self.fstar) / fstar_new.T
        
        # Geometric denominator  1 + k̂·n̂_ij  (parent sign convention)
        denom = 1.0 + kdotr[:, link_ord]   # (nsrc, 6, n)

        # ── Full velocity-corrected response ──────────────────────────────────
        gs = (
            asum[:, [0, 1, 2, 0, 1, 2], :]
            / (2.0 * denom)
            * (
                B_tx * jnp.exp(-1j * phi_tx)
                - B_rx * jnp.exp(-1j * (omega_L + phi_rx))
            )
        )  # (nsrc, 6, n),  convention [12, 23, 31, 21, 32, 13]

        # Reorder → parent output convention [12, 23, 31, 13, 32, 21]
        return gs[:, [0, 1, 2, 5, 4, 3]]
    
    def get_data_minus_template(
        self,
        params: jax.Array,
        data: jax.Array,
        kmin: int | None = None,
        kmax: int | None = None,
        t_init: float = None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Compute data minus template
        
        Parameters
        ----------
        params : jax.Array
            Physical parameters (8,) or (n_sources, 8)
        kmin : int
            Minimum frequency index
        kmax : int
            Maximum frequency index
        t_init : float, optional
            Initialization time
            
        Returns
        -------
        loglike : jnp.ndarray
            Log-likelihood value
        """
        kmins = self.get_kmin(f0=params[:, 0], fdot=params[:, 1], t_init=t_init)

        # Compute residual: data - sum(templates)
        _x, _y, _z = self.get_tdi(params,t_init=t_init,**kwargs)
        templates = -jnp.stack([_x, _y, _z],axis=1)
        
        return vadd(data, templates, kmins-kmin, kmax, self.n)

    # ------------------------------------------------------------------

    def inner_product(
        self,
        h1: jax.Array,
        h2: jax.Array,
        inv_cov: jax.Array,
    ) -> jax.Array:
        """Noise-weighted inner product  <h1 | h2> = 4 Re[∑_f h1†·S⁻¹·h2] Δf.

        Parameters
        ----------
        h1, h2 : jax.Array, shape ``(N_batch, 3, N_freq)``
            TDI arrays, batch-first.
        inv_cov : jax.Array, shape ``(N_freq, 3, 3)``
            Inverse noise covariance per frequency bin.

        Returns
        -------
        jax.Array, shape ``(N_batch,)``
            Real inner product per batch entry.
        """
        df = 1.0 / self.t_obs
        # h1, h2: (N_batch, 3, N_freq), inv_cov: (N_freq, 3, 3) → (N_batch,)
        return 4.0 * jnp.einsum("bcf,fcd,bdf->b", h1.conj(), inv_cov, h2).real * df

    # ------------------------------------------------------------------

    def log_likelihood(
        self,
        params: jax.Array,
        data: jax.Array,
        inv_cov: jax.Array,
        kmin: int,
        kmax: int,
        max_batch_size: int | None = None,
        **kwargs,
    ) -> jnp.ndarray:
        """Per-walker log-likelihood  -½ <d−h | d−h>.

        Parameters
        ----------
        params : jax.Array, shape ``(N_walkers, 8)``
            Source parameters per walker.
        data : jax.Array, shape ``(N_walkers, 3, N_freq)``
            Per-walker data windowed to the frequency band ``[kmin, kmax)``.
        inv_cov : jax.Array, shape ``(N_freq, 3, 3)``
            Inverse noise covariance per frequency bin.
        kmin, kmax : int
            Absolute frequency-bin limits of the data window.
        max_batch_size : int, optional
            When given, ``params`` and ``data`` are zero-padded to exactly
            this length before the JAX call so that the compiled shape stays
            fixed across MCMC iterations.  The returned array is always
            sliced back to ``N_walkers``.
        **kwargs
            Forwarded to :meth:`get_data_minus_template` (e.g.
            ``tdi_generation``, ``tdi_combination``).

        Returns
        -------
        jnp.ndarray, shape ``(N_walkers,)``
            Log-likelihood values.
        """
        n_actual = len(params)
        if max_batch_size is not None and n_actual < max_batch_size:
            pad = max_batch_size - n_actual
            params = jnp.concatenate(
                [params, jnp.zeros((pad, params.shape[-1]), dtype=params.dtype)],
                axis=0,
            )
            data = jnp.concatenate(
                [data, jnp.zeros((pad,) + data.shape[1:], dtype=data.dtype)],
                axis=0,
            )

        d_m_h = self.get_data_minus_template(params, data, kmin, kmax, **kwargs)
        # d_m_h: (N_batch, 3, N_freq) — matches inner_product convention directly
        results = -0.5 * self.inner_product(d_m_h, d_m_h, inv_cov)
        return results[:n_actual]

    # ------------------------------------------------------------------

    def mismatch(
        self,
        h1: jax.Array,
        h2: jax.Array,
        inv_cov: jax.Array,
        maximise_phase: bool = False,
    ) -> jax.Array:
        """Phase-maximised mismatch  1 − |<h1|h2>| / √(<h1|h1> <h2|h2>).

        Parameters
        ----------
        h1, h2 : jax.Array, shape ``(N_batch, 3, N_freq)``
            Pre-computed TDI template arrays, batch-first
            (stack ``[A, E, T]`` along ``axis=1``).
        inv_cov : jax.Array, shape ``(N_freq, 3, 3)``
            Inverse noise covariance per frequency bin.
        maximise_phase : bool, optional
            If True, the inner product is taken in absolute value to maximise
            over any overall phase offset between the two templates.

        Returns
        -------
        jax.Array, shape ``(N_batch,)``
            Phase-maximised mismatch per batch entry.
        """
        df = 1.0 / self.t_obs
        # h1, h2: (N_batch, 3, N_freq), inv_cov: (N_freq, 3, 3) → (N_batch,)
        h1h2 = 4.0 * jnp.einsum("bcf,fcd,bdf->b", h1.conj(), inv_cov, h2) * df
        h1h1 = 4.0 * jnp.einsum("bcf,fcd,bdf->b", h1.conj(), inv_cov, h1).real * df
        h2h2 = 4.0 * jnp.einsum("bcf,fcd,bdf->b", h2.conj(), inv_cov, h2).real * df
        if maximise_phase:
            h1h2 = jnp.abs(h1h2)
        else:
            h1h2 = h1h2.real
        return 1.0 - (h1h2 / jnp.sqrt(h1h1 * h2h2))
