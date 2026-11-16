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
        window: str | tuple | jax.Array | None = "boxcar",
        old_implementation: bool = False,  # for testing only
    ) -> None:
        super().__init__(orbits, t_obs=t_obs, t0=t0, n=n, window=window)
        self.old_implementation = old_implementation

        self.velocity = (
            self.orbits.compute_velocity(self.t0 + self.tm, [1, 2, 3])
            .swapaxes(0, 1)
            .swapaxes(1, 2)
        ) / c  # (3, 3, n)

        self.ltts = self.orbits.compute_ltt(self.t0 + self.tm)

        if self.old_implementation:
            print(
                "WARNING: JaxGBFull velocity corrections are disabled. "
                "This option is for testing only."
            )

    # ------------------------------------------------------------------

    def _construct_slow_part(
        self,
        f0_fdot_phi0: jax.Array,
        k: jax.Array,
        apac_ssb: tuple[jax.Array, jax.Array],
        epec_ssb: tuple[jax.Array, jax.Array],
    ) -> jax.Array:
        """Heterodyned slow response using the velocity-corrected kernel.

        Parameters
        ----------
        f0_fdot_phi0 : jax.Array, shape (nsrc, 3)
            Per-source ``[f0, fdot, phi0]``.
        k : jax.Array, shape (3, nsrc)
            GW propagation unit vectors.
        apac_ssb : tuple of complex jax.Array, each (nsrc,)
            Plus and cross strain amplitudes in the SSB frame.
        epec_ssb : tuple of jax.Array, each (nsrc, 3, 3)
            Plus and cross polarisation tensors in the SSB frame.

        Returns
        -------
        jax.Array, shape (nsrc, 6, n)
            Slow heterodyned link responses.
            Link order: ``[12, 23, 31, 13, 32, 21]``.
        """
        f0, fdot, phi0 = f0_fdot_phi0[:, 0], f0_fdot_phi0[:, 1], f0_fdot_phi0[:, 2]
        receiver_sc = jnp.array([0, 1, 2, 0, 2, 1])
        transmitter_sc = jnp.array([1, 2, 0, 2, 0, 1])

        # JaxGB 0.2.2 defines r as receiver position minus transmitter position.
        # Its three independent arms are ordered 12, 23, 31; reverse them to
        # obtain the six directed links in the public output order.
        r = jnp.concatenate((self.r, -self.r[::-1]), axis=0)
        kdotr = jnp.einsum("sx,lxt->slt", k.T, r)

        kdotp_raw = jnp.einsum("sx,cxt->sct", k.T, self.position) / c
        xi = self.tm[None, None, :] - kdotp_raw
        fi = f0[:, None, None] + fdot[:, None, None] * xi
        fonfs = fi[:, receiver_sc, :] / self.fstar

        eplus, ecross = epec_ssb
        a_ij = (
            jnp.einsum("sij,ljt->slit", eplus, r)
            * r[None, :, :, :]
            * apac_ssb[0][:, None, None, None]
            + jnp.einsum("sij,ljt->slit", ecross, r)
            * r[None, :, :, :]
            * apac_ssb[1][:, None, None, None]
        ).sum(axis=2)

        q = jnp.rint(f0 * self.t_obs)
        df = 2.0 * np.pi * (q / self.t_obs)
        om = 2.0 * np.pi * f0
        arg_s = (
            -phi0[:, None] + (om - df)[:, None] * self.tm[None, :]
        )[:, None, :] + np.pi * fdot[:, None, None] * xi**2
        phase = om[:, None, None] * kdotp_raw - arg_s
        phase_rx = phase[:, receiver_sc, :]

        alpha = 0.5 * fonfs * (1.0 - kdotr)
        denominator = 1.0 - kdotr

        velocity = jnp.asarray(self.velocity)
        kdotv = jnp.einsum("sx,xct->sct", k.T, velocity)
        kdotv_tx = kdotv[:, transmitter_sc, :]
        kdotv_rx = kdotv[:, receiver_sc, :]
        ndotv_tx = (r * velocity[transmitter_sc]).sum(axis=1)
        ndotv_rx = (r * velocity[receiver_sc]).sum(axis=1)

        b_rx = 1.0 - kdotv_tx + ndotv_rx[None, :, :]
        b_tx = (
            1.0
            - kdotv_rx
            + (ndotv_tx - 2.0 * ndotv_rx)[None, :, :]
        )
        if self.old_implementation:
            b_tx = jnp.ones_like(b_tx)
            b_rx = jnp.ones_like(b_rx)

        base_response = (
            -0.5j
            * a_ij
            * fonfs
            * jnp.sinc(alpha / np.pi)
            * jnp.exp(-1j * (alpha + phase_rx))
        )
        velocity_correction = (
            -a_ij
            / (2.0 * denominator)
            * (
                (b_tx - 1.0) * jnp.exp(-1j * phase_rx)
                - (b_rx - 1.0) * jnp.exp(-1j * (2.0 * alpha + phase_rx))
            )
        )
        return base_response + velocity_correction
    
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
        kmins = self.get_kmin(f0=params[:, 0])

        # Compute residual: data - sum(templates)
        _x, _y, _z = self.get_tdi(params,**kwargs)
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
