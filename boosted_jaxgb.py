"""JaxGB response with the first-order (in v/c) velocity-corrected link kernel.

The single-link transfer function follows Sec. II A of the paper (``main.tex``,
Eq. ``response_kernel_db``), which rewrites Eq. (39) of arXiv:2509.10038,

    G = G_stat + G_red + G_pa1 + G_pa2 + G_defl + G_sc ,

with, for a link from the emitter j to the receiver i (C_p = xi_p / (2 kappa),
kappa = 1 - n.k, Delta_beta = beta_i - beta_j, a = Delta_beta - (n.Delta_beta) n),

    G_stat = C [e^{-i Phi_1} - e^{-i Phi_2}]
    G_red  = C [beta_i.k e^{-i Phi_2} - beta_j.k e^{-i Phi_1}]
    G_pa1  = C beta_j.n [e^{-i Phi_1} - e^{-i Phi_2}]
    G_pa2  = -C Delta_beta.n e^{-i Phi_1}
    G_defl = -e^{-i Phi_2} S(Delta_Phi) [zeta(n, a) + C k.a]
    G_sc   = zeta(n, beta_i) e^{-i Phi_2} - zeta(n, beta_j) e^{-i Phi_1}

where Phi_1 = Phi_2 + Delta_Phi, Delta_Phi = 2 pi f L kappa / c, and
S(x) = (1 - e^{-i x}) / (i x). All quantities are evaluated at reception time,
except the emitter position, taken at the emission event (retarded geometry).

Conventions are those of :mod:`jaxgb`: k is the GW propagation direction,
n_ij points from the emitter j to the receiver i, and links are ordered
(12, 23, 31, 13, 32, 21).
"""

from __future__ import annotations

from collections.abc import Iterable
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

#: Terms of the single-link kernel (see the module docstring).
ALL_TERMS = ("stat", "red", "pa1", "pa2", "defl", "sc")

#: Links ordered (12, 23, 31, 13, 32, 21); link ij goes from j (emitter) to i (receiver).
LINK_RECEIVER = np.array([0, 1, 2, 0, 2, 1])
LINK_EMITTER = np.array([1, 2, 0, 2, 1, 0])


def _active_terms(model) -> set[str]:
    """Terms to include for ``model``, honouring the legacy ``include_*`` flags."""
    terms = getattr(model, "terms", None)
    terms = set(ALL_TERMS if terms is None else terms)
    if not getattr(model, "include_sc_motion", True):
        terms.discard("sc")
    if not getattr(model, "include_lensing", True):
        terms.discard("defl")
    return terms


def _xi_over_kappa(xi: jax.Array, nk: jax.Array, n_perp_sq: jax.Array) -> jax.Array:
    """Regular evaluation of xi_p / kappa for polarisation-weighted ``xi``.

    ``xi`` is proportional to ``|n_perp|^2 = kappa (1 + n.k)``, so the ratio is
    finite for ``n -> k``: xi / kappa = (1 + n.k) xi / |n_perp|^2. It is
    returned as 0 at exactly ``n = k`` (the limit depends on the direction of
    approach), and the guard keeps gradients finite.
    """
    ok = n_perp_sq > 1e-30
    safe = jnp.where(ok, n_perp_sq, 1.0)
    return jnp.where(ok, (1.0 + nk) * xi / safe, 0.0)


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
    """Galactic binary response with the velocity-corrected single-link kernel.

    Drop-in replacement for :class:`JaxGB`. It supplies retarded link geometry
    and overrides :meth:`_construct_slow_part`; public response methods remain
    inherited from JaxGB.

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
    old_implementation
        Use the parent JaxGB kernel for baseline comparisons.
    include_sc_motion, include_lensing
        Switch off ``G_sc`` and ``G_defl``, respectively. Both default to True.
    terms
        Subset of :data:`ALL_TERMS` to include (default: all). ``("stat",)``
        gives the stationary kernel evaluated with the retarded geometry.
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
        include_sc_motion: bool = True,
        include_lensing: bool = True,
        terms: Iterable[str] | None = None,
    ) -> None:
        super().__init__(orbits, t_obs=t_obs, t0=t0, n=n, window=window)
        self.old_implementation = old_implementation
        self.include_sc_motion = include_sc_motion
        self.include_lensing = include_lensing
        self.terms = None if terms is None else tuple(terms)
        unknown = set(ALL_TERMS if terms is None else terms) - set(ALL_TERMS)
        if unknown:
            raise ValueError(f"unknown kernel terms {sorted(unknown)}; use {ALL_TERMS}")

        self.velocity = (
            self.orbits.compute_velocity(self.t0 + self.tm, [1, 2, 3])
            .swapaxes(0, 1)
            .swapaxes(1, 2)
        ) / c  # (3, 3, n)

        self.ltts = self.orbits.compute_ltt(self.t0 + self.tm)  # (n, 6)
        if self.old_implementation:
            self.position_emit = None
        else:
            transmitter_sc = np.array([1, 2, 0, 2, 1, 0])
            emission_times = (
                self.t0 + self.tm[:, None] - np.asarray(self.ltts)
            )
            self.position_emit = self.orbits.compute_position(
                emission_times, transmitter_sc + 1
            ).transpose(1, 2, 0)  # (6, 3, n), emitter evaluated at emission

        if self.old_implementation:
            print(
                "WARNING: JaxGBFull uses the parent JaxGB response. "
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

        Notes
        -----
        Following ``main.tex``, each link (receiver i, emitter j) is written as
        ``gs = e^{-i Phi_2} [G_stat + G_red + ...] / e^{-i Phi_2}`` with
        ``e^{-i Phi_2}`` the heterodyned GW phase at the receiver, and
        ``e^{-i Phi_1} = e^{-i Phi_2} (1 + s)`` where
        ``s = e^{-i Delta_Phi} - 1 = -2 i alpha S`` is evaluated through the
        sinc form (no cancellation for ``f -> 0`` or ``kappa -> 0``);
        ``alpha = Delta_Phi / 2``. The emitter position is taken at the emission
        time (``position_emit``), so that ``L = c * ltt`` and
        ``r_i - L n = r_j(t - L / c)``. The orbital velocities are those at
        reception time and are assumed constant over the light travel time.
        """
        if self.old_implementation:
            return super()._construct_slow_part(
                f0_fdot_phi0, k, apac_ssb, epec_ssb
            )
        terms = _active_terms(self)

        f0, fdot, phi0 = f0_fdot_phi0[:, 0], f0_fdot_phi0[:, 1], f0_fdot_phi0[:, 2]
        k_s = k.T  # (nsrc, 3)

        # --- geometry of the photon path: n_ij and betas, shapes (6, 3, n) ---
        r_i = self.position[LINK_RECEIVER]
        r_j = jnp.asarray(self.position_emit)  # emitter at emission
        n = r_i - r_j
        n = n / jnp.linalg.norm(n, axis=1, keepdims=True)  # n_ij
        beta = jnp.asarray(self.velocity)  # (3, 3, n), beta_I = v_I / c
        beta_i, beta_j = beta[LINK_RECEIVER], beta[LINK_EMITTER]
        delta_beta = beta_i - beta_j
        a = delta_beta - n * jnp.sum(n * delta_beta, axis=1, keepdims=True)

        # --- polarisation-weighted tensor and scalar products (once) ---------
        eplus, ecross = epec_ssb
        h_ab = (
            apac_ssb[0][:, None, None] * eplus + apac_ssb[1][:, None, None] * ecross
        )  # sum_p A_p e^p_ab, shape (nsrc, 3, 3)
        h_n = jnp.einsum("sij,ljt->slit", h_ab, n)  # (nsrc, 6, 3, n)

        def zeta(w):
            """sum_p A_p zeta_p(n, w); zeta(n) is the sum_p A_p xi_p."""
            return jnp.einsum("slit,lit->slt", h_n, w)

        def dot_k(w):
            return jnp.einsum("sx,lxt->slt", k_s, w)

        nk = dot_k(n)  # n.k
        n_perp = n[None] - nk[:, :, None, :] * k_s[:, None, :, None]
        c_p = 0.5 * _xi_over_kappa(
            zeta(n), nk, jnp.sum(n_perp**2, axis=2)
        )  # C = xi / (2 kappa), summed over polarisations

        # --- phases ------------------------------------------------------------
        kdotp = dot_k(self.position) / c  # k.r_I / c, (nsrc, 3, n)
        xi_wave = self.tm[None, None, :] - kdotp
        f_rx = (f0[:, None, None] + fdot[:, None, None] * xi_wave)[:, LINK_RECEIVER]
        q = jnp.rint(f0 * self.t_obs)
        df = 2.0 * np.pi * (q / self.t_obs)
        om = 2.0 * np.pi * f0
        arg_s = (-phi0[:, None] + (om - df)[:, None] * self.tm[None, :])[
            :, None, :
        ] + np.pi * fdot[:, None, None] * xi_wave**2
        phase_rx = (om[:, None, None] * kdotp - arg_s)[:, LINK_RECEIVER]  # Phi_2
        e_rx = jnp.exp(-1j * phase_rx)  # e^{-i Phi_2}

        ltt = jnp.asarray(self.ltts).T  # (6, n), L / c
        omega_ltt = 2.0 * np.pi * f_rx * ltt[None]  # 2 pi f L / c
        alpha = 0.5 * omega_ltt * (1.0 - nk)  # Delta_Phi / 2
        s_fac = jnp.sinc(alpha / np.pi) * jnp.exp(-1j * alpha)  # S(Delta_Phi)
        s = -2.0j * alpha * s_fac  # e^{-i Delta_Phi} - 1
        e_tx = 1.0 + s  # e^{-i Phi_1} / e^{-i Phi_2}

        kb_i, kb_j = dot_k(beta_i), dot_k(beta_j)  # k.beta_I
        nb_j = jnp.sum(n * beta_j, axis=1)[None]  # n.beta_j
        ndb = jnp.sum(n * delta_beta, axis=1)[None]  # n.Delta_beta

        # --- G / e^{-i Phi_2} ---------------------------------------------------
        g = jnp.zeros_like(alpha, dtype=complex)
        if "stat" in terms:  # C [e^{-i Phi_1} - e^{-i Phi_2}] = C s
            g += -0.5j * zeta(n) * omega_ltt * s_fac
        if "red" in terms:  # C [beta_i.k - beta_j.k e^{-i Delta_Phi}]
            g += c_p * (kb_i - kb_j * e_tx)
        if "pa1" in terms:  # C beta_j.n [e^{-i Delta_Phi} - 1]
            g += c_p * nb_j * s
        if "pa2" in terms:  # -C Delta_beta.n e^{-i Delta_Phi}
            g -= c_p * ndb * e_tx
        if "defl" in terms:  # -S [zeta(n, a) + C k.a]
            g -= s_fac * (zeta(a) + c_p * dot_k(a))
        if "sc" in terms:  # zeta(n, beta_i) - zeta(n, beta_j) e^{-i Delta_Phi}
            g += zeta(beta_i) - zeta(beta_j) * e_tx

        return e_rx * g  # (nsrc, 6, n)
    
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
