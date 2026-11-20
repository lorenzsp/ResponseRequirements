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

The kernel in :meth:`JaxGBFull._construct_slow_part` was checked against the
segwo single-link kernel (``_link_response_boosted``, same equations) for every
term separately and for the sum: agreement to ~1e-10 of the stationary term,
the residual being the O(beta^2) difference between the exact retarded
direction used here and the first-order expression used in segwo.
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


def _add_to(tdi_one: jax.Array, tdi_sum: jax.Array, kmin_off: jax.Array, n: int):
    """Add ``tdi_sum`` (3, n) to the window ``tdi_one`` (3, nf) starting at bin ``kmin_off``.

    Bins of ``tdi_sum`` falling outside ``[0, nf)`` are dropped. The indices are
    clipped before the scatter (JAX clamps out-of-range indices and wraps negative
    ones), and the dropped samples are zeroed, so that nothing is written to the
    edge bins.
    """
    nf = tdi_one.shape[-1]
    idx = jnp.arange(n) + kmin_off
    valid = (idx >= 0) & (idx < nf)
    return tdi_one.at[:, jnp.clip(idx, 0, nf - 1)].add(jnp.where(valid, tdi_sum, 0.0))


vadd = jax.vmap(_add_to, in_axes=(0, 0, 0, None))


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
        gives the stationary kernel evaluated with the geometry selected by
        ``retarded``.
    retarded
        If True (default), the link direction ``n_ij`` is built from the emitter
        position at the emission event, ``r_j(t - L_ij / c)``. If False, the
        instantaneous separation ``r_i(t) - r_j(t)`` is used, as for stationary
        spacecraft (same meaning as the ``retarded`` flag of segwo's
        ``_link_response_boosted``). With zero velocities and ``retarded=False``
        the kernel reduces to the stationary response of Baghi et al. Note that
        the retarded geometry comes from the orbit positions, not from the
        velocities, so ``retarded=True`` is kept even if all ``beta`` were zero.
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
        terms: Iterable[str] | None = None,
        retarded: bool = True,
    ) -> None:
        super().__init__(orbits, t_obs=t_obs, t0=t0, n=n, window=window)
        self.old_implementation = old_implementation
        self.retarded = retarded
        self.terms = None if terms is None else tuple(terms)
        unknown = set(ALL_TERMS if terms is None else terms) - set(ALL_TERMS)
        if unknown:
            raise ValueError(f"unknown kernel terms {sorted(unknown)}; use {ALL_TERMS}")

        self.velocity = (
            self.orbits.compute_velocity(self.t0 + self.tm, [1, 2, 3])
            .swapaxes(0, 1)
            .swapaxes(1, 2)
        ) / c  # (3, 3, n), beta_I = v_I / c at reception time

        self.ltts = self.orbits.compute_ltt(self.t0 + self.tm)  # (n, 6)
        if self.old_implementation or not self.retarded:
            self.position_emit = None
        if self.old_implementation:
            print(
                "WARNING: JaxGBFull uses the parent JaxGB response. "
                "This option is for testing only."
            )
        elif self.retarded:
            self.position_emit = self._emission_positions()  # (6, 3, n)

    def _emission_positions(self) -> np.ndarray:
        """Emitter position at the emission event, r_j(t - L_ij / c), shape (6, 3, n).

        One call per link with a 1-D time vector (as in ``JaxGBaccurate``), which does
        not rely on any broadcasting convention of ``orbits.compute_position``.
        """
        ltts = np.asarray(self.ltts)  # (n, 6)
        out = []
        for link, emitter in enumerate(LINK_EMITTER):
            t_emit = self.t0 + self.tm - ltts[:, link]
            pos = np.asarray(self.orbits.compute_position(t_emit, [int(emitter) + 1]))
            out.append(pos.reshape(self.n, 3).T)  # (3, n)
        return np.stack(out)

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
        time (``position_emit``) if ``retarded`` (default), so that ``L = c * ltt``
        and ``r_i - L n = r_j(t - L / c)``; with ``retarded=False`` the instantaneous
        separation defines ``n``, while ``L = c * ltt`` is kept. The orbital velocities are those at
        reception time and are assumed constant over the light travel time.
        """
        if self.old_implementation:
            return super()._construct_slow_part(f0_fdot_phi0, k, apac_ssb, epec_ssb)
        terms = _active_terms(self)

        f0, fdot, phi0 = f0_fdot_phi0[:, 0], f0_fdot_phi0[:, 1], f0_fdot_phi0[:, 2]
        k_s = k.T  # (nsrc, 3)

        # --- geometry of the photon path: n_ij and betas, shapes (6, 3, n) ---
        r_i = self.position[LINK_RECEIVER]
        if self.retarded:
            r_j = jnp.asarray(self.position_emit)  # emitter at emission
        else:
            r_j = self.position[LINK_EMITTER]  # instantaneous emitter position
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

        self._phase_rx = phase_rx  # kept for tests: gs * exp(1j * phase_rx) = G / e^{-i Phi_2}
        return e_rx * g  # (nsrc, 6, n)

    # ------------------------------------------------------------------

    def get_data_minus_template(
        self,
        params: jax.Array,
        data: jax.Array,
        kmin: int,
        kmax: int | None = None,
        **kwargs,
    ) -> jnp.ndarray:
        """Data minus template, for data windowed to the bins ``[kmin, kmax)``.

        Parameters
        ----------
        params : jax.Array, shape (n_walkers, 8)
            Source parameters.
        data : jax.Array, shape (n_walkers, 3, N_freq)
            Data in the band ``[kmin, kmax)``, ``N_freq = kmax - kmin``.
        kmin : int
            Absolute index of the first frequency bin of ``data``.
        kmax : int, optional
            Unused (the window length is taken from ``data``); kept for API
            compatibility.
        **kwargs
            Forwarded to :meth:`get_tdi`.

        Returns
        -------
        jnp.ndarray, shape (n_walkers, 3, N_freq)
            ``data - template``; template bins outside the window are dropped.
        """
        kmins = self.get_kmin(f0=params[:, 0])
        _x, _y, _z = self.get_tdi(params, **kwargs)
        templates = -jnp.stack([_x, _y, _z], axis=1)  # (n_walkers, 3, n)
        return vadd(data, templates, kmins - kmin, self.n)

    # ------------------------------------------------------------------

    def inner_product(
        self,
        h1: jax.Array,
        h2: jax.Array,
        inv_cov: jax.Array,
    ) -> jax.Array:
        """Noise-weighted inner product  <h1 | h2> = 4 Re[sum_f h1^dagger S^-1 h2] Delta f.

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
        """Per-walker log-likelihood  -1/2 <d-h | d-h>.

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
        """Mismatch  1 - Re<h1|h2> / sqrt(<h1|h1><h2|h2>), optionally phase-maximised.

        Evaluated as ``1/2 <d - s | d - s>`` with the unit-normalised templates
        ``d = h1 / |h1|``, ``s = h2 / |h2|``, which equals ``1 - Re<d|s>`` exactly but
        is free of the ``~1e-16`` cancellation floor of the naive expression (needed
        for the mismatches of 1e-13 and below of orbit-uncertainty studies).

        Parameters
        ----------
        h1, h2 : jax.Array, shape ``(N_batch, 3, N_freq)``
            Pre-computed TDI template arrays, batch-first
            (stack ``[A, E, T]`` along ``axis=1``).
        inv_cov : jax.Array, shape ``(N_freq, 3, 3)``
            Inverse noise covariance per frequency bin.
        maximise_phase : bool, optional
            If True, ``h2`` is first rotated by the overall phase that maximises
            the overlap, i.e. ``1 - |<h1|h2>| / sqrt(<h1|h1><h2|h2>)`` is returned.

        Returns
        -------
        jax.Array, shape ``(N_batch,)``
            Mismatch per batch entry.
        """
        df = 1.0 / self.t_obs
        ip = lambda a, b: 4.0 * jnp.einsum("bcf,fcd,bdf->b", a.conj(), inv_cov, b) * df
        d = h1 / jnp.sqrt(ip(h1, h1).real)[:, None, None]
        s = h2 / jnp.sqrt(ip(h2, h2).real)[:, None, None]
        if maximise_phase:
            s = s * jnp.exp(-1j * jnp.angle(ip(d, s)))[:, None, None]
        diff = d - s
        return 0.5 * ip(diff, diff).real