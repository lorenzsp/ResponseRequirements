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
    ) -> None:
        super().__init__(orbits, t_obs=t_obs, t0=t0, n=n)
        
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
        # r convention: [r_12, r_13, r_23, r_31]  (identical to parent)
        # r = jnp.zeros((4, 3, self.n))
        # r = r.at[0].set(self.position[1] - self.position[0])
        # r = r.at[1].set(self.position[2] - self.position[0])
        # r = r.at[2].set(self.position[2] - self.position[1])
        # r = r.at[3].set(-r[1])
        # r /= self.arm_length
        
        # # option 1
        # # MOSAS = np.array([12, 23, 31, 13, 32, 21])
        # r = jnp.zeros((4, 3, self.n))
        # r = r.at[0].set((self.position[1] - self.position[0])/(self.ltts[:,0] * c))
        # r = r.at[1].set((self.position[2] - self.position[0])/(self.ltts[:,3] * c))
        # r = r.at[2].set((self.position[2] - self.position[1])/(self.ltts[:,1] * c))
        # r = r.at[3].set(-r[1]/(self.ltts[:,2] * c))
        
        # option 2
        r = jnp.zeros((4, 3, self.n))
        r = r.at[0].set((self.position[1] - self.position[0])/np.linalg.norm(self.position[1] - self.position[0], axis=0))
        r = r.at[1].set((self.position[2] - self.position[0])/np.linalg.norm(self.position[2] - self.position[0], axis=0))
        r = r.at[2].set((self.position[2] - self.position[1])/np.linalg.norm(self.position[2] - self.position[1], axis=0))
        r = r.at[3].set(-r[1]/np.linalg.norm(self.position[1] - self.position[0], axis=0) / c)

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
        vel = self.velocity  # (3_sc, 3_xyz, n), v/c dimensionless

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