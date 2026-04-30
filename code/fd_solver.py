"""Finite-difference solver for 1-D MFGs on a periodic spatial domain.

Solves the coupled HJB--Fokker-Planck system with periodic BC on x in [0,1]:
    HJB (backward): partial_t phi + nu*Lap phi - H_eff(x, partial_x phi) + f0(x,rho) = 0
                    phi(T, x) = g(x)
    FP  (forward):  partial_t rho = nu*Lap rho + partial_x(rho * grad_p_H_eff)
                    rho(0, x) = rho0(x)

Uses Lax-Friedrichs numerical viscosity for stability of the HJ nonlinearity,
plus adaptive CFL sub-stepping when needed.
"""

import numpy as np


# ------------------------------------------------------------------ #
# HJB backward solver (Lax-Friedrichs)                                #
# ------------------------------------------------------------------ #

def solve_hjb(ham, rho_for_Heff, rho_for_f0, g, dx, dt, nu):
    """Solve the HJB equation backward in time using Lax-Friedrichs.

    rho_for_Heff : (Nt,Nx)  density used in H_eff = H0 + eps*R(rho_for_Heff, p)
                            (frozen ``rho_bar`` in the outer Picard iteration)
    rho_for_f0   : (Nt,Nx)  density used in the f0 source term
                            (current iterate during inner fictitious play)
    """
    Nt, Nx = rho_for_Heff.shape
    phi = np.zeros((Nt, Nx))
    phi[-1] = g.copy()

    for n in range(Nt - 2, -1, -1):
        phi[n] = _hjb_step_lf(
            phi[n + 1], rho_for_Heff[n + 1], rho_for_f0[n + 1],
            ham, dx, dt, nu
        )

    return phi


def _hjb_step_lf(phi_next, rho_Heff, rho_f0, ham, dx, dt, nu):
    """One backward step of HJB with Lax-Friedrichs viscosity, sub-stepped if needed."""
    p_est = _grad_x_central(phi_next, dx)
    v_est = ham.grad_p_H_eff(rho_Heff, p_est)
    alpha = max(float(np.max(np.abs(v_est))), 1e-6)
    n_sub = max(1, int(np.ceil(dt * alpha / (0.5 * dx))))
    dt_sub = dt / n_sub

    phi = phi_next.copy()
    for _ in range(n_sub):
        p = _grad_x_central(phi, dx)
        H_val = ham.H_eff(rho_Heff, p)
        v = ham.grad_p_H_eff(rho_Heff, p)
        alpha_local = max(float(np.max(np.abs(v))), 1e-6)
        lf_visc = 0.5 * alpha_local * dx * _laplacian(phi, dx)
        diff = nu * _laplacian(phi, dx)
        f_val = ham.f0(rho_f0)        # f0 uses current iterate density
        phi = phi + dt_sub * (diff + lf_visc - H_val + f_val)
    return phi


# ------------------------------------------------------------------ #
# Fokker-Planck forward solver (upwind)                               #
# ------------------------------------------------------------------ #

def solve_fp(ham, rho_frozen, phi, rho0, dx, dt, nu):
    """Solve the FP equation forward in time using upwind for advection."""
    Nt, Nx = phi.shape
    rho = np.zeros((Nt, Nx))
    rho[0] = rho0.copy()
    mass = rho0.sum() * dx

    for n in range(Nt - 1):
        rho[n + 1] = _fp_step(rho[n], rho_frozen[n], phi[n], ham, dx, dt, nu, mass)

    return rho


def _fp_step(rho_n, rho_bar_n, phi_n, ham, dx, dt, nu, mass_target):
    """One forward step of FP with conservative upwind advection.

    The paper's FP equation is  partial_t rho = nu*Lap rho + div(rho * grad_p H_eff),
    which is the standard continuity equation  partial_t rho + div(rho * u) = nu*Lap rho
    with physical agent velocity  u = -grad_p H_eff = -v.
    We discretize using conservative upwind on faces i+1/2.
    """
    p = _grad_x_central(phi_n, dx)
    v = ham.grad_p_H_eff(rho_bar_n, p)
    u = -v  # physical agent velocity

    alpha = max(float(np.max(np.abs(u))), 1e-6)
    n_sub = max(1, int(np.ceil(dt * alpha / (0.5 * dx))))
    dt_sub = dt / n_sub

    rho = rho_n.copy()
    for _ in range(n_sub):
        # Face-centred velocity at i+1/2
        u_face = 0.5 * (u + np.roll(u, -1))
        u_face_p = np.maximum(u_face, 0)
        u_face_m = np.minimum(u_face, 0)
        # Conservative upwind flux at face i+1/2:
        # F_{i+1/2} = rho_i * u+ + rho_{i+1} * u-
        F_face = rho * u_face_p + np.roll(rho, -1) * u_face_m
        # advection contribution: -(F_{i+1/2} - F_{i-1/2})/dx  (FV update)
        adv = -(F_face - np.roll(F_face, 1)) / dx
        diff = nu * _laplacian(rho, dx)
        rho = rho + dt_sub * (adv + diff)
        rho = np.maximum(rho, 1e-12)
        # Mass renormalisation (periodic domain conserves total mass exactly
        # in continuum; we correct for accumulated discretisation drift)
        rho *= mass_target / (rho.sum() * dx)
    return rho


# ------------------------------------------------------------------ #
# Inner fixed-point iteration                                          #
# ------------------------------------------------------------------ #

def inner_solve(ham, rho_bar, rho0, g, dx, dt, nu,
                n_inner=60, tol=1e-6, damping=0.5):
    """Solve the separable MFG sub-problem with frozen rho_bar via damped fictitious play.

    The damped update  rho <- (1-d)*rho + d*rho_new  is standard for
    converging fictitious play in MFGs; without damping, vanilla alternation
    typically oscillates.  See Cardaliaguet (2013), Section 5.
    """
    Nt, Nx = rho_bar.shape
    rho = np.tile(rho0, (Nt, 1))
    phi = np.zeros((Nt, Nx))
    residuals = []

    for _ in range(n_inner):
        # HJB: H_eff uses frozen rho_bar, f0 uses CURRENT density iterate
        phi_new = solve_hjb(ham, rho_bar, rho, g, dx, dt, nu)
        rho_new = solve_fp(ham, rho_bar, phi_new, rho0, dx, dt, nu)
        # Damped update
        rho_next = (1 - damping) * rho + damping * rho_new
        res = float(np.sqrt(np.mean((rho_next - rho) ** 2)) * dx)
        residuals.append(res)
        rho = rho_next
        phi = phi_new
        if res < tol:
            break

    return phi, rho, residuals


# ------------------------------------------------------------------ #
# Stencils (periodic BC)                                               #
# ------------------------------------------------------------------ #

def _grad_x_central(u, dx):
    return (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

# alias kept for backward compatibility / metrics module
_grad_x = _grad_x_central


def _laplacian(u, dx):
    return (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx ** 2


def _div_upwind_flux(rho, v, dx):
    """Upwind divergence of flux (rho * v): returns d(rho*v)/dx with upwinding on v."""
    v_plus  = np.maximum(v, 0)
    v_minus = np.minimum(v, 0)
    flux_left  = rho * v_plus + np.roll(rho, -1) * v_minus  # flux at i+1/2
    flux_right = np.roll(rho, 1) * np.roll(v_plus, 1) + rho * np.roll(v_minus, 1)  # flux at i-1/2
    # Note: this is -d(rho*v)/dx in conservative form; sign chosen to match
    # FP equation partial_t rho = +div(rho*v) (paper convention)
    return (flux_left - flux_right) / dx


def _div_upwind(flux, dx):
    """Backward-compat alias used in metrics.py."""
    return (flux - np.roll(flux, 1)) / dx
