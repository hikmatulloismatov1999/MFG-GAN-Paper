"""Metrics for comparing MFG solutions."""

import numpy as np


def l2_error(rho1, rho2, dx=1.0):
    """Time-averaged L2 error: sqrt(1/T * int ||rho1-rho2||^2 dt)."""
    return np.sqrt(np.mean((rho1 - rho2) ** 2)) * dx


def wasserstein1_1d(rho1, rho2, dx):
    """1-Wasserstein distance for 1-D densities on a uniform grid.
    W1 = integral |CDF1 - CDF2| dx.
    """
    cdf1 = np.cumsum(rho1) * dx
    cdf2 = np.cumsum(rho2) * dx
    # Normalise to proper CDFs
    cdf1 /= cdf1[-1]
    cdf2 /= cdf2[-1]
    return np.sum(np.abs(cdf1 - cdf2)) * dx


def wasserstein2_1d(rho1, rho2, dx):
    """2-Wasserstein distance for 1-D densities (Brenier formula).
    W2^2 = integral (F1^{-1}(u) - F2^{-1}(u))^2 du
         = sorted L2 norm of quantile functions.
    For discrete distributions, sort particles and compare.
    """
    # Build empirical CDFs and invert via interpolation
    x = np.arange(len(rho1)) * dx
    u = np.linspace(0, 1, 1000)

    def quantile(rho, u_vals):
        cdf = np.cumsum(rho) * dx
        cdf = cdf / cdf[-1]
        # Prepend 0 so cdf starts at 0
        cdf_ext = np.concatenate([[0.0], cdf])
        x_ext   = np.concatenate([[x[0] - dx], x])
        return np.interp(u_vals, cdf_ext, x_ext)

    q1 = quantile(rho1, u)
    q2 = quantile(rho2, u)
    return np.sqrt(np.mean((q1 - q2) ** 2))


def contraction_rate(errors):
    """Estimate empirical contraction rate from a sequence of errors.
    Returns median of consecutive ratios (robust to noise).
    """
    ratios = [errors[k + 1] / errors[k]
              for k in range(len(errors) - 1) if errors[k] > 1e-14]
    return float(np.median(ratios)) if ratios else float("nan")


def hjb_residual(phi, rho, ham, dx, dt, nu):
    """Mean-squared HJB residual ||partial_t phi + nu*Lap phi - H_eff + f0||^2."""
    from fd_solver import _grad_x, _laplacian
    Nt, Nx = phi.shape
    res_sq = []
    for n in range(1, Nt - 1):
        dt_phi = (phi[n + 1] - phi[n - 1]) / (2 * dt)
        p      = _grad_x(phi[n], dx)
        H      = ham.H_eff(rho[n], p)
        lap    = nu * _laplacian(phi[n], dx)
        f      = ham.f0(rho[n])
        r      = dt_phi + lap - H + f
        res_sq.append(np.mean(r ** 2))
    return float(np.sqrt(np.mean(res_sq)))


def fp_residual(rho, phi, ham, dx, dt, nu):
    """Mean-squared FP residual ||partial_t rho - nu*Lap rho - div(rho*v)||^2."""
    from fd_solver import _grad_x, _laplacian, _div_upwind
    Nt, Nx = rho.shape
    res_sq = []
    for n in range(1, Nt - 1):
        dt_rho   = (rho[n + 1] - rho[n - 1]) / (2 * dt)
        p        = _grad_x(phi[n], dx)
        v        = ham.grad_p_H_eff(rho[n], p)
        lap      = nu * _laplacian(rho[n], dx)
        div_flux = _div_upwind(rho[n] * v, dx)
        r        = dt_rho - lap - div_flux
        res_sq.append(np.mean(r ** 2))
    return float(np.sqrt(np.mean(res_sq)))
