"""Extended-MFGAN outer loop.

Implements Algorithm 1 from the paper:
    for k = 0, 1, ..., K-1:
        H^k(x,p) = H0(x,p) + eps * R(x, rho^k, p)
        (phi^{k+1}, rho^{k+1}) = InnerSolver(H^k, f0, g, rho0)
    return (phi^K, rho^K)

The inner solver is the FD fictitious-play from fd_solver.inner_solve.
"""

import numpy as np
from fd_solver import inner_solve


def run(ham, rho0, g, dx, dt, T_steps,
        K_outer=15, n_inner=30, tol_inner=1e-6,
        damping=0.3, rho_true=None, verbose=True):
    """Run the Extended-MFGAN outer iteration.

    Parameters
    ----------
    ham       : MFGHamiltonian
    rho0      : (Nx,) initial density
    g         : (Nx,) terminal cost
    dx, dt    : grid spacings
    K_outer   : number of outer iterations
    n_inner   : max fictitious-play steps per inner solve
    tol_inner : inner-loop tolerance
    rho_true  : (Nt, Nx) ground-truth density (optional; for error tracking)
    verbose   : print per-iteration info

    Returns
    -------
    results : dict with keys
        'rho'         — final density (Nt, Nx)
        'phi'         — final value function (Nt, Nx)
        'outer_errors'— list of ||rho^{k+1} - rho^k||_L2 per outer step
        'true_errors' — list of ||rho^k - rho*||_L2  (if rho_true provided)
        'kappa_emp'   — empirical contraction factor list
    """
    Nx = len(rho0)
    Nt = T_steps

    rho_bar = np.tile(rho0, (Nt, 1))  # initial frozen density
    phi = np.zeros((Nt, Nx))

    outer_errors = []
    true_errors  = []
    kappa_emp    = []

    for k in range(K_outer):
        phi_new, rho_new, _ = inner_solve(
            ham, rho_bar, rho0, g, dx, dt, ham.nu,
            n_inner=n_inner, tol=tol_inner, damping=damping
        )

        # Outer convergence: ||rho^{k+1} - rho^k||_L2 (time-averaged)
        err = np.sqrt(np.mean((rho_new - rho_bar) ** 2)) * dx
        outer_errors.append(err)

        # Empirical contraction factor
        if k >= 1 and outer_errors[-2] > 1e-14:
            kappa_emp.append(outer_errors[-1] / outer_errors[-2])

        # True error (if ground truth available)
        if rho_true is not None:
            terr = np.sqrt(np.mean((rho_new - rho_true) ** 2)) * dx
            true_errors.append(terr)

        if verbose:
            kappa_str = f"  kappa_emp={kappa_emp[-1]:.3f}" if kappa_emp else ""
            terr_str  = f"  ||rho-rho*||={true_errors[-1]:.4e}" if true_errors else ""
            print(f"  Outer k={k+1:2d}  ||delta_rho||={err:.4e}{kappa_str}{terr_str}")

        rho_bar = rho_new
        phi     = phi_new

    return {
        "rho":          rho_bar,
        "phi":          phi,
        "outer_errors": outer_errors,
        "true_errors":  true_errors,
        "kappa_emp":    kappa_emp,
    }


def mfdgm_run(ham, rho0, g, dx, dt, T_steps,
              K_outer=15, n_inner=30, tol_inner=1e-6,
              damping=0.3, rho_true=None, verbose=True):
    """Picard baseline (NOT a faithful MFDGM implementation).

    A faithful MFDGM \\citep{assouline2023deep} parameterises both
    rho and phi as neural networks and minimises the joint HJB+FP
    residual; that requires a full PINN training loop and is left to
    future work.  This routine is kept as a Picard-on-full-H baseline
    for code-level comparison: it runs the same inner FD solve as
    Extended-MFGAN but updates rho_bar to the most recent outer
    iterate rather than freezing it.  For density-only coupling and
    for inner solves that converge to the SAME fixed point on each
    outer step, the two outer iterations coincide; for non-separable
    couplings they can differ.
    """
    Nx = len(rho0)
    Nt = T_steps

    rho = np.tile(rho0, (Nt, 1))
    phi = np.zeros((Nt, Nx))

    outer_errors = []
    true_errors  = []
    kappa_emp    = []

    for k in range(K_outer):
        # In MFDGM, H depends on the current rho (not a frozen copy).
        # We re-solve with ham but pass rho as the frozen rho_bar.
        phi_new, rho_new, _ = inner_solve(
            ham, rho, rho0, g, dx, dt, ham.nu,
            n_inner=n_inner, tol=tol_inner, damping=damping
        )

        err = np.sqrt(np.mean((rho_new - rho) ** 2)) * dx
        outer_errors.append(err)

        if k >= 1 and outer_errors[-2] > 1e-14:
            kappa_emp.append(outer_errors[-1] / outer_errors[-2])

        if rho_true is not None:
            terr = np.sqrt(np.mean((rho_new - rho_true) ** 2)) * dx
            true_errors.append(terr)

        if verbose:
            kappa_str = f"  kappa_emp={kappa_emp[-1]:.3f}" if kappa_emp else ""
            print(f"  MFDGM  k={k+1:2d}  ||delta_rho||={err:.4e}{kappa_str}")

        rho = rho_new
        phi = phi_new

    return {
        "rho":          rho,
        "phi":          phi,
        "outer_errors": outer_errors,
        "true_errors":  true_errors,
        "kappa_emp":    kappa_emp,
    }
