"""Hamiltonian definitions for MFG experiments.

Each class implements H(x, rho, p) = H0(x,p) - f0(x,rho) + eps*R(x,rho,p).
All operations are vectorised over a 1-D spatial grid.
"""

import numpy as np


class MFGHamiltonian:
    """Abstract base: override H0, grad_p_H0, f0, R, grad_p_R."""

    def __init__(self, eps: float = 0.0, nu: float = 0.0, lam: float = 1.0):
        self.eps = eps
        self.nu = nu
        self.lam = lam          # Lasry-Lions monotonicity constant for f0

    # ------------------------------------------------------------------ #
    # Subclass interface                                                   #
    # ------------------------------------------------------------------ #
    def H0(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad_p_H0(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def f0(self, rho: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def R(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.zeros_like(p)

    def grad_p_R(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.zeros_like(p)

    # ------------------------------------------------------------------ #
    # Derived quantities                                                   #
    # ------------------------------------------------------------------ #
    def H_eff(self, rho_bar: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Effective separable Hamiltonian H0(p) + eps*R(rho_bar, p)."""
        return self.H0(p) + self.eps * self.R(rho_bar, p)

    def grad_p_H_eff(self, rho_bar: np.ndarray, p: np.ndarray) -> np.ndarray:
        """grad_p of effective Hamiltonian."""
        return self.grad_p_H0(p) + self.eps * self.grad_p_R(rho_bar, p)

    def H_full(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Full non-separable Hamiltonian."""
        return self.H0(p) - self.f0(rho) + self.eps * self.R(rho, p)


# ------------------------------------------------------------------ #
# Concrete Hamiltonians                                               #
# ------------------------------------------------------------------ #

class QuadraticHamiltonian(MFGHamiltonian):
    """H0(p)=0.5 p^2, f0(rho)=lam*rho, R=0 (pure separable MFG).
    Used for Experiment 1 (analytic solution validation).
    """

    def H0(self, p):
        return 0.5 * p ** 2

    def grad_p_H0(self, p):
        return p

    def f0(self, rho):
        return self.lam * rho


class TrafficHamiltonian(MFGHamiltonian):
    """MFG-LWR traffic flow:
        H(x,rho,p) = 0.5 p^2 - (1-rho) p
        Decomposition:
            H0(p)      = 0.5 p^2 - p        (c0 = 1)
            R(rho, p)  = rho * p             (bilinear, LR = u_max, L_{R,p} = rho_max)
            f0(rho)    = lam * rho           (Lasry-Lions with constant lam)
    """

    def H0(self, p):
        return 0.5 * p ** 2 - p

    def grad_p_H0(self, p):
        return p - 1.0

    def f0(self, rho):
        return self.lam * rho

    def R(self, rho, p):
        return rho * p

    def grad_p_R(self, rho, p):
        return rho


class DensityOnlyHamiltonian(MFGHamiltonian):
    """H0(p)=0.5 p^2, R(rho,p)=LR*rho (independent of p), f0=lam*rho.
    Used for phase-transition experiment (L_{R,p}=0 case).
    The fixed-point map has exactly kappa = eps*LR/lam.
    """

    def __init__(self, LR: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.LR = LR

    def H0(self, p):
        return 0.5 * p ** 2

    def grad_p_H0(self, p):
        return p

    def f0(self, rho):
        return self.lam * rho

    def R(self, rho, p):
        # density-only: R(rho) = LR * rho, no p dependence
        return self.LR * rho

    def grad_p_R(self, rho, p):
        return np.zeros_like(p)
