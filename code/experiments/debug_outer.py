"""Debug script: trace one Extended-MFGAN run with small grid."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from hamiltonians import DensityOnlyHamiltonian
from fd_solver import inner_solve

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)


def perturbed_uniform(x, amp=0.1, k=1):
    rho = 1.0 + amp * np.cos(2 * np.pi * k * x)
    rho = np.maximum(rho, 1e-3)
    rho /= rho.sum() * (x[1] - x[0])
    return rho


# Small grid for fast iteration
Nx = 32
T  = 0.5
Nt = 30
x  = np.linspace(0, 1, Nx, endpoint=False)
dx = x[1] - x[0]
dt = T / (Nt - 1)
lam = 1.0
LR  = 1.0
nu  = 0.05
eps = 0.3   # well below threshold; expect kappa ~ 0.3

rho0 = perturbed_uniform(x, amp=0.1, k=1)
g    = np.zeros(Nx)
ham  = DensityOnlyHamiltonian(LR=LR, eps=eps, nu=nu, lam=lam)

print(f"Setup: Nx={Nx}, Nt={Nt}, dx={dx:.4f}, dt={dt:.4f}, nu={nu}, eps={eps}")
print(f"Initial rho: mean={rho0.mean():.4f}, std={rho0.std():.4f}, range=[{rho0.min():.3f},{rho0.max():.3f}]")

rho_bar = np.tile(rho0, (Nt, 1))
outer_errors = []

for k in range(6):
    print(f"\n--- Outer k={k} ---")
    phi, rho, residuals = inner_solve(ham, rho_bar, rho0, g, dx, dt, ham.nu,
                                       n_inner=40, tol=1e-8, damping=0.3)
    print(f"  Inner residuals: {[f'{r:.2e}' for r in residuals[::5]]}")
    print(f"  rho range: [{rho.min():.3f}, {rho.max():.3f}]  std={rho.std():.4f}")
    diff = float(np.sqrt(np.mean((rho - rho_bar) ** 2)) * dx)
    outer_errors.append(diff)
    if k > 0:
        ratio = outer_errors[-1] / outer_errors[-2]
        print(f"  ||rho^k - rho^{{k-1}}|| = {diff:.4e}    ratio={ratio:.3f}")
    else:
        print(f"  ||rho^k - rho^{{k-1}}|| = {diff:.4e}")
    rho_bar = rho.copy()

print(f"\nTheoretical kappa = eps*LR/lam = {eps:.3f}")
print(f"Empirical kappa (last 3 ratios) =")
for k in range(max(1, len(outer_errors)-3), len(outer_errors)):
    print(f"   step {k}: {outer_errors[k]/outer_errors[k-1]:.3f}")
