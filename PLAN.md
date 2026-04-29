# Research Plan: Extended-MFGAN for Weakly Non-Separable Hamiltonians

## Target Venue
- **Primary:** NeurIPS 2025 / ICML 2025  
- **Fallback:** JMLR or SIAM Journal on Applied Mathematics  
- **Deadline:** TBD

---

## Core Claim (One Sentence)
GAN-based MFG solvers outperform physics-informed methods (MFDGM) in the
weakly non-separable regime `ε·L_R < λ`, where we prove Wasserstein-2
convergence with explicit rate `(ε·L_R/λ)^k`.

---

## Paper Status

| Section | Status | Notes |
|---------|--------|-------|
| Abstract | DRAFT | needs revision after experiments |
| 1. Introduction | DRAFT | contributions listed |
| 2. Preliminaries | DRAFT | GAN vs MFDGM comparison table |
| 3. Weak Non-Separable H | DRAFT | Definition, classification, traffic flow |
| 4. Algorithm (Extended-MFGAN) | DRAFT | Algorithm box complete |
| 5. Convergence | DRAFT | Main theorem + proof sketch written |
| 6. Experiments | SKELETON | TODO: run experiments, fill figures |
| 7. Conclusion | DRAFT | — |
| Appendix A (Proofs) | DRAFT | Needs rigorous Step 2 (Lasry-Lions estimate) |

---

## Open Mathematical Questions (Priority Order)

### Q1 — Rigorous Lasry-Lions Estimate (CRITICAL)
The proof of Theorem 5.1 Step 2 uses an informal bound.
Need to rigorously derive:
```
λ·‖ρ₁ - ρ₂‖²_L² ≤ C·ε·L_R·‖ρ̄₁ - ρ̄₂‖_L²·‖ρ₁ - ρ₂‖_L²
```
This requires careful use of the weak formulation of the FP equation
and the duality between HJB and FP (Lasry-Lions duality argument).

**Reference:** Lasry-Lions (2007), Theorem 1.1 + stability estimates.

### Q2 — Equivalence of L² and W₂ for Bounded Densities
The proof sketch uses "equivalence of L² and W₂ for bounded densities."
Need precise statement + reference (or proof).
Villani (2009) Ch. 6 should suffice.

### Q3 — Bilinear Case: Exact vs. Approximate
For R = ρ·p (traffic flow), linearization of R around ρ^k is exact.
Does this mean the outer loop converges in fewer steps (or even 1 step)?
Write a proposition about this.

### Q4 — Necessity of ε·L_R < λ
Is the GAN-tractable condition sufficient or also necessary?
Construct a counterexample where ε·L_R ≥ λ and the iteration diverges.

---

## Experiments TODO

### Experiment 1 (Analytic Comparison)
- [ ] Implement APAC-Net baseline (Python/JAX or PyTorch)
- [ ] Run with ε=0, verify single-step convergence
- [ ] Report: W₂ error, L² relative error, wall-clock time

### Experiment 2 (Traffic Flow — MFG-LWR)
- [ ] Implement Extended-MFGAN outer loop
- [ ] Run deterministic (ν=0) and stochastic (ν=0.5)
- [ ] Compare vs. MFDGM (use code from Assouline & Missaoui if available)
- [ ] Plot: convergence curves, density snapshots, fundamental diagram

### Experiment 3 (Phase Transition)
- [ ] Vary ε ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2}
- [ ] Measure empirical contraction κ_emp at each ε
- [ ] Plot κ_emp vs ε, overlay theoretical bound κ = ε·L_R/λ
- [ ] Mark phase transition at ε* = λ/L_R

### Experiment 4 (High-Dimensional Scaling)
- [ ] Test d = 2, 50, 100, 200 with ε=0.5
- [ ] Report: relative error, iterations to convergence, wall-clock

---

## Implementation Plan

```
MFG_GAN_Paper/
├── code/
│   ├── apac_net.py          # APAC-Net inner GAN solver
│   ├── mfdgm.py             # MFDGM baseline
│   ├── extended_mfgan.py    # Our algorithm (outer loop)
│   ├── hamiltonians.py      # H₀, f₀, R definitions
│   ├── experiments/
│   │   ├── exp1_analytic.py
│   │   ├── exp2_traffic.py
│   │   ├── exp3_phase_transition.py
│   │   └── exp4_highdim.py
│   └── utils/
│       ├── wasserstein.py   # W₂ distance computation
│       └── plotting.py
```

**Framework:** JAX (preferred, given existing Jax work in your directory)

---

## Writing Schedule

| Week | Task |
|------|------|
| 1 | Rigorous proof of Theorem 5.1 (Q1 + Q2) |
| 2 | Implement APAC-Net + Extended-MFGAN code |
| 3 | Run Experiments 1 & 2 |
| 4 | Run Experiments 3 & 4 + phase transition analysis |
| 5 | Fill figures into paper, revise intro + abstract |
| 6 | Polish, related work, rebuttal prep |

---

## Key References to Read

- [ ] Lasry-Lions (2007) — monotonicity + stability theory
- [ ] Lin et al. (APAC-Net) — APAC-Net algorithm details
- [ ] Lavigne & Pfeiffer (2023) — policy iteration for non-separable H
- [ ] Villani (2009) Ch. 6 — W₂ vs L² equivalence
- [ ] Sirignano & Spiliopoulos (DGM) — Deep Galerkin method
