# Project Summary — Extended-MFGAN

> **Drop this file into any LLM or send it to a collaborator.**
> It contains the full research context in ~10 minutes of reading time.

---

## 1. Authors and venue

- **Hikmatullo Ismatov** (`hikmatullo.ismatov@kaust.edu.sa`)
- **Prof. Diogo A. Gomes** (`diogo.gomes@kaust.edu.sa`)
- King Abdullah University of Science and Technology (KAUST), Saudi Arabia
- **Target venue**: SIAM Journal on Control and Optimization (**SICON**). Backup: ESAIM:COCV, Numerische Mathematik, JMLR.
- **GitHub**: <https://github.com/hikmatulloismatov1999/MFG-GAN-Paper>

---

## 2. One-paragraph elevator pitch

Existing deep learning solvers for Mean Field Games (MFGs) fall into two
incompatible families: GAN-based solvers (APAC-Net, MFGAN) which give
Wasserstein convergence but only for **separable** Hamiltonians, and
physics-informed solvers (MFDGM) which handle any Hamiltonian but converge
only in `L^p` for `p < 2` with no explicit rate. We introduce a class of
**weakly non-separable Hamiltonians**, where the coupling between density
and momentum is measured by a single parameter `ε`, and propose
**Extended-MFGAN**: a fixed-point iteration that freezes ρ inside the
non-separable residual, so each outer iteration reduces to a separable
sub-problem solvable by any existing GAN-based inner solver. We prove an
`L²` contraction theorem with explicit, sharp rate `κ = εL_R/λ`, yielding
Wasserstein-2 convergence at rate `κ^(k/2)`. The condition `εL_R < λ` is
shown to be both **necessary and sufficient** for linear convergence in
the linearised regime via a counterexample on the torus.

---

## 3. Key research question

> Can we get **all three** of {non-separable Hamiltonians, distributional
> representation of `ρ`, explicit Wasserstein convergence rate} **simultaneously**,
> in some useful regime?

Existing methods give at most two of three. Our contribution: yes, in
the **weakly non-separable regime** `εL_R < λ`.

---

## 4. The mathematical setup in one screen

### 4.1 Mean Field Game system

For a domain `Ω ⊂ ℝ^d`, horizon `T > 0`, the MFG system is the coupled
backward HJB and forward Fokker–Planck:
```
−∂_t φ − ν Δφ + H(x, ρ, ∇φ) = 0           (HJB,  φ(T,x) = g(x,ρ(T,x)))
∂_t ρ  − ν Δρ  − div(ρ ∇_p H(x, ρ, ∇φ)) = 0 (FP,   ρ(0,x) = ρ_0(x))
```

### 4.2 Weakly non-separable Hamiltonian

```
H(x, ρ, p) = H_0(x, p)  −  f_0(x, ρ)  +  ε·R(x, ρ, p)
            |_________| |__________| |__________________|
            c_0-strongly  λ-Lasry-     Lipschitz residual,
            convex in p   Lions monot. constants L_R, L_{R,p}, L_{pρ}
```

- `ε = 0`: separable MFG (recovers APAC-Net regime)
- `R(x,ρ,p) = ρ·p`: bilinear traffic flow (`H_LWR` at `ε = 1`)
- `R(x,ρ)`: density-only sub-case (`L_{R,p} = 0`)

### 4.3 GAN-tractable regime

For density-only coupling (simplified):
```
ε · L_R  <  λ
```

For general `R` (full version in Sec. 5 of the paper):
```
α_ε := c_0 ρ_min − 3 ε L_{R,p} ρ_max  >  0,
κ_ε := [ε L_R + √(ε² L_R² + λ ε² L_{pρ}² ρ_max²/α_ε)] / (2λ)  <  1
```

### 4.4 Extended-MFGAN algorithm

```
Initialise ρ^0 ← ρ_0
for k = 0, 1, …, K−1:
    H^k(x, p) := H_0(x, p) + ε · R(x, ρ^k(x,t), p)        # frozen ρ → separable
    (φ^{k+1}, ρ^{k+1}) ← APAC-Net(H^k, f_0, g, ρ_0)        # inner GAN solve
return (φ^K, ρ^K)
```

### 4.5 Main theorem

**Theorem (L² contraction).** Under Lasry–Lions monotonicity, strong
convexity of `H_0`, bounded densities `ρ_min ≤ ρ ≤ ρ_max`, bounded
momentum `‖∇φ‖_∞ ≤ u_max`, and the GAN-tractable condition,
```
‖ρ^{k+1} − ρ*‖_{L²} ≤ κ_ε · ‖ρ^k − ρ*‖_{L²}.
```

**Corollary (W₂ convergence).** Under the same assumptions,
```
W₂(ρ^k(t,·), ρ*(t,·)) ≤ C(Ω, ρ_max) · κ_ε^{k/2}.
```

**Proposition (sharpness, bilinear case).** For `R = ρ·p` (traffic
flow), the linearised outer map at the no-drift uniform equilibrium
has operator norm exactly `ε|1−ε|/λ`, so the upper bound in the main
theorem is attained.

**Proposition (necessity).** On the ergodic torus example with
`f_0 = λρ`, `R = L_R·ρ`, when `εL_R ≥ λ` the linearised outer iteration
fails to contract. So `εL_R < λ` is sharp.

### 4.6 Proof technique (one paragraph)

**Lasry–Lions duality.** Compute `d/dt ∫ δφ · δρ dx` where δ denotes
the difference between two MFG solutions. Six steps: substitute the
PDEs; diffusion cancels between HJB and FP; integrate by parts; integrate
over `[0,T]` with `δρ(0) = 0` and `δφ(T) = 0` killing boundary terms;
split the resulting bilinear form into a base-Hamiltonian piece (non-
positive by strong convexity) and a coupling-residual piece (controlled
by `L_R`); combine with Lasry–Lions monotonicity. Result:
```
λ ‖δρ‖² ≤ εL_R ‖δρ̄‖ ‖δρ‖ + B_ε ‖δρ̄‖²
```
which on division gives the contraction with the announced rate.
Banach fixed-point theorem closes the iteration.

---

## 5. The deep-learning landscape this paper sits in

| Method | Non-sep. `H` | `ρ` rep. | Convergence | Rate |
|---|---|---|---|---|
| APAC-Net (Lin et al. 2021) | ✗ | distributional | `W_1` | ✗ |
| MFGAN (Ruthotto et al. 2020) | ✗ | distributional | `W_1` | ✗ |
| MFDGM (Assouline–Missaoui 2023) | ✓ | pointwise NN | `L^p`, `p < 2` | ✗ |
| **Extended-MFGAN (ours)** | weakly | distributional | `L²` and `W₂` | **`κ^k`** |

**Trade-off**: we are restricted to `εL_R < λ`. Above this threshold, MFDGM
remains preferred.

---

## 6. Repository layout

```
MFG_GAN_Paper/
├── main_sicon.tex             ← SICON submission entry point (SIAM class)
├── main.tex                   ← Plain article version
├── main_neurips.tex           ← NeurIPS-style version
├── presentation.tex           ← Original 30-min talk (for the paper itself)
├── presentation_mfdgm.tex     ← Teaching presentation about MFDGM + our work (38 slides)
├── speech.tex                 ← Speaker script for our paper talk
├── speech_mfdgm.tex           ← Speaker script for the MFDGM talk
├── cover_letter_sicon.tex     ← Editor cover letter
├── SICON_SUBMISSION.md        ← Submission checklist
├── PROJECT_SUMMARY.md         ← (this file)
├── CHECKLIST.md               ← Open / closed tasks
├── PLAN.md                    ← Original research plan
├── README.md                  ← Repo overview
├── references.bib             ← Bibliography
├── sections/
│   ├── abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_preliminaries.tex
│   ├── 02b_related_work.tex
│   ├── 03_weak_nonseparable.tex   ← Definition 3.1, Prop 3.2 (sharpness), Prop 3.3 (necessity)
│   ├── 04_algorithm.tex             ← Extended-MFGAN pseudocode
│   ├── 05_convergence.tex           ← Assumptions, GAN-tractable cond, Thm 5.1, Cor W₂, Lasry–Lions stability lemma
│   ├── 06_experiments.tex           ← Numerical validation of linearised rate
│   └── 07_conclusion.tex
├── appendix/
│   └── A_proofs.tex                 ← Step 5 expansion, bilinear verification, inexact solves, W₂ vs L²
├── figures/
│   ├── exp1_analytic.pdf
│   ├── exp2_traffic.pdf
│   └── exp3_phase_transition.pdf
└── code/
    ├── hamiltonians.py              ← Quadratic, Traffic, DensityOnly Hamiltonians
    ├── fd_solver.py                 ← Lax-Friedrichs HJB + upwind FP + damped fictitious play
    ├── extended_mfgan.py            ← Outer Picard iteration + MFDGM-style Picard baseline
    ├── metrics.py                   ← L², W₂, residuals
    └── experiments/
        ├── exp1_analytic.py         ← separable limit verification
        ├── exp2_traffic.py          ← bilinear MFG-LWR linearised sweep
        ├── exp3_phase_transition.py ← density-only phase transition
        └── debug_outer.py
```

---

## 7. Numerical experiments — what is and what is NOT done

### What IS done
Three experiments validate the **linearised** rate of the outer fixed-point
map. Each one iterates the analytically derived linearised map and matches
empirical to theoretical κ to four decimal places:

- **Exp 1** (separable limit, ε=0): error → machine zero in one step
- **Exp 2** (bilinear MFG-LWR, λ=0.2): two-sided phase transition at
  ε* ≈ {0.276, 0.724}, empirical κ = ε|1−ε|/λ to 4 digits
- **Exp 3** (density-only sweep, λ=L_R=1): sharp transition at ε*=1,
  empirical κ = ε to 4 digits across `ε ∈ [0.1, 1.5]`

### What is NOT done (explicit gap)
- A real GAN inner solver. The code uses an analytic linearised map, not
  APAC-Net training. Adding a neural-network inner solver is engineering
  work, not theoretical.
- A pipeline-level head-to-head against MFDGM.
- High-dimensional scaling (d ≥ 50).

These are listed transparently in §6 ("What is *not* done here") and §7
(future work). A reviewer will probably ask for at least a small full-NN
demo as a revision request.

---

## 8. Key decisions made (and why)

| Decision | Why |
|---|---|
| **Outer Picard iteration** (not direct nonlinear PDE solve) | Decouples non-separable problem into separable sub-problems; each is GAN-tractable |
| Add **Assumption (bounded momentum)** `‖∇φ‖_∞ ≤ u_max` | `R = ρp` is not uniformly Lipschitz without it; needed for L_R to be finite on the admissible region |
| Add **non-vacuum condition** `ρ ≥ ρ_min > 0` | Required for the p-variation absorption in Step 5 of the proof |
| **Factor 3** in the second GAN-tractable condition | Comes from a three-way split in Step 5; could be reduced but kept for clarity |
| State convergence as `L²` then derive `W₂` corollary | The proof actually gives `L²` at rate κ^k; W₂ follows at κ^(k/2) via the L^1-to-W₂ inequality. Stating W₂ at rate κ^k (as we did initially) would have been wrong |
| **Linearised regime** for numerics, with full disclosure in §6 | The contraction theorem is provably tight in this regime. Avoids overclaiming about the nonlinear regime |

---

## 9. Reviewer issues and how they were addressed

We did an honest self-review and a follow-up deep review. Six issues were
fixed:

1. **W₂ rate misstated** as κ^k (it's κ^(k/2)) — fixed in Thm 5.1 + new Cor 5.2
2. **R Lipschitz uniform in p** was false for R=ρp — added bounded-momentum Assumption
3. **Factor-3 inconsistency** between main text and appendix — aligned to 3
4. **Prop 3.2 proof secretly switched** from bilinear to density-only — rewrote correctly for R=ρp
5. **Traffic ε parameterisation** inconsistent — defined family H_ε(x,ρ,p) = ½p² − p + ερp explicitly
6. **Experiments overstated** as "Extended-MFGAN convergence" — reframed as "linearised rate validation"

Plus minor:
- "GAN methods provably outperform" → "provide stronger convergence guarantees within the GAN-tractable regime"
- exp3 ε=1 labeled "threshold" not "diverging"
- references.bib `huang2019mfg` corrected to actual authors (Huang/Di/Du/Chen 2020)

---

## 10. Submission plan

**Target**: SIAM Journal on Control and Optimization (SICON).

**Why SICON**:
- Lasry–Lions duality is control-theoretic, central to SICON's scope
- Convergence rate result for a fixed-point iteration on HJB-FP system
- MFG-LWR traffic flow is a recurring SICON example
- Diogo Gomes is well known in the SICON community

**Backup venues**: ESAIM:COCV, Numerische Mathematik, JMLR.

**Suggested editors**: Pierre Cardaliaguet, Beatrice Acciaio, Alain Bensoussan, René Carmona.
**Suggested referees**: Yves Achdou, Mathieu Laurière, Lars Ruthotto, Levon Nurbekyan.

**Open items before submission** (see CHECKLIST.md):
- Diogo's signoff on rewritten Prop 3.2 and bounded-momentum assumption
- Verify "first explicit rate" claim against Lavigne–Pfeiffer 2023
- Optional: add small finite-difference reference experiment for bilinear MFG-LWR
- arXiv preprint (math.OC + cs.LG cross-list)
- SICON portal submission with cover letter

---

## 11. Five things to NOT forget

1. **The sharpness is in the linearised regime.** We do NOT claim it in the full nonlinear regime. Be careful in talks.
2. **`κ^k` is `L²`. `W₂` is `κ^(k/2)`.** Square-root scaling. Common reviewer trap.
3. **Factor-3** in the second GAN-tractable condition is honest, not a typo. Came from the three-way split of the current term.
4. **Bounded momentum `u_max` is an assumption**, not a derived fact. Standard for viscosity solutions of coercive HJB, but we should explicitly cite Lasry–Lions or Carmona–Delarue when invoking it.
5. **Extended-MFGAN and MFDGM are complementary**, not competing. MFDGM is the right tool outside the GAN-tractable regime. Frame the contribution as filling a gap, not as a replacement.

---

## 12. Quick recipe for resuming work

1. **Pull the repo:** `cd ~/Desktop/Learning/claude/MFG_GAN_Paper && git pull`
2. **Read this file + CHECKLIST.md** — current state, open items
3. **Read sections/05_convergence.tex** — the math heart of the paper
4. **Read sections/06_experiments.tex** — what numerical claims we make
5. **Run experiments to verify:**
   ```
   cd code/experiments
   python3 exp1_analytic.py
   python3 exp2_traffic.py
   python3 exp3_phase_transition.py
   ```
   Each runs in <30 seconds; checks match the tables in §6.
6. **Resume from the highest-priority open item in CHECKLIST.md.**

---

## 13. Glossary (one-line definitions)

- **MFG** (Mean Field Game): continuum-of-rational-agents game described by coupled HJB+FP.
- **HJB**: Hamilton–Jacobi–Bellman, backward PDE for the value function.
- **FP**: Fokker–Planck, forward PDE for the density.
- **Hamiltonian**: the Legendre transform of the running cost; determines optimal velocity via `∇_p H`.
- **Separable**: `H(x,ρ,p) = H_0(x,p) − f_0(x,ρ)`; the GAN family's hard requirement.
- **GAN-tractable regime**: `ε L_R < λ` (the regime our theorem applies in).
- **APAC-Net**: GAN-based MFG solver of Lin et al. 2021; separable H only.
- **MFDGM**: PINN-based MFG solver of Assouline–Missaoui 2023; any H but weak guarantee.
- **Wasserstein-2 (W₂)**: optimal-transport distance between distributions; geometrically meaningful.
- **Lasry–Lions duality**: the integration-by-parts identity used to prove MFG uniqueness; central tool of our proof.
- **Hopf formula**: rewrites an HJ equation as a min-max; requires separability.

---

*End of summary. Total reading time: ~10 minutes. For technical depth, read sections 03 (definitions), 05 (theorem + proof), and the appendix.*
