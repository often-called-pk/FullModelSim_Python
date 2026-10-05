# FullModelSim - Speed & Capability Development Roadmap

_Generated 2026-10-04. Built from FullModelSim's own architecture plus publicly published transient-lap-time-simulation methodology and first principles. No third-party proprietary material used._

## Executive summary

Why the global space-domain collocation NLP is inherently slow for parameter exploration: FullModelSim transcribes the WHOLE lap into one simultaneous nonlinear program and solves it once with IPOPT (transcription.py:317 "nlp = {\"f\": J, \"x\": w, \"g\": g}"; :347 "return ca.nlpsol(\"solver\", \"ipopt\", nlp, opts)"). Problem size grows linearly with track length through N=round(L/OPT_ds) on a uniform mesh (transcription.py:35 "N = int(round(s[-1] / OPT_ds))"; :36 linspace), and the collocation block alone adds nx*N*OPT_d state variables (transcription.py:198 "Xkj = SX.sym(\"Xkj\", nx, N * OPT_d)") plus (OPT_d+1)*N*nx defect/continuity constraints (transcription.py:309). For Barcelona (~155 intervals, nx=23) that is ~15k variables / ~17k constraints in one KKT system. Every IPOPT iteration pays BOTH (a) a sparse KKT factorisation of that system and (b) an exact-Hessian evaluation of the full symbolic Pacejka-5.2 combined-slip tyre at all N*OPT_d collocation points, the tyre is ~60% of the per-knot graph and is built from nested atan/sin/cos/exp/sign (vehModel.py:348 "_fx0 ... D * ca.sin(Cx * ca.atan(B*k - E*(B*k - ca.atan(B*k)))) + Sv"; :334 sign; :338 exp). Total time = per-iteration cost x iteration count, and iteration count is a volatile multiplier on this nonconvex problem (max_iter=6000, userOpts.py:196). Crucially, exploring P parameters today means P fully independent cold two-stage solves: each MLTP() cold-runs the 7-state warm start (MLTP.py:149) then a primal-only-seeded 23-state solve with no dual/barrier reuse (userOpts.py:258-272 set no warm_start_init_point), and a converged full result cannot even be re-injected because the warm_start branch only reads a 7-state data.init (MLTP.py:156-160) and the saved data dict stores no duals (MLTP.py:255-264). So an optimal-control NLP, built for one high-accuracy answer, is the wrong instrument for sweeping hundreds of setups.

Headline of the plan: keep the 23-state collocation NLP as the accuracy anchor, but stop using it as the exploration tool. The single highest-leverage change is to ADD a fast quasi-steady-state (QSS/GGV) screening solver (a numpy-only envelope precompute + forward/backward speed march on the existing discretise() grid, ~1e2-1e4x faster per setup) and TABULATE the tyre/aero evaluation (replace the symbolic Pacejka graph with a C2 B-spline ca.interpolant), so sweeps run on the cheap tier and the NLP is reserved for confirming the shortlist. Around those two anchors, quick wins (CSE+JIT, f_dyn.map, curvature-weighted mesh, screening tolerances + IPOPT warm-start/dual reuse) cut both per-iteration cost and the cold-start penalty immediately, and a formal multi-fidelity ladder with cross-tier warm-starting ties it together.

## Priority table

| # | Item | Phase | Effort | Impact | Files | Expected gain |
|---|------|-------|--------|--------|-------|---------------|
| 1 | CSE + JIT-compile the dynamics/constraint Functions (f_dyn, f_sf, h_eq) | quick-win | S | medium | MLTP.py, MLTP_initial.py, functions/transcription.py, MLTP_paramOptim.py | Constant-factor reduction in NLP function/derivative (func_s) time per iteration; CSE comm... |
| 2 | Vectorise the transcription with f_dyn.map(N) instead of the Python inline loop | quick-win | M | medium | functions/transcription.py, MLTP.py | Lower NLP build time and memory (the one-time assembly cost), and often faster derivative ... |
| 3 | Curvature-weighted non-uniform collocation mesh (single pass) | quick-win | M | high | functions/transcription.py, userOpts.py | Equal-accuracy N reduction of ~30-60% on straight-dominated tracks → proportional drop in ... |
| 4 | Screening solver options + IPOPT warm-start/dual reuse + scaling confirmation | quick-win | S | medium | userOpts.py, functions/transcription.py, functions/hsl.py | Confirming exact Hessian avoids a silent 2-5x iteration blow-up; scaling alignment gives a... |
| 5 | Fast QSS / g-g-v screening solver (new fast tier), HEADLINE | mid-term | L | high | functions/ggv.py, MLTP_screen.py, userOpts.py, plotSDI.py | Per-setup lap-time estimate in ~10-100 ms vs tens of seconds to minutes for a 23-state sol... |
| 6 | Tabulate the tyre as a C2 B-spline interpolant (replace symbolic Pacejka 5.2), HEADLINE | mid-term | L | high | vehModel.py, vehParams.py, functions/tyre_table.py, MLTP.py | Cuts the dominant func_s (tyre eval + AD) component substantially, commonly 2-5x on the t... |
| 7 | Parametric warm-start reuse across sweeps: persist + re-inject primals AND duals | mid-term | M | high | MLTP.py, functions/transcription.py, functions/importfile.py, userOpts.py | Per-setup iteration count from O(50-200) cold to O(5-20) warm → 3-10x fewer iterations acr... |
| 8 | Automated DoE / setup-sweep orchestrator (screen cheap, confirm expensive) | mid-term | M | high | setup_sweep.py, MLTP_screen.py, MLTP.py, MLTP_paramOptim.py | Explore 1e3-1e4 setups per full-solve budget instead of a handful; embarrassingly parallel... |
| 9 | Iterative hp / adaptive mesh refinement with warm-started outer loop | mid-term | L | medium | functions/transcription.py, MLTP.py, userOpts.py | Equal-accuracy total-unknown reduction (stacks on the curvature-mesh item); on high-speed ... |
| 10 | Formalise the two-stage solve into an N-rung multi-fidelity ladder with homotopy | structural | L | high | MLTP.py, MLTP_initial.py, functions/transcription.py, userOpts.py | Typically 1.5-3x fewer IPOPT iterations on the 23-state solve from a QSS-seeded, bicycle-r... |
| 11 | Free-trajectory QSS optimal-control tier (QSS speed, but the line is optimised) | structural | L | high | functions/ggv.py, vehModel_qss.py, functions/transcription.py, MLTP.py | Near-NLP line quality at roughly QSS-order cost (seconds, not minutes) once the envelope i... |
| 12 | Energy / ERS deployment co-optimisation (energy budget as a state + constraint) | structural | L | medium | vehModel.py, MLTP.py, userOpts.py, MLTP_paramOptim.py | Unlocks energy-deployment optimisation and SoC-constrained lap times (qualy vs race-stint ... |
| 13 | Transient time-marching solver with a driver model (middle fidelity tier) | structural | XL | medium | transient_solve.py, functions/driver_model.py, vehModel.py, functions/ggv.py | Seconds per lap (one forward pass + a few driver-tuning iterations) vs the full NLP's tens... |
| 14 | Tyre thermal + wear states as extra ODEs (capability, top-rung only) | structural | XL | low | vehModel.py, vehParams.py, userOpts.py, MLTP.py | New analyses rather than speed: per-corner temperature traces, optimal warm-up strategy, l... |

## Recommended sequencing

Recommended order (impact-over-effort, with dependencies):

PHASE 1, quick wins, do first and in parallel (no dependencies, all low risk):
1. CSE + JIT (establishes the measured baseline, do before any tabulation so the tabulation win is attributable).
2. f_dyn.map(N) vectorisation.
3. Curvature-weighted mesh (single pass).
4. Screening solver options + warm_start_init_point scaffold + exact-Hessian assertion + scaling test.
Gate: re-benchmark with bench_linear_solver.py on BCN and a synthetic track; confirm lap time unchanged (items 1,2) or validated-shifted (item 3).

PHASE 2, mid-term anchors:
5. QSS/GGV screener (HEADLINE), depends on nothing in the NLP; build the g-g-v envelope module first because items 8, 10 and 11 all consume it.
6. Tyre B-spline tabulation (HEADLINE), independent of item 5; the offline MF-sweep script it needs can also feed the item-5 envelope, so build the sweep once and share it.
7. Parametric warm-start + dual persistence, depends on the warm_start_init_point scaffold from item 4.
8. DoE/sweep orchestrator, depends on item 5 (screen) and item 7 (warm-started confirmation solves).
9. Iterative hp/adaptive mesh, depends on item 3 (reuses the non-uniform s_knot path).

PHASE 3, structural:
10. Multi-fidelity ladder + homotopy, consumes item 5 (QSS rung-0) and item 7 (cross-tier warm-starting); should wrap items 5/6/7 into one spec.
11. Free-line QSS tier, depends on item 5's envelope; slots into the item-10 ladder as a new rung.
12. Energy/ERS co-optimisation, independent capability; pair with item 10 homotopy for convergence.
13. Transient time-marching + driver, depends on item 5 (target speed profile) and benefits from item 6 (fast tyre eval in the RHS).
14. Tyre thermal/wear, last; depends on item 10 (homotopy) and item 4 (scaling) for convergence; highest effort, narrowest payoff.

Critical path for 'make setup exploration fast' (the stated goal): items 4 → 7 → 5 → 8, with 6 in parallel. Items 1-3 are free accelerators that apply to every tier.

---

# FullModelSim, Prioritized Development Roadmap

**Goal:** make FullModelSim faster (especially for setup exploration) and more capable, without copying any third-party product. Every item maps onto real files and is justified by public methodology or first principles.

---

## Executive summary, why the global NLP is slow to explore, and the highest-leverage fix

FullModelSim transcribes the **whole lap into one simultaneous NLP** and solves it once with IPOPT (`functions/transcription.py:317` `nlp = {"f": J, "x": w, "g": g}`; `:347` `return ca.nlpsol("solver", "ipopt", nlp, opts)`). That is the right instrument for **one** high-accuracy answer and the wrong one for **sweeping many setups**, for three compounding reasons:

1. **Size scales with track length.** `N = round(L/OPT_ds)` on a uniform mesh (`transcription.py:35`; `:36` `linspace`), and the collocation block adds `nx*N*OPT_d` state variables (`transcription.py:198` `Xkj = SX.sym("Xkj", nx, N * OPT_d)`) plus `(OPT_d+1)*N*nx` defect/continuity constraints (`transcription.py:309`). Barcelona (~155 intervals, nx=23) ≈ 15k vars / 17k constraints in one KKT system.
2. **Every iteration is expensive twice over.** Each IPOPT iteration both factorises that sparse KKT system **and** evaluates the exact Hessian of the full symbolic Pacejka-5.2 combined-slip tyre at all `N*OPT_d` points. The tyre is ~60% of the per-knot graph and is nested `atan/sin/cos/exp/sign` (`vehModel.py:348` `_fx0 ... D*ca.sin(Cx*ca.atan(B*k - E*(B*k - ca.atan(B*k)))) + Sv`; `:334` `ca.sign`; `:338` `ca.exp`), exactly the fill-generating terms in the Jacobian/Hessian.
3. **Iteration count is a volatile multiplier and nothing is reused across setups.** `max_iter=6000` (`userOpts.py:196`), nonconvex problem. Worse, exploring P parameters = P independent **cold** solves: each `MLTP()` cold-runs the 7-state warm start (`MLTP.py:149` `if warm_start is None:`) then a **primal-only**-seeded 23-state solve with no dual/barrier reuse (`userOpts.py:258-272` set no `warm_start_init_point`), and a converged full result **cannot even be re-injected**, the warm-start branch reads only a 7-state `data.init` (`MLTP.py:156-160`) and the saved dict stores no duals (`MLTP.py:255-264`).

**Highest-leverage change:** keep the 23-state NLP as the accuracy anchor, but add a **fast QSS/g-g-v screening solver** (numpy-only envelope + forward/backward speed march on the existing `discretise()` grid, ~1e2–1e4× faster per setup) and **tabulate the tyre** (C2 B-spline `ca.interpolant` replacing the symbolic Pacejka graph). Sweeps then run on the cheap tier; the NLP confirms the shortlist. Quick wins (CSE+JIT, `f_dyn.map`, curvature mesh, warm-start/dual reuse) cut per-iteration cost and the cold-start penalty immediately.

---

## Priority table (ranked by impact ÷ effort)

| # | Item | Phase | Effort | Impact | Expected gain |
|---|------|-------|--------|--------|---------------|
| 1 | CSE + JIT `f_dyn`/`h_eq` | quick-win | S | medium | const-factor func-eval speedup, zero accuracy risk |
| 2 | `f_dyn.map(N)` vectorised transcription | quick-win | M | medium | lower NLP build time/memory |
| 3 | Curvature-weighted mesh (single pass) | quick-win | M | high | 30–60% fewer intervals at equal accuracy |
| 4 | Screening options + IPOPT warm-start/scaling | quick-win | S | medium | unlocks resolve speedups; guards exact Hessian |
| 5 | **QSS / g-g-v screening solver** | mid-term | L | high | ~1e2–1e4× per setup |
| 6 | **Tyre B-spline tabulation** | mid-term | L | high | 2–5× on the tyre portion of each iteration |
| 7 | Parametric warm-start + dual persistence | mid-term | M | high | 3–10× fewer iterations per swept setup |
| 8 | DoE/sweep orchestrator | mid-term | M | high | 1e3–1e4 setups per full-solve budget |
| 9 | Iterative hp / adaptive mesh | mid-term | L | medium | further equal-accuracy unknown reduction |
| 10 | Multi-fidelity ladder + homotopy | structural | L | high | 1.5–3× fewer iters + robustness |
| 11 | Free-trajectory QSS OCP tier | structural | L | high | near-NLP line quality at QSS cost |
| 12 | Energy/ERS deployment co-opt | structural | L | medium | EV deploy optimisation (new capability) |
| 13 | Transient time-marching + driver | structural | XL | medium | seconds/lap with transient fidelity |
| 14 | Tyre thermal/wear states | structural | XL | low | warm-up / degradation analyses |

---

## Phase 1, Quick wins (days; low risk; apply to every tier)

### 1. CSE + JIT-compile the dynamics/constraint Functions
Build `f_dyn`/`h_eq` with `{'cse': True, 'jit': True, 'compiler':'shell', 'jit_options':{'flags':['-O3']}}`. Today `MLTP.py:169` builds `f_dyn` with neither. CSE trims the duplicated `atan/sin` sub-trees shared across the four axle-symmetric corners; JIT compiles the transcendental evaluation. **Zero fidelity change**, the correct baseline to measure tabulation against. *Basis: Andersson et al., CasADi (Math. Prog. Comp. 2019); CasADi cse/jit docs.*

### 2. Vectorise transcription with `f_dyn.map(N)`
Replace the per-interval inline loop (`transcription.py:232` `for k in range(N):` → `:249` `dXkj, Qk = f_dyn(*args)`) with one mapped evaluation, so the dynamics graph is built **once** rather than inlined N times into the NLP, its Jacobian and Hessian. Preserve the column-major packing and the linear-input branch (`:238-244`). *Basis: CasADi `Function.map`/`.expand`; Betts.*

### 3. Curvature-weighted non-uniform mesh (single pass)
Replace `transcription.py:36` `s_knot = np.linspace(...)` with a monitor-function equidistribution `M(s)=1+a|kappa|+b|dkappa/ds|`. Everything downstream already consumes arbitrary `s_knot`/`dsk`. Keep `MLTP.py:34 _interp_to`'s uniform init grid separate; cap `max dsk` so a straight can't swallow a brake point. *Basis: Betts; Patterson & Rao, GPOPS-II.*

### 4. Screening options + IPOPT warm-start/scaling confirmation
`userOpts.py:258-272` omits three levers: add `warm_start_init_point='yes'` (resolves only) with small `warm_start_bound_push`, **assert** `hessian_approximation` stays exact, and test `nlp_scaling_method='none'` (trust the manual `x_s/u_s` normalisation applied at `vehModel.py:553`) vs gradient-based. Expose looser `tol`/`max_iter` for a screening mode (`acceptable_tol=1e-3` already at `:262`). *Basis: Wächter & Biegler, IPOPT (2006); IPOPT options reference.*

---

## Phase 2, Mid-term anchors (the exploration workflow)

### 5. Fast QSS / g-g-v screening solver, HEADLINE
New numpy-only `functions/ggv.py` + `MLTP_screen.py`: precompute the g-g(-v) envelope from the friction-circle + power/aero balance (the integrated form of the per-knot constraints at `MLTP.py:44-49`), find apices on `discretise()`'s `k_knot` (`transcription.py:51`), set `v_apex = sqrt(a_y,max(v)/|kappa|)`, forward/backward march, `T = sum ds/v`. Reuses `discretise()` (`transcription.py:26`) and `ctx.vp/aero/pt`, replaces the IPOPT NLP entirely. **Screen only**, it fixes the line at centreline and ignores the transient/suspension states that are the 23-state model's purpose. *Basis: Brayshaw & Harrison (IMechE 2005); Siegler/Deakin/Crolla (SAE 2000-01-3563); Milliken & Milliken RCVD.*

### 6. Tabulate the tyre as a C2 B-spline, HEADLINE
Offline-sweep MF over (slip ratio, slip angle, Fz, camber); in `vehModel.py` replace the symbolic `_fx0`/`_fy0` + combined-slip assembly with `ca.interpolant('…','bspline', grids, vals)`, keeping load transfer/aero symbolic. Use **bspline (C2) only**, `linear` is C0 and will stall IPOPT. Watch the SX↔MX port (model is SX). Keep the symbolic path for `MLTP_TyreOptim` runs that co-optimise tyre coefficients. *Basis: CasADi interpolant/bspline docs; Gillis, OptiSpline.*

### 7. Parametric warm-start + dual persistence
Save `sol['lam_g']`/`sol['lam_x']` in the result `.mat`, add an `MLTP` branch that accepts a full result and feeds `x0/lam_g0/lam_x0` into `solver()` (available but discarded today, `transcription.py:320,322`; result stored without duals at `MLTP.py:255-264`). Requires item 4's `warm_start_init_point` and identical problem structure between chained solves. *Basis: Pirnay/López-Negrete/Biegler sIPOPT (2012); Zavala & Biegler (Automatica 2009).*

### 8. Automated DoE / sweep orchestrator
`setup_sweep.py`: take `param_specs` like `optimise_design` (`MLTP_paramOptim.py:34`, `:48-51`), sample via `scipy.stats.qmc`, rank thousands on the QSS screener (item 5) in parallel, re-solve top-k on the warm-started NLP (item 7), optionally finish with `optimise_design`. Pure orchestration over existing entry points + `vp_overrides` (`userOpts.py:192`). *Basis: Dal Bianco & Lot QSS sensitivity (VSD 2019); hierarchical screen-then-refine (arXiv:2003.04882).*

### 9. Iterative hp / adaptive mesh refinement
Wrap `build_and_solve_nlp` in a refine loop (error indicator → bisect/merge → warm-started re-solve), reusing item 3's non-uniform path. Allow per-element `OPT_d` (C,D,B already built for arbitrary degree, `transcription.py:32-33`). Keep elements short near hard braking / active-aero switching. *Basis: Betts; Patterson & Rao, GPOPS-II.*

---

## Phase 3, Structural (capability + convergence)

### 10. Multi-fidelity ladder + homotopy
Refactor the hardcoded 7→23 warm start (`MLTP.py:149`, `:181`; constant seeds at `:106-113`) into a data-driven tier list `[qss, m7, m23]`, each interpolated onto the next grid, with optional continuation on `OPT_e` (`userOpts.py:195`, threaded to `transcription.py:219-222`) or config switches. 1.5–3× fewer iterations + robustness on stiff configs. *Basis: Christ/Wischnewski/Heilmeier/Lohmann; Dal Bianco/Lot/Gadola GP2 (2018); Betts continuation.*

### 11. Free-trajectory QSS optimal-control tier
Keep the cheap envelope constraint but re-introduce `n(s), eps(s), v(s)` as decision variables on the existing transcription scaffolding (`MLTP(..., fidelity='qss_freeline')`). Closest fast paradigm to what FullModelSim already is (`transcription.py:255` objective); fixes QSS's fixed-line weakness at QSS-order cost. *Basis: Veneri & Massaro, free-trajectory QSS (VSD 2020); Lot & Dal Bianco curvilinear.*

### 12. Energy / ERS deployment co-optimisation
Promote the post-solve energy integral (`MLTP.py:247`) into an in-the-loop state with a per-lap budget + power-limit constraints in `build_path_constraints` (`MLTP.py:53,75-78`); optionally promote the budget via `optimise_design`. First-order EV question no QSS screen can answer. Pair with item 10 homotopy; regularise bang-bang deploy via existing `ctx.rdu/rdu2`. *Basis: Massaro & Limebeer survey (VSD 2021); Ebbesen/Salazar hybrid energy-optimal control.*

### 13. Transient time-marching + driver model
Integrate `vehModel`'s `dx` (`vehModel.py:551-553`) forward in time (RK4) with a path/speed-tracking driver clipped to the g-g envelope; exercises the transient suspension/tyre-deflection states (`vehModel.py:557`) QSS is blind to. A middle tier for damper/kerb studies, **not** a true optimum. Mind the `sf` singularity when `vx·cos(eps)−vy·sin(eps)→0`. *Basis: Kelly & Sharp; Siegler/Deakin/Crolla.*

### 14. Tyre thermal / wear states
Add per-axle temperature + wear ODEs scaling the Pacejka D-factor, slotting into the state vector with their own `sf`-multiplied derivatives. Enables warm-up / intra-stint analyses. Highest effort, narrowest payoff; last, behind the homotopy ladder and scaling work (`hsl.py:165`). *Basis: West & Limebeer, optimal tyre management; Massaro & Limebeer.*

---

## Validation discipline (applies throughout)
- Items 1–2 must leave lap time **bit-for-bit** unchanged; items 3, 6, 9 shift it slightly, validate each against a fine-uniform / symbolic reference on 2–3 circuits (BCN + a synthetic).
- Benchmark every solver-side change with `bench_linear_solver.py` reporting the **per-iteration** linear-solver cost (the apples-to-apples metric; CLAUDE.md notes MA57 ≈ 4–5× faster/iter than MUMPS).
- Keep one output schema across all tiers so `plotSDI`, the `.mat` saver and the co-opt wrappers work unchanged.
- The fast tiers must be **labelled estimates**; a cheaper tyre or fixed line must never be reported as the optimum.

## Critical path for "fast setup exploration" (the stated goal)
**Item 4 → 7 → 5 → 8**, with **item 6** in parallel. Items 1–3 are free accelerators that apply to every tier.

---

## Feasibility review (adversarial check against the real code)

**Overall:** All load-bearing file:line citations in the roadmap and executive summary check out against the actual code in D:\IRP\FullModelSim_Python (transcription.py, MLTP.py, vehModel.py, userOpts.py, functions/collocation.py, functions/hsl.py, MLTP_initial.py, MLTP_paramOptim.py). The headline is feasible: because vehModel builds m.dx as one explicit sf-scaled ODE vertcat divided by x_s (vehModel.py:551-553), a QSS screener and a time-marcher can both be added alongside the NLP, and because discretise() is casadi-free numpy they can reuse the existing grid. Two genuine (and self-flagged) obstacles survive scrutiny: (a) the g-g-v envelope is not literally numpy-only, MF5.2 lives only as symbolic CasADi (vehModel.py:321-348), so it must be re-derived in numpy or evaluated numerically from ctx.mf; (b) tyre B-spline tabulation hits a real SX/MX wall, the whole model is pure SX (vehModel.py:40) while ca.interpolant is MX, needing low-level ca.bspline or an MX port. Three items need wording/scope adjustment: Item 3 understates scope (warmstart_guesses grid at MLTP.py:100 also assumes uniform knots), Item 9 understates hp effort (single global Xkj/C/D/B at transcription.py:190,198 make per-element degree invasive, not a data-structure tweak), and Item 13's 'D-factor is constant' should read 'temperature/wear-independent' (D is load-dependent, vehModel.py:323-348). No item is infeasible as stated. The corrected top-three re-orders the plan to lead with full-fidelity warm-start/dual reuse (lowest-risk, highest exploration leverage) before the QSS and tabulation anchors."

**Corrected top-three priorities:**
- 1) Parametric warm-start/dual reuse across sweeps (Item 7) BUNDLED with its enabler warm_start_init_point + exact-Hessian assertion (Item 4). This is the single highest risk-adjusted lever for the stated goal (exploration): it keeps full 23-state fidelity, every mechanism is confirmed present (solver+sol already returned at transcription.py:322; only primal x0 passed today at :320; no duals saved at MLTP.py:255-263; no warm_start_init_point in userOpts.py:258-272), and CasADi nlpsol natively accepts lam_g0/lam_x0. Turns O(50-200) cold iters into O(5-20) per neighbouring setup with zero accuracy compromise, unlike QSS it never mis-ranks transient setups. Effort M, low risk.
- 2) Fast QSS / g-g-v screening solver (Item 5, the headline). Confirmed feasible: discretise() is casadi-free (transcription.py:26-58) and reusable, the friction-circle form exists (MLTP.py:44-49), and it cleanly replaces nlpsol (transcription.py:347). It delivers the 1e2-1e4x throughput that makes a real DoE (Item 8) possible. One correction baked into the plan: the combined-slip envelope must either re-implement MF5.2 in numpy or numerically evaluate the existing symbolic CasADi tyre (vehModel.py:321-348) from ctx.mf, it is not literally 'numpy-only' for free. Strictly a screen (fixed line n=0, no transients).
- 3) CSE + JIT the dynamics/constraint Functions (Item 1), ahead of tyre B-spline tabulation (Item 6). CSE+JIT is zero-fidelity, low-effort, and compounds on EVERY remaining solve, the cold anchor, all warm-started resolves from priority 1, and the NLP confirmations of QSS-shortlisted setups, whereas tabulation carries a real SX/MX port obstacle (vehModel is pure SX at vehModel.py:40; ca.interpolant is MX) and a grid-coarseness risk that can shift the optimum. Sequence tabulation (bigger per-iteration ceiling) and the curvature-weighted mesh (Item 3, high impact but needs the warmstart-grid fix at MLTP.py:100) immediately after, as the second wave.

| Item | Verdict | Finding | Location |
|------|---------|---------|----------|
| HEADLINE: QSS/time-marching screener addable alongside NLP +... | sound | Confirmed feasible. m.dx is one explicit ODE set, "dx = ca.vertcat(dvx, dvy, dr, ... dzt_rr) / ca.DM(x_s)" (vehModel.py:551-553), so it can be wrapped in a ca... | vehModel.py:551 |
| Item1 CSE+JIT on f_dyn/h_eq (quick-win) | sound | Confirmed: "f_dyn = ca.Function(\"f_dyn\", [m.x, m.u, m.pv], [m.dx, L], ...)" (MLTP.py:169) and h_eq (MLTP.py:174) are built with NO cse/jit; equivalents exist ... | MLTP.py:169 |
| Item2 Vectorise transcription with f_dyn.map(N) | sound | Confirmed: the Python loop "for k in range(N):" (transcription.py:232) re-inlines "dXkj, Qk = f_dyn(*args)" (:249) per interval, and no .map()/.expand() exists ... | functions/transcription.py:249 |
| Item3 Curvature-weighted non-uniform mesh (single pass) | needs-adjustment | Downstream support is confirmed: s_col/start handle arbitrary dsk (transcription.py:40-41), curvature via np.interp (:51-53). BUT the claim "only the s_knot con... | MLTP.py:100 |
| Item4 Screening knobs + warm_start_init_point + scaling/Hess... | sound | Confirmed: the ipopt dict (userOpts.py:258-272) sets tol (default 1e-4 via :197, applied :261), "acceptable_tol": 1e-3 (:262), mu_init 1e-1, mu_strategy adaptiv... | userOpts.py:258 |
| Item5 Fast QSS / g-g-v screener (HEADLINE) | sound | Feasible: discretise() (transcription.py:26) + k_col resample (:51) are casadi-free, friction-circle form exists (MLTP.py:44-49), and it can replace nlpsol (tra... | functions/transcription.py:26 |
| Item6 Tabulate tyre as C2 B-spline interpolant (HEADLINE) | sound | Target confirmed: "_fx0 ... D * ca.sin(Cx * ca.atan(B*k - E*(B*k - ca.atan(B*k)))) + Sv" (vehModel.py:348), with ca.sign (:334) and ca.exp (:338); Fz is availab... | vehModel.py:40 |
| Item7 Parametric warm-start reuse: persist + re-inject prima... | sound | All premises confirmed: cold start at MLTP.py:149 ("if warm_start is None:"), warm_start branch reads only 7-state data.init (MLTP.py:156-160), solve passes pri... | transcription.py:320 |
| Item8 Automated DoE / setup-sweep orchestrator | sound | Confirmed: "def optimise_design(param_specs, tag, ...)" (MLTP_paramOptim.py:34) promotes each vp field to an SX symbol ("sym = ca.SX.sym(field); setattr(vp, fie... | MLTP_paramOptim.py:34 |
| Item9 Iterative hp / adaptive mesh refinement | needs-adjustment | h-refinement is fine (warm-started outer loop, arbitrary dsk supported). But "per-element degree is mostly a data-structure change" understates it: the transcri... | functions/transcription.py:198 |
| Item10 Multi-fidelity ladder with homotopy | sound | Confirmed: two-rung warm start auto-runs MLTP_initial then warmstart_guesses interpolates onto the 23-state grid (MLTP.py:149,181); the 14 non-bicycle states ar... | MLTP.py:106 |
| Item11 Free-trajectory QSS optimal-control tier | sound | Feasible: build_and_solve_nlp is generic in (m.nx,m.nu,m.ny) and minimises "J = J + ca.mtimes(Qk, B) * dsk[k] ..." (transcription.py:255), so a tiny ~[n,eps,v] ... | functions/transcription.py:255 |
| Item12 Energy/ERS state + budget constraint | sound | Confirmed: energy is currently post-solve only ("E[i+1] = E[i] + 0.5*(P[i]+P[i+1])*...(t...)*2.7778e-4 / pt.eff", MLTP.py:247) and per-knot power limits already... | MLTP.py:247 |
| Item13 Tyre thermal + wear states | needs-adjustment | Mechanism feasible, extra states slot into the sf-scaled dx (vehModel.py:551-553) and couple through the D/mu term. Minor correction: "vehModel.py:348 D-factor... | vehModel.py:348 |

---

## Addendum 2026-10-04: UX, ease-of-use, and language vs algorithm

---

# Addendum, Ease-of-use, UI/UX, and the "C++ would be faster" question

_Appended 2026-10-04. Builds on the 14-item speed & capability roadmap above; does not duplicate it. Scope: the PySide6 desktop app (`app/*.py`) and the performance premise. All claims cite the repo as `file:line`._

This addendum answers a specific brief: fold ease-of-use and UI/UX into the simulator, and reconcile the roadmap with the belief that the reference commercial sim is "super-smooth and fast because it was written in C++." It has three parts: (1) a performance note settling the C++-vs-algorithm question, (2) a prioritised usability tier mapped to `app/*.py`, and (3) sharpened speed priorities that fall out of the premise analysis.

## Part 1, Performance note: language is not the bottleneck, the method is

**The hypothesis on the table:** "the commercial sim is fast because it is C++; FullModelSim feels slower because it is Python, so a C++ rewrite would close the gap." Tested against this repo, the premise is **false**: the compute that dominates a lap solve is *already* compiled C++/Fortran. Python only assembles the problem once. The lever is the **algorithm**, not the language.

### Where the wall-clock actually goes

A solve is: build one big symbolic NLP, hand it to IPOPT, let IPOPT iterate. The two costly things per iteration, a sparse KKT factorisation and an exact-Hessian evaluation of the dynamics+tyre, are both native:

1. **The solver is CasADi's bundled IPOPT (C++).** `functions/transcription.py:347` `return ca.nlpsol("solver", "ipopt", nlp, opts)`. Once `solver(...)` is called (`functions/transcription.py:320`), control leaves Python entirely until convergence, up to `max_iter=6000` iterations (`userOpts.py:196`).
2. **The KKT linear factorisation is Fortran.** MA57/MA97/MUMPS, selected in `functions/hsl.py` and wired at `functions/transcription.py:343-347`. `requirements.txt` notes MA57 factorises "~4–5× faster per IPOPT iteration than MUMPS", the same kernels a C++ sim would call.
3. **The dynamics + Pacejka 5.2 tyre and its AD Hessian are a CasADi-generated C++ graph.** `MLTP.py:169` `f_dyn = ca.Function("f_dyn", [m.x, m.u, m.pv], [m.dx, L], ...)` compiles the pure-SX model (`vehModel.py:40`) into a native graph. The transcendental-heavy tyre (`vehModel.py:348` `_fx0 ... D * ca.sin(Cx * ca.atan(B*k - E*(B*k - ca.atan(B*k)))) + Sv`) is evaluated in C++ every iteration.
4. **Python's only job is one-time assembly.** The interval loop `functions/transcription.py:232` `for k in range(N):` → `:249` `dXkj, Qk = f_dyn(*args)`, the `np.vstack`/`reshape` packing (`:280-315`), and post-solve reconstruction run **once per solve** on numpy (BLAS/LAPACK-backed).

### Verdict: do not rewrite in C++ for solve speed

Rewriting the Python glue shaves a constant factor off the *construction* phase (seconds, once) and leaves the *dominant* cost, iterations × (factorisation + exact-Hessian eval), already native, untouched. For a solve that spends 95%+ of wall-clock inside IPOPT, **Amdahl's law caps the gain at a few percent**, against re-implementing and re-validating thousands of lines. The commercial sim's smoothness is almost certainly **a cheaper algorithm**, a forward/backward speed-march over a g-g-v envelope, O(N) in milliseconds, not a cheaper language. That is exactly the headline the roadmap already names (Item 5, QSS/g-g-v screener, ~1e2–1e4× per setup). **Method, not C++.**

### UI smoothness is a separate axis, and already correct

- **PySide6 *is* native C++ Qt.** The toolkit is the same Qt a C++ app would use; a C++/Qt rewrite would render identically.
- **The solve already runs out-of-process.** `app/solve_runner.py:28` `self._proc = QProcess(self)` starts the solver as a subprocess, so the Qt event loop never blocks on IPOPT. The UI is responsive *by design*. Residual jank is engineering (debounce the log-pane append, `app/solve_runner.py:35-38`), never a language problem.

### The one place a compiled inner loop *does* help

When the roadmap's QSS screener (Item 5) and transient marcher (Item 13) are built, those are genuine O(N) numerical loops run per-setup across a large DoE (Item 8). **If** numpy vectorisation of the march is too slow, a `numba.njit` kernel (preferred, stays in Python) or a small Cython/C/Rust extension for *that single ~100-line loop* is justified, targeted, not a rewrite, and only after the fast tier exists. Separately, roadmap Item 1 (`{'cse':True,'jit':True}` on `f_dyn`/`h_eq`) is the real "compile it" win: it C-compiles the dynamics graph for a per-iteration constant-factor gain with zero fidelity change, without leaving Python.

## Part 2, Usability / ease-of-use tier

The roadmap above optimises *how long one solve takes*; this tier optimises *how many solves a user can set up, launch, read and compare per hour*. It is all in `app/*.py`, around a backend that is already responsive by design. Items are ranked by **impact ÷ effort**; the top three are S-effort/high-impact.

### Tier U1, quick, high-leverage (do first)

- **U1.1 Make the child process actually stream (S / high).** `app/solve_runner.py:16` launches the solver with no `-u`/`PYTHONUNBUFFERED`, and `headless_solve.py:61` admits stdout is "block-buffered to the pipe," so the "live" log arrives in bursts. Set `python -u` / `PYTHONUNBUFFERED=1` on the `QProcess` environment. Prerequisite for every progress feature. *`app/solve_runner.py`.*
- **U1.2 Save/load the whole run setup + ship scenario templates (S / high).** Presets persist vehicle params only (`app/mainwindow.py:212`), so they cannot reproduce a scenario. `RunConfig` already round-trips a full session (`app/runconfig.py:56` `to_dict`, `:60` `from_dict`). Wire Save/Load to `RunConfig.to_dict`, seed `user_presets_dir` with 3–4 shipped templates, surface them in a combo. *`app/mainwindow.py`, `app/runconfig.py`.*
- **U1.3 Reproducibility snapshot per run (S / high).** The cfg goes to one temp file every run overwrites (`app/mainwindow.py:351`); the `.mat` path is keyed only by circuit+config (`app/results.py:16`), so setups collide and overwrite. Write `cfg.json` + a `provenance.json` into a per-run folder on success, and add "Load config" (via `RunConfig.from_dict`). *`app/mainwindow.py`, `app/runconfig.py`, `app/results.py`.*

### Tier U2, core workflow (do next)

- **U2.1 Keyboard accelerators (S / medium).** None exist (`app/mainwindow.py:62` is click-only). Add `Ctrl+R`/`F5` run, `Esc` cancel, `Ctrl+S`/`Ctrl+O`, `Ctrl+1..4`, and `&` mnemonics. *`app/mainwindow.py`.*
- **U2.2 Graceful cancel + alive indicator (S–M / medium).** `app/solve_runner.py:42` `self._proc.kill()` is a bare SIGKILL with no confirm/liveness cue. Add an elapsed "solving…" indicator, confirm-on-cancel, close-time cleanup; a "Resume" pointing next `warm_start` (`app/runconfig.py:27`) at the last `.mat` reinforces roadmap Item 7. *`app/solve_runner.py`, `app/mainwindow.py`.*
- **U2.3 Setup search/filter box (M / high).** ~110 params in one flat loop (`app/mainwindow.py:122`) with no search. Add a `QLineEdit` substring-matching label+key+unit, auto-expanding hits. Highest discoverability fix. *`app/mainwindow.py`.*
- **U2.4 Essentials-vs-Expert split (M / high).** Only one section defaults open (`app/mainwindow.py:134`); 59 Pacejka coefficients sit at equal weight. Add a `tier` to `ParamMeta`, gate Pacejka/numerical behind a collapsed "Expert" switch. *`app/vp_params.py`, `app/mainwindow.py`.*
- **U2.5 Inline validation; kill the silent-default fallback (M / high).** Bad input is swallowed (`app/widgets.py:28`) and replaced by the default (`app/mainwindow.py:324`), a silent-wrong-result mode; `lo/hi` is unenforced for sci fields (`setRange` only on the spin-box branch, `app/mainwindow.py:150`). Enforce bounds, style errors, and **block Run** instead of correcting. *`app/widgets.py`, `app/mainwindow.py`.*
- **U2.6 Parsed IPOPT progress + convergence chart (M / high).** The pane gets raw bytes, no parsing (`app/solve_runner.py:36`). Regex the iteration row (iter, objective, inf_pr, inf_du) against `max_iter`, drive a progress bar + "iter N" + elapsed + a small infeasibility plot. *`app/solve_runner.py`, `app/mainwindow.py`.*
- **U2.7 Structured result card + classified failure banner (M / high).** Success is a one-line label that throws "Results unreadable" with no `.mat` (`app/mainwindow.py:380`); failure is a raw traceback + exit code (`headless_solve.py:63`, `app/mainwindow.py:368`). Emit a structured block (IPOPT status, iters, wall-clock, lap time, verified-unit energy) extending `parse_summary` (`app/results.py:24`); classify failures into a banner. *`app/results.py`, `headless_solve.py`, `app/mainwindow.py`.*

### Tier U3, deeper / foundational

- **U3.1 Inert Cd/Cl marker + unified unknown-key handling (M / medium).** Cd/Cl are editable but overwritten by `DATA_AA.mat` (`app/vp_params.py:120`); badge them. Load silently drops unknown keys while run hard-raises (`app/runconfig.py:53`), make both warn-and-skip. *`app/mainwindow.py`, `app/vp_params.py`, `app/runconfig.py`.*
- **U3.2 Embed plots in-app (M / medium).** The deliverable opens only in a browser via a bare path (`app/mainwindow.py:266`); the HTML already exists (`app/results.py:38`). Embed in a `QWebEngineView`, browser as fallback. *`app/mainwindow.py`.*
- **U3.3 Explain EM4/ATD conflict + theme styling (S / low).** ATD greys out with no rationale (`app/mainwindow.py:308`); add a tooltip. Replace hardcoded `#b30000` (`app/mainwindow.py:177`, `app/widgets.py:41`) with palette colours; compare changed-state with tolerance, not `!=` (`app/mainwindow.py:169`). *`app/mainwindow.py`, `app/widgets.py`.*
- **U3.4 Run history + A/B compare (L / high).** Every run clears the log (`app/mainwindow.py:357`) and collides on one filename (`app/results.py:16`), A/B data is destroyed. On U1.3's per-run folders, add a runs ledger, click-to-reload, and two-run compare. Pairs with roadmap Item 8. *`app/mainwindow.py`, `app/runconfig.py`, `app/results.py`.*
- **U3.5 Pacejka labels/units/tooltips (L / medium).** ~59 coefficients fall through to `app/vp_params.py:134` (`label=key, unit=""`), reading as bare symbols. Populate real meta or a reference view. Lower priority once U2.4 hides them. *`app/vp_params.py`.*
- **U3.6 Presets-as-diff + per-field reset + undo (L / medium).** Full-value presets pin untouched params; reset is all-or-section only (`app/mainwindow.py:187`). Save diffs with metadata, add per-row reset + undo. *`app/mainwindow.py`, `app/runconfig.py`.*

### UX priority table (ranked by impact ÷ effort)

| # | UX item | Files | Effort | Impact | I/E |
|---|---------|-------|--------|--------|-----|
| U1.1 | Unbuffered child output (real streaming) | `app/solve_runner.py` | S | high | ~3.0 |
| U1.2 | Full-session presets + shipped templates | `app/mainwindow.py`, `app/runconfig.py` | S | high | ~3.0 |
| U1.3 | Reproducibility snapshot (cfg+provenance per run) | `app/mainwindow.py`, `app/runconfig.py`, `app/results.py` | S | high | ~3.0 |
| U2.1 | Keyboard accelerators & mnemonics | `app/mainwindow.py` | S | medium | ~2.0 |
| U2.2 | Graceful cancel + alive indicator | `app/solve_runner.py`, `app/mainwindow.py` | S–M | medium | ~2.0 |
| U2.3 | Setup search/filter box | `app/mainwindow.py` | M | high | ~1.5 |
| U2.4 | Essentials-vs-Expert split (gate Pacejka) | `app/vp_params.py`, `app/mainwindow.py` | M | high | ~1.5 |
| U2.5 | Inline validation; kill silent-default fallback | `app/widgets.py`, `app/mainwindow.py` | M | high | ~1.5 |
| U2.6 | Parsed IPOPT progress + convergence chart | `app/solve_runner.py`, `app/mainwindow.py` | M | high | ~1.5 |
| U2.7 | Structured result card + classified failure banner | `app/results.py`, `headless_solve.py`, `app/mainwindow.py` | M | high | ~1.5 |
| U3.1 | Inert Cd/Cl marker + unified key handling | `app/mainwindow.py`, `app/vp_params.py`, `app/runconfig.py` | M | medium | ~1.0 |
| U3.2 | Embed Plotly plots in-app (QWebEngineView) | `app/mainwindow.py` | M | medium | ~1.0 |
| U3.3 | EM4/ATD explanation + dark-mode styling | `app/mainwindow.py`, `app/widgets.py` | S | low | ~1.0 |
| U3.4 | Run history + A/B compare | `app/mainwindow.py`, `app/runconfig.py`, `app/results.py` | L | high | ~1.0 |
| U3.5 | Pacejka labels/units/tooltips | `app/vp_params.py` | L | medium | ~0.7 |
| U3.6 | Presets-as-diff + per-field reset + undo | `app/mainwindow.py`, `app/runconfig.py` | L | medium | ~0.7 |

## Part 3, Sharpened speed priorities (from the premise analysis)

The premise analysis does not overturn the roadmap's ordering, it *confirms* it and adds guardrails:

1. **Keep the corrected top-three as written**, warm-start + dual reuse (Item 7 + its Item 4 enabler), then CSE+JIT (Item 1), then the QSS/g-g-v screener (Item 5). These are algorithm levers; Amdahl gives every reason not to swap them for a rewrite.
2. **Record "rewrite in C++/Rust for solve speed" as an explicit non-goal.** The hot loop is already compiled (IPOPT C++ `transcription.py:347`, MA57/MUMPS Fortran via `hsl.py`, CasADi C++ tyre graph `MLTP.py:169`); a rewrite is capped at a few percent. Writing it down keeps the question closed.
3. **Promote Item 1 (CSE + JIT) as the canonical "compile it without leaving Python" win,** done before any tabulation so its per-iteration gain is measured and attributable, the real answer to "make the compute faster," zero fidelity change.
4. **Gate the only justified compiled kernel.** Implement the QSS screener (Item 5) and transient marcher (Item 13) inner loop in vectorised numpy first; add a `numba.njit` kernel (preferred) or a small C/Rust extension for that single ~100-line loop **only if** numpy proves too slow across a large DoE (Item 8).
5. **Decouple UI smoothness from solve speed.** The solve is already off the GUI thread (`solve_runner.py:28`); remaining smoothness is throttling the log-pane append (`solve_runner.py:35-38`) and keeping parsing off the paint path, tracked under UX items U1.1/U2.6, not the speed roadmap.
6. **Sequence reproducibility/run-history UX (U1.3 → U3.4) with the DoE orchestrator (Item 8).** Per-run cfg+result folders and a runs ledger are the natural front-end for screen-cheap/confirm-expensive sweeps; building them together avoids a second pass over the `results.py` path scheme.

---

## Phase 1 status (2026-10-04)

Phase 1 was implemented on branch `worktree-speed-phase1` and measured on one machine (IPOPT 3.14.11, CasADi 3.7.2, HSL ma57 with MC64 scaling). The benchmark case is Sturn (`OPT_ds` 30, N=18) unless stated. Tables, caveats and raw numbers: [phase1_findings_2026-10-04.md](phase1_findings_2026-10-04.md).

**What changed the plan.**

* *The cost is iterations times factorisation, not function evaluation.* In the cold 23-state solve, MA57 factorisation is 81 to 87% of IPOPT time and NLP function evaluation 9 to 13%; the NLP build takes about 3.5 to 4 s on Sturn and the Python assembly loop 0.02 to 0.17 s ([where the time goes](phase1_findings_2026-10-04.md#where-the-time-goes)). Per-evaluation items (1, 2, 6) can therefore gain at most a few percent. The levers are iteration count and refactorisations: warm start with duals, the curvature mesh, and the QSS screen.
* *The shipped tyre set has no lateral force.* Its lateral block (`pKy4 = 0`) is the MATLAB 'Test' case, not the set MATLAB runs. The Sturn solve needs 3209 iterations for a non-physical 25.57 s; with the MATLAB-run set (`tyre_set='MF205'`) it needs 489 iterations for 18.009 s (all pre-fix NLP; after the 2026-10-04 NLP fix the MF205 solve takes 179 iterations for 18.0086 s), within 0.05% of MATLAB ([tyre-set defect](phase1_findings_2026-10-04.md#tyre-set-defect-and-matlab-reference)); the five-coefficient proxy used for the benchmark runs needs 257 iterations for 18.016 s (pre-fix). The owner decision was taken on 2026-10-04: the default is now `tyre_set='MF205'` (the MATLAB-run set), and `'CopyB'` is kept as the legacy shipped set.
* *"Rewrite in C++ for solve speed" stays a non-goal.* IPOPT and MA57 are already native, and the remaining cost is iteration count and factorisation work.

### Status of items 1 to 14

| # | Item | Status | Measured outcome | Details |
|---|------|--------|------------------|---------|
| 1 | CSE + JIT | Done, opt-in (low value) | CSE is a mathematical no-op (f, g, Jacobian bit-identical, Hessian 1 ulp) but moves IPOPT to a different optimum on this path-sensitive NLP (shipped-tyre lap 32.6 vs 25.57 s) and fails on VirtualTrack; function evaluation is only ~10% of wall. JIT gives 1.1 to 1.2x on direct evaluation and nothing on the inlined SX NLP. `functions/casadi_opts.py`, `MLTP_CSE=1`, `userOpts(cse=, jit=)` | [findings](phase1_findings_2026-10-04.md#per-item-results) |
| 2 | `f_dyn.map` vectorised transcription | Done: SX route default, MX route opt-in (low value) | The SX route is bit-identical to the old loop (3209 iterations, same lap). The MX route (`MLTP_SYM_TYPE=MX`) cuts the BCN NLP build from 27 s to 1.9 s but evaluates derivatives ~8x slower and lands on a different optimum. The Python loop was never the cost (0.02 to 0.17 s) | [findings](phase1_findings_2026-10-04.md#per-item-results) |
| 3 | Curvature-weighted mesh | Done, default `mesh='auto'` (curvature for tracks >= 2000 m, uniform otherwise) | BCN: curvature N=103 matches uniform N=155 (117.40 vs 117.42 s) in 392 vs 968 s wall (2.5x) and 497 vs 852 iterations (pre-fix NLP, five-coefficient proxy; after the NLP fix the default BCN call, `auto` = curvature N=155, takes 247 iterations, 116.441 s and ~250 s of solve, against 614, 116.523 s and 566 s before). Caveat: `auto` keeps N = round(L/OPT_ds), so this 2.5x (curvature `OPT_ds` 45, N=103 vs uniform `OPT_ds` 30, N=155) needs `OPT_ds=45` set explicitly. Sturn N=18: lap error vs a fine N=36 reference +0.16% curvature vs +0.72% uniform. `functions/mesh.py`, `userOpts(mesh='curvature')` | [findings](phase1_findings_2026-10-04.md#per-item-results) |
| 4 | Screening options, IPOPT warm-start scaffold, scaling check | Done (preset evidence is Sturn only) | Options study: keep the current defaults, never disable MC64. `userOpts(screening=True)` loosens five tolerances (196 vs 257 iterations in the head-to-head pair with the five-coefficient proxy and 290 vs 489 with `tyre_set='MF205'`, but -41% to +16% across five pairs; all pre-fix NLP, not re-measured since); `userOpts(ipopt_overrides=)`; dual warm-start recipe in `functions/warmstart.py`; exact-Hessian warning in `userOpts.py` | [findings](phase1_findings_2026-10-04.md#ipopt-options-study-and-recommendation) |
| 5 | QSS / g-g-v screening solver | Done | ~7 ms per setup on Sturn and ~30 ms on BCN; +3.9% (Sturn) and +11.1% (BCN) vs the corrected-tyre NLP (+4.1% and +11.2% vs the default NLP laps after the 2026-10-04 fix), mostly from the fixed centreline. `functions/ggv.py`, `MLTP_screen.py`, `test_screen.py` | [findings](phase1_findings_2026-10-04.md#per-item-results) |
| 6 | Tyre B-spline tabulation | Dropped (low value) | Function evaluation is ~10% of wall, so a faster tyre cannot gain more than a few percent, and it would also have to cross the SX/MX wall. Revisit only after iteration count and factorisation cost are cut | [findings](phase1_findings_2026-10-04.md#where-the-time-goes) |
| 7 | Parametric warm start + dual persistence | Done | After a +3% setup change (pre-fix NLP, five-coefficient proxy): 11 iterations / 5.8 s vs 364 / 41.6 s cold (`alpha_RW`) and 32 / 8.3 s vs 327 / 36.3 s (`mb`); identical re-solve 0 iterations / 4.7 s, bit-identical (still 0 iterations after the NLP fix: the `setup_sweep` hub check). `functions/warmstart.py`, `MLTP(warm_start=, warm_start_duals=True)`, `data["nlp"]` | [findings](phase1_findings_2026-10-04.md#per-item-results) |
| 8 | DoE / sweep orchestrator | Done | `setup_sweep.py`, `functions/sweep.py`, `MLTP_screen.screen_batch`, `test_setup_sweep.py`: Sobol/LHS design, batched QSS screen (exact parity with `screen_sweep`, ~126 setups/s on Sturn in-process), inert and unknown fields classified (`brkB`/`Tdist` dropped with ATD On), one hub solve then star-topology confirmations warm-started with duals (median 48 iterations per hop, ~850 confirmations/hour with 7 workers). Sturn, 64 samples, top-4 + 4 probes: Spearman 1.0, Kendall 1.0, resolved-pair concordance 36/36, QSS deltas over-stated ~1.5x. Outputs under `Results/sweeps/<name>/`, deterministic and resumable | [Phase 2 status](#phase-2-status-2026-10-05) |
| 9 | Iterative hp / adaptive mesh | Done, opt-in (h-refinement only) | `MLTP(refine=...)`, `functions/refine.py`, `test_refine.py`: integrated-defect indicator on n and eps, bisection above `tol` 1e-2, reseeded warm restarts, result stem `_meshAdaptive`. Sturn from N=12 (`OPT_ds` 45): N 12 -> 21 -> 27 at +0.007% vs the N=36 uniform optimum (625 iterations; uniform N=18 is +0.85%). BCN: no gain (116.747 s in 363 s from N=78 vs 116.441 s in 315 s at the default N=155) | [Phase 2 status](#phase-2-status-2026-10-05) |
| 10 | Multi-fidelity ladder + homotopy | Next (inputs ready) | The QSS rung (item 5), the 7-state rung, warm start with duals (item 7) and the mesh reseeding of item 9 exist; item 8 is done | [Phase 2 status](#phase-2-status-2026-10-05) |
| 11 | Free-trajectory QSS tier | Not started | Evidence of value: the fixed line is the main screening error (Sturn +3.9%, all from the centreline; BCN ~6% line plus ~5% corner speed) | [findings](phase1_findings_2026-10-04.md#next-steps) |
| 12 | Energy / ERS co-optimisation | Not started | The prerequisite EM4 power-cap review is done (2026-10-04): the per-motor cap is ~52x looser than the single-motor cap, inherited from `vehModel.m` and kept for parity | [findings](phase1_findings_2026-10-04.md#update-2026-10-05-follow-ups) |
| 13 | Transient time-marching + driver model | Not started | Unchanged | [findings](phase1_findings_2026-10-04.md#next-steps) |
| 14 | Tyre thermal / wear states | Not started | Unchanged | [findings](phase1_findings_2026-10-04.md#next-steps) |

### Revised critical path

This supersedes the "Corrected top-three priorities" and the "Critical path" paragraphs above. The original path was items 4 -> 7 -> 5 -> 8 with item 6 in parallel and items 1 to 3 as free accelerators. Phase 1 completed items 4, 7 and 5, built items 1 and 2 as opt-in accelerators (their measured gain is small or path-changing), built item 3 as the curvature mesh that `mesh='auto'` now selects by default on tracks >= 2000 m, and dropped item 6. For "make setup exploration fast" the path is now:

1. **Settle the tyre default** (done 2026-10-04: the default is now `tyre_set='MF205'` and `'CopyB'` is kept as legacy, see [decisions](phase1_findings_2026-10-04.md#decisions-taken-2026-10-04)). Every benchmark, sweep and warm start rests on a well-posed model; with the legacy `'CopyB'` set a Sturn solve takes 6.6x (vs `tyre_set='MF205'`, 489) to 12.5x (vs the five-coefficient proxy, 257) the iterations (pre-fix counts) and is non-physical.
2. **Model review pass** on the issues found while auditing ([model observations](phase1_findings_2026-10-04.md#model-observations)): the EM4 per-motor power cap (~52x looser than the single motor), `brkB` and `Tdist` being inert with `ATD=On` although they are default parameters of `MLTP_paramOptim`, the unsprung-mass load accounting (26.0 kN total tyre load vs m*g = 20.5 kN), and the front/rear Cl split having no steady-state effect on axle loads. A sweep ranks setups by the model, so any parameter these touch must be settled first. **Done 2026-10-04:** all four are inherited from `vehModel.m` and kept for parity, and two NLP-assembly port bugs were found and fixed ([findings update](phase1_findings_2026-10-04.md#update-2026-10-05-follow-ups)).
3. **Item 8, sweep orchestrator**: screen with `screen_sweep`, confirm the shortlist with warm-started `MLTP` runs chained on a fixed mesh. Chain the warm starts: cold solves scatter by about 0.01 s between local optima, which is more than many setup effects (an `alpha_RW` +3% change reads as -6.7 ms in a cold-vs-cold comparison and +2.2 ms warm-vs-base). **Done 2026-10-05** ([Phase 2 status](#phase-2-status-2026-10-05)): the confirmations form a star around one hub solution rather than a chain.
4. **Item 10, ladder** (QSS rung 0, 7-state, 23-state), consuming items 5 and 7.
5. **Item 11** (free-line QSS) if screening rank quality proves insufficient, since the fixed line is the dominant QSS error; items 9 (done 2026-10-05) and 12 after that.

Ordering of levers by measured value: iteration count and factorisation (warm start with duals, mesh, tyre fix) > screening tier (QSS) > per-evaluation cost (CSE, JIT, `f_dyn.map`, tabulation).

## Phase 2 status (2026-10-05)

Items 8 and 9 are done (same branch, machine and tools as Phase 1; measured on the NLP after the 2026-10-04 constraint fix, where the default Sturn solve takes 179 iterations for 18.0086 s and BCN 247 iterations for 116.441 s, see the [findings update](phase1_findings_2026-10-04.md#update-2026-10-05-follow-ups)). The model review pass closed on 2026-10-04: the four quirks (EM4 power cap, unsprung-mass loads, aero split, inert `brkB`/`Tdist`) are inherited from `vehModel.m` and kept for parity, and only two NLP-assembly items were port bugs.

* **Item 8, `setup_sweep.py`** (with `functions/sweep.py`, `MLTP_screen.screen_batch`, `test_setup_sweep.py`). `setup_sweep([(field, lower, upper), ...], n_samples=256, circuit='Sturn', top_k=8, n_probes=4, base=None, confirm=True, finish=False)` returns a `SweepResult` whose `best['vp_overrides']` feeds `MLTP(vp_overrides=...)`; `confirm=False` stops after the screen and shortlist (no NLP, no casadi). The stages are a Sobol or Latin hypercube design over the specs, a batched QSS screen with exact parity to `screen_sweep` (about 126 setups/s on Sturn in-process), inert and unknown fields dropped (`brkB`/`Tdist` with ATD On), one hub solve, then a star of confirmations warm-started with primal and duals from the hub (median 48 iterations per hop, about 850 confirmations/hour with 7 workers; a chain of warm starts depends on its order, a star row does not), rank-agreement metrics, and an optional `finish=True` through `optimise_design`. Outputs go to `Results/sweeps/<name>/` (`plan.json`, `samples.csv`, `hub.mat`, `rows/`, `confirmed.csv`, `summary.json`, `report.html`); a sweep is deterministic and resumable. Sturn, 64 samples, top-4 + 4 probes: Spearman 1.0, Kendall 1.0, resolved-pair concordance 36/36, and the QSS deltas are over-stated about 1.5x, so quote the confirmed NLP deltas. Rank agreement is measured on Sturn only.
* **Item 9, `MLTP(refine=...)`** (`functions/refine.py`, `test_refine.py`; off by default, `refine=True` or a dict with defaults `passes` 2, `tol` 1e-2, `max_N` 4 x N0; result stem suffix `_meshAdaptive`). An integrated-defect indicator on the path states n and eps (the stiff tyre and suspension states make an all-state defect unusable), bisection of the intervals above `tol` 1e-2 (nested knots, `OPT_d` fixed) and reseeded warm restarts. Sturn from N=12 (`OPT_ds` 45): N 12 -> 21 -> 27 at +0.007% against the uniform N=36 optimum (625 iterations; uniform N=18 is +0.85%). BCN gains nothing: from N=78, 116.747 s in 363 s against 116.441 s in 315 s for the default N=155 solve. Fine meshes are fragile (2 of 8 measured passes did not converge).

**Revised next steps.**

1. **Item 10, the ladder** (QSS rung 0, 7-state, 23-state): next, with every input in place (QSS rung, 7-state rung, dual warm start, mesh reseeding). The cold 7-state start is the fragile part: the co-optimisation from it takes 220 iterations to a worse local optimum (18.244 s), from the converged full solution 5 iterations to 18.008 s.
2. **Item 11, free-line QSS**: the fixed centreline is still the main screening error (+4.1% on Sturn, +11.2% on BCN against the NLP laps), but rank agreement on the Sturn sweep is already perfect, so measure the screen's rank quality on BCN before committing to it.
3. **Item 12, energy / ERS co-optimisation**: its prerequisite review is closed (the EM4 per-motor power cap is inherited from `vehModel.m`, not a bug), so it can start from the parity model and pair with the item 10 homotopy.

Items 13 and 14 are unchanged and item 6 stays dropped. Open housekeeping: re-measure the screening preset and the other study defaults on the fixed NLP, and re-test CSE on VirtualTrack.
