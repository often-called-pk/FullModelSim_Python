---
type: "query"
date: "2026-10-03T18:14:59.786720+00:00"
question: "Why does MainWindow connect Windows GUI App Design to IPOPT & HSL Linear Solver, App Entry & Packaging, Vehicle-Param Editor Config, Qt Main Window & Widgets, Results Parsing, Collapsible Section Widget?"
contributor: "graphify"
outcome: "corrected"
correction: "ma86 rationale belongs on functions/hsl.py _SUPPORTED_HSL (L28-32), not MainWindow; MainWindow->IPOPT link is the runtime linear_solver data flow via cfg.json + subprocess (mainwindow.py L333 -> runconfig.py L26 -> headless_solve.py L25 -> MLTP.py L138 -> userOpts.py L270 -> transcription.py L343 -> hsl.py L144/L156). README edges are module-level, not symbol-level."
source_nodes: ["MainWindow", "ma86 excluded from defaults", "apply_linear_solver()", "_make_solver()", "RunConfig", "build_solve_kwargs()", "userOpts()", "SolveRunner", "parse_summary()", "README.md - Quick-start and Status"]
---

# Q: Why does MainWindow connect Windows GUI App Design to IPOPT & HSL Linear Solver, App Entry & Packaging, Vehicle-Param Editor Config, Qt Main Window & Widgets, Results Parsing, Collapsible Section Widget?

## Answer

Expanded from original query via vocab: [mainwindow, linear, solver, hsl, ipopt, defaults, excluded, runconfig, results, collapsible, section, entry]. Traversed MainWindow (app/mainwindow.py L43) neighbours grouped by community; verified by 12 agents against source. ANSWER: MainWindow bridges 7 communities because every cross-module call lives in its METHODS (class body L44-419 is only defs): __init__ -> SolveRunner (L47), RunConfig (L50); _build_setup_tab -> all_vp_defaults (L108), CollapsibleSection (L123), meta_for (L127); _build_advanced_tab -> ScientificField (L294); _save_preset/_load_preset -> user_presets_dir (L206/L222); _show_results -> results.result_mat_path/parse_summary/plot_dir/list_plots (L371/374/381/384, reached via _on_finished L366); main() constructs MainWindow at app/main.py L22. The graph hangs these on the class node = summarising method-level edges. AST missed the results.* calls (attribute-style via 'from app import results, paths' L20). IPOPT/HSL link CORRECTED: the graph's only edge there ('ma86 excluded from defaults' rationale_for MainWindow, INFERRED 0.85) is mis-targeted. The ma86 segfault rationale is encoded in code ONLY at functions/hsl.py L28-32 (_SUPPORTED_HSL = (ma27, ma57, ma97)) and enforced at L144-152 (unsupported ma* -> RuntimeWarning + mumps), tested test_hsl.py L120-125. mainwindow.py L30 SOLVERS = [ma57, ma97, ma27, mumps] is a bare copy of the HSL plan L19 / userOpts.py L190 comment list, no comment, no spec link. The REAL GUI->IPOPT link is a runtime data flow invisible to AST: combo text (mainwindow.py L274/L333) -> RunConfig.linear_solver (runconfig.py L26 default ma57) -> rc.write_cfg (mainwindow.py L352, cfg.json) -> QProcess subprocess (solve_runner.py L15-16) -> headless_solve.py L25 cfg.get('linear_solver','ma57') -> MLTP(**useropts_kwargs) MLTP.py L138/L144 and L150-152 (also forwarded to MLTP_initial warm start; HSL plan L733 'not forwarded' limitation is STALE since commit 27836b5) -> userOpts.py L190/L270/L274 ctx.opts -> build_and_solve_nlp -> _make_solver transcription.py L343-347 -> hsl.apply_linear_solver / resolve_hsl_dir L58-61 / probe_linear_solver (cached per process) -> ca.nlpsol('ipopt'). Nothing validates solver names before hsl.py; GUI restricts by non-editable combo only. No test pins SOLVERS or the combo->RunConfig wiring. Frozen-exe caveat: sys._MEIPASS shadows seeded default dir; if Coin-HSL not bundled, ma57 silently becomes mumps while combo shows ma57. Doc staleness: build/README.md L53 says Main tab (code: Advanced); HSL spec L233 omits ma27 though GUI offers it; userOpts.py L11 and CLAUDE.md L11 still say MUMPS. README edges (INFERRED 0.95) overstate: README names module basenames (L164-165), never MainWindow/RunConfig/parse_summary symbols. Dead import: 'paths' at mainwindow.py L20.

## Outcome

- Signal: corrected
- Correction: ma86 rationale belongs on functions/hsl.py _SUPPORTED_HSL (L28-32), not MainWindow; MainWindow->IPOPT link is the runtime linear_solver data flow via cfg.json + subprocess (mainwindow.py L333 -> runconfig.py L26 -> headless_solve.py L25 -> MLTP.py L138 -> userOpts.py L270 -> transcription.py L343 -> hsl.py L144/L156). README edges are module-level, not symbol-level.

## Source Nodes

- MainWindow
- ma86 excluded from defaults
- apply_linear_solver()
- _make_solver()
- RunConfig
- build_solve_kwargs()
- userOpts()
- SolveRunner
- parse_summary()
- README.md - Quick-start and Status