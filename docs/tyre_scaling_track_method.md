<!--
Claim intent manifest (v3.8). Emitted before the first prose block. Audit metadata, not report content.
The time of day was not recorded, so emitted_at gives the date only.
{
  "manifest_version": "1.0",
  "manifest_id": "M-2026-10-20T00:00:00Z-r2c8",
  "emitted_by": "report_compiler_agent",
  "emitted_at": "2026-10-20T00:00:00Z",
  "claims": [
    {"claim_id": "C-001", "claim_text": "In this search, every validated on-track identification of MF parameters or scaling factors measured force-level or vehicle-state signals.", "intended_evidence_kind": "empirical", "planned_refs": ["A2", "A3", "A7", "A8"]},
    {"claim_id": "C-002", "claim_text": "This search found no study that fits MF scaling factors to lap times plus tyre temperature and pressure readings. The absence is Moderate evidence because the stopping rules were not met.", "intended_evidence_kind": "empirical", "planned_refs": ["A9", "A10"]},
    {"claim_id": "C-003", "claim_text": "Track-fitted scaling factors carry the car, suspension, controller, manoeuvre and driver as well as the tyre.", "intended_evidence_kind": "empirical", "planned_refs": ["A2", "A7", "A8", "A10"]},
    {"claim_id": "C-004", "claim_text": "Published temperature and pressure forms are normalised multipliers equal to 1 at a reference. They fit the slots of LMUX, LKX, LMUY and the PKY1 part of LKY, not the load-location terms.", "intended_evidence_kind": "empirical", "planned_refs": ["B1", "B4", "B11"]},
    {"claim_id": "C-005", "claim_text": "Cornering stiffness falls with temperature in four sources against one grade A null. For peak lateral friction, two sources give a window, one a fall in full sliding and one a rise, and one source conflicts with itself. Both direction ratings stay Low to Moderate, and the friction conflict is unresolved.", "intended_evidence_kind": "empirical", "planned_refs": ["A5", "A11", "B4", "B6", "B9", "B10", "B12", "B13", "B14"]},
    {"claim_id": "C-006", "claim_text": "Pressure effects on cornering stiffness are load-dependent and include a grade A null. The pressure effect on peak friction is uncertain, and no source compares both friction channels on one tyre. A pressure factor fitted to lap times is a lumped quantity.", "intended_evidence_kind": "empirical", "planned_refs": ["B1", "B2", "B3", "B4", "B10", "C1"]},
    {"claim_id": "C-007", "claim_text": "Tyre temperature and hot pressure are outcomes of the lap and two readings of one thermal state. The post-session reading is a second outcome and a safety check. It neither enters the fit nor selects the data.", "intended_evidence_kind": "empirical", "planned_refs": ["A5", "B2", "B4", "B8", "B11", "B14", "C1", "C3"], "negative_constraints": [{"constraint_id": "NC-C007-1", "rule": "No claim that a fit of lap time on measured tyre temperature estimates a causal temperature effect."}, {"constraint_id": "NC-C007-2", "rule": "No use of the tyre-temperature reading as a data filter in the primary analysis."}]},
    {"claim_id": "C-008", "claim_text": "Calibration needs a functional dependence of the parameters on inputs plus a discrepancy term with realistic priors. Two-stage random-effects models handle unbalanced sessions and session-specific slopes.", "intended_evidence_kind": "theoretical", "planned_refs": ["C7", "C8", "C9", "C10", "C12"]},
    {"claim_id": "C-009", "claim_text": "For the effect of cold pressure or the association with track temperature, the measured tyre temperature stays out of the model. A tyre-temperature effect needs a pace measure that the core data lack.", "intended_evidence_kind": "theoretical", "planned_refs": ["B14", "C3", "C4"]},
    {"claim_id": "C-010", "claim_text": "Through the QSS, LKX and LKY are structurally non-identifiable, and no measurement reaches them. Fixed factors and Fz0_shift bias the free ones unless they sit at their true values.", "intended_evidence_kind": "theoretical", "planned_refs": ["A15", "C13"]},
    {"claim_id": "C-011", "claim_text": "Per circuit, the core data can identify one effective grip scale and its total change with the designed cold pressure, plus an association with track temperature across sessions. LFZO, LKX, LKY and the Cd and Cl scales are not identifiable from them.", "intended_evidence_kind": "theoretical", "planned_refs": ["A2", "A3", "A7", "A9", "A10", "A11", "A14", "B3", "B6", "B13", "C3", "C8", "C11", "C12", "C13"], "negative_constraints": [{"constraint_id": "NC-C011-1", "rule": "No verdict is stated as final. All verdicts are candidates until strand B2 or team data confirm them."}, {"constraint_id": "NC-C011-2", "rule": "No claim that the pressure shape transfers to a circuit without a passed transfer check, or across drivers, forward models or thermal conditions outside its tags."}, {"constraint_id": "NC-C011-3", "rule": "No causal label on the track-temperature term."}]},
    {"claim_id": "C-012", "claim_text": "The track protocol uses a designed, rotated cold-pressure sweep with equal relative offsets on gauge readings, scrubbed sets, per-corner readings when the car stops in the pit box, a confounder log, safety rules that override the sweep and pre-set rules for lap validity and for stints that end early.", "intended_evidence_kind": "normative", "planned_refs": ["A6", "B3", "B4", "C11", "C15"]},
    {"claim_id": "C-013", "claim_text": "The estimation procedure inverts stint median lap times through the QSS and fits a stint-level REML model with session effects, a session-specific slope and one trend term, t-based small-sample intervals, collinearity checks, a full refit after each session, a minimum session count before a success stop and leave-one-session-out validation on within-session contrasts against a simulator-free model.", "intended_evidence_kind": "theoretical", "planned_refs": ["A15", "C9", "C12", "C13", "C15"]},
    {"claim_id": "C-014", "claim_text": "Use the simulator-free model for the pressure decision. Add LMUX and LMUY linked to one value at 1, and use the pressure shape only in the QSS after four gates, on calibration circuits or circuits with a passed transfer check. Keep the effective level outside the vehicle model. Do not add LKX, LKY, LFZO and the Cd and Cl scales now, and keep Fz0_shift unchanged.", "intended_evidence_kind": "normative", "planned_refs": ["A2", "A10", "A15", "B3", "C13"]}
  ],
  "manifest_negative_constraints": [
    {"constraint_id": "MNC-1", "rule": "No unqualified causal language about tyre-temperature or track-temperature effects estimated from the core data."},
    {"constraint_id": "MNC-2", "rule": "No number from an abstract-only source unless its abstract states it."},
    {"constraint_id": "MNC-3", "rule": "The held-out records A4, A12, A13, B5 and C6, the leads L1 to L5, the safety documents S1 to S4, the reserve records and the Phase 1 scoping anchors are not cited as evidence."},
    {"constraint_id": "MNC-4", "rule": "No fitted temperature coefficient of B11 is reported as a verified magnitude or direction."},
    {"constraint_id": "MNC-5", "rule": "No session count is stated as evidence-based."},
    {"constraint_id": "MNC-6", "rule": "No safety rule is presented as a finding of the evidence set."}
  ]
}
-->

# Estimating Magic Formula Scaling Factors Without a Tyre Rig: A Candidate Track Method for a Small Race Team

Prepared for the team owner of FullModelSim_Python by AI agents (Claude, Anthropic), 20 October 2026. Revision 2 after two review rounds (Appendix A), not reviewed by a human. All verdicts are candidates.

## Executive Summary

**The answer.** Every validated on-track identification in this search measured forces or vehicle states (e.g., Bogdanski & Best, 2018)<!--ref:A3--><!--anchor:quote:CG%20longitudinal%20and%20lateral%20accelerations%20and%20body%20roll%20rate-->. Lap times give only an effective grip of tyre, car, driver and model. This is because fitted coefficients absorb the suspension (Braghin et al., 2011)<!--ref:A2--><!--anchor:quote:implicitly%20compensates%20for%20suspension%20geometry%20and%20compliance--> and riders on the same tyres reach different speeds (Roziere et al., 2011)<!--ref:A10--><!--anchor:quote:values%20that%20differed%20substantially%20in%20terms%20of%20the%20speeds%20and%20camber%20angles%20measured-->. Per circuit, the core data can fix one effective grip scale and its change with a designed cold-pressure offset, the pressure shape g_p (inference). Track temperature gives only an association. Sliding friction heats the tread (Harsh & Shyrokau, 2019)<!--ref:B11--><!--anchor:section:Eq3-->. So the tyre temperature reading is a second outcome and a safety check, never a regressor or a data filter. In the QSS screener, slip stiffness cannot change the lap time (project fact P1).

**Code.** Add LMUX and LMUY, linked to one value with default 1. Leave LKX, LKY, LFZO and the Cd and Cl scales out, or at 1, and keep Fz0_shift at 1.0. Use the simulator-free model M0b for the cold-pressure choice. Put g_p into LMUX and LMUY only in the QSS, after four gates. The gates are a success stop, the driver rule, a held-out contrast check, and a calibration circuit or a passed transfer check.

**Method in plain steps.**

1. Get the tyre maker's limits and the warmer and pit-lane rules. Write a risk assessment and the lap-validity rules.
2. Choose two or three cold-pressure levels, with one relative offset on all corners.
3. Run every level in every session, one stint each, in rotated order (pilot: R-L-R).
4. Set cold pressures on cold tyres, and log the tyre temperature and barometric pressure. Do not bleed in a sweep stint unless safety requires it.
5. When the car stops in the pit box, read hot pressures, then three-point probe temperatures, with clock times.
6. Log track and ambient temperature, wind, tyre set and age, driver, setup, mass and battery state.
7. After each session, turn stint median lap times into grip scales with the QSS. Refit the scales on the offset with session effects and a time term.
8. Stop after a minimum number of sessions when the effect is precise enough, or when the budget runs out.

**Accuracy.** No source gives a number. The pilot and the planned simulation study (strand B2) must supply it. Precision depends mostly on the number of sessions.

## Abstract

Small race teams rarely own a tyre test rig. This report asks which Magic Formula (MF) scaling factors such a team can estimate from lap times plus tyre temperature and pressure readings. The factors are for a four-motor electric race car simulator. It synthesises 36 verified sources, tested by three devil's advocate checkpoints and two review rounds. Every validated on-track identification found used force or vehicle-state signals, and tyre temperature and hot pressure are outcomes of the lap. Published temperature and pressure forms are normalised multipliers with conflicting, mostly passenger-tyre magnitudes. Per circuit, the core data can identify one effective grip scale and its change with a designed cold-pressure offset. They cannot identify LKX or LKY through the quasi-steady-state screener, nor LFZO or the aerodynamic scales. The report gives a track protocol, an estimation procedure and code recommendations, led by a simulator-free pressure model. All verdicts stay candidates until a simulation study and team data confirm them.

**Keywords:** Magic Formula, scaling factors, inflation pressure, tyre temperature, lap-time simulation, identifiability

## Introduction

FullModelSim_Python is a Python port of a MATLAB minimum-lap-time simulator for a four-motor electric race car. It has a 23-state collocation model (IPOPT) and a quasi-steady-state (QSS) g-g-v screener. The screener ranks setups in tens of milliseconds from steady loads and peak friction. On 19 October 2026, its tyre model was MF 5.2 combined slip with a generic 205/60 R15 road-tyre set, without pressure or temperature terms. The owner describes task C6-2 as approved. The task adds the scaling factors LFZO (nominal load), LMUX and LMUY (peak friction), and LKX and LKY (longitudinal and cornering slip stiffness). It also adds scales on the drag and lift coefficients (Cd, Cl). All of them are at 1, and Fz0_shift stays unchanged. The project status note (docs/laptime_sim_improvements.md) still lists them as waiting for sign-off.

The team has no rig. It measures tyre temperature with a probe or pyrometer and inflation pressure cold and hot. It always has lap times, while GPS and acceleration logging are optional. Sub-questions SQ1 to SQ5 cover methods, effects, protocol, estimation and identifiability. Hypothesis H1 of the research brief: the core data identify one combined peak-friction scale as a low-order function of temperature and pressure. H1 adds that the core data identify no other factor.

## Method

**Design.** A blueprint within the confirmed scope set an evidence synthesis plus a read-only analysis of the project equations. A numerical identifiability study (strand B2) awaits the owner's approval, so no verdict is final.

**Search.** Three slices covered on-track methods (Theme A), temperature and pressure effects (Theme B), and measurement, estimation and identifiability (Theme C). All searches ran on 19 October 2026 in OpenAlex, Semantic Scholar, Crossref, arXiv, Europe PMC, repositories and web search, with counter-evidence strings. A task cap limited each slice to 15 annotated sources (Table 1).

**Table 1**

*Search and Verification Counts*

|Stage|Theme A|Theme B|Theme C|Total|
|:---|:---|:---|:---|:---|
|Screened (title and abstract)|687 or fewer|383|about 365|about 1,435|
|Assessed (abstract or full text)|53|58|66|177|
|Annotated (cap of 15 per theme)|15|15|15|41 unique|
|Screened by source verification| | | |41|
|Verified and cited (5 held out)| | | |36|

*Note.* Four papers appear in two themes, so 45 entries give 41 unique sources.

**Verification.** Six batches checked existence and each claim against its source. All 41 sources exist, and none was rejected. A task decision held out five (A4, A12, A13, B5 and C6), although the verifiers recommended them with corrected wording. They are listed after the References, and the text notes where they bear on a finding. Verifiers read 17 sources in full or preprint text, one patent description and 18 abstracts or records. They took no number from an abstract that does not state it. Reviews matched every quote anchor to a source text and found no cited DOI in the Retraction Watch data.

**Grading.** Sources are graded A to F and findings rated Strong, Moderate, Low or Very low (adapted from GRADE). No quantity had three comparable studies, so only directions were counted. Appendix B lists project facts P1 to P7 from read-only code checks.

## Findings

### SQ1: Methods That Identify Tyre Parameters From Track Data

**In this search, every validated method measured forces or vehicle states (Strong).** The smallest validated set took the steer angle and wheel speeds as inputs (Bogdanski & Best, 2018)<!--ref:A3--><!--anchor:quote:steer%20angle%20and%20the%20four%20wheelspeeds-->. Its identification set was one 100 s run with "some extreme high slip inputs" (Bogdanski & Best, 2018)<!--ref:A3--><!--anchor:quote:was%20used%20as%20the%20identification%20set-->. Parameters diverge if they "are not independent in their influence on the recorded outputs" (Bogdanski & Best, 2018)<!--ref:A3--><!--anchor:quote:are%20not%20independent%20in%20their%20influence%20on%20the%20recorded%20outputs-->.

**No lap-time method was found (Moderate, by absence).** Cappello and Hoegh (2026)<!--ref:A9--><!--anchor:quote:we%20lean%20on%20moderately%20strong%20priors%20since%20we%20are%20relatively%20data%20poor--> estimated a latent tyre pace rather than a tyre parameter from timing data, with "moderately strong priors" (preprint text). A two-wheeler circle-test patent states that speed and angle are "more accurate than determining the mean lap time" (Roziere et al., 2011)<!--ref:A10--><!--anchor:quote:more%20accurate%20than%20determining%20the%20mean%20lap%20time-->. The stopping rules were not met and practitioner books stayed unread, so the absence rates Moderate.

**A track-fitted factor belongs to the whole system (Moderate).** Track identification "implicitly compensates for suspension geometry and compliance" (Braghin et al., 2011)<!--ref:A2--><!--anchor:quote:implicitly%20compensates%20for%20suspension%20geometry%20and%20compliance-->. The peak lateral force depends on the manoeuvre, "likely due to temperature differences and to vibrations at large slip angles" (Albinsson et al., 2019)<!--ref:A7--><!--anchor:quote:likely%20due%20to%20temperature%20differences%20and%20to%20vibrations%20at%20large%20slip%20angles-->. Sagmeister et al. (2024)<!--ref:A8--><!--anchor:quote:the%20respective%20scaling%20factors%20in%20the%20magic%20formula--> fitted the scaling factors of a road-tyre set to closed-loop race-car data. Inference: the team's factors will mix tyre mismatch, temperature and pressure effects, driver and controller.

### SQ2: Forms and Magnitudes of Temperature and Pressure Effects

**Published forms fit some MF slots (Moderate).** Singh and Sivaramakrishnan (2023)<!--ref:B4--><!--anchor:section:Eq1-Eq2--> reproduce the MF pressure multipliers on peak longitudinal friction and slip stiffness, printing px3 for ppx3. Besselink et al. (2010)<!--ref:B1--><!--anchor:quote:has%20been%20adapted%20to%20cope%20with%20large%20camber%20angles%20and%20inflation%20pressure%20changes--> adapted MF 5.2 to large pressure changes. Harsh and Shyrokau (2019)<!--ref:B11--><!--anchor:section:Eq24--> give temperature multipliers, with lateral terms in their Equation 25 (Harsh & Shyrokau, 2019)<!--ref:B11--><!--anchor:section:Eq25-->. With dpi the normalised pressure offset and dT = (T - T_ref) / T_ref:

```
mu_x*(1 + ppx3*dpi + ppx4*dpi^2)     Kxk*(1 + ppx1*dpi + ppx2*dpi^2)                     (1)
Dx*(1 + TX3*dT + TX4*dT^2)    Kx*(1 + TX1*dT + TX2*dT^2)    Dy*(1 + TY3*dT + TY4*dT^2)
PKY1*(1 + TY1*dT)             PKY2*(1 + TY2*dT)                                           (2)
```

Each multiplier equals 1 at the reference, so a normalised shape fits LMUX, LKX, LMUY and the PKY1 part of LKY. LKY cannot carry the TY2 "Temperature effect on location of cornering stiffness peak" (Harsh & Shyrokau, 2019)<!--ref:B11--><!--anchor:quote:Temperature%20effect%20on%20location%20of%20cornering%20stiffness%20peak-->. It also cannot carry a pressure term on the load at maximum cornering stiffness (Singh & Sivaramakrishnan, 2023)<!--ref:B4--><!--anchor:section:Eq9-->.

**Magnitudes are few and conflict (Low).** Twelve of the sixteen SQ2 sources are passenger-tyre, bench or model sources, partly because of the cap, and each defines temperature differently (Table 2).

**Table 2**

*Reported Temperature and Pressure Effects*

|Channel|Finding (certainty)|Source and conditions|
|:---|:---|:---|
|LMUY, T|Window mu_y = mu_ymax + 1 - cosh((T - T_opt) / T_disp), with 1.1, 88 deg C and 50 deg C. DERIVED: about 7% lost 20 deg C off the optimum (Moderate)|Alcazar Vargas et al. (2022)<!--ref:B10--><!--anchor:section:Table4-->, summer tyre, belt bench|
|LMUY, T|Full-sliding mu_r = 1.070 - 0.003967 T. DERIVED: 4.2% per 10 deg C at 30 deg C (Low)|Kobayashi et al. (2024)<!--ref:B13--><!--anchor:section:TableI-->, drifting, 30 to 80 deg C|
|LKY and LMUY, T|Stiffness -20% to -25% cold to hot (Low). Grip -10% in their Table 3 but about +26% in their Figure 18(b), same run (Singh & Sivaramakrishnan, 2023)<!--ref:B4--><!--anchor:section:Fig18b--> (Very low, self-conflicting)|Singh and Sivaramakrishnan (2023)<!--ref:B4--><!--anchor:section:Table3-->, one new tyre, no break-in|
|LMUY, T|Optimum 90 to 105 deg C, no data source (Very low)|West and Limebeer (2022)<!--ref:B14--><!--anchor:section:2.1-->, F1 model, super-soft compound|
|LMUY and LKY, T|Hot: higher grip, lower stiffness (Low)|Farroni (2014)<!--ref:A5--><!--anchor:quote:higher%20grip%20and%20lower%20cornering%20stiffness-->, tyre type not stated|
|LKY, T|About exponential fall, model-derived form (Moderate)|Lu et al. (2022)<!--ref:B9--><!--anchor:quote:decreases%20gradually%20with%20increasing%20temperature-->, soaked 0 to 53 deg C, brand given as Hankook and as ATLAS|
|LKY, T|C_alpha = p2 / (T - p1) + p3 on asphalt temperature, which cut the in-sample SD by 55% to 65% (Moderate)|Savant et al. (2023)<!--ref:A11--><!--anchor:section:3.2-->, road cars, linear range|
|LKY, T and p|No temperature influence, slight pressure effect without a tendency (Moderate nulls)|Alcazar Vargas et al. (2022)<!--ref:B10--><!--anchor:quote:practically%20no%20influence%20on%20any%20of%20the%20parameters%20studied%2C%20while%20pressure%20seems%20to%20have%20a%20slight%20effect-->, soaked 30 to 90 deg C|
|LKY and LMUY, p|Stiffness +10% per 20% change from nominal, sign changes with load. Peak friction effect not very dominant (Low)|Singh and Sivaramakrishnan (2023)<!--ref:B4--><!--anchor:section:Table3-->, 33 to 45 psi|
|LKX and LMUX, p|Significant, no sign or size (Moderate for existence)|Hoepping et al. (2018)<!--ref:B2--><!--anchor:quote:a%20significant%20influence%20of%20the%20inflation%20pressure%20on%20longitudinal%20tire%20characteristics%20as%20slip%20stiffness%20or%20maximum%20traction%20force-->, roller rig|

*Note.* DERIVED values are arithmetic on the published fits. Harsh and Shyrokau (2019)<!--ref:B11--><!--anchor:section:Eq24--> give forms only. Held-out record B5: rough tracks generally give two friction peaks, so one optimum is an approximation.

**Directions.** Four Table 2 sources report that cornering stiffness falls with temperature, against the null of the only grade A rig source. For peak lateral friction, Alcazar Vargas et al. (2022)<!--ref:B10--><!--anchor:section:Eq8--> fit a window, and West and Limebeer (2022)<!--ref:B14--><!--anchor:quote:operating%20window%20will%20vary%20between%20different%20tyre%20compounds--> model one. Kobayashi et al. (2024)<!--ref:B13--><!--anchor:section:TableI--> report a fall in full sliding and Farroni (2014)<!--ref:A5--><!--anchor:quote:higher%20grip%20and%20lower%20cornering%20stiffness--> a rise. Singh and Sivaramakrishnan (2023)<!--ref:B4--><!--anchor:quote:Typically%2C%20the%20interaction%20forces%20reach%20their%20maximum%20values%20only%20within%20a%20narrow%20temperature%20range--> state a window only as a general point, citing others, and their own run conflicts with itself. Both direction ratings stay Low to Moderate. The window and full-sliding fits disagree over about 45 to 80 deg C, where both data sets lie (DERIVED). Flash heating in full sliding may explain part of the disagreement (inference), since local heating makes rubber friction fall with sliding velocity (Persson, 2006)<!--ref:B6--><!--anchor:quote:the%20temperature%20increase%20results%20in%20a%20decrease%20in%20rubber%20friction%20with%20increasing%20sliding%20velocity-->. A vendor simulation found that higher tyre temperature or rolling speed mostly degraded handling (Lugaro et al., 2020)<!--ref:B12--><!--anchor:quote:a%20higher%20tire%20temperature%20and%20%2F%20or%20higher%20rolling%20speed%20results%20in%20a%20degradation%20of%20the%20vehicle%20handling%20response-->. So the conflict stays unresolved, and the data must straddle an optimum before anyone reads a sign.

**Pressure.** Higher pressure lowers cornering stiffness at low loads and raises it at high loads (Singh & Sivaramakrishnan, 2023)<!--ref:B4--><!--anchor:quote:A%20lower%20cornering%20stiffness%20at%20low%20vertical%20loads%20and%20a%20higher%20cornering%20stiffness%20at%20high%20vertical%20loads-->. On constant-radius tracks, pressure changed steering demand, lateral acceleration and self-steering (Jagicza et al., 2026)<!--ref:B3--><!--anchor:quote:pressure-dependent%20changes%20in%20steering%20angle%20demand%2C%20achievable%20lateral%20acceleration%2C%20and%20self-steering%20characteristics-->. The grade A rig found only a slight effect without a tendency (Table 2). Slip stiffness and traction force respond significantly, with no sign or size given (Hoepping et al., 2018)<!--ref:B2--><!--anchor:quote:a%20significant%20influence%20of%20the%20inflation%20pressure%20on%20longitudinal%20tire%20characteristics%20as%20slip%20stiffness%20or%20maximum%20traction%20force-->. For peak lateral friction, however, earlier studies showed no clear general relation (Singh & Sivaramakrishnan, 2023)<!--ref:B4--><!--anchor:quote:the%20peak%20lateral%20friction%20characteristics%20do%20not%20show%20a%20clear%20general%20relation%20for%20the%20inflation%20pressure-->. No source compares both friction channels on one tyre, so one shared value for LMUX and LMUY is an untested assumption (inference). Pressure also changes the loaded radius (Besselink et al., 2010)<!--ref:B1--><!--anchor:quote:the%20effect%20of%20inflation%20pressure%20on%20the%20loaded%20radius%20and%20the%20tyre%20enveloping%20properties%20is%20discussed--> and "every aspect of tire performance" (Kasprzak et al., 2006)<!--ref:C1--><!--anchor:quote:Inflation%20pressure%20affects%20every%20aspect%20of%20tire%20performance-->. So a lap-time pressure factor lumps these channels (inference). No source gives a temperature or pressure effect on the load sensitivity of peak friction, the LFZO channel.

### SQ3: Evidence for a Practical Protocol

**Temperature and hot pressure are outcomes of the lap (Strong).** Sliding friction heats the tread (Harsh & Shyrokau, 2019)<!--ref:B11--><!--anchor:section:Eq3-->. At constant gas volume, "the pressure is directly proportional to the Kelvin temperature" (Harsh & Shyrokau, 2019)<!--ref:B11--><!--anchor:quote:the%20pressure%20is%20directly%20proportional%20to%20the%20Kelvin%20temperature-->. In the equation below, p_cold is the pressure at ambient temperature (Harsh & Shyrokau, 2019)<!--ref:B11--><!--anchor:quote:pcold%20is%20the%20pressure%20of%20the%20tire%20at%20ambient%20temperature-->:

```
p_hot = p_cold * (T_gas + 273) / (T_amb + 273)      (3)
```

Equation 3 needs absolute pressure, so add the logged barometric pressure to gauge readings (inference). DERIVED at 1.013 bar: 1.60 bar gauge set at 20 deg C becomes about 2.14 bar gauge at 80 deg C. But gauge values give 1.93 bar, which understates the rise by about 40%. In sun or warmers, use the tyre temperature at the cold set (T_set) for T_amb (inference). The driver aims to keep surface temperatures in the window (West & Limebeer, 2022)<!--ref:B14--><!--anchor:quote:control%20objective%20is%20to%20get%20the%20tyre%20surface%20temperatures%20into%20this%20optimal%20operating-->. Pressure follows the air temperature in the tyre (Kasprzak et al., 2006)<!--ref:C1--><!--anchor:quote:as%20the%20air%20temperature%20within%20the%20tire%20varies%20during%20use%20so%20does%20the%20inflation%20pressure-->. So hot pressure and probe temperature are two readings of one thermal state (inference).

**Pressure can be designed, running temperature cannot (Moderate).** Rig studies set pressure by design (Hoepping et al., 2018)<!--ref:B2--><!--anchor:quote:covers%20a%20wide%20range%20of%20different%20wheel%20loads%20and%20inflation%20pressures%20for%20three%20different%20tire%20variations--> and separated it from temperature (Angrick et al., 2014)<!--ref:B8--><!--anchor:quote:the%20effects%20of%20temperature%20and%20inner%20pressure%20on%20tire%20characteristics%20is%20investigated%20separately-->. The one published track lever for temperature is the driving pattern (Farroni, 2014)<!--ref:A5--><!--anchor:quote:defined%20in%20order%20to%20collect%20data%20at%20different%20tyre%20temperature%20and%20wear%20conditions-->. Warmers, where allowed, set only the start temperature (inference).

**Measurement is the weakest link (Very low).** Friction follows the surface temperature at the road interface (Singh & Sivaramakrishnan, 2023)<!--ref:B4--><!--anchor:quote:is%20only%20influenced%20by%20the%20surface%20temperature%20of%20the%20tire%20at%20the%20road%20interface-->. The carcass changes "much more slowly" than the tread (West & Limebeer, 2022)<!--ref:B14--><!--anchor:quote:the%20carcass%20temperature%20varies%20much%20more%20slowly-->. So no post-session reading equals the contact temperature. No peer-reviewed study compares probe, pyrometer and infrared readings at a track or gives the cool-down after pit entry, so the pilot must measure it. Tread readings can be combined with weights w_i = (T_i - T_amb) / sum_j (T_j - T_amb) (Alcazar Vargas et al., 2022)<!--ref:B10--><!--anchor:section:Eq3-Eq4-->.

**Confounders.** Cappello and Hoegh (2026)<!--ref:A9--><!--anchor:quote:a%20skewed-t%20observation%20model%20to%20account%20for%20asymmetric%20driver%20errors--> used a skewed-t model for asymmetric driver errors. Their fuel-mass term (Cappello & Hoegh, 2026)<!--ref:A9--><!--anchor:quote:Lap%20times%20are%20modeled%20as%20a%20function%20of%20fuel%20mass%20and%20latent%20tire%20pace--> maps to battery state and derating here (inference). Mizuno et al. (2005)<!--ref:B7--><!--anchor:quote:often%20vary%20depending%20on%20the%20season%20or%20with%20differences%20in%20the%20road%20surface%20temperature--> note that test results vary with season and road temperature, and outer and loaded tyres heat first (Harsh & Shyrokau, 2019)<!--ref:B11--><!--anchor:quote:start%20heating%20up%20first-->. One rig run used new tyres "without a warm-up or break-in procedure" (Singh & Sivaramakrishnan, 2023)<!--ref:B4--><!--anchor:quote:running%20the%20test%20on%20new%20tires%20without%20a%20warm-up%20or%20break-in%20procedure-->, which mixes heating with conditioning (inference). No source quantifies track evolution, wind, air density, derating or the sessions needed. So the protocol logs them under one design rule: a designed input must not change in step with time.

### SQ4: Evidence for an Estimation Procedure

**The calibration structure is known (Strong for statistics, Low for transfer).** Calibration attempts to "correct for any inadequacy of the model which is revealed by a discrepancy" (Kennedy & O'Hagan, 2001)<!--ref:C7--><!--anchor:quote:attempt%20to%20correct%20for%20any%20inadequacy%20of%20the%20model%20which%20is%20revealed%20by%20a%20discrepancy-->. Without a discrepancy term, an analysis "may lead to biased and over-confident parameter estimates" (Brynjarsdottir & O'Hagan, 2014)<!--ref:C8--><!--anchor:quote:may%20lead%20to%20biased%20and%20over-confident%20parameter%20estimates%20and%20predictions-->. With a discrepancy term, discrepancy and parameters stay confounded unless priors are meaningful (Brynjarsdottir & O'Hagan, 2014)<!--ref:C8--><!--anchor:quote:which%20will%20only%20be%20resolved%20with%20meaningful%20priors-->. Brown and Atamturktur (2018)<!--ref:C9--><!--anchor:quote:a%20framework%20for%20modeling%20the%20calibration%20parameters%20as%20functions%20of%20the%20control%20inputs--> model parameters as functions of control inputs, without a discrepancy term (Brown & Atamturktur, 2018)<!--ref:C9--><!--anchor:quote:is%20solely%20due%20to%20misspecified%20parameters%20in%20the%20computer%20model%20and%20measurement%20error-->. They show that a missing function can mimic a need for a discrepancy term (Brown & Atamturktur, 2018)<!--ref:C9--><!--anchor:quote:can%20mislead%20a%20researcher%20into%20thinking%20a%20calibrated%20model%20is%20in%20need%20of%20an%20empirical%20discrepancy%20term-->. A review names unidentifiability under model inadequacy as a challenge (Sung & Tuo, 2024)<!--ref:C10--><!--anchor:quote:the%20challenges%20of%20the%20unidentifiability%20issue%20while%20accommodating%20model%20inadequacy-->, and two-stage random-effects models suit unbalanced data (Laird & Ware, 1982)<!--ref:C12--><!--anchor:quote:two-stage%20random-effects%20models%20can%20be%20used%20easily-->.

**The target decides what enters the model.** For the cold-pressure effect and the track-temperature association, the tyre temperature is an effect of both the exposure and pace. By a conventional criterion, "necessary although not sufficient", a confounder is "not a descendant (effect) of the exposure or outcome" (Greenland et al., 1999)<!--ref:C3--><!--anchor:quote:not%20a%20descendant%20%28effect%29%20of%20the%20exposure%20or%20outcome-->. Within strata of a common effect, the association of its causes "will almost certainly differ from the marginal association" (Greenland et al., 1999)<!--ref:C3--><!--anchor:quote:will%20almost%20certainly%20differ%20from%20the%20marginal%20association-->. So the reading stays out of the model and must not select the data (inference). A tyre-temperature effect itself needs a pace measure that the core data lack. Instruments need stated assumptions (Angrist et al., 1996)<!--ref:C4--><!--anchor:quote:under%20some%20simple%20and%20easily%20interpretable%20assumptions-->, and rotation already gives the total effect of the designed offset (inference). Slip power heats the tread (West & Limebeer, 2022)<!--ref:B14--><!--anchor:section:Eq4-->, so the reading shows how hard the tyre worked.

## Track Protocol

The protocol assumes no GPS or acceleration logging, and Table 3 states what each field serves. Pit entry means the moment the car stops in the team's pit box. Minimality stays untested until B2 runs the element-removal test.

### Safety

These operating rules are not findings. Circuit, event and car-maker rules come first, and the stricter rule wins. Safety documents S1 to S4, listed after the References and outside the evidence set, support them.

1. Write a risk assessment for each test day. Take readings only in the team's pit box with the car stopped, under the event's pit-lane rules (S1). Never approach a moving car to keep a clock time.
2. Before anyone touches the car, a person trained for its high-voltage system makes the drive system safe. That person also checks the high-voltage status by the maker's procedure, or by one written into the risk assessment (S4). After a fault, nobody touches the car until that person clears it.
3. Treat tyres, wheels, brakes and motor housings as hot, and wear heat-resistant gloves. Keep hands and tools off the wheel motors and their high-voltage cables (P7).
4. Keep pressures and tyre temperatures inside the tyre maker's limits, with a margin. Safety overrides the sweep. Correct a pressure outside a limit before the car runs again, drop that level and record why. Inflate with a regulated supply, never standing over the tyre, and wear safety glasses (S2). Only a qualified person mounts tyres.
5. Torque wheel fasteners to the maker's value after a change, and check them after the first run (S3).
6. Run limit tests only where the venue approves them. Asymmetric pressures change the balance (Jagicza et al., 2026)<!--ref:B3--><!--anchor:quote:Asymmetric%20front-rear%20pressure%20distributions%20were%20found%20to%20modify%20the%20understeer-oversteer%20balance-->, so brief the driver and raise the speed in steps.
7. The driver may stop any run. Stop at once if a tyre behaves abnormally.
8. Tell drivers what is recorded and why. Keep the driver-code key apart, and use the data only for tyre calibration. Ask an ethics office before academic use, and keep downloaded source texts private.

### Before the First Session

1. Get the owner inputs: tyre-maker limits, warmer and pit-lane rules, drivers, sensors and sector timing. Also get dt_dec (the smallest lap-time difference that changes a decision) and N_avail (sessions per season for calibration).
2. Ask the tyre maker for a TIR file, the MF property file of the race tyre. The factors then correct a tyre-specific baseline, as with MF "models provided by the tire manufacturer" (Cortivo et al., 2024)<!--ref:A6--><!--anchor:quote:models%20provided%20by%20the%20tire%20manufacturer-->.
3. Fix the lap-validity rules. Exclude out-laps, in-laps and laps in traffic or under a yellow or red flag. Mark laps more than 3 raw median absolute deviations slower than the stint median for a sensitivity fit.
4. Scrub each new set before it enters a sweep stint, or keep its first stint out of the sweep. In one rig run on new tyres without break-in, peak grip changed by about a quarter over the first sweeps (Singh & Sivaramakrishnan, 2023)<!--ref:B4--><!--anchor:section:Fig18b-->.
5. Choose two or three cold-pressure levels including the reference, with the largest step inside the limits. Use one relative offset on all corners, on gauge readings.
6. Use one calibrated gauge, one probe at a fixed insertion depth and a clock synchronised with the timing.

### In Every Session

7. Run each level in its own stint, with the reference in every session, in rotated order (R-L-H, L-H-R, H-R-L). Spread sessions over the day so that track temperature does not move with the event day.
8. Set cold pressures on cold tyres. Record the reading, soak time, tyre temperature at the set, sun or shade and warmer use. With warmers, the series rules and a stated set temperature define the level.
9. To change level within a session, cool the tyres to near their cold-set temperature. As an alternative, use a second set and rotate sets across levels over the sessions. If no air was lost since the cold set, a warm reset needs no gas temperature. Reset to p_w2 = (p_w1 + p_atm) * (p_c2 + p_atm) / (p_c1 + p_atm) - p_atm (DERIVED, Appendix D). All pressures are gauge values, with p_c1 the last cold set, p_c2 the next level and p_w1 the warm reading. Check the reset once in the pilot against a later cold reading, and record the method and heat cycles.
10. Do not bleed or top up between the cold set and the pit-entry reading of a sweep stint. This step applies unless Safety rule 4 requires a bleed or top-up.
11. Drive the same pre-declared number of flying laps per stint, and record the lap pattern before pit entry.
12. At pit entry, read the corners in a fixed order. Read the hot pressure first, then the probe at the inner, centre and outer tread. Record clock times and the time since the last flying lap.
13. Option: a mid-session stop for a second set of readings.

### After Every Session

14. In the pilot, read one corner several times at logged delays (cool-down), and run the reference twice, R-L-R (stint-to-stint scatter). These values replace the assumptions of B2.
15. Keep every stint under its assigned level. If a stint ends early or a safety correction interrupts it, keep its valid laps before the event. Log the reason, and fit with and without such stints, with their count per level. A stint outside the maker's operating window is a result of its level. A reading outside a safety limit follows Safety rule 4.

**How many sessions.** No source gives a number. B2 must compute it from the pilot scatter, with a design that states "aims, data-generating mechanisms, estimands, methods, and performance measures" (Morris et al., 2019)<!--ref:C15--><!--anchor:quote:aims%2C%20data-generating%20mechanisms%2C%20estimands%2C%20methods%2C%20and%20performance%20measures-->. Each full session adds one within-session contrast per non-reference level, and precision depends mostly on the number of sessions. The sweep runs until the stopping rule holds or N_avail is used.

**Optional runs and extras.** Timed circles at two radii and timed braking runs per level may split lateral and longitudinal grip if the driver reaches the limit (inference). Circle tests respond to pressure (Jagicza et al., 2026)<!--ref:B3--><!--anchor:quote:pressure-dependent%20changes%20in%20steering%20angle%20demand%2C%20achievable%20lateral%20acceleration%2C%20and%20self-steering%20characteristics-->. Front-only or rear-only offsets may show which axle limits (inference). Sector times would add a pace measure without tyre temperature and a braking-cornering split (inference). In-session IR sensors would add the running surface temperature per tyre (Kobayashi et al., 2024)<!--ref:B13--><!--anchor:section:IV.A-->. TPMS, ride height and coast-down runs have no verified source [MATERIAL GAP]. The gain of each extra ranges "from minimal to substantial" and must be measured (Arendt et al., 2012)<!--ref:C11--><!--anchor:quote:by%20an%20amount%20that%20ranges%20from%20minimal%20to%20substantial-->. Held-out record A13: segment times of two drivers differed in 3 of 9 segments and cancelled over the lap.

**Table 3**

*Log Sheet Template and What Each Field Serves*

|Block|Fields|When|Serves (status)|
|:---|:---|:---|:---|
|Session|Date, circuit, session ID and type, start time, event day, driver codes, combined mass of car, driver and ballast, setup ID, green or washed track|Start|Session and driver terms, QSS mass (log)|
|Stint|Track and ambient temperature, wind, barometric pressure|Stint start|b1, air density, Equation 3 (log)|
|Stint|Level (R, L or H), change method, set ID, laps on set, heat cycles, warmer use and set temperature|Before|Design, set term, tyre age (core)|
|Each corner|Cold pressure (gauge), gauge ID and calibration date, soak time, tyre temperature at set, sun or shade, clock time|Before|dp, Equation 3 (core)|
|Stint|Battery state and any derating|Start and end|QSS power limit (log)|
|Stint|Lap pattern before pit entry, time since last flying lap, any bleed, top-up or early stop and why|Pit entry|Stint validity (core)|
|Each corner|Hot pressure, then probe at inner, centre and outer tread, with clock times and insertion depth|Pit entry|Safety, second outcome, Equation 3 check (core)|
|Lap|Lap number, lap time, time in session, validity and reason|Every lap|Stint median, time term (core)|

*Note.* Core fields serve the estimate or safety, log fields the confounder model. Use one pressure unit and driver codes in shared files.

## Estimation Procedure

The QSS is the forward model, because a run takes tens of milliseconds.

**Inputs and stint values.** Index j is the stint, k the session and d the driver. Pressures are gauge values, as read:

```
dp_j  = (p_cold,j - p_cold,ref) / p_cold,ref                      (4)
dTt_k = mean stint track temperature of session k - T_track,ref   (5)
```

For each valid stint, set the QSS to the logged air density, power limit, setup and mass. Then find the grip scale s_jk (LMUX and LMUY together, Fz0_shift at 1) at which the QSS lap equals the stint's median valid lap. This needs lap time to fall steadily as grip rises (for B2 to confirm).

**Model.**

```
y_jk    = ln s_jk = alpha_k + (a1 + w_k)*dp_j + a2*dp_j^2 + a3*dp_j*dTt_k
          + c1*t_jk (+ c2*age_jk) + g_set + e_jk                     (6)
alpha_k = ln s_eff,d + b1*dTt_k + u_k                                (7)
g_p(dp) = exp(a1*dp + a2*dp^2),   g_p(0) = 1
```

Here t_jk is the stint-midpoint time in session, age_jk the laps on the set and g_set a tyre-set effect. In addition, w_k is a session slope deviation with variance tau^2, u_k a session effect and e_jk the stint error. Start with a1 and the time term. Add age_jk only if the set plan separates it from time and the collinearity index of the pair stays low. Add a2 (three levels), a3 or b1 only if held-out sessions support them. The tyre temperature stays out.

**Fitting steps.**

1. Freeze protocol, estimands, margins and code before team data exist. This extends the advice to "write detailed simulation protocols before writing code" (Morris et al., 2019)<!--ref:C15--><!--anchor:quote:write%20detailed%20simulation%20protocols%20before%20writing%20code--> (inference). Use the 'corrected' physics mode, with 'matlab' as a check (P6).
2. Fit Equation 6 by restricted maximum likelihood (REML), with fixed session effects and valid laps per stint as weights. This way, a1 rests on within-session contrasts that session effects cannot bias (inference).
3. Estimate tau from the spread of per-session slopes. This is because parameters "vary across individuals, with a distribution specified at the second stage" (Laird & Ware, 1982)<!--ref:C12--><!--anchor:quote:some%20random-effects%20parameters%20vary%20across%20individuals%2C%20with%20a%20distribution%20specified%20at%20the%20second%20stage-->.
4. Fit b1 on the K session effects with a driver term (Equation 7), and report K. Time of day, rubber, tyre sets or wind can move with track temperature, so b1 is an association.
5. With several drivers, add a driver term and apply the driver rule.
6. Validate by leave-one-session-out prediction. For g_p, predict the within-session contrasts of the held-out session (each stint median minus that of the reference stint). In these contrasts, the unknown session level cancels. For the level and b1, predict the session mean. Score both by mean absolute error against M0 (factors at 1) and M0b. M0b uses dp, dTt and time with session and driver effects, and no simulator.

**Uncertainty.**

- Use t-based 95% intervals with about K - 1 degrees of freedom for a1 and the driver term, or the Kenward-Roger approximation (lead L4). Likelihood-based intervals are "considered superior to asymptotic confidence intervals for finite samples" (Raue et al., 2009)<!--ref:C13--><!--anchor:quote:considered%20superior%20to%20asymptotic%20confidence%20intervals%20for%20finite%20samples-->. But with their chi-square threshold (Raue et al., 2009)<!--ref:C13--><!--anchor:quote:quantile%20of%20the-->, they cover less than 95% at a handful of sessions (DERIVED). Their coverage is about 88% at K = 4 (Appendix D).
- Use the profile likelihood to test practical identifiability: a parameter fails "if the likelihood-based confidence region is infinitely extended" (Raue et al., 2009)<!--ref:C13--><!--anchor:quote:if%20the%20likelihood-based%20confidence%20region%20is%20infinitely%20extended-->. Check parameter subsets with the collinearity index of Brun et al. (2001)<!--ref:A15--><!--anchor:quote:near-linear%20dependence%20of%20sensitivity%20functions%20of%20parameter%20subsets--> (Appendix D). Its critical values "lie in the range of 5-20 according to our experience" (Brun et al., 2001)<!--ref:A15--><!--anchor:quote:lie%20in%20the%20range%20of%205-20%20according%20to%20our%20experience-->. The index depends on "changes in the experimental layout" (Brun et al., 2001)<!--ref:A15--><!--anchor:quote:changes%20in%20the%20experimental%20layout-->.
- The standard error of a1 is roughly sqrt((tau^2 + se^2) / K), with se the typical within-session slope error (DERIVED). So the effective sample size lies between the numbers of sessions and stints and approaches K when tau dominates. Stint-median scatters of 0.1%, 0.2% and 0.4% in lap time equal about 0.4%, 0.7% and 1.5% of grip per stint (DERIVED from P5).
- Performance measures "should usually be considered jointly" (Morris et al., 2019)<!--ref:C15--><!--anchor:quote:performance%20measures%20should%20usually%20be%20considered%20jointly-->. So B2 reports bias, width and 95% coverage, and it simulates the stopping rule as a sequential procedure.

**Update and stopping.** Refit Equation 6 on all valid stints after each session, and let the refit govern. Between refits, per-session slopes may be pooled with weights 1/(se_k^2 + tau^2), but the pool must not stop the sweep. Published forms serve only as wide bounds, because bounds "are obviously influential in determining posterior sample paths" (Brown & Atamturktur, 2018)<!--ref:C9--><!--anchor:quote:are%20obviously%20influential%20in%20determining%20posterior%20sample%20paths-->. Stop with success only after K_min full sessions (set by B2). The t-based 95% interval of a1 must also be finite and plausible, with a lap-time half-width of at most dt_dec. Stop for budget at N_avail sessions. If the pressure effect on lap time lies inside plus or minus dt_dec, report no decision-relevant effect. Then, or at budget, keep LMUX = LMUY = 1 and report the interval.

**Gates for g_p.** g_p enters LMUX and LMUY in the QSS only after four gates, all fixed before data:

1. A success stop.
2. The driver rule: the driver-by-pressure term passes when its 95% interval lies inside a margin from dt_dec. This interval is in lap time over the level range. A non-significant test does not pass. With one driver, g_p holds for that driver only.
3. The contrast check: on the contrast target, g_p beats M0 by a superiority margin. It also stays within a non-inferiority margin of M0b. Both margins are set from dt_dec, with power from B2. Superiority over M0b would rarely pass, because on a calibration circuit both models carry the same pressure information (inference).
4. Transfer: g_p, tagged by inversion model (QSS, centreline, physics mode), driver, circuit and temperature span, serves only calibration circuits. A new circuit joins after one session there with the reference and one other level. It joins if the QSS-predicted contrast with g_p lies within a margin from dt_dec of the measured contrast.

Keep g_p out of the full model until B2 gives its elasticity at s = 0.98, 1.00, 1.02 and s_eff. This is because the QSS runs a fixed centreline. In the full model, g_p would carry a stiffness-and-balance response as uniform peak grip, unfit for setup optimisation (inference).

**Into the simulator.** Store s_eff with its tags (model, tyre set, physics mode, driver, circuit) in a correlation record outside the vehicle model. Predict a measured lap as r times the simulated lap, with r = t_QSS(s_eff) / t_QSS(1). This prediction holds if the lap-time response near grip 1 matches that at s_eff (a B2 check). For pressure decisions, use t_QSS(s_eff * g_p) / t_QSS(s_eff). Run setup work at the nominal level. Rank candidate setups at s = 1 and at s = s_eff, and act only on rankings that agree.

## Identifiability (SQ5)

Verdict codes: V1 is a function of temperature and pressure, V2 a constant per tyre and season and V3 a fixed literature shape. V4 is "keep at 1" and V5 "needs an optional extra". All are candidates.

- Per circuit and set of conditions, the data have one lap time per lap and one designed lever. These give at most one grip direction plus its pressure response (inference). Pooled circuits could in principle separate more, but each circuit needs its own level.
- LKX and LKY cannot change the QSS lap time (P1). This non-identifiability "is related to the model structure independent of experimental data" (Raue et al., 2009)<!--ref:C13--><!--anchor:quote:is%20related%20to%20the%20model%20structure%20independent%20of%20experimental%20data-->. The only remedy is "a qualitatively new measurement" (Raue et al., 2009)<!--ref:C13--><!--anchor:quote:The%20only%20remedy%20is%20a%20qualitatively%20new%20measurement-->, but no measurement reaches factors that the QSS does not contain. Column (c) of Table 4 assumes a model that resolves slip, such as the full model, whose sensitivity nobody has measured.
- For the four-motor car, the QSS limits take the smaller axle capacity (P4), so lap times cannot give per-axle factors through the QSS.
- In the QSS, LFZO shares the peak-friction channel with LMUX, LMUY and the Cl scale (P3). Its effect grows with load (DERIVED, Appendix D), so it is only near-collinear on one circuit.
- With Fz0 fixed, 1/Fz0_shift and the combined scale are one direction in the QSS (P2). Keep Fz0_shift at 1 and refit s if it changes, for example through MLTP_TyreOptim. Fixed factors and Fz0_shift bias the free ones unless they "are at the true parameter values" (Brun et al., 2001)<!--ref:A15--><!--anchor:quote:are%20at%20the%20true%20parameter%20values--> (inference from the general result).

**Table 4**

*Identifiability of Each Factor by Data Set*

|Factor|(a) Core data|(b) Plus GPS|(c) Plus accelerations|Reason and sources|
|:---|:---|:---|:---|:---|
|LMUY|Weakly, inside one scale with LMUX|Weakly, from corner speeds (inference)|Effective lateral peak if the driver reaches the limit|Timing data needed strong priors (Cappello & Hoegh, 2026)<!--ref:A9--><!--anchor:quote:we%20lean%20on%20moderately%20strong%20priors%20since%20we%20are%20relatively%20data%20poor-->, manoeuvre-dependent peak (Albinsson et al., 2019)<!--ref:A7--><!--anchor:quote:The%20maximum%20lateral%20force%20is%20sensitive%20to%20the%20chosen%20manoeuvres-->|
|LMUX|Weakly, circuit-dependent share (inference)|Weakly, from braking (inference)|Effective braking peak at the limit|Identified with hard braking and high slip (Bogdanski & Best, 2018)<!--ref:A3--><!--anchor:quote:including%20some%20extreme%20high%20slip%20inputs-->, P4|
|LKY|Not, structural in the QSS|Not, no slip angle|Weakly, with steer, wheel speeds and aggressive runs, in a slip model|Needs slip-angle information, measured as in Braghin et al. (2011)<!--ref:A2--><!--anchor:quote:vehicle%20sideslip%20angle%2C%20yaw%20rate%2C%20lateral%20acceleration%2C%20speed%20and%20steer%20angle--> or estimated (inference). A framework study gives no magnitudes (Zarate Villazon et al., 2026)<!--ref:A14--><!--anchor:quote:a%20subset%20of%20parameters%20critical%20to%20combined%20slip%20of%20the%20MF%20tire%20model-->. P1|
|LKX|Not, structural in the QSS|Not|Weakly, with motor speeds, a speed reference and hard runs, in a slip model|Needs the slip ratio (Bogdanski & Best, 2018)<!--ref:A3--><!--anchor:quote:steer%20angle%20and%20the%20four%20wheelspeeds-->, P1|
|LFZO|Not, near-collinear on one circuit, mode-dependent (P6)|Weakly at best|Weakly, with a known aero load|Load channel (inference from P3)|
|Cd scale|Not: about 0.1% of lap per 20% drag, confounded with power, wind and rolling resistance|Weakly, from coasting (inference)|Weakly, from coast-down deceleration (inference)|P5|
|Cl scale|Not, collinear with LMUY on one circuit|Weakly, fast against slow corners (inference)|Weakly to identifiable, envelope against speed (inference)|Inference: aliased with speed-dependent friction (Persson, 2006)<!--ref:B6--><!--anchor:quote:the%20temperature%20increase%20results%20in%20a%20decrease%20in%20rubber%20friction%20with%20increasing%20sliding%20velocity-->|
|g_p|In principle, as the total effect of the designed cold pressure, lumped and tagged|In principle, partial split (inference)|In principle, with split (inference)|Total effect of a designed exposure (Greenland et al., 1999)<!--ref:C3--><!--anchor:quote:the%20net%20%28total%29%20effect%20of%20exposure-->, session structure (Laird & Ware, 1982)<!--ref:C12--><!--anchor:quote:two-stage%20random-effects%20models%20can%20be%20used%20easily-->, vehicle-level response (Jagicza et al., 2026)<!--ref:B3--><!--anchor:quote:pressure-dependent%20changes%20in%20steering%20angle%20demand%2C%20achievable%20lateral%20acceleration%2C%20and%20self-steering%20characteristics-->|
|g_T|Only b1, an association across sessions|Weakly, as a direct effect given a measured pace|As (b)|Pace heats the tyre (Kobayashi et al., 2024)<!--ref:B13--><!--anchor:section:Eq22-->. Exogenous precedent (Savant et al., 2023)<!--ref:A11--><!--anchor:section:3.2-->|
|Level s_eff|Effective only, tagged|Same|Same, envelope as a lower bound|Riders differ in level (Roziere et al., 2011)<!--ref:A10--><!--anchor:quote:values%20that%20differed%20substantially%20in%20terms%20of%20the%20speeds%20and%20camber%20angles%20measured-->. Discrepancy confounded (Brynjarsdottir & O'Hagan, 2014)<!--ref:C8--><!--anchor:quote:which%20will%20only%20be%20resolved%20with%20meaningful%20priors-->|

*Note.* "Effective" includes driver margin and model mismatch. Set (a) has a designed cold pressure. Held-out record A12, for LMUY and LMUX: a peak term was well estimated once data covered the peak. In the same record, on race data, the term fitted on both axles, although the rear did not pass its peak. Held-out record C6, for Cl: a staged speed-trace split on fabricated telemetry, with interacting stages.

**What is lost without a rig.** Inference: a controlled tyre state, because vehicle tests give a manoeuvre-dependent peak (Albinsson et al., 2019)<!--ref:A7--><!--anchor:quote:The%20maximum%20lateral%20force%20is%20sensitive%20to%20the%20chosen%20manoeuvres-->. Also lost are an independent force check (Braghin et al., 2011)<!--ref:A2--><!--anchor:quote:showing%20a%20successful%20tire%20model%20identification--> and a soaked uniform temperature (Lu et al., 2022)<!--ref:B9--><!--anchor:quote:the%20tire%20is%20first%20placed%20in%20an%20incubator%20equipment%20for%20several%20hours-->. Temperature separated from pressure (Angrick et al., 2014)<!--ref:B8--><!--anchor:quote:the%20effects%20of%20temperature%20and%20inner%20pressure%20on%20tire%20characteristics%20is%20investigated%20separately--> and pressure and load sweeps (Singh & Sivaramakrishnan, 2023)<!--ref:B4--><!--anchor:quote:The%20sensitivity%20study%20was%20conducted%20for%20a%20high-performance%20summer%20tire--> are lost too. A final loss is an indoor baseline with all factors at 1 (Arosio et al., 2005)<!--ref:A1--><!--anchor:quote:with%20scaling%20factors%20assumed%20to%20be%20one-->.

## Discussion

The evidence supports H1 in part: one effective grip scale and the pressure shape are candidates, and a temperature function is not.

**The most reliable answer is simulator-free.** M0b answers the pressure question on the calibration circuits. The MF slot earns its cost only where the simulator must predict unrun conditions. No step has tested that use yet, hence the transfer gate. Setup optimisation in the full model is where the lumped shape is least trustworthy, because pressure acts on balance too (Jagicza et al., 2026)<!--ref:B3--><!--anchor:quote:Asymmetric%20front-rear%20pressure%20distributions%20were%20found%20to%20modify%20the%20understeer-oversteer%20balance--> (inference).

**The level stays outside the tyre.** Shapes fit the slots, because published forms equal 1 at a reference (Harsh & Shyrokau, 2019)<!--ref:B11--><!--anchor:section:Eq24-->. But the level absorbs suspension compliance (Braghin et al., 2011)<!--ref:A2--><!--anchor:quote:implicitly%20compensates%20for%20suspension%20geometry%20and%20compliance-->, the rider (Roziere et al., 2011)<!--ref:A10--><!--anchor:quote:values%20that%20differed%20substantially%20in%20terms%20of%20the%20speeds%20and%20camber%20angles%20measured--> and the physics mode. The physics mode has an 11% lap offset (P6), which maps through P5 to grip of order 40% (DERIVED, rough). Inside an ideal-driver simulator, the level would weaken the tyre and misdirect setup work. On the other hand, setups ranked at grip 1 can rank differently at the car's real grip. Hence the two-level ranking check.

**Strength and transfer.** The pressure result rests on within-session contrasts, so it is stronger than the track-temperature association (inference). The tyre responds to the running pressure, but the team sets the cold one. So the cold-offset slope of the MF pressure multiplier moves with the thermal state, and more when the pressure term is curved (DERIVED, Appendix D). Hence tau, a3 and the temperature-span tag, and the code must state whether its MF pressures are gauge or absolute. A hot-pressure target design would reduce this dependence, at the cost of a less clean lever.

## Limitations

- B2 has not run, so no interval, session count, coverage or estimator score exists.
- No step yet tests g_p on an unrun circuit or setup. A reviewer lead outside the verified set (L3), on spatially resolved grip limits, bears on this.
- The stopping rules were not met, Semantic Scholar was rate-limited and practitioner books stayed unread. The cap left out two eligible race-tyre records (L1, L2). One is on an MF-based model with temperature, pressure and wear terms, and the other on slick against passenger tyre sensitivity. The pipeline verified neither them nor the other reviewer leads.
- Eighteen of the 36 sources were checked at abstract level, and most magnitudes come from passenger tyres on rigs. No source quantifies reading lag, pressure fall after pit entry or track evolution.
- Singh and Sivaramakrishnan (2023)<!--ref:B4--><!--anchor:section:Fig18b--> conflict with themselves on the temperature direction of peak grip (Table 2), which earlier phases missed.
- The causal methods come from epidemiology and economics. Greenland et al. (1999)<!--ref:C3--><!--anchor:quote:so%20that%20issues%20of%20measurement%20error%20and%20random%20variation%20will%20not%20arise--> set measurement error aside, and Angrist et al. (1996)<!--ref:C4--><!--anchor:section:abstract--> treat a binary treatment, with the published comments unread.
- Held-out records hold counter-evidence that this report cannot cite. This counter-evidence covers two friction peaks on rough tracks (B5) and segment-time differences that cancel over a lap (A13). It also covers a staged speed-trace split on fabricated telemetry (C6) and race-car excitation data (A12).
- The fitted temperature coefficients in the appendix of Harsh and Shyrokau (2019)<!--ref:B11--><!--anchor:section:AppendixA--> are unverified.
- Fifteen cited sources have industry links, nine with tyre makers or tyre-model developers (Appendix C).

## Recommendations

1. Use the simulator-free model M0b for the team's cold-pressure choice, because it needs no factor slot.
2. Apply the code actions of Table 5.

**Table 5**

*Recommended Code Action per Factor*

|Factor|Code action now|Candidate verdict|Better verdict needs|
|:---|:---|:---|:---|
|LMUX and LMUY|Add, linked, default 1. Set to g_p(dp) only in the QSS after the four gates. Keep at 1 in the full model until B2 gives its elasticity, and not for pressure-balance setup work.|V1 or V2 candidate, one QSS-equivalent shape|Sector times or GPS speed (V5)|
|Level s_eff|Keep out of the vehicle model, as a lap-time ratio|Effective only|Not possible from timing data|
|LKX|Do not add now (no QSS effect, P1), or hold at 1|V4 candidate|Steer, motor speeds, IMU, hard runs (V5)|
|LKY|As LKX|V4 candidate. V3 only after a verified recount, a B2 check and owner sign-off|Steer, wheel speeds, IMU, side-slip sensor (V5)|
|LFZO|Do not add now, or hold at 1. Never fit it from lap times|V4 candidate|A source for a temperature or pressure effect|
|Cd and Cl scales|Do not add now, or hold at 1|V4 candidates|Coast-down runs (Cd), ride height or GPS (Cl), leads only|
|Fz0_shift|No change (1.0). Refit s if it changes, including through MLTP_TyreOptim|Fixed by decision|Out of scope|

**Owner actions.**

1. Decide on B2 (a few hundred QSS runs and at most 12 full-model solves per circuit, one at a time). Include the elasticity at s = 0.98, 1.00, 1.02 and s_eff, and the coverage of the sequential stopping rule. Also include the power of the contrast check and one cross-circuit transfer case.
2. Supply dt_dec, N_avail, the tyre-maker limits, the warmer rules and set temperature, and the pit-lane rules. Also supply the sensors and the drivers per session. Say whether sector times, a mid-session stop, a circle area and a TIR file exist.
3. Choose the purpose of the MF slots: QSS prediction under unrun conditions, or setup optimisation. Setup optimisation needs more than the core data.
4. Choose a cold-pressure or hot-pressure target design, and decide on re-admitting the held-out sources.
5. Start with one pilot session, and feed its scatter and cool-down data into B2.
6. If adding one instrument, start with sector times, then GPS speed (no verified source measures their gain).

## AI Disclosure

AI Disclosure: This report was produced with AI-assisted research tools. The research pipeline included AI-powered literature search, source verification, evidence synthesis, and report drafting. In this run, AI agents (Claude models, Anthropic) did every step of the deep-research workflow of the academic-research-skills plugin, version 3.19.0. The agents ran in Claude Code on 19 and 20 October 2026. The steps included source verification, three devil's advocate checkpoints (two rounds at the third) and two rounds of editorial, ethics and claim-reference review. This revision and the round-2 ethics review ran on claude-opus-5-5, as did revision 1 by its record. The record does not name the other models. No human has reviewed the synthesis, the draft or the revisions, and the owner confirmed only the research scope. AI agents can misread a source, mistranscribe a quote or overstate a finding, and all reviews came from one pipeline, without a cross-model check. The team owner decides how to use this report and is responsible for that use.

**Funding and interests.** None declared by the pipeline. The team owner must confirm this statement.

## References

Albinsson, A., Bruzelius, F., Jacobson, B., & Bakker, E. (2019). Evaluation of vehicle-based tyre testing methods. *Proceedings of the Institution of Mechanical Engineers, Part D: Journal of Automobile Engineering, 233*(1), 4-17. https://doi.org/10.1177/0954407018760953

Alcazar Vargas, M., Perez Fernandez, J., Sanchez Andrades, I., Cabrera Carrillo, J. A., & Castillo Aguilar, J. J. (2022). Modeling of the influence of operational parameters on tire lateral dynamics. *Sensors, 22*(17), Article 6380. https://doi.org/10.3390/s22176380

Angrick, C., van Putten, S., & Prokop, G. (2014). Influence of tire core and surface temperature on lateral tire characteristics. *SAE International Journal of Passenger Cars - Mechanical Systems, 7*(2), 468-481. https://doi.org/10.4271/2014-01-0074

Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of causal effects using instrumental variables. *Journal of the American Statistical Association, 91*(434), 444-455. https://doi.org/10.1080/01621459.1996.10476902

Arendt, P. D., Apley, D. W., Chen, W., Lamb, D., & Gorsich, D. (2012). Improving identifiability in model calibration using multiple responses. *Journal of Mechanical Design, 134*(10), Article 100909. https://doi.org/10.1115/1.4007573

Arosio, D., Braghin, F., Cheli, F., & Sabbioni, E. (2005). Identification of Pacejka's scaling factors from full-scale experimental tests. *Vehicle System Dynamics, 43*(sup1), 457-474. https://doi.org/10.1080/00423110500229683

Besselink, I. J. M., Schmeitz, A. J. C., & Pacejka, H. B. (2010). An improved Magic Formula/Swift tyre model that can handle inflation pressure changes. *Vehicle System Dynamics, 48*(sup1), 337-352. https://doi.org/10.1080/00423111003748088

Bogdanski, K., & Best, M. C. (2018). Kalman and particle filtering methods for full vehicle and tyre identification. *Vehicle System Dynamics, 56*(5), 769-790. https://doi.org/10.1080/00423114.2017.1337914

Braghin, F., Cheli, F., & Sabbioni, E. (2011). Identification of tire model parameters through full vehicle experimental tests. *Journal of Dynamic Systems, Measurement, and Control, 133*(3), Article 031006. https://doi.org/10.1115/1.4003093

Brown, D. A., & Atamturktur, S. (2018). Nonparametric functional calibration of computer models. *Statistica Sinica, 28*(2), 721-742. https://doi.org/10.5705/ss.202015.0344

Brun, R., Reichert, P., & Kuensch, H. R. (2001). Practical identifiability analysis of large environmental simulation models. *Water Resources Research, 37*(4), 1015-1030. https://doi.org/10.1029/2000WR900350

Brynjarsdottir, J., & O'Hagan, A. (2014). Learning about physical parameters: The importance of model discrepancy. *Inverse Problems, 30*(11), Article 114007. https://doi.org/10.1088/0266-5611/30/11/114007

Cappello, C., & Hoegh, A. (2026). A state-space approach to modeling tire degradation in Formula 1 racing. *Journal of Sports Analytics, 12*, Article 22150218261446170. https://doi.org/10.1177/22150218261446170

Cortivo, D., Meneghetti, G., Massaro, M., Dindo, L., & Vendramin, M. (2024). Development of a tire characterization procedure from track acquisitions with an instrumented race vehicle. In W. Huang & M. Ahmadian (Eds.), *Advances in dynamics of vehicles on roads and tracks III* (Vol. 2, pp. 390-399). Springer. https://doi.org/10.1007/978-3-031-66968-2_38

Farroni, F. (2014). *Development of a grip and thermodynamics sensitive tyre/road interaction forces characterization procedure employed in high-performance vehicles simulation* [Doctoral dissertation, Universita degli Studi di Napoli Federico II]. fedOA. https://www.fedoa.unina.it/9766/

Greenland, S., Pearl, J., & Robins, J. M. (1999). Causal diagrams for epidemiologic research. *Epidemiology, 10*(1), 37-48. https://doi.org/10.1097/00001648-199901000-00008

Harsh, D., & Shyrokau, B. (2019). Tire model with temperature effects for Formula SAE vehicle. *Applied Sciences, 9*(24), Article 5328. https://doi.org/10.3390/app9245328

Hoepping, K., Augsburg, K., & Buechner, F. (2018). Extending the Magic Formula tire model for large inflation pressure changes by using measurement data from a corner module test rig. *SAE International Journal of Passenger Cars - Mechanical Systems, 11*(2), 103-118. https://doi.org/10.4271/06-11-02-0009

Jagicza, M., Nagy, L. I., & Lakatos, I. (2026). Influence of tire pressure distribution on vehicle cornering and self-steering behavior. *Vehicles, 8*(7), Article 154. https://doi.org/10.3390/vehicles8070154

Kasprzak, E. M., Lewis, K. E., & Milliken, D. L. (2006). *Inflation pressure effects in the nondimensional tire model* (SAE Technical Paper 2006-01-3607). SAE International. https://doi.org/10.4271/2006-01-3607

Kennedy, M. C., & O'Hagan, A. (2001). Bayesian calibration of computer models. *Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63*(3), 425-464. https://doi.org/10.1111/1467-9868.00294

Kobayashi, T., Weber, T. P., & Gerdes, J. C. (2024). Trajectory planning using tire thermodynamics for automated drifting. In *2024 IEEE Intelligent Vehicles Symposium (IV)* (pp. 2103-2109). IEEE. https://doi.org/10.1109/IV55156.2024.10588753

Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal data. *Biometrics, 38*(4), 963-974. https://doi.org/10.2307/2529876

Lu, L., Lu, D., Wu, H., Wang, W., Li, L., & Lv, Y. M. (2022). Research and modeling of tire cornering characteristics considering temperature based on UniTire model. *Proceedings of the Institution of Mechanical Engineers, Part D: Journal of Automobile Engineering, 236*(4), 497-511. https://doi.org/10.1177/09544070211024443

Lugaro, C., Alirezaei, M., Konstantinou, I., & Behera, A. (2020). *A study on the effect of tire temperature and rolling speed on the vehicle handling response* (SAE Technical Paper 2020-01-1235). SAE International. https://doi.org/10.4271/2020-01-1235

Mizuno, M., Sakai, H., Oyama, K., & Isomura, Y. (2005). Development of a tyre force model incorporating the influence of the tyre surface temperature. *Vehicle System Dynamics, 43*(sup1), 395-402. https://doi.org/10.1080/00423110500140187

Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. *Statistics in Medicine, 38*(11), 2074-2102. https://doi.org/10.1002/sim.8086

Persson, B. N. J. (2006). Rubber friction: Role of the flash temperature. *Journal of Physics: Condensed Matter, 18*(32), 7789-7823. https://doi.org/10.1088/0953-8984/18/32/025

Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmueller, U., & Timmer, J. (2009). Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics, 25*(15), 1923-1929. https://doi.org/10.1093/bioinformatics/btp358

Roziere, J.-F., Perard, B., & Roch, B. (2011). *Method of estimating the transverse grip of a pair of tyres by comparative analysis* (European Patent Application No. EP 2300797 A1). European Patent Office. https://patents.google.com/patent/EP2300797A1/en

Sagmeister, S., Kounatidis, P., Goblirsch, S., & Lienkamp, M. (2024). Analyzing the impact of simulation fidelity on the evaluation of autonomous driving motion control. In *2024 IEEE Intelligent Vehicles Symposium (IV)* (pp. 230-237). IEEE. https://doi.org/10.1109/IV55156.2024.10588858

Savant, S., De Carvalho Pinheiro, H., Sacchi, M. E., Conti, C., & Carello, M. (2023). Tires and vehicle lateral dynamic performance: A corrective algorithm for the influence of temperature. *Machines, 11*(6), Article 654. https://doi.org/10.3390/machines11060654

Singh, K. B., & Sivaramakrishnan, S. (2023). *Extended Pacejka tire model for enhanced vehicle stability control* (arXiv:2305.18422) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2305.18422

Sung, C.-L., & Tuo, R. (2024). A review on computer model calibration. *WIREs Computational Statistics, 16*(1), Article e1645. https://doi.org/10.1002/wics.1645

West, W. J., & Limebeer, D. J. N. (2022). Optimal tyre management for a high-performance race car. *Vehicle System Dynamics, 60*(1), 1-19. https://doi.org/10.1080/00423114.2020.1802047

Zarate Villazon, A. M., Brown, I., Balchanos, M., & Mavris, D. (2026). *Lap time performance sensitivity to tire model parameters: An optimal control framework for model selection and calibration* (SAE Technical Paper 2026-01-0214). SAE International. https://doi.org/10.4271/2026-01-0214

*Version notes.* Seven entries were read in another version. For the entry by Lu and colleagues, the direction, the form and the test details come from the 2020 Research Square preprint (https://doi.org/10.21203/rs.3.rs-37902/v1). For the entry by Cappello and Hoegh, the statement on priors comes from arXiv preprint 2512.00640. For the patent entry by Roziere and colleagues, the description comes from the US family member US8408058B2. For the entries by Sagmeister and colleagues and by Kobayashi and colleagues, the verifiers read the arXiv accepted manuscripts 2602.07984 and 2407.12989. For the entry by West and Limebeer, they read a University of Pretoria repository manuscript, whose page numbers can differ from the version of record. For the entry by Persson, the quote in Appendix C comes from arXiv preprint cond-mat/0605273.

## Works Named but Not Used as Evidence

The report names these works so that readers can find and check them. It cites none of them as evidence. The pipeline's verifiers checked the five held-out records. For the other works, reviewers confirmed existence through Crossref, DataCite or a saved copy of the page, but nobody verified their content for this report.

*Held out after verification by a task decision (IDs as in the text)*

- A4. Farroni, F. (2016). T.R.I.C.K.-Tire/Road Interaction Characterization & Knowledge: A tool for the evaluation of tire and vehicle performances in outdoor test sessions. *Mechanical Systems and Signal Processing, 72-73*, 808-831. https://doi.org/10.1016/j.ymssp.2015.11.019
- A12. Goblirsch, S., Ruhland, B., Betz, J., & Lienkamp, M. (2025). Bayesian optimization-based tire parameter and uncertainty estimation for real-world data. In *2025 IEEE Intelligent Vehicles Symposium (IV)* (pp. 1114-1120). IEEE. https://doi.org/10.1109/IV64158.2025.11097410
- A13. Kegelman, J. C., Harbott, L. K., & Gerdes, J. C. (2017). Insights into vehicle trajectories at the handling limits: Analysing open data from race car drivers. *Vehicle System Dynamics, 55*(2), 191-207. https://doi.org/10.1080/00423114.2016.1249893
- B5. Grosch, K. A. (1963). The relation between the friction and visco-elastic properties of rubber. *Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences, 274*(1356), 21-39. https://doi.org/10.1098/rspa.1963.0112
- C6. Canopy Simulations. (2016, August 12). *Getting started 3: Matching telemetry* [Technical article]. Michelin Simulation Services. https://simulation.michelin.com/canopy/technical-articles/getting-started-3-matching-telemetry

*Leads, content not verified by the pipeline*

- L1. Sakhnevych, A. (2022). Multiphysical MF-based tyre modelling and parametrisation for vehicle setup and control strategies optimisation. *Vehicle System Dynamics, 60*(10), 3462-3483. https://doi.org/10.1080/00423114.2021.1977833
- L2. Romagnuolo, F., Aratri, R., De Pinto, S., Farroni, F., de Bellis, S. A., Bottiglione, F., Mantriota, G., & Sakhnevych, A. (2025). *From temperature to wear: A quantitative analysis of tire influence on vehicle cornering using handling key performance indicators* (SAE Technical Paper 2025-01-5060). SAE International. https://doi.org/10.4271/2025-01-5060
- L3. Werner, F., Schwehn, A.-K., Lienkamp, M., & Betz, J. (2025). *GripMap: An efficient, spatially resolved constraint framework for offline and online trajectory planning in autonomous racing* (arXiv:2504.12115) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2504.12115
- L4. Kenward, M. G., & Roger, J. H. (1997). Small sample inference for fixed effects from restricted maximum likelihood. *Biometrics, 53*(3), 983. https://doi.org/10.2307/2533558
- L5. Siemens Digital Industries Software. (2023). *MF-Tyre/MF-Swift help* [Third-party manual hosted by Altair]. https://2024.help.altair.com/2024/hwdesktop/hwx/topics/motionview/MFTyre-MFSwift_Help.pdf

*Note.* For L4, Crossref and OpenAlex give only the first page. For C6, the page header names Michelin Simulation Services and the page body names Canopy Simulations.

*Safety documents (operating guidance, outside the evidence set)*

- S1. British Racing & Sports Car Club. (2023). *Event briefing notes: Silverstone 24hr, 19-21 May 2023* [Briefing notes]. https://brscc.co.uk/wp-content/uploads/2023/05/Briefing-Notes-C1-24HR-19-21-May-2023-ISSUED-16-MAY-2023.pdf
- S2. Continental. (n.d.-a). *Tire mounting safety instruction*. Retrieved October 19, 2026, from https://www.continental-tires.com/products/b2c/car/technical-services/tire-mounting-safety-instruction/
- S3. Continental. (n.d.-b). *Retorquing wheels*. Retrieved October 19, 2026, from https://www.continental-tires.com/products/b2c/tire-knowledge/retorquing-wheels/
- S4. Federation Internationale de l'Automobile. (n.d.). *FIA safety bulletin 44* [Copy hosted by AKK Motorsport]. Retrieved October 19, 2026, from https://www.autourheilu.fi/site/assets/files/49408/fia_safety_bulletin_44.pdf

*Note.* S1 is one club's event briefing (pit-lane working area and speed limit). S2 and S3 are a tyre maker's instructions on inflation, eye protection and re-torquing. S4 sets high-voltage procedures and protective equipment for each role, teams included. They are examples only. The owner must get the documents that apply to the team's events and car.

## Appendix A: Revision Notes

Revision 2 answers four round-2 reviews of revision 1. They include the editorial review (Ed, 07_editor_r2.md) and devil's advocate checkpoint 3 in its second round (DA, 07_da3_r2.md). They also include the claim-reference audit (CR, 07_claimref_r2.md) and the ethics review (Eth, 07_ethics_r2.md). No review raised a critical item. Table A1 gives an action for every major, conditional and minor item. Table A2 keeps the revision 1 log. Structural changes: the optional-extras table became a paragraph of the Track Protocol. So the identifiability table is now Table 4 and the code table Table 5. Derived results moved to Appendix D, and a list of works named but not used as evidence follows the References.

**Table A1**

*Revision 2 Log*

|#|Source and severity|Feedback|Action|Status|
|:---|:---|:---|:---|:---|
|1|Ed R2-M1, CR MIN-4, major|B4 direction contradicted by its own figure, direction count wrong|Table 2 gives both B4 results, "one new tyre, no break-in", rated Very low. Directions recounted: two windows, one fall, one rise, B4 self-conflicting, its window sentence attributed to a figure from its references. Limitations bullet. Protocol step 4 scrubs new sets and cites Figure 18(b). Manifest C-005 updated|Resolved|
|2|Ed R2-M2, DA M1, major|Held-out check cannot detect g_p, transfer untested, margin direction unstated, gate lists differ|Validation scores within-session contrasts for g_p and session means for the level and b1. Contrast check: superiority over M0, non-inferiority to M0b, margins from dt_dec, power from B2. g_p serves calibration circuits only until a transfer check passes. One list of four gates in the Executive Summary, the Estimation Procedure and Table 5. Limitation added|Resolved. Margins and power wait for the owner and B2|
|3|DA M2, Ed R2-m1, R2-m6, CR MIN-2, major|Large-sample intervals at small K, effective sample size, undefined prior and posterior|REML fit, t intervals with about K - 1 degrees of freedom or Kenward-Roger (lead L4), and DERIVED coverage (Appendix D). K_min before a success stop, coverage and a sequential simulation in B2, and a corrected effective-sample-size sentence. The prior-posterior condition is dropped (the editor's option), because the dt_dec condition sets the precision target|Resolved. K_min waits for B2|
|4|Eth R2-E4, CR MIN-11, conditional|Seven works without attribution, capped records described beyond their verification|List of works named but not used as evidence. Capped records described by topic only. Held-out IDs lead to the list|Resolved. Re-admission stays an owner decision|
|5|Eth R2-E5, conditional|No safety override, pit-lane and high-voltage gaps, no eye protection, safety sources unreachable|Safety rules rewritten: override in rule 4, pit-lane rules, trained high-voltage person, wheel motors (P7), safety glasses, S1 to S4 listed. Steps 10 and 15 and Executive Summary step 4 reworded. Pit entry defined as the car stopped in the pit box|Resolved|
|6|Ed R2-m2, DA m6|Time and tyre-age terms collinear|One trend term first. Age only if the set plan separates it and the collinearity index allows|Resolved|
|7|Ed R2-m3, DA m5|Dropping exception or early-stopped stints selects on outcomes|Step 15 keeps valid laps under the assigned level, logs the reason and fits with and without, with counts per level|Resolved|
|8|Ed R2-m4|Cold-equivalent reset needs the gas temperature|Ratio form in step 9, derivation and test case in Appendix D, pilot check|Resolved|
|9|Ed R2-m5|Three gate lists|One list (row 2)|Resolved|
|10|Ed R2-m7|Costs of the level placement not weighed|Lap ratio valid only if the response at s_eff matches (B2 check). Pressure effect computed at s_eff. Setups ranked at s = 1 and s = s_eff. Opposite risk named in the Discussion|Resolved|
|11|Ed R2-m8, CR MIN-1, DA m4|Slope formula ignores the pressure basis|dpi_0 for absolute and gauge bases (Appendix D). Equation 4 and step 5 use gauge readings. The code must state its basis|Resolved|
|12|Ed R2-m9, DA m1|"At most one grip direction" overreaches|Per circuit and set of conditions, with one level per circuit|Resolved|
|13|Ed R2-m10, DA O2|P2 needs Fz0 fixed, TyreOptim route, scope of P2 and P4|P2 and P4 rewritten. MLTP_TyreOptim named in the Identifiability section and Table 5|Resolved|
|14|Ed R2-m11|"Flag" has three meanings|Track flags only. Slow laps are marked for a sensitivity fit. The tyre reading is a second outcome|Resolved|
|15|DA m2|Raue remedy misapplied|No measurement reaches factors absent from the QSS. Column (c) of Table 4 assumes a slip model|Resolved|
|16|DA m3|Three identifiability labels too strong|LMUX circuit-dependent, g_p "in principle" with GPS or IMU, LFZO near-collinear with a DERIVED load dependence|Resolved|
|17|DA m7|Checkpoint count out of date|Abstract and AI Disclosure updated|Resolved|
|18|CR MIN-3|Table 1 total|"about 1,435"|Resolved|
|19|CR MIN-5|Held-out A13 in an Evidence cell|Moved to a held-out note in the optional-extras paragraph|Resolved|
|20|CR MIN-6|Unlabelled inferences and weak anchors|B4 procedure labelled inference. A9 cited only for strong priors. A2 "measured as in". Brun links labelled inference from the general result|Resolved|
|21|CR MIN-7|Verbs stronger than the sources|A3 "identification set", A10 "states", B12 "vendor simulation"|Resolved|
|22|CR MIN-8, Ed line-level|C3 criterion framing|"By a conventional criterion, necessary although not sufficient", with "or outcome"|Resolved|
|23|CR MIN-9|Caveats missing|B9 brand and model-derived form, B14 super-soft compound, A7 vibrations, A15 experimental layout|Resolved|
|24|CR MIN-10, Eth A1, A6|Appendix C wording, locators and gaps|"Appear in the sources or their records". A5 quote anchors. A1 partners. B11 Rimac and no-conflict statement. Siemens through lead L5. Eight intellectual links named|Resolved|
|25|CR MIN-12|Precision items|2.14 bar, "wheelspeeds" anchor, S_K definition (Appendix D), B11 appendix anchor, P6 item labels|Resolved|
|26|Ed line-level|Abstract counts, approval status, non-reference level, P3 tense|Applied|Resolved|
|27|Eth A2, A3, A4, A5, A7|Manifest time, "open-source", driver weight, disclosure wording, quotation marks|Date-only timestamp with a note. "Open-source" removed. Combined mass logged. "In this run, AI agents did every step". The suspension phrase is quoted|Resolved|
|28|Ed and DA suggestions, DA frame-lock|Scrubbing, blind levels, Method sentence, warmers, within-stint slope|Scrubbing (step 4), the quote-anchor sentence (Method) and warmers (SQ3, owner action 2) adopted. Blind levels not adopted, because Safety rule 8 tells drivers what is recorded. Within-stint slope not adopted, for length|Partly adopted|
|29|DA sources, O1, O3|Three sources found by the devil's advocate|Two listed as leads L3 and L4, not cited. The third, an F1 timing preprint, not used, because the report's own logic carries its points|Acknowledged|
|30|Length|Body above 7,000 words after the fixes|Optional-extras table folded into Table 4 and one paragraph. Derivations moved to Appendix D. The 6.4 s tread time constant, the front-rear tread difference and the A1 premise sentence removed. The B14 window in Safety rule 3 removed (Table 2 keeps the window)|Resolved|

**Table A2**

*Revision 1 Log (Answers to the Round-1 Reviews of 06_report_draft.md)*

|#|Source and severity|Feedback|Action|Status|
|:---|:---|:---|:---|:---|
|1|Ed M1, DA3 M2, major|The temperature gate selects on a common effect|Gate removed as a data filter. The reading stays a safety check and a second outcome|Resolved|
|2|Ed M2, major|No stint effect|Stint medians as the unit, fixed session effects for a1, stint error, R-L-R pilot, scatter stated per stint|Resolved|
|3|Ed M3, Eth E5, CR MIN-10, major|Designed pressure unprotected, gauge values in the gas law|Absolute-pressure note with a DERIVED example, the B11 assumptions, T_set, cold-equivalent reset, set rotation and set term, no-bleed rule, warmer inputs and log fields|Resolved|
|4|Ed M4, Eth E4, major|B10 pressure null missing, exclusion statements and reason|B10 null in Table 2 and SQ2. Held-out statements by ID. Reason in the Method|Partly resolved, completed in revision 2|
|5|Ed M5, major|Cap undisclosed|Cap in the Method, Table 1 and the Abstract. Two race-tyre leads named in the Limitations|Partly resolved: leads unverified|
|6|Ed M6, DA3 M4, major|Path from estimate to code|Interval rule, one-driver case, s_eff as a lap-time ratio, transfer tags, QSS-only use of g_p, setup-optimisation limit, M0b first|Resolved|
|7|Ed M7, DA3 m1, major and minor|Minimality claimed without traceability|Title says "Candidate". Table 3 states what each field serves. Minimality marked untested|Resolved, test needs B2|
|8|DA3 M1, Ed m6, major|Track-temperature term labelled a total effect|Association across sessions, with its identifying assumption, K reported, logged start time and event day|Resolved|
|9|DA3 M3, major|Pressure slope treated as fixed|tau, optional a3 term, tau in the update rule, thermal-span tag, DERIVED slope formula|Resolved|
|10|CR MAJ-1, Ed m9, major|False B2 against B4 contradiction|Channels separated, effect called uncertain, assumption behind one shared value stated|Resolved|
|11|CR MAJ-2, Eth E2, DA3 m5, major|Industry-link count of six|Fifteen sources in two groups (Appendix C), summarised in the Limitations|Resolved|
|12|Eth E1, DA3 m10, conditional|AI disclosure incomplete|Tool, version, date, known models, AI verifiers, error modes, same-pipeline reviews|Partly resolved: earlier-phase models not on record|
|13|Eth E3, CR MIN-3, conditional|A1 misstated|Reworded. A1 removed from the claim on validated methods|Resolved|
|14|Eth E5, conditional|Safety notes missing|Eight safety and data-handling rules|Partly resolved, completed in revision 2|
|15|Ed m1, DA3 m4|P2 incomplete|P2 and the Identifiability bullet updated|Resolved|
|16|Ed m2|P6 cause and the 40% step|Six fixes named, values per circuit, 40% as an order of magnitude|Resolved|
|17|Ed m3, CR MIN-13|6.4 s used outside its regime|Reworded, and the pilot measures the cool-down|Resolved|
|18|Ed m4, m5, DA3 m8|Stopping and update rules|Success stop, budget stop, null action, full refit governs|Resolved|
|19|Ed m7, DA3 m6|Outlier rule pools levels|Within stint, raw MAD, fits with and without|Resolved|
|20|Ed m8, m10, m11, m15 to m18, DA3 m11|Table rows, two-wheeler patent, A9 wording, scope, C4 role, log fields, small corrections, counts|Applied|Resolved|
|21|Ed m12|Undefined terms|H1, TIR, S_n, V1 to V5, factor definitions and code location|Resolved|
|22|Ed m13, CR MIN-21, Eth A3|B4 page warning|Page anchors replaced by quote anchors|Resolved|
|23|Ed m14|No validation metric|Mean absolute error of the stint median, margin from dt_dec|Resolved, target changed in revision 2|
|24|CR MIN-1, MIN-2, MIN-4 to MIN-9, MIN-11 to MIN-20|Locators, weak anchors, wording, caveats, labels, version notes|Applied|Resolved|
|25|DA3 m2, m3, m7, m9|QSS qualifiers, argument order, relative offsets, split runs|Applied|Resolved|
|26|Eth A1, A2, A4 to A8, A10|Wording, anchors, driver data, interests, checks, private texts|Applied|Resolved|
|27|DA3 O1, sources added by Ed and Eth|New sources outside the verified set|Not cited|Acknowledged|
|28|Ed suggestions|Pit-wall card, fixed session effect, worked example, reference text|Fixed session effect adopted. The Executive Summary steps serve as the card|Partly resolved|
|29|DA3 frame-lock check|Hot-pressure design, sector times|Owner actions 2 and 4|Resolved|

**Unresolved issues.**

1. Strand B2 has not run, so all verdicts are candidates. Without B2, K_min, the margins, the power of the contrast check, the coverage and the session count do not exist.
2. Owner inputs are missing: dt_dec, N_avail, the tyre maker's limits, the warmer rules and set temperature, and the pit-lane rules. Other missing inputs are the sensors, sector timing, the number of drivers, a mid-session stop and a circle area. A TIR file, the purpose of the MF slots and the approval status of task C6-2 are missing too. The project status note lists the task as waiting for sign-off.
3. The fitted temperature coefficients of B11 wait for a verifier.
4. Re-admission of the five held-out sources is open, and the leads L1 to L5 are unverified.
5. 04_verify_3.md has no correction note for the B4 conflict between Table 3 and Figure 18(b) (orchestrator decision).
6. The models of the phases before revision 1 are not on record.
7. No Material Passport exists in the workspace. So the claim intent manifest is an HTML comment at the top of this file, with a date-only timestamp.
8. The standard AI disclosure text was extended. Its sentences on verification of all findings and on human oversight throughout the process would be false for this run. So they are replaced.
9. The safety documents S1 to S4 are examples. The owner must get the documents that apply to the team's events and car.
10. The new or changed anchors have had no claim-reference audit. They are A1, A3, A5, A15, B4, B6, B11, C3, C13 and C15, many in Appendices C and D.
11. After the STE edit, the body has 7,116 words, 116 above the 7,000 target. Any addition needs an equal cut.

**Word count.** From the title to the end of the funding statement, without the manifest comment: 7,116 whitespace-separated tokens. Before the STE edit split the long sentences, the count was 6,997. This count includes headings, table cells, list numbers and equation symbols. References and the list of works not used as evidence: about 1,680 tokens. Appendices: about 3,110 tokens. The manifest comment holds about 940 tokens.

## Appendix B: Project Facts From the Code

**Table B1**

*Project Facts From Read-Only Code Checks of 19 and 20 October 2026*

|ID|Fact|
|:---|:---|
|P1|The QSS uses peak friction only and ignores the slip shape and the vertical shifts (functions/ggv.py, docstring).|
|P2|With Fz0 fixed, Fz0_shift divides the load that feeds Dx, Dy, Kxk and Kya in the full model. This load also feeds the vertical shifts, Ky_gamma0 and the combined-slip terms (vehModel.py, lines 349 to 486). In the QSS, D = mu*Fz/Fz0_shift on both axes (functions/ggv.py, lines 51 and 234 to 236). So 1/Fz0_shift and a combined scale on LMUX and LMUY are one direction. MLTP_TyreOptim.py sets Fz0 = 4905*Fz0_shift when it changes the shift (lines 6 to 15), which also moves the nominal-load channel.|
|P3|LFZO, once added, will rescale Fz0 = 4905 N (vehParams.py, line 48) in the load offset dfz. In the full model, Fz0 also enters the cornering stiffness (vehModel.py, line 449).|
|P4|For the four-motor car, the QSS lateral and braking limits take the smaller axle capacity (functions/ggv.py, lines 351 to 354 and 384 to 388). With ATD on and EM4 off, the QSS braking limit sums the two axles.|
|P5|In the QSS, 5% more grip shortens the lap by 1.3% to 1.4%, and a 20% drag error moves it by about 0.1% (docs/laptime_sim_improvements.md).|
|P6|In the default 'matlab' mode, static tyre loads sum to about 26.0 kN against m g = 20.5 kN. The 'corrected' mode applies six fixes (fix items A1 to A6 of docs/laptime_sim_improvements.md), this one included. Its full-model laps are 11.1% (Sturn) and 11.7% (BCN) slower (same file, status of 18 October 2026).|
|P7|In the four-motor configuration, the full model gives each wheel its own motor torque (vehModel.py, lines 133 to 136).|

## Appendix C: Industry and Intellectual Links of the Cited Sources

Fifteen of the 36 cited sources have industry authors, funders or partners, and all links appear in the sources or their records. Nine involve tyre makers or tyre-model developers:

- Pirelli and Nokian, with Porsche and CRF among the other partners of the project tasks (Arosio et al., 2005)<!--ref:A1--><!--anchor:quote:The%20partners%20involved%20in%20these%20tasks%20are%20Pirelli%2C%20Nokian%2C%20Porsche%2C%20CRF-->.
- Ferrari GT and Bridgestone Europe (Farroni, 2014)<!--ref:A5--><!--anchor:quote:Ferrari%20GT%20and%20Bridgestone%20Europe-->, with data normalised under confidentiality agreements (Farroni, 2014)<!--ref:A5--><!--anchor:quote:For%20industrial%20confidentiality%20agreements%2C%20plots%2C%20diagrams%20and%20data%20will%20be%20provided%20as%20normalized-->.
- Michelin (Roziere et al., 2011)<!--ref:A10--><!--anchor:section:applicants-->.
- TNO, whose MF-Tyre model the paper extends (Besselink et al., 2010)<!--ref:B1--><!--anchor:quote:extensions%20to%20the%20widely%20used%20TNO%20MF-Tyre%205.2%20Magic%20Formula%20tyre%20model-->.
- Goodyear, testing its own tyre (Singh & Sivaramakrishnan, 2023)<!--ref:B4--><!--anchor:section:affiliations-->.
- Apollo Vredestein (Harsh & Shyrokau, 2019)<!--ref:B11--><!--anchor:section:1-->, with a first author who lists Rimac Automobili (Harsh & Shyrokau, 2019)<!--ref:B11--><!--anchor:section:affiliations-->. The paper declares no conflict of interest (Harsh & Shyrokau, 2019)<!--ref:B11--><!--anchor:section:ConflictsOfInterest-->.
- Siemens, which supplies MF-Tyre (lead L5), in Lugaro et al. (2020)<!--ref:B12--><!--anchor:section:affiliations-->.
- Toyota Research Institute and Bridgestone (Kobayashi et al., 2024)<!--ref:B13--><!--anchor:quote:The%20authors%20also%20thank%20Toyota%20Research%20Institute%20and%20Bridgestone%20Corporation-->.
- Milliken Research Associates, developer of the extended tyre model (Kasprzak et al., 2006)<!--ref:C1--><!--anchor:section:affiliations-->.

Six link to automakers or engineering firms. They include Jaguar Land Rover (Bogdanski & Best, 2018)<!--ref:A3--><!--anchor:section:funding-->, Michelotto Engineering (Cortivo et al., 2024)<!--ref:A6--><!--anchor:section:affiliations--> and Volvo Cars (Albinsson et al., 2019)<!--ref:A7--><!--anchor:section:affiliations-->. The others are Stellantis (Savant et al., 2023)<!--ref:A11--><!--anchor:section:affiliations-->, Toyota (Mizuno et al., 2005)<!--ref:B7--><!--anchor:section:affiliations--> and China FAW Group (Lu et al., 2022)<!--ref:B9--><!--anchor:section:affiliations-->.

The verifier records also note minor intellectual links for eight cited sources. These links mostly concern authors who present, extend or review their own methods, theory or software. The sources include Sagmeister et al. (2024)<!--ref:A8--><!--anchor:section:affiliations-->, who share an author and a group with held-out record A12. They also include Persson (2006)<!--ref:B6--><!--anchor:quote:Recently%20I%20have%20developed%20a%20theory%20which%20accurately%20describes%20this%20energy%20dissipation-->, Greenland et al. (1999)<!--ref:C3--><!--anchor:quote:The%20theory%20of%20causal%20graphs%20is%20equivalent%20to%20the%20G-computation%20theory%20of%20Robins--> and Angrist et al. (1996)<!--ref:C4--><!--anchor:section:abstract-->. Another is Brynjarsdottir and O'Hagan (2014)<!--ref:C8--><!--anchor:section:affiliations-->, whose second author co-wrote the calibration method that they qualify. The remaining sources are Sung and Tuo (2024)<!--ref:C10--><!--anchor:section:affiliations-->, Raue et al. (2009)<!--ref:C13--><!--anchor:quote:An%20implementation%20is%20freely%20available%20for%20MATLAB%20and%20the%20PottersWheel%20modeling%20toolbox--> and Morris et al. (2019)<!--ref:C15--><!--anchor:quote:Note%20that%20the%20Stata%20package%20simsum-->.

## Appendix D: Derived Results

These results are arithmetic on stated values or on the report's own equations (DERIVED). No source states them.

- **Coverage at few sessions.** Consider K equal-weight per-session slopes with normal errors and unknown variance. The nominal 95% profile-likelihood interval for the mean slope then accepts values with t^2 at most (K - 1) * (exp(3.84 / K) - 1). Here t has K - 1 degrees of freedom. Its coverage is about 85% at K = 3, 88% at K = 4, 91% at K = 6 and 93% at K = 10. At K = 4, the exact t interval is 45% wider (3.18 against 2.20 standard errors). For one parameter, the large-sample threshold is a log-likelihood drop of 3.84 / 2 = 1.92.
- **Collinearity index.** gamma_K = 1 / sqrt(lambda_min(S_K' S_K)), with S_K the sensitivity columns of the parameter subset K, each divided by its norm (Brun et al., 2001)<!--ref:A15--><!--anchor:section:Eq12-Eq13-->.
- **Lap-time uncertainty.** sd(t) is about |dt/ds| * s * sd(ln g_p). From P5, the QSS lap changes by 0.26% to 0.28% per 1% of grip. So stint-median scatters of 0.1%, 0.2% and 0.4% equal about 0.4%, 0.7% and 1.5% of grip.
- **LFZO in the QSS.** Let D be proportional to pD1 + pD2*dfz, with dfz = (Fz - LFZO*Fz0) / (LFZO*Fz0). Then d ln D / d LFZO = -pD2 * (Fz / Fz0) / (pD1 + pD2*dfz) at LFZO = 1. The effect grows with load, so it differs between slow and fast corners but stays near-collinear with peak grip on one circuit.
- **Pressure slope.** Let R_k be the thermal ratio of Equation 3 in session k, and p_nom the MF nominal pressure. Let c_k = R_k * p_cold,ref / p_nom, and let dpi_0 be the running MF pressure offset at dp = 0. With absolute pressures, dpi_0 = c_k - 1. With gauge pressures, dpi_0 = c_k - 1 + (R_k - 1) * p_atm / p_nom. The slope of the Equation 1 multiplier in dp at dp = 0 is c_k * (ppx3 + 2*ppx4*dpi_0). It moves with the thermal state through c_k even when ppx4 = 0, and more when the pressure term is curved.
- **Warm reset (step 9).** Use Equation 3 with absolute pressures and an unchanged gas temperature during the change. This gives p_w2 = (p_w1 + p_atm) * (p_c2 + p_atm) / (p_c1 + p_atm) - p_atm, all gauge. Test case: p_c1 = 1.6 bar, p_c2 = 1.4 bar, T_set = 20 deg C and T_gas = 60 deg C. These values give p_w1 = 1.957 bar. Both Equation 3 and the ratio form give a target of 1.730 bar. Bleeding cools the gas slightly, hence the pilot check.
- **Gas-law example (SQ3).** (1.60 + 1.013) * 353 / 293 - 1.013 = 2.135 bar gauge, against 1.60 * 353 / 293 = 1.928 bar from gauge values. The rises are 0.535 and 0.328 bar, so the gauge form understates the rise by 39%.
- **Other values.** Table 2: 2.1 - cosh(0.4) = 1.019, which is 7.4% below 1.1, and 0.03967 / 0.951 = 4.2% per 10 deg C. The window fit gives 0.7, the start of the measured warm-up, at 88 - 50 * acosh(1.4) = 44.6 deg C. The full-sliding data end at 80 deg C. An 11% lap offset divided by 0.26 to 0.28 gives grip of order 40%.
