Mathematical supplement to Online Resource 1

Article: Enhancing Reinforcement Learning with Large Language Models:
A Survey of Grounded Reward, Policy, and Environment Interfaces
Journal: Artificial Intelligence Review
Author: Yanqi Li, University of Southern Queensland
Correspondence: Yanqi.Li@unisq.edu.au

Read mathematical_checks.pdf. It is self-contained: notation, source formulas,
constructed examples, enumeration, and boundaries are included. Its section and
equation references are internal; it does not require the manuscript's labels.

Contents
- mathematical_checks.tex / .pdf: standalone analytical note.
- references.bib: the five source references used by this note.
- reproduce_checks.py: original numerical examples, exact arithmetic and full
  group enumeration, including the finite-group ProCredit comparison.
- results.json: output from reproduce_checks.py, including all 45 count states
  for each ProCredit G=8 standard-deviation convention.
- reproduce_calibration_family.py: calibrated-family identities, negative
  region, fixed-evaluator slice, GSPO collinearity, and coordinate boundaries.
- reproduce_calibration_family.results.json: output from the family program.
Both programs use Python 3.9+ and its standard library; no installation required.

Reproduce
    python reproduce_checks.py
    python reproduce_calibration_family.py
On Windows, py can replace python. To choose an output location:
    python reproduce_checks.py --output comparison_results.json
    python reproduce_calibration_family.py --output family_results.json
The programs run independently and print PASS after successful checks.

reproduce_checks.py performs exact rational calculations where possible and
exhaustive count enumeration with floating-point square roots for standardized
groups. It has no random sampling. The SWE-RM kappa table uses population
standardization on mixed-score groups and zero advantages on constant-score
groups, equivalently the positive-stabilizer limit described in the note.

reproduce_calibration_family.py uses fixed seed 20261021. It checks 1,200 exact
rational base policies, 6,000 numerical sign cases, 80 finite differences with
scores frozen, and 128 independent multinomial GSPO enumerations. GSPO cases use
G=2,3,8,16, population/sample standard deviations, and positive stabilizers
1e-6 and 0.05. It also verifies the fixed-evaluator a<1/7 threshold, the slice
root, a contained negative rectangle, and coordinate examples. The analytical
proof covers the stated domains; these finite checks corroborate that proof
and do not estimate the prevalence of a behavior in trained systems.
Neither program trains a policy, downloads data, or requires a GPU.

Compile
    tectonic mathematical_checks.tex --keep-logs
Alternatively use a standard LaTeX + BibTeX build with plainnat/natbib.

Scope
These are survey-owned mathematical constructions based on published formulas,
not reproduced original-paper training results. ARM-FM concerns discounted task
preferences. SWE-RM concerns current-policy calibration, the initial GSPO
surrogate, and explicitly identified optimizer conditions. ProCredit concerns
the conversion of its one-turn score into its actual combined advantage and
the control obtained by removing the standard-deviation divisor.

For the calibrated family, each base policy determines an evaluator whose
scores are then frozen before differentiating in the independent logit space.
The fixed q=2/3 slice uses one evaluator over a one-dimensional calibration
curve. This does not assert calibration of one fixed evaluator throughout an
open neighborhood of the full policy simplex. The GSPO calculation assumes
iid one-token groups, G>=2, a positive stabilizer, initial inactive clipping,
and no additional loss gradient. Equal positive lengths add a common factor.

The positive/negative directions are local expected directions in the stated
parameterization. The idealized full-gradient Adam and reward-tilt comparisons
are analytical boundaries, not claims about source implementations. The source
formula's unspecified numerical choices are explicit sensitivities.

Source versions checked
ARM-FM: ICLR 2026 proceedings PDF, Algorithm 1 (p19), Appendix A.5 (p21),
Appendix A.9 optional intermediate events (p32).
SWE-RM: ICLR 2026 proceedings PDF, Appendix C.3 (p16), E.1 (pp20-21).
ProCredit: arXiv:2609.27532v3, Sections 3.1-3.3 (pp5-6), Proposition 3
(p23), Table I1 (p24). Progress coefficient is called b in the note to avoid
confusion with the binary correctness variable c.

Related normalization analyses: Bereket and Leskovec, arXiv:2508.11800v1,
Sections 4 and A.1-A.3; Wang, arXiv:2607.21273v2, Sections 3.2 and 5.3.
These precedents are credited in the note before the finite-group comparison.
