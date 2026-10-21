# LLM–RL Survey: Sources, Evidence, and Analytical Checks

Companion materials for the revised manuscript:

**Enhancing Reinforcement Learning with Large Language Models: A Survey of Grounded Reward, Policy, and Environment Interfaces**

This repository connects the survey's comparisons to the original studies. It contains the source register and evidence matrix supplied in **Online Resource 1**, together with the mathematical supplement and its reproduction code.

Current manuscript version: **21 October 2026**.

## Start here

| Material | Contents |
| --- | --- |
| [Source register](public_sources.csv) · [JSON](public_sources.json) | 52 cited works: 37 primary method or empirical studies, 14 related surveys, and one foundational theory paper. Records identify the publication, the version read, public links, and its role in the manuscript. |
| [Evidence matrix](evidence_matrix.csv) · [JSON](evidence_matrix.json) | 82 records connecting specific claims to source pages, figures, tables, equations, and comparison conditions. A source can support several records. |
| [Data guide](DATA_GUIDE.txt) | Field descriptions, page-number conventions, evidence identifiers, and suggested routes through the comparisons. |
| [References](references.bib) | BibTeX entries for the 52 works in the source register. |
| [Mathematical supplement](math/mathematical_checks.pdf) | The reward-to-update analyses, with source formulas, derivations, and constructed examples. [TeX source](math/mathematical_checks.tex) and [reproduction instructions](math/README.txt) are included. |

The evidence matrix supports the survey's analysis of guidance quality and learning utility, policy adaptation and transfer, and the use of generated tasks and curricula. Each record distinguishes a study's reported comparison from the interpretation developed in the survey. These materials document the sources used in this manuscript; they are not an exhaustive bibliography of the field.

## Reproduce the analytical checks

Use Python 3.9 or later. The programs require only the standard library.

```sh
python math/reproduce_checks.py --output comparison_results.json
python math/reproduce_calibration_family.py --output family_results.json
```

On Windows, `py` can replace `python`. Both programs print `PASS` when their checks succeed. The reference outputs are [results.json](math/results.json) and [reproduce_calibration_family.results.json](math/reproduce_calibration_family.results.json).

These programs check the survey's mathematical constructions. They do not run the original papers' policy-training experiments. The assumptions and source versions are described in the [mathematical supplement](math/README.txt).

## Relationship to the June inventory

The June 2026 version contained 38 study-level entries: 27 in-scope studies and 11 adjacent exemplars. The October register follows the revised manuscript's full bibliography, including related surveys and theory. The two totals therefore describe different sets of sources.

The former `inventory.csv` and `inventory.json` have been replaced by `public_sources.*` and `evidence_matrix.*`. The June files remain available in Git history under the tag `june-2026-inventory`. Use the current register for the revised manuscript; the historical inventory has not been retrospectively corrected. See the [change log](CHANGELOG.md) for the migration details.

## Source access and corrections

Public links identify the original papers and the versions consulted. Third-party paper PDFs are not redistributed here; the included PDF is the survey's own mathematical supplement.

To report a correction, open an issue with the `citation_key` or `evidence_id`, the relevant source page, and the proposed change.
