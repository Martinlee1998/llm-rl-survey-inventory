# LLM-RL Survey Online Inventory

This repository provides the linked study inventory for the survey manuscript:

**Enhancing Reinforcement Learning with Large Language Models: A Survey of Grounded Reward, Policy, and Environment Interfaces**

The repository contains metadata only. It does not redistribute copyrighted paper PDFs.

## Files

- `inventory.csv`: tabular linked inventory for review and spreadsheet use.
- `inventory.json`: the same inventory in structured JSON form.
- `references.bib`: selected BibTeX entries corresponding to the inventory rows.

## Current Coverage

Last updated: 2026-06-22.

- Total inventory rows: 38
- In-scope studies: 27
- Adjacent exemplars: 11
- Primary interface counts among in-scope rows: Reward 13, Policy 9, Environment 5
- DOI coverage: 37 of 38 rows

`Progress Reward Model for Reinforcement Learning via Large Language Models` is linked to its official OpenReview page, but its DOI is left blank because no verified DOI was found during this update.

## Columns

- `study`: short study name used in the manuscript table.
- `citation_key`: BibTeX key from the manuscript source.
- `title`, `authors`, `year`, `venue`: bibliographic metadata.
- `primary_interface`: Reward, Policy, or Environment.
- `secondary_tags`: compact methodological tags from the appendix inventory.
- `domain_benchmark_family`: domain or benchmark family used in the study.
- `rl_update`: whether the paper reports a reinforcement-learning update rather than only planning, prompting, or evaluation.
- `status`: `in scope` or `adjacent exemplar`.
- `inventory_group`: manuscript appendix group.
- `doi`, `doi_url`: verified DOI and resolver link when available.
- `url`: stable public landing page, publisher page, OpenReview page, PMLR page, NeurIPS page, or arXiv page.
- `link_type`: broad source type for the public link.
- `metadata_source`, `metadata_notes`: provenance notes for the linked metadata.

## DOI Policy

DOIs are included only when they were already present in the manuscript BibTeX or could be verified through exact-title public metadata. For arXiv records, the DOI uses the standard `10.48550/arXiv.<id>` form only when the arXiv identifier was verified. If no DOI could be verified, the DOI field is intentionally blank rather than guessed.

Some papers have both a conference/publisher page and an arXiv version. When no publisher DOI was found, the inventory uses the verified arXiv DOI while preserving the most stable public landing page in `url`.
