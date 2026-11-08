# Data audit

Before any modelling, the dataset is audited for completeness, plausibility, provenance and variability; the modelling population is stated explicitly, and expectations for the models are recorded before they are built.
This report states the findings and decisions; [`notebooks/audit.ipynb`](../notebooks/audit.ipynb) shows the methods and figures, and the committed artifacts — [`qc_summary.json`](qc_summary.json), [`curve_diagnostics.csv`](curve_diagnostics.csv), produced by [`scripts/write_qc_report.py`](../scripts/write_qc_report.py) — hold the exact numbers.
To reproduce: install the package (`pip install -e ".[dev]"`), fetch the data as described in [`data/README.md`](../data/README.md), and run the script.

**Dataset.** 517 blade-coated MAPbI₃₋ₓClₓ thin films, published by
P. G. Abrego Martínez and A. N. Corpus Mendoza on [Zenodo](https://doi.org/10.5281/zenodo.19237616) (2026)
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with a companion paper ([DOI 10.1002/pssr.70177](https://doi.org/10.1002/pssr.70177)).
Checksums, instruments and the auxiliary dataset are documented in
[`data/README.md`](../data/README.md).

## Curves

517 curves on a uniform 300–1000 nm grid at 2 nm (351 points), as optical density in the archive's sense, `OD(λ) = A(λ)/t` with thickness `t` in nm. `OD` has units of nm⁻¹, not dimensionless absorbance. Unmeasured points are `NaN`, never imputed.

- **Missingness:** 2,400 of 181,467 points (1.3%), all 50-point tails of the 48 films that stop at exactly 900 nm; no late starts, no interior gaps, no empty curves. All 517 curves are complete over 300–900 nm; 469 over the full grid.
- **Values:** No negatives (minimum 1.14×10⁻⁴ nm⁻¹), no constant and no all-zero curves. No cleaning or clipping policy is needed.
- **Thickness:** Complete and positive for all films (107.5–1970.5 nm), so the normalization is usable everywhere.

## Grid provenance

The paper reports acquisition at a 10 nm scanning step over 300–900 nm; the archive ships 300–1000 nm at 2 nm, with the 48 truncated curves ending exactly at 900 nm. A second-difference probe over the grid phases finds no piecewise-linear interpolation signature (median |second difference| flat at 0.000031–0.000033 nm⁻¹; method in the notebook). A smoothed resampling would leave no trace, so provenance stays formally open. The 2 nm spacing is treated as a storage grid instead of a verified optical resolution.

## Parameter completeness

All 517 films carry all five integrated-OD windows, a positive measured thickness, a "curve evaluated" flag that agrees with the curves everywhere, and parseable fabrication dates spanning 2023-09-06 to 2024-11-12. The population is chemically coherent — one precursor composition, one solvent system — with measured variation in the controlled factors: additive recipe (NaLES/DDABr 202 films, none 169, NaLES 146), substrate stack (glass 313, corning 191, TiO₂-based stacks 8 and 5), and blade-coating parameters.
86 films (≈17%) fall below the source paper's `ODint < 2` a.u. low-quality marker. They are kept, with the label recorded as a subgroup flag.

## Variability

Over the fully measured 300–900 nm span: one principal component carries 50% of the variance, two carry 90%, four carry 99%. The population is strikingly low-dimensional. The median distance to the nearest other curve is 4.9×10⁻³ nm⁻¹ (5th percentile 1.7×10⁻³), roughly half a typical curve's peak amplitude: genuinely diverse, no near-duplicates (closest pair `BPD_sample_294`/`296` at 4.3×10⁻⁴ nm⁻¹). Euclidean distance is amplitude-dominated, so this is a diversity estimate, not a shape comparison.

## The published 245-film subset

The source paper builds it in two steps: drop films with missing target integrated OD (removes nothing here), then exclude additive categories "with too few observations" (threshold unstated). That rule removes whole arms at a time, so from arms of {202, 169, 146} films only the following subset sizes are reachable:

| Threshold τ | Arms kept | Films kept |
|---|---|---|
| τ ≤ 146 | all three | 517 |
| 147–169 | NaLES/DDABr + none | 371 |
| 170–202 | NaLES/DDABr | 202 |
| τ ≥ 203 | none | 0 |

The published 245 is in none of them. The subset cannot be rebuilt from the shipped database. Provenance limitation (derivation in the notebook).

## Population statement

Modelling uses **all 517 films on the 300–900 nm span**, where every curve is complete. The `ODint < 2` label is retained as a recorded subgroup flag for stratified analysis, never a silent exclusion. Any future restriction will be stated with its numbers.

## Expectations recorded before modelling

- A compact linear model will be a strong baseline: with 99% of the variance in four components, a PCA generator is expected to match a small neural model on most metrics. Matching it is a valid outcome. The neural model must show measurable advantage — in distribution tails, covariance structure or conditional behaviour — to justify its complexity.
- Memorization is checked against the measured diversity scale (median 4.9×10⁻³ nm⁻¹ between genuinely different films): a generator that reproduces training curves would show much smaller generated-to-training distances.
- Evaluation stays at the film level. Each curve is one independent film, and splits respect that.

## Conclusion

A coherent, well-documented population of 517 independent films with complete curves over 300–900 nm, no plausibility issues, and genuine, low-dimensional but substantial variability. The dataset comfortably supports the intended small-data generative study, with no film excluded.

## Limitations

- 2 nm grid provenance is unresolved; edge analyses must state any smoothing.
- The paper's 245-film subset cannot be rebuilt, so per-subset comparisons at film level are not possible.
- 48 curves end at 900 nm. 300–900 nm is the working span.
- Single composition and solvent system: conclusions are scoped to MAPbI₃₋ₓClₓ blade-coated films.
- Raw spectra are third-party measurements (CC BY 4.0). The repository ships derived artifacts only, attributed as above.
