# Absynth

**Generative modelling of experimental perovskite thin film absorption spectra with physically interpretable representations.**

Absynth (absorption + synthesis) asks whether compact generative models can reproduce the *distribution* of measured UV–vis spectra of perovskite thin films, i.e. new curves that vary the way real films vary.

The data are absorption spectra of 517 blade-coated MAPbI₃₋ₓClₓ films from a published Zenodo dataset (CC BY 4.0). Provenance, checksums and fetch instructions: [data/README.md](data/README.md).

## Notebooks

The project's findings are presented as a series of notebooks, each building on the last:

1. [Audit](notebooks/audit.ipynb) — What the dataset contains: 517 complete spectra, one coherent population, strikingly low-dimensional variation.
2. [Features](notebooks/features.ipynb) — What actually varies between films, and what the thickness normalization `q = A/t` does and does not do.
3. [Baselines](notebooks/baselines.ipynb) — Before a learned model: how far copying and a PCA Gaussian get, and where the Gaussian assumption already falls short.
4. [Conditioning](notebooks/conditioning.ipynb) — The thickness-conditioning comparison: better films, or just better numbers?

## Findings

- **A linear baseline holds.** A 4-component PCA generator reconstructs held-out films about as well as any learned model in the study. On a population this low-dimensional, this is an expected outcome.
- **Fidelity alone cannot discriminate generators.** A PCA generator scores well on fidelity, but its new curves overshoot the measured spread. The measure that separates generators is calibrated novelty: new curves whose distances to the training set match those of genuinely new films.
- **Conditioning improves coupling, not reconstruction.** A thickness input does not make the generated curves fit better. The conditioned VAE's films do couple curve to thickness the way measured films do, while keeping calibrated novelty.
- **The measured coupling is mostly arithmetic.** The anticorrelation between normalized absorbance and thickness sits at the level that the definition of `q` produces by division alone. The physical signal is the sublinear absorbance scaling, with candidate causes that band-resolved analysis could separate.
- **Two model failures.** The conditioned autoencoder's films initially ignored their stated thickness. Sampling code and thickness jointly restored the coupling, though not the realism. The variational model diverged in a minority of seeds; a rerun of one failed seed behaved like the converged seeds before failing.

## Limitations

- One composition (MAPbI₃₋ₓClₓ), one coating process, one laboratory. The findings describe this archive, not perovskite films in general.
- The spectra are stored from 300–1000 nm at 2 nm spacing, while the source paper reports measurements from 300–900 nm at a 10 nm step. This discrepancy is unexplained (see the [audit notebook](notebooks/audit.ipynb)).
- The apparent sublinear absorbance scaling (β̂ ≈ 0.33) cannot be decomposed into instrumental opacity, interface offsets, or thickness-correlated morphology from these curves alone.
- All models are compact by design. Nothing here tests large generative architectures.
- The variational model's training diverged at two of five seeds. A learning-rate search might stabilize it, but none was run.

## Reproducing

Every number is computed from frozen records with pinned seeds, and the notebooks re-derive the prose numbers from those records. Per-slice errors are reported in both normalized and absorbance units, because either space alone can hide an error. Full protocol: [REPRODUCING.md](REPRODUCING.md). Quick version, with Python ≥ 3.11:

```bash
python -m pip install -r requirements-torch-cpu.txt   # CPU-only torch, where needed
python -m pip install -e ".[dev]"
# Fetch and verify the dataset first - commands in data/README.md
python scripts/run_pilot.py             # baselines record and figures
python scripts/run_vae.py               # trained-model record and figures
python scripts/run_thickness.py         # thickness-conditioning record and figures
python scripts/run_seed_diagnostics.py  # seed-sweep diagnostics record
```


## Attribution and licences

- **Data:** P. G. Abrego Martínez and A. N. Corpus Mendoza, *UV–Vis Absorption Dataset for Multivariable Analysis of Blade-Coated Perovskite Films*, Zenodo, CC BY 4.0 — [DOI](https://doi.org/10.5281/zenodo.19237616). Companion paper: Abrego-Martínez et al., *Optical Density of Perovskite Films as a Novel Parameter in Statistical Modeling for Perovskite Solar Cells*, Phys. Status Solidi RRL — [DOI](https://doi.org/10.1002/pssr.70177).
