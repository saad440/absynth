# Absynth

**Generative modelling of experimental perovskite thin film absorption spectra with physically interpretable representations.**

Absynth (name derived from **abs**orption + **synth**esis) asks whether a compact generative model can reproduce the *distribution* of measured UV–vis spectra of perovskite thin films.

The data used is absorption spectra from 517 blade-coated MAPbI₃₋ₓClₓ films, from a published Zenodo dataset (CC BY 4.0). Provenance, checksums and fetch instructions: [data/README.md](data/README.md).

## Notebooks (currently finished)

- [Baselines](notebooks/baselines.ipynb): The frozen evaluation — how far copying and a PCA Gaussian get, and where the Gaussian assumption falls short.
- [Features](notebooks/features.ipynb) and [Audit](notebooks/audit.ipynb): The groundwork — what actually varies between films, and what the dataset contains.

## Run locally

With Python 3.11 or newer:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python scripts/run_pilot.py   # refit the baselines, rewrite the record and figures; needs the fetched data (see above)
```

On Linux x86_64 or Windows machines without an NVIDIA GPU, install the CPU-only PyTorch build first. The default wheel otherwise pulls several gigabytes of CUDA libraries this project never uses. macOS and Linux ARM wheels are already CPU-only:

```bash
python -m pip install -r requirements-torch-cpu.txt
python -m pip install -e ".[dev]"
```


**Attribution:** the spectra are third-party measurements used under CC BY 4.0. Sources are cited in [data/README.md](data/README.md).
