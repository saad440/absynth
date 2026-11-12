# Data

Experimental UV–vis spectra from independent published measurements.
Every spectrum here is a third-party measurement, used under the
licences below.

Raw files are not committed to this repository (`.gitignore` excludes
`data/raw/`). Everything needed to reproduce the local copies
— sources, checksums, fetch commands — lives in this file.

## Dataset: UNAM blade-coated perovskite films

- **Citation:** P. G. Abrego Martínez and A. N. Corpus Mendoza, *UV
  Vis Absorption Dataset for Multivariable Analysis of Blade-Coated
  Perovskite Films*, Zenodo (2026).
  DOI: [10.5281/zenodo.19237616](https://doi.org/10.5281/zenodo.19237616)
- **Related paper:** Abrego-Martínez et al., *Optical Density of
  Perovskite Films as a Novel Parameter in Statistical Modeling for
  Perovskite Solar Cells*, Phys. Status Solidi RRL (2026).
  DOI: [10.1002/pssr.70177](https://doi.org/10.1002/pssr.70177)
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  stated in the Zenodo record metadata and in the dataset's `README`.
- **Archive:** `Dataset.zip`, 702,653 bytes,
  MD5 `6599f206e94c527a8a4a6b534e99967a`.

**Contents.**

- `blade_perovskite_database.csv` — 517 films (one row each) with
  substrate, synthesis, deposition and metric variables.
- `blade_perovskite_metadata.csv` — descriptions of the 45 variables.
- `curvedata_opticaldensity.csv` — measured optical-density curves,
  one column per film (`BPD_sample_001`…`BPD_sample_517`) on a uniform
  wavelength grid of 300–1000 nm at 2 nm (351 rows, column
  `Wavelength_nm`).
- `README.md` — the authors' description and citation request.

**Measurement context (reported by the paper).** Films blade-coated on
2.5 × 2.5 cm substrates and annealed at 100 °C for 5 minutes; spectra
on a Shimadzu UV-1800; thickness by stylus profilometry (Ambios
Technology XP-200). The paper reports acquisition over 300–900 nm with
a 10 nm scanning step — coarser and narrower than the shipped curves
(300–1000 nm at 2 nm, of which 48 stop at 900 nm). The archive does
not explain the difference, so the shipped 2 nm spacing is treated as
a storage grid, not a stated optical resolution.

**Fetch and extract (verified).**

```bash
python -c "from absynth.data import DATASETS, download; \
s = DATASETS['unam_blade_coated']; \
download(s['url'], 'data/raw/Dataset.zip', s['checksum'])"
unzip data/raw/Dataset.zip -d data/raw
```

The download streams to a `.part` file, verifies the MD5 while it
arrives, and renames only on success. An existing archive is
re-verified rather than re-downloaded. The archive contains a
top-level `Dataset/` folder, so the extraction lands the four files in
`data/raw/Dataset/` — the path every module and notebook reads.

## Attribution

CC BY 4.0 requires attribution. Every public figure or table derived
from the dataset cites the sources above, and the repository README
links here alongside any results.

The source *papers* are a different matter: their text and figures are
published under CC BY-NC, so this repository links to them rather than
reproducing them. Everything shipped here is generated from the
CC BY 4.0 datasets themselves.
