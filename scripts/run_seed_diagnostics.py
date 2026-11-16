"""Diagnose the robustness sweep's diverged seed.

Usage (from the repository root, after ``pip install -e ".[dev]"``):

    python scripts/run_seed_diagnostics.py
        [--config configs/seed_diagnostics.yaml] [--report-dir reports]
        [--frozen-record reports/thickness_experiments.json]

The thickness comparison's robustness sweep retrains the conditioned
models across seeds and records a seed whose training diverges as the
instability it measures. One conditioned variational seed diverged
part-way through training; the seed statistics in the frozen record
are computed over the seeds whose training converged. This script
asks the survivor question: are the converged seeds representative,
or did they earn their place by never failing?

Training seeds model init and batch order from the configuration seed
alone, so retraining with ``epochs`` truncated to before the recorded
divergence reproduces the original trajectory exactly up to that
epoch: the truncated rerun of the diverged seed ends in its genuine
pre-divergence state. Three comparisons land in the record:

- **Same horizon.** Every robustness seed retrained at the truncated
  horizon, evaluated with the sweep's own metrics — coupling,
  memorization ratio, q-space covariance, reconstruction in both
  spaces, latent KL, and the tail of the per-epoch KL history. The
  diverged seed's pre-divergence values read against the converged
  seeds' values at the identical horizon, so horizon and seed are not
  confounded.
- **Full horizon, converged seeds.** The converged seeds retrained to
  the full horizon and compared field-by-field with the frozen
  record's robustness entries. Exact agreement is both a determinism
  check on training and a check that this script's metric composition
  mirrors the runner's.
- **Full horizon, diverged seed.** Retrained to the full horizon to
  confirm the divergence reproduces with the same signature. If it
  did not, the trajectory would not be reproducible and the truncated
  comparison would lose its meaning.

The frozen record is read, never written: the diagnostics live in
their own record, ``reports/seed_diagnostics.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.functional import mse_loss

from absynth.data import DATASETS
from absynth.evaluation import (
    coupling_record,
    covariance_record,
    memorization_record,
    nearest_training_distances,
)
from absynth.loader import load_dataset
from absynth.preprocessing import require_complete, restrict_to_span
from absynth.provenance import git_provenance, package_versions
from absynth.splits import date_block_holdout
from absynth.train import (
    TrainedThicknessConditionedVariationalAutoEncoder,
    TrainingDivergenceError,
    VariationalAutoEncoderConfig,
    train_thickness_conditioned_variational_autoencoder,
)

# Metrics the same-horizon comparison reads, one entry per seed.
_COMPARISON_METRICS = (
    "coupling_pearson_r",
    "memorization_median_ratio",
    "covariance_relative_frobenius_q_space",
    "heldout_reconstruction_standardized",
    "heldout_reconstruction_nm_squared",
    "heldout_reconstruction_a_space_nm_squared",
)


def _integrated_q(curves: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """Integrated attenuation per curve over the working span, nm⁻¹·nm."""
    return np.trapezoid(curves, wavelengths_nm, axis=1)


def _per_film_errors(rebuilt: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Mean squared error per film over wavelengths, in the input's unit squared."""
    return np.mean((rebuilt - reference) ** 2, axis=1)


def _reconstruction_entry(
    trained: TrainedThicknessConditionedVariationalAutoEncoder,
    held_out: np.ndarray,
    held_out_thickness_nm: np.ndarray,
) -> tuple[float, float, float]:
    """Deterministic held-out reconstruction: posterior means, no sampling.

    Mirrors the thickness runner's conditioned reconstruction, the
    absorbance conversion included: converting with ``A = q·t`` is
    applied to each side before differencing, exactly as the runner
    writes it, so the numbers can agree bit-for-bit.
    """
    standardized_curves = torch.from_numpy(
        trained.standardizer.transform(held_out).astype(np.float32)
    )
    standardized_thickness = torch.from_numpy(
        trained.thickness_standardizer.transform(held_out_thickness_nm).astype(np.float32)
    )
    with torch.no_grad():
        mean, _std = trained.model.encode(standardized_curves)
        rebuilt = trained.model.decode(mean, standardized_thickness)
    standardized_mse = float(mse_loss(rebuilt, standardized_curves))
    rebuilt_nm = trained.standardizer.inverse_transform(rebuilt.numpy())
    per_film_nm2 = _per_film_errors(rebuilt_nm, held_out)
    thickness_nm = held_out_thickness_nm[:, np.newaxis]
    per_film_a2 = _per_film_errors(rebuilt_nm * thickness_nm, held_out * thickness_nm)
    return standardized_mse, float(per_film_nm2.mean()), float(per_film_a2.mean())


def _entry_metrics(
    trained: TrainedThicknessConditionedVariationalAutoEncoder,
    seed: int,
    sample_seed: int,
    n_samples: int,
    train: np.ndarray,
    held_out: np.ndarray,
    held_out_thickness_nm: np.ndarray,
    wavelengths_nm: np.ndarray,
    heldout_distances: np.ndarray,
    epochs: int,
) -> dict[str, object]:
    """The robustness sweep's per-seed entry, plus the KL-history tail.

    Field names and composition mirror the thickness runner's
    robustness entries, so the full-horizon entries can be compared
    with the frozen record field by field. ``heldout_distances`` is
    the held-out films' nearest-training-curve distances, computed
    once; the memorization ratio reads a generated population against
    them.
    """
    drawn_curves, drawn_thickness = trained.sample(n_samples, sample_seed)
    standardized_mse, nm2, a_space_nm2 = _reconstruction_entry(
        trained, held_out, held_out_thickness_nm
    )
    return {
        "seed": seed,
        "epochs": epochs,
        "sampling_seed": sample_seed,
        "heldout_reconstruction_standardized": standardized_mse,
        "heldout_reconstruction_nm_squared": nm2,
        "heldout_reconstruction_a_space_nm_squared": a_space_nm2,
        "latent_kl": [float(value) for value in trained.latent_kl],
        "coupling_pearson_r": coupling_record(
            _integrated_q(drawn_curves, wavelengths_nm), drawn_thickness
        )["pearson_r"],
        "memorization_median_ratio": memorization_record(
            nearest_training_distances(drawn_curves, train), heldout_distances
        )["median_ratio"],
        "covariance_relative_frobenius_q_space": covariance_record(held_out, drawn_curves)[
            "covariance_relative_frobenius"
        ],
        "epoch_kl_tail10": [float(value) for value in trained.epoch_kl[-10:]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the conditioned variational model's robustness seeds at a "
            "truncated horizon and at the full horizon, and write the diagnostics record."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/seed_diagnostics.yaml"),
        help="diagnostics configuration",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports"), help="directory for the JSON record"
    )
    parser.add_argument(
        "--frozen-record", type=Path, default=Path("reports/thickness_experiments.json"),
        help="the thickness record whose robustness entries the full-horizon runs reproduce",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frozen_record = json.loads(args.frozen_record.read_text(encoding="utf-8"))
    frozen_entries = {
        int(entry["seed"]): entry for entry in frozen_record["robustness"]["variational"]
    }

    dataset = load_dataset(Path(config["data"]["archive_dir"]))
    span_curves = restrict_to_span(dataset.curves)
    require_complete(span_curves)
    curves = span_curves.optical_density
    wavelengths_nm = span_curves.wavelengths_nm
    thickness_nm = dataset.thickness_nm.to_numpy(dtype=float)

    split = date_block_holdout(
        dataset.parameters,
        test_fraction=float(config["split"]["test_fraction"]),
        seed=int(config["split"]["seed"]),
        rare_stack_max_films=int(config["split"]["rare_stack_max_films"]),
    )
    train = curves[split.train_mask]
    train_thickness = thickness_nm[split.train_mask]
    held_out = curves[split.test_mask]
    held_out_thickness = thickness_nm[split.test_mask]

    seeds = [int(seed) for seed in config["diagnostics"]["seeds"]]
    truncated_epochs = int(config["diagnostics"]["truncated_epochs"])
    full_horizon_epochs = int(config["diagnostics"]["full_horizon_epochs"])
    n_samples = int(config["sampling"]["n_samples"])
    base_seed = int(config["sampling"]["seed"])
    headline_seed = int(config["variational"]["seed"])
    # Sample seeds follow the thickness runner's robustness rule with
    # its one exception: the runner reuses the headline run's own draw
    # for the headline seed, whose robustness entry carries the
    # headline sampling seed, base_seed + 5, the conditioned
    # variational model's position in the record's generator list.
    # Every other seed draws with base seed + 100 + seed. Mirroring
    # both cases is what makes the full-horizon entries reproduce the
    # frozen record's sampling exactly.
    def sample_seed_for(seed: int) -> int:
        if seed == headline_seed:
            return base_seed + 5
        return base_seed + 100 + seed


    heldout_distances = nearest_training_distances(held_out, train)

    truncated_entries = []
    for seed in seeds:
        print(
            f"diagnostics: training the conditioned variational model at "
            f"seed = {seed}, epochs = {truncated_epochs} (truncated) ...",
            flush=True,
        )
        trained = train_thickness_conditioned_variational_autoencoder(
            train,
            train_thickness,
            VariationalAutoEncoderConfig(
                **{**config["variational"], "seed": seed, "epochs": truncated_epochs}
            ),
        )
        truncated_entries.append(
            _entry_metrics(
                trained,
                seed,
                sample_seed_for(seed),
                n_samples,
                train,
                held_out,
                held_out_thickness,
                wavelengths_nm,
                heldout_distances,
                epochs=truncated_epochs,
            )
        )

    full_horizon_entries = []
    for seed in seeds:
        print(
            f"diagnostics: training the conditioned variational model at "
            f"seed = {seed}, epochs = {full_horizon_epochs} (full horizon) ...",
            flush=True,
        )
        try:
            trained = train_thickness_conditioned_variational_autoencoder(
                train,
                train_thickness,
                VariationalAutoEncoderConfig(
                    **{**config["variational"], "seed": seed, "epochs": full_horizon_epochs}
                ),
            )
        except TrainingDivergenceError as error:
            full_horizon_entries.append({"seed": seed, "diverged": True, "detail": str(error)})
            continue
        entry = _entry_metrics(
            trained,
            seed,
            sample_seed_for(seed),
            n_samples,
            train,
            held_out,
            held_out_thickness,
            wavelengths_nm,
            heldout_distances,
            epochs=full_horizon_epochs,
        )
        # Field-by-field comparison with the frozen record's entry for
        # this seed, when one exists. Exact agreement certifies both
        # training determinism and the metric mirror above.
        frozen_entry = frozen_entries.get(seed)
        if frozen_entry is None:
            entry["frozen_record_match"] = None
        else:
            match = {
                key: bool(value == frozen_entry[key])
                for key, value in entry.items()
                if key in frozen_entry
            }
            entry["frozen_record_match"] = match
            entry["frozen_record_match_all"] = all(match.values())
        full_horizon_entries.append(entry)

    truncated_comparison = {
        key: {str(entry["seed"]): entry[key] for entry in truncated_entries}
        for key in _COMPARISON_METRICS
    }

    record = {
        "provenance": {
            "config_path": str(args.config),
            "frozen_record": str(args.frozen_record),
            "git": git_provenance(args.config, Path(__file__)),
            "package_versions": package_versions(),
            "dataset": {
                "doi": DATASETS["unam_blade_coated"]["doi"],
                "checksum": DATASETS["unam_blade_coated"]["checksum"],
                "licence": DATASETS["unam_blade_coated"]["licence"],
            },
            "split": split.provenance,
        },
        "question": (
            "Are the converged robustness seeds representative? One conditioned "
            "variational seed diverged during the thickness comparison's robustness "
            "sweep, and the seed statistics in the frozen record are computed over "
            "the seeds whose training converged. Because training is seeded from the "
            "configuration alone, a run truncated to before the recorded divergence "
            "reproduces the diverged trajectory up to that epoch; its final state is "
            "the diverged seed's genuine pre-divergence state, compared here with "
            "the converged seeds at the identical horizon."
        ),
        "protocol": {
            "seeds": seeds,
            "truncated_epochs": truncated_epochs,
            "full_horizon_epochs": full_horizon_epochs,
            "n_samples": n_samples,
            "sampling_seed_rule": "base seed + 100 + seed",
            "sampling_base_seed": base_seed,
        },
        "truncated": truncated_entries,
        "full_horizon": full_horizon_entries,
        "truncated_comparison": truncated_comparison,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.report_dir / "seed_diagnostics.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"record: {record_path}")


if __name__ == "__main__":
    main()
