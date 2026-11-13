"""Run the pilot baselines and write the frozen evaluation record.

Usage (from the repository root, after ``pip install -e ".[dev]"``):

    python scripts/run_pilot.py [--config configs/pilot.yaml]
        [--report-dir reports] [--figure-dir figures]

The script is orchestration only. All logic lives in ``absynth``.
It fits the two statistical baselines on the training films selected
by :func:`absynth.splits.date_block_holdout`, evaluates them against
held-out films with :mod:`absynth.evaluation`, and writes one JSON
record plus two figures. The record carries the configuration path,
git state (revision, plus whether tracked files are modified), seeds,
package versions and the dataset checksum, so the pilot can be
reproduced and its provenance audited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from absynth.baselines import bootstrap_resample, fit_pca_generator
from absynth.data import DATASETS
from absynth.evaluation import (
    component_scores,
    covariance_record,
    mean_curve_deviation,
    memorization_record,
    nearest_training_distances,
    quantile_band,
)
from absynth.explore import nearest_neighbour_distances
from absynth.loader import load_dataset
from absynth.preprocessing import require_complete, restrict_to_span
from absynth.provenance import git_provenance, package_versions
from absynth.splits import date_block_holdout


def _mean_curve_figure(
    wavelengths_nm: np.ndarray,
    train: np.ndarray,
    synthetic: np.ndarray,
    path: Path,
) -> None:
    """Median curves and 10–90% bands: measured vs PCA synthetic."""
    band_real = quantile_band(train, quantiles=(0.1, 0.5, 0.9))
    band_synthetic = quantile_band(synthetic, quantiles=(0.1, 0.5, 0.9))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(
        wavelengths_nm,
        band_real[0],
        band_real[2],
        color="0.55",
        alpha=0.35,
        label="measured (training days), 10–90% band",
    )
    ax.plot(wavelengths_nm, band_real[1], color="0.15", label="measured median")
    ax.fill_between(
        wavelengths_nm,
        band_synthetic[0],
        band_synthetic[2],
        color="tab:orange",
        alpha=0.30,
        label="PCA synthetic, 10–90% band",
    )
    ax.plot(
        wavelengths_nm, band_synthetic[1], "--", color="tab:orange", label="PCA synthetic median"
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Optical density $q$ (nm$^{-1}$)")
    ax.set_title("PCA synthetic population against measured films")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _nn_distance_figure(
    floor_distances: np.ndarray,
    synthetic_distances: np.ndarray,
    heldout_distances: np.ndarray,
    population_median: float,
    path: Path,
) -> None:
    """Cumulative distance profiles: the memorization comparison.

    The floor sits at exactly zero, since it only copies. Held-out
    films give the novelty scale of a genuinely new film. A generator
    is read against both.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    profiles = (
        (floor_distances, "bootstrap floor (copies)", "0.4", "-"),
        (synthetic_distances, "PCA synthetic", "tab:orange", "--"),
        (heldout_distances, "held-out measured films", "tab:blue", "-"),
    )
    for values, label, color, style in profiles:
        ordered = np.sort(values)
        ax.plot(
            ordered,
            np.arange(1, ordered.size + 1) / ordered.size,
            label=label,
            color=color,
            linestyle=style,
        )
    ax.axvline(population_median, color="0.3", linestyle=":", label="measured population median")
    ax.set_xlabel(r"Distance to nearest training film (nm$^{-1}$)")
    ax.set_ylabel("Cumulative fraction of curves")
    ax.set_ylim(0, 1)
    ax.set_title("Novelty: distance from generated curves to the training set")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the pilot baselines and write the evaluation record and figures."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pilot.yaml"), help="pilot configuration"
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports"), help="directory for the JSON record"
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("figures"), help="directory for the PNG figures"
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = load_dataset(Path(config["data"]["archive_dir"]))
    span_curves = restrict_to_span(dataset.curves)
    require_complete(span_curves)
    curves = span_curves.optical_density

    split = date_block_holdout(
        dataset.parameters,
        test_fraction=float(config["split"]["test_fraction"]),
        seed=int(config["split"]["seed"]),
        rare_stack_max_films=int(config["split"]["rare_stack_max_films"]),
    )
    train = curves[split.train_mask]
    held_out = curves[split.test_mask]

    n_samples = int(config["sampling"]["n_samples"])
    floor = bootstrap_resample(
        train, n_samples, np.random.default_rng(int(config["sampling"]["floor_seed"]))
    )
    generator = fit_pca_generator(train, n_components=int(config["pca"]["n_components"]))
    synthetic = generator.sample(n_samples, np.random.default_rng(int(config["sampling"]["seed"])))

    floor_distances = nearest_training_distances(floor, train)
    synthetic_distances = nearest_training_distances(synthetic, train)
    heldout_distances = nearest_training_distances(held_out, train)
    population_median = float(np.median(nearest_neighbour_distances(curves)[0]))

    record = {
        "provenance": {
            "config_path": str(args.config),
            "git": git_provenance(args.config, Path(__file__)),
            "package_versions": package_versions(),
            "dataset": {
                "doi": DATASETS["unam_blade_coated"]["doi"],
                "checksum": DATASETS["unam_blade_coated"]["checksum"],
                "licence": DATASETS["unam_blade_coated"]["licence"],
            },
            "split": split.provenance,
        },
        "baselines": {
            "bootstrap_floor": {
                "memorization": memorization_record(floor_distances, heldout_distances),
                "mean_curve_deviation": mean_curve_deviation(train, floor),
            },
            "pca": {
                "n_components": generator.n_components,
                "scales_nm_inverse": [float(scale) for scale in generator.scales],
                "real_score_std_nm_inverse": [
                    float(standard_deviation)
                    for standard_deviation in component_scores(
                        train, generator.mean, generator.components
                    ).std(axis=0, ddof=1)
                ],
                "memorization": memorization_record(synthetic_distances, heldout_distances),
                "mean_curve_deviation": mean_curve_deviation(train, synthetic),
                "covariance": covariance_record(train, synthetic),
            },
        },
        "reference_metrics": {
            "n_films": int(curves.shape[0]),
            "n_training_films": int(train.shape[0]),
            "n_heldout_films": int(held_out.shape[0]),
            "population_nearest_neighbour_median_nm_inverse": population_median,
        },
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.report_dir / "pilot_baselines.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _mean_curve_figure(
        span_curves.wavelengths_nm, train, synthetic, args.figure_dir / "pilot_mean_curves.png"
    )
    _nn_distance_figure(
        floor_distances,
        synthetic_distances,
        heldout_distances,
        population_median,
        args.figure_dir / "pilot_nn_distances.png",
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"record: {record_path}")


if __name__ == "__main__":
    main()
