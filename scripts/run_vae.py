"""Run the trained-model experiments and write the evaluation record.

Usage (from the repository root, after ``pip install -e ".[dev]"``):

    python scripts/run_vae.py [--config configs/vae.yaml]
        [--report-dir reports] [--figure-dir figures]

The script trains the compact autoencoder and the variational
autoencoder at several KL weights on the training films selected by
:func:`absynth.splits.date_block_holdout`, the same frozen split as the
pilot. Five generators are then evaluated with
:mod:`absynth.evaluation`: the bootstrap floor, the PCA generator, the
autoencoder's KL-free code-sampling generator, and each variational
model drawing from its prior. The record carries the configuration
path, git state, seeds, package versions and the dataset checksum, so
the experiments can be reproduced and audited.

Distributional metrics compare generated populations against the
held-out films. Unlike the pilot record, which measured against the
training population, these generators have no held-out information in
them, so comparing against held-out films is what tests generalization.
Memorization keeps the pilot's semantics throughout: distances from
generated curves to the nearest training curve, read against the same
profile for held-out films. Each trained model also reports its
reconstruction of the held-out films, in standardized units for the
trained models and in nm⁻² for all models, because the reconstruction
question comes before the sampling question.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.nn.functional import mse_loss

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
from absynth.train import (
    AutoEncoderConfig,
    VariationalAutoEncoderConfig,
    train_autoencoder,
    train_variational_autoencoder,
)


def _distribution_metrics(
    synthetic: np.ndarray, train: np.ndarray, held_out: np.ndarray, sampling_seed: int
) -> dict[str, object]:
    """Population-level comparison of one synthetic set against held-out films."""
    return {
        "sampling_seed": sampling_seed,
        "memorization": memorization_record(
            nearest_training_distances(synthetic, train),
            nearest_training_distances(held_out, train),
        ),
        "mean_curve_deviation_vs_heldout": mean_curve_deviation(held_out, synthetic),
        "covariance_vs_heldout": covariance_record(held_out, synthetic),
    }


def _reconstruction_mse(rebuilt: np.ndarray, reference: np.ndarray) -> float:
    """Mean squared reconstruction error over films and wavelengths, nm⁻²."""
    return float(np.mean((rebuilt - reference) ** 2))


def _pca_reconstruction(curves: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Rebuild curves from their PCA coordinates: the linear reconstruction."""
    scores = component_scores(curves, mean, components)
    return scores @ components + mean


def _mean_curves_figure(
    wavelengths_nm: np.ndarray,
    held_out: np.ndarray,
    generators: list[tuple[str, np.ndarray, str]],
    path: Path,
) -> None:
    """Held-out measured bands against each generator's median curve."""
    band_real = quantile_band(held_out, quantiles=(0.1, 0.5, 0.9))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(
        wavelengths_nm,
        band_real[0],
        band_real[2],
        color="0.55",
        alpha=0.35,
        label="held-out measured, 10–90% band",
    )
    ax.plot(wavelengths_nm, band_real[1], color="0.15", label="held-out measured median")
    for label, synthetic, color in generators:
        band = quantile_band(synthetic, quantiles=(0.1, 0.5, 0.9))
        ax.plot(wavelengths_nm, band[1], "--", color=color, label=label)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Optical density $q$ (nm$^{-1}$)")
    ax.set_title("Generated populations against held-out measured films")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _nn_distance_figure(
    profiles: list[tuple[str, np.ndarray, str, str]],
    population_median: float,
    path: Path,
) -> None:
    """Cumulative distance profiles: the memorization comparison.

    The floor sits at exactly zero, since it only copies. Held-out
    films give the novelty scale of a genuinely new film. Each
    generator is read against both.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, values, color, style in profiles:
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
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _training_figure(
    ae_losses: list[float],
    vae_reconstruction: list[float],
    vae_kl: list[float],
    vae_kl_weight: float,
    path: Path,
) -> None:
    """Training histories: the autoencoder loss and the anchor variational run."""
    epochs = range(1, len(ae_losses) + 1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(epochs, ae_losses, color="tab:blue")
    left.set_xlabel("Epoch")
    left.set_ylabel("Reconstruction MSE (standardized units²)")
    left.set_title("Autoencoder training loss")
    right.plot(epochs, vae_reconstruction, color="tab:orange")
    right.set_xlabel("Epoch")
    right.set_ylabel("Reconstruction MSE (standardized units²)", color="tab:orange")
    kl_axis = right.twinx()
    kl_axis.plot(epochs, vae_kl, color="tab:green")
    kl_axis.set_ylabel("KL per curve", color="tab:green")
    right.set_title(f"Variational training at kl_weight = {vae_kl_weight}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the autoencoder and the variational models, evaluate all "
            "generators, and write the record and figures."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/vae.yaml"), help="experiment configuration"
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
    base_seed = int(config["sampling"]["seed"])
    floor_seed = int(config["sampling"]["floor_seed"])

    print("training the autoencoder ...", flush=True)
    autoencoder = train_autoencoder(train, AutoEncoderConfig(**config["autoencoder"]))

    variational: list[tuple[int, float, object]] = []
    variational_values = {
        key: value for key, value in config["variational"].items() if key != "kl_weights"
    }
    for position, kl_weight in enumerate(config["variational"]["kl_weights"]):
        print(
            f"training the variational model at kl_weight = {float(kl_weight):.5g} ...", flush=True
        )
        vae_config = VariationalAutoEncoderConfig(
            **{**variational_values, "kl_weight": float(kl_weight)}
        )
        trained = train_variational_autoencoder(train, vae_config)
        variational.append((position, float(kl_weight), trained))

    generator = fit_pca_generator(train, n_components=int(config["pca"]["n_components"]))
    floor = bootstrap_resample(train, n_samples, np.random.default_rng(floor_seed))

    sampling_seeds = {
        "bootstrap_floor": floor_seed,
        "pca": base_seed,
        "autoencoder_codes": base_seed + 1,
        **{
            f"variational_kl_{weight:.5g}": base_seed + 2 + position
            for position, weight, _ in variational
        },
    }
    synthetic = {
        "pca": generator.sample(n_samples, np.random.default_rng(sampling_seeds["pca"])),
        "autoencoder_codes": autoencoder.sample(n_samples, sampling_seeds["autoencoder_codes"]),
        **{
            f"variational_kl_{weight:.5g}": trained.sample(
                n_samples, sampling_seeds[f"variational_kl_{weight:.5g}"]
            )
            for position, weight, trained in variational
        },
    }

    print("evaluating ...", flush=True)
    heldout_distances = nearest_training_distances(held_out, train)
    population_median = float(np.median(nearest_neighbour_distances(curves)[0]))

    standardized_held_out = torch.from_numpy(
        autoencoder.standardizer.transform(held_out).astype(np.float32)
    )
    with torch.no_grad():
        autoencoder_rebuilt = autoencoder.model(standardized_held_out)
        variational_rebuilt = {
            position: trained.model.decode(trained.model.encode(standardized_held_out)[0])
            for position, _, trained in variational
        }

    anchor_position = len(variational) // 2
    anchor_weight = variational[anchor_position][1]

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
            "sampling_seeds": sampling_seeds,
        },
        "reference_metrics": {
            "n_films": int(curves.shape[0]),
            "n_training_films": int(train.shape[0]),
            "n_heldout_films": int(held_out.shape[0]),
            "population_nearest_neighbour_median_nm_inverse": population_median,
        },
        "calibration": {
            "kl_weight_anchor": anchor_weight,
            "anchor_implied_noise_variance": variational[anchor_position][2].implied_noise_variance,
            "autoencoder_heldout_reconstruction_standardized": float(
                mse_loss(autoencoder_rebuilt, standardized_held_out)
            ),
        },
        "autoencoder": {
            "config": config["autoencoder"],
            "parameter_count": autoencoder.model.parameter_count,
            "epoch_losses": [float(value) for value in autoencoder.epoch_losses],
            "heldout_reconstruction_nm_squared": _reconstruction_mse(
                autoencoder.standardizer.inverse_transform(autoencoder_rebuilt.numpy()),
                held_out,
            ),
            "heldout_reconstruction_standardized": float(
                mse_loss(autoencoder_rebuilt, standardized_held_out)
            ),
            **_distribution_metrics(
                synthetic["autoencoder_codes"],
                train,
                held_out,
                sampling_seeds["autoencoder_codes"],
            ),
        },
        "variational": [
            {
                "kl_weight": weight,
                "implied_noise_variance": trained.implied_noise_variance,
                "config": {**config["variational"], "kl_weight": weight},
                "parameter_count": trained.model.parameter_count,
                "epoch_reconstruction": [float(value) for value in trained.epoch_reconstruction],
                "epoch_kl": [float(value) for value in trained.epoch_kl],
                "latent_kl": [float(value) for value in trained.latent_kl],
                "heldout_reconstruction_nm_squared": _reconstruction_mse(
                    trained.standardizer.inverse_transform(variational_rebuilt[position].numpy()),
                    held_out,
                ),
                "heldout_reconstruction_standardized": float(
                    mse_loss(variational_rebuilt[position], standardized_held_out)
                ),
                **_distribution_metrics(
                    synthetic[f"variational_kl_{weight:.5g}"],
                    train,
                    held_out,
                    sampling_seeds[f"variational_kl_{weight:.5g}"],
                ),
            }
            for position, weight, trained in variational
        ],
        "pca": {
            "n_components": generator.n_components,
            "heldout_reconstruction_nm_squared": _reconstruction_mse(
                _pca_reconstruction(held_out, generator.mean, generator.components),
                held_out,
            ),
            **_distribution_metrics(synthetic["pca"], train, held_out, sampling_seeds["pca"]),
        },
        "bootstrap_floor": {
            "memorization": memorization_record(
                nearest_training_distances(floor, train), heldout_distances
            ),
            "mean_curve_deviation_vs_heldout": mean_curve_deviation(held_out, floor),
        },
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.report_dir / "vae_experiments.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    anchor_entry = record["variational"][anchor_position]
    _mean_curves_figure(
        span_curves.wavelengths_nm,
        held_out,
        [
            ("PCA synthetic median", synthetic["pca"], "tab:green"),
            ("AE code Gaussian median", synthetic["autoencoder_codes"], "tab:purple"),
            (
                f"VAE median, kl = {anchor_weight:.4g}",
                synthetic[f"variational_kl_{anchor_weight:.5g}"],
                "tab:orange",
            ),
        ],
        args.figure_dir / "vae_mean_curves.png",
    )
    _nn_distance_figure(
        [
            ("bootstrap floor (copies)", nearest_training_distances(floor, train), "0.4", "-"),
            (
                "PCA synthetic",
                nearest_training_distances(synthetic["pca"], train),
                "tab:green",
                "--",
            ),
            (
                "AE code Gaussian",
                nearest_training_distances(synthetic["autoencoder_codes"], train),
                "tab:purple",
                "--",
            ),
            (
                f"VAE, kl = {anchor_weight:.4g}",
                nearest_training_distances(synthetic[f"variational_kl_{anchor_weight:.5g}"], train),
                "tab:orange",
                "--",
            ),
            ("held-out measured films", heldout_distances, "tab:blue", "-"),
        ],
        population_median,
        args.figure_dir / "vae_nn_distances.png",
    )
    _training_figure(
        [float(value) for value in autoencoder.epoch_losses],
        anchor_entry["epoch_reconstruction"],
        anchor_entry["epoch_kl"],
        anchor_weight,
        args.figure_dir / "vae_training.png",
    )

    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"record: {record_path}")


if __name__ == "__main__":
    main()
