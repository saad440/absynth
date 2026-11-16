"""Run the thickness-aware comparison and write the evaluation record.

Usage (from the repository root, after ``pip install -e ".[dev]"``):

    python scripts/run_thickness.py [--config configs/thickness.yaml]
        [--report-dir reports] [--figure-dir figures]

Films are curve-and-thickness pairs: the archive stores optical
density ``q = A/t``, and converting a generated curve back to
absorbance, ``A = q·t``, needs a thickness again. This script trains
the compact autoencoder and the variational model at its calibrated
KL weight twice each, curve-only and thickness-conditioned, on the
same frozen date-block holdout as the earlier records, and fits the
thickness-slice PCA generator. Every generator is then evaluated on
three fronts:

- **Coupling.** The correlation between integrated ``q`` and
  thickness in generated pairs, read against held-out films, the
  joint resampling floor, a labelled demonstration of independent
  pairing (the PCA curves with independently drawn thicknesses),
  which erases the coupling by construction, and the arithmetic
  coupling, which pairs integrated absorbance with independently
  drawn thicknesses and converts back through ``q = A/t``, so the
  division alone drives the whole correlation. The training
  population's absorbance scaling ``∫A ∝ t^β``, a log-log slope with
  a bootstrap band, is reported alongside.
- **Thickness resolution.** Per-film held-out reconstruction errors
  averaged inside thickness slices, per model, in q-space and after
  conversion to absorbance, where a thick film's error is
  re-inflated by its thickness.
- **Space.** Population-level comparisons in q-space and absorbance
  space, where the ranking of models can differ: dividing by
  thickness shrinks a thick film's errors, multiplying re-inflates
  them.

The pair generators emit films: the joint resampling floor, the
thickness-slice generator and the conditioned models draw the
thickness from the measured training distribution and the curve from
a model of ``q`` given ``t``. The labelled independent-pairing
demonstration draws the two separately on purpose, to show the
failure. The curve-only generators are evaluated as curve
populations, in q-space only. The record carries the configuration
path, git state, seeds, package versions and the dataset checksum; a
robustness sweep repeats the conditioned models across seeds, and a
seed whose training diverges is recorded as the instability it
measures.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.nn.functional import mse_loss

from absynth.baselines import fit_pca_generator
from absynth.data import DATASETS
from absynth.evaluation import (
    arithmetic_coupling_pairs,
    component_scores,
    coupling_record,
    covariance_record,
    mean_curve_deviation,
    memorization_record,
    nearest_training_distances,
    quantile_band,
    scaling_exponent,
    sensitivity_decomposition,
    thickness_slice_errors,
)
from absynth.explore import nearest_neighbour_distances
from absynth.loader import load_dataset
from absynth.models import CurveStandardizer
from absynth.preprocessing import absorbance_from_q, require_complete, restrict_to_span
from absynth.provenance import git_provenance, package_versions
from absynth.splits import date_block_holdout
from absynth.thickness import fit_thickness_slice_generator, resample_joint
from absynth.train import (
    AutoEncoderConfig,
    TrainedThicknessConditionedVariationalAutoEncoder,
    TrainingDivergenceError,
    VariationalAutoEncoderConfig,
    train_autoencoder,
    train_thickness_conditioned_autoencoder,
    train_thickness_conditioned_variational_autoencoder,
    train_variational_autoencoder,
)


def _integrated_q(curves: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """Integrated attenuation per curve over the working span, nm⁻¹·nm."""
    return np.trapezoid(curves, wavelengths_nm, axis=1)


def _curve_distribution_metrics(
    synthetic: np.ndarray, train: np.ndarray, held_out: np.ndarray, sampling_seed: int
) -> dict[str, object]:
    """Population-level comparison of one synthetic curve set, q-space."""
    return {
        "sampling_seed": sampling_seed,
        "memorization": memorization_record(
            nearest_training_distances(synthetic, train),
            nearest_training_distances(held_out, train),
        ),
        "mean_curve_deviation_vs_heldout": mean_curve_deviation(held_out, synthetic),
        "covariance_vs_heldout": covariance_record(held_out, synthetic),
    }


def _pair_distribution_metrics(
    curves: np.ndarray,
    thickness_nm: np.ndarray,
    train: np.ndarray,
    held_out: np.ndarray,
    held_out_thickness_nm: np.ndarray,
    wavelengths_nm: np.ndarray,
    sampling_seed: int,
) -> dict[str, object]:
    """Pair-level comparison: q-space, absorbance space, and coupling.

    The two spaces answer different questions. q-space scores curve
    shapes; absorbance space is the film-level reading, where
    converting with ``A = q·t`` weights a curve error by the
    thickness it belongs to.
    """
    synthetic_a = absorbance_from_q(curves, thickness_nm)
    held_out_a = absorbance_from_q(held_out, held_out_thickness_nm)
    return {
        "sampling_seed": sampling_seed,
        "memorization": memorization_record(
            nearest_training_distances(curves, train),
            nearest_training_distances(held_out, train),
        ),
        "mean_curve_deviation_vs_heldout_q_space": mean_curve_deviation(held_out, curves),
        "covariance_vs_heldout_q_space": covariance_record(held_out, curves),
        "mean_curve_deviation_vs_heldout_a_space": mean_curve_deviation(held_out_a, synthetic_a),
        "covariance_vs_heldout_a_space": covariance_record(held_out_a, synthetic_a),
        "coupling_pearson_r": coupling_record(
            _integrated_q(curves, wavelengths_nm), thickness_nm
        )["pearson_r"],
    }


def _per_film_errors(rebuilt: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Mean squared error per film over wavelengths, in the input's unit squared."""
    return np.mean((rebuilt - reference) ** 2, axis=1)


def _pca_reconstruction(curves: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Rebuild curves from their PCA coordinates: the linear reconstruction."""
    scores = component_scores(curves, mean, components)
    return scores @ components + mean


def _curve_reconstruction(
    model,
    standardizer: CurveStandardizer,
    standardized_held_out: torch.Tensor,
    held_out: np.ndarray,
    held_out_thickness_nm: np.ndarray,
    variational: bool,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Deterministic reconstruction of held-out curves by one trained model.

    The variational model uses posterior means, no sampling. Returns
    the mean squared error in standardized units, the overall
    reconstruction error in nm⁻², and the per-film errors in nm⁻² and
    in absorbance units squared: the film-level reading, where an
    error at thickness ``t`` counts multiplied by ``t``.
    """
    with torch.no_grad():
        if variational:
            mean, _std = model.encode(standardized_held_out)
            rebuilt = model.decode(mean)
        else:
            rebuilt = model(standardized_held_out)
    standardized_mse = float(mse_loss(rebuilt, standardized_held_out))
    rebuilt_nm = standardizer.inverse_transform(rebuilt.numpy())
    per_film_nm2 = _per_film_errors(rebuilt_nm, held_out)
    per_film_a2 = _per_film_errors(
        rebuilt_nm * held_out_thickness_nm[:, np.newaxis],
        held_out * held_out_thickness_nm[:, np.newaxis],
    )
    return standardized_mse, float(per_film_nm2.mean()), per_film_nm2, per_film_a2


def _conditioned_reconstruction(
    model,
    standardizer: CurveStandardizer,
    thickness_standardizer,
    held_out: np.ndarray,
    held_out_thickness_nm: np.ndarray,
    variational: bool,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Reconstruction of held-out films at their measured thicknesses.

    The conditional reconstruction diagnostic: the encoder maps the
    held-out curve, the decoder rebuilds it at the film's own measured
    thickness, so no information about the film's thickness is
    missing from the rebuild. Returns the same four quantities as
    :func:`_curve_reconstruction`.
    """
    standardized_curves = torch.from_numpy(standardizer.transform(held_out).astype(np.float32))
    standardized_thickness = torch.from_numpy(
        thickness_standardizer.transform(held_out_thickness_nm).astype(np.float32)
    )
    with torch.no_grad():
        if variational:
            mean, _std = model.encode(standardized_curves)
            rebuilt = model.decode(mean, standardized_thickness)
        else:
            rebuilt = model(standardized_curves, standardized_thickness)
    standardized_mse = float(mse_loss(rebuilt, standardized_curves))
    rebuilt_nm = standardizer.inverse_transform(rebuilt.numpy())
    per_film_nm2 = _per_film_errors(rebuilt_nm, held_out)
    per_film_a2 = _per_film_errors(
        rebuilt_nm * held_out_thickness_nm[:, np.newaxis],
        held_out * held_out_thickness_nm[:, np.newaxis],
    )
    return standardized_mse, float(per_film_nm2.mean()), per_film_nm2, per_film_a2


def _conditioning_sensitivity(
    trained,
    training_thickness_nm: np.ndarray,
    seed: int,
    held_out: np.ndarray,
    wavelengths_nm: np.ndarray,
) -> dict[str, float]:
    """How much the thickness input moves the decoding, at fixed codes.

    Sixty-four codes, drawn the way each model's own generator draws
    them, are decoded at the 10th and 90th percentile of the measured
    training thicknesses. The mean absolute change across wavelengths
    is read against the held-out films' mean per-wavelength spread,
    the population's own variability scale: a ratio near 0 means the
    conditioning never attached and the model ignores the input. The
    change is then split along the held-out mean-curve direction, the
    curve-height axis. Amplitude means the thickness input rescales
    curves; shape means it moves wavelengths differently. A residual
    coherent in wavelength would carry thickness-dependent physics
    that a pure rescaling cannot express.
    """
    thickness_lo, thickness_hi = np.quantile(training_thickness_nm, [0.1, 0.9])
    n_codes = 64
    if isinstance(trained, TrainedThicknessConditionedVariationalAutoEncoder):
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            codes = torch.randn(n_codes, trained.model.n_latent, generator=generator)
    else:
        rng = np.random.default_rng(seed)
        drawn = rng.multivariate_normal(
            mean=trained.codes.mean(axis=0),
            cov=np.cov(trained.codes, rowvar=False),
            size=n_codes,
        )
        codes = torch.from_numpy(drawn.astype(np.float32))
    decoded = {}
    for label, thickness in (("lo", thickness_lo), ("hi", thickness_hi)):
        standardized = torch.from_numpy(
            trained.thickness_standardizer.transform(np.full(n_codes, thickness)).astype(np.float32)
        )
        with torch.no_grad():
            decoded[label] = trained.model.decode(codes, standardized).numpy()
    delta = decoded["hi"] - decoded["lo"]
    sensitivity = float(np.abs(delta).mean())
    spread = float(held_out.std(axis=0).mean())
    decomposition = sensitivity_decomposition(delta, held_out.mean(axis=0), wavelengths_nm)
    return {
        "thickness_lo_nm": float(thickness_lo),
        "thickness_hi_nm": float(thickness_hi),
        "mean_abs_decoding_change_nm_inverse": sensitivity,
        "heldout_mean_per_wavelength_std_nm_inverse": spread,
        "ratio_to_heldout_spread": sensitivity / spread,
        "amplitude_fraction": decomposition["amplitude_fraction"],
        "amplitude_fraction_per_code_mean": decomposition["amplitude_fraction_per_code_mean"],
        "residual_wavelength_pearson_r": decomposition["residual_wavelength_pearson_r"],
    }


def _coupling_figure(
    series: list[tuple[str, np.ndarray, np.ndarray, str, str, float]], path: Path
) -> None:
    """Integrated q against thickness: measured films and generated pairs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, thickness, integrated, color, marker, alpha in series:
        ax.scatter(
            thickness, integrated, s=12, color=color, marker=marker, alpha=alpha,
            label=label, linewidths=0,
        )
    ax.set_xlabel("Thickness (nm)")
    ax.set_ylabel(r"Integrated $q$ over 300–900 nm (nm$^{-1}\cdot$nm)")
    ax.set_title("Curve–thickness coupling: measured films and generated pairs")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _slice_error_figure(
    slice_edges_nm: np.ndarray,
    models: list[tuple[str, list[float], str, str]],
    path: Path,
) -> None:
    """Grouped bars: per-slice held-out reconstruction error, per model."""
    n_slices = len(slice_edges_nm) - 1
    n_models = len(models)
    width = 0.8 / n_models
    positions = np.arange(n_slices)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, (label, errors, color, hatch) in enumerate(models):
        offset = positions + index * width - 0.4 + width / 2
        ax.bar(
            offset, errors, width, label=label, color=color, hatch=hatch,
            edgecolor="white", linewidth=0.3,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{lo:.0f}–{hi:.0f}" for lo, hi in pairwise(slice_edges_nm)])
    ax.set_xlabel("Held-out thickness slice (nm)")
    ax.set_ylabel(r"Reconstruction MSE (nm$^{-2}$)")
    ax.set_title("Where the reconstruction error lives along thickness")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _absorbance_curves_figure(
    wavelengths_nm: np.ndarray,
    held_out_a: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    path: Path,
) -> None:
    """Held-out absorbance bands against each pair generator's median."""
    band_real = quantile_band(held_out_a, quantiles=(0.1, 0.5, 0.9))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(
        wavelengths_nm, band_real[0], band_real[2], color="0.55", alpha=0.35,
        label="held-out measured, 10–90% band",
    )
    ax.plot(wavelengths_nm, band_real[1], color="0.15", label="held-out measured median")
    for label, synthetic_a, color in series:
        band = quantile_band(synthetic_a, quantiles=(0.1, 0.5, 0.9))
        ax.plot(wavelengths_nm, band[1], "--", color=color, label=label)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Absorbance $A$ (dimensionless)")
    ax.set_title("Generated films in absorbance space, $A = q\\cdot t$")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the curve-only and thickness-conditioned models, evaluate all "
            "generators, and write the record and figures."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/thickness.yaml"), help="experiment configuration"
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
    # Held-out films enter absorbance space through their measured
    # thicknesses, the same conversion the generated films undergo.
    held_out_a = absorbance_from_q(held_out, held_out_thickness)

    n_slices = int(config["thickness_slices"]["n_slices"])
    # The evaluation partition comes from the held-out thicknesses
    # themselves: equal film counts per slice, full coverage, and the
    # same partition for every model, so their per-slice errors are
    # directly comparable.
    evaluation_edges = np.quantile(held_out_thickness, np.linspace(0.0, 1.0, n_slices + 1))

    n_samples = int(config["sampling"]["n_samples"])
    # The arithmetic coupling only needs enough draws for a tight
    # value. 20,000 draws put one standard error near ±0.007, small
    # next to the scatter of any real comparison.
    n_reference_draws = 20_000
    base_seed = int(config["sampling"]["seed"])
    floor_seed = int(config["sampling"]["floor_seed"])
    kl_weight = float(config["variational"]["kl_weight"])
    variational_name = f"variational_kl_{kl_weight:.5g}"
    conditioned_variational_name = f"conditioned_{variational_name}"

    print("training the curve-only autoencoder ...", flush=True)
    autoencoder = train_autoencoder(train, AutoEncoderConfig(**config["autoencoder"]))
    print(f"training the curve-only variational model at kl_weight = {kl_weight:.5g} ...", flush=True)
    variational = train_variational_autoencoder(
        train, VariationalAutoEncoderConfig(**config["variational"])
    )
    print("training the thickness-conditioned autoencoder ...", flush=True)
    conditioned_autoencoder = train_thickness_conditioned_autoencoder(
        train, train_thickness, AutoEncoderConfig(**config["autoencoder"])
    )
    print(
        f"training the thickness-conditioned variational model at kl_weight = {kl_weight:.5g} ...",
        flush=True,
    )
    conditioned_variational = train_thickness_conditioned_variational_autoencoder(
        train, train_thickness, VariationalAutoEncoderConfig(**config["variational"])
    )

    generator = fit_pca_generator(train, n_components=int(config["pca"]["n_components"]))
    slice_generator = fit_thickness_slice_generator(
        train,
        train_thickness,
        n_slices=n_slices,
        n_components=int(config["thickness_slices"]["n_components"]),
    )

    sampling_seeds = {
        "joint_floor": floor_seed,
        "pca": base_seed,
        "autoencoder_codes": base_seed + 1,
        variational_name: base_seed + 2,
        "thickness_slices": base_seed + 3,
        "conditioned_autoencoder_codes": base_seed + 4,
        conditioned_variational_name: base_seed + 5,
        "independent_marginal_pca": base_seed + 6,
        # Drives the code draws of the conditioning sensitivity check.
        "conditioning_sensitivity": base_seed + 7,
        # The arithmetic-coupling pairing draw and the absorbance-
        # scaling bootstrap, both drawn from the measured training
        # distributions.
        "arithmetic_coupling": base_seed + 8,
        "absorbance_scaling": base_seed + 9,
    }

    print("drawing synthetic films ...", flush=True)
    floor_curves, floor_thickness = resample_joint(
        train, train_thickness, n_samples, np.random.default_rng(sampling_seeds["joint_floor"])
    )
    pca_curves = generator.sample(n_samples, np.random.default_rng(sampling_seeds["pca"]))
    ae_curves = autoencoder.sample(n_samples, sampling_seeds["autoencoder_codes"])
    vae_curves = variational.sample(n_samples, sampling_seeds[variational_name])
    slice_curves, slice_thickness = slice_generator.sample(
        n_samples, np.random.default_rng(sampling_seeds["thickness_slices"])
    )
    conditioned_ae_curves, conditioned_ae_thickness = conditioned_autoencoder.sample(
        n_samples, sampling_seeds["conditioned_autoencoder_codes"]
    )
    conditioned_vae_curves, conditioned_vae_thickness = conditioned_variational.sample(
        n_samples, sampling_seeds[conditioned_variational_name]
    )
    # The independent-pairing demonstration keeps the PCA curve samples
    # and draws their thicknesses separately: same marginals, erased
    # coupling.
    independent_thickness = train_thickness[
        np.random.default_rng(sampling_seeds["independent_marginal_pca"]).integers(
            0, train_thickness.size, size=n_samples
        )
    ]
    independent_a = absorbance_from_q(pca_curves, independent_thickness)

    heldout_distances = nearest_training_distances(held_out, train)
    population_median = float(np.median(nearest_neighbour_distances(curves)[0]))
    heldout_coupling = coupling_record(
        _integrated_q(held_out, wavelengths_nm), held_out_thickness
    )["pearson_r"]
    # The film-level reading of the same association works out, per
    # film, to ∫A = t·∫q exactly, because the trapezoid integral is
    # linear in the curve and each film's thickness is one constant
    # factor. The arithmetic coupling and the scaling exponent read
    # the training distributions, the ones every generator draws from.
    training_integrated_a = _integrated_q(train, wavelengths_nm) * train_thickness

    print("evaluating ...", flush=True)
    coupling = {"heldout_measured": heldout_coupling}
    reference_q, reference_t = arithmetic_coupling_pairs(
        training_integrated_a,
        train_thickness,
        n_reference_draws,
        np.random.default_rng(sampling_seeds["arithmetic_coupling"]),
    )
    coupling["arithmetic_coupling"] = coupling_record(reference_q, reference_t)[
        "pearson_r"
    ]
    absorbance_scaling = scaling_exponent(
        training_integrated_a,
        train_thickness,
        np.random.default_rng(sampling_seeds["absorbance_scaling"]),
    )
    absorbance_scaling["population"] = "training"
    absorbance_scaling["absorbance_coupling_pearson_r"] = coupling_record(
        training_integrated_a, train_thickness
    )["pearson_r"]
    generation = {
        "pca": _curve_distribution_metrics(pca_curves, train, held_out, sampling_seeds["pca"]),
        "autoencoder_codes": _curve_distribution_metrics(
            ae_curves, train, held_out, sampling_seeds["autoencoder_codes"]
        ),
        variational_name: _curve_distribution_metrics(
            vae_curves, train, held_out, sampling_seeds[variational_name]
        ),
    }
    pair_samples = {
        "joint_floor": (floor_curves, floor_thickness),
        "thickness_slices": (slice_curves, slice_thickness),
        "conditioned_autoencoder_codes": (conditioned_ae_curves, conditioned_ae_thickness),
        conditioned_variational_name: (conditioned_vae_curves, conditioned_vae_thickness),
        "independent_marginal_pca": (pca_curves, independent_thickness),
    }
    for name, (curves_drawn, thickness_drawn) in pair_samples.items():
        coupling[name] = coupling_record(
            _integrated_q(curves_drawn, wavelengths_nm), thickness_drawn
        )["pearson_r"]
        generation[name] = _pair_distribution_metrics(
            curves_drawn,
            thickness_drawn,
            train,
            held_out,
            held_out_thickness,
            wavelengths_nm,
            sampling_seeds[name],
        )
    # The demonstration reuses the PCA curve samples, so its record
    # states both seeds: the curves' and the thicknesses'.
    generation["independent_marginal_pca"]["curve_sampling_seed"] = sampling_seeds["pca"]
    # Each code-based pair generator carries its pairing scheme: the
    # conditioned autoencoder cosamples codes and thicknesses from one
    # joint Gaussian, the conditioned variational model draws prior
    # codes against resampled measured thicknesses, and the
    # demonstration pairs independently on purpose.
    pairing_schemes = {
        "conditioned_autoencoder_codes": "joint_cosampling",
        conditioned_variational_name: "prior_codes_plus_measured_t",
        "independent_marginal_pca": "independent_marginals",
    }
    for name, scheme in pairing_schemes.items():
        generation[name]["code_pairing"] = scheme

    standardized_held_out = torch.from_numpy(
        autoencoder.standardizer.transform(held_out).astype(np.float32)
    )

    pca_rebuilt = _pca_reconstruction(held_out, generator.mean, generator.components)
    pca_per_film_nm2 = _per_film_errors(pca_rebuilt, held_out)
    pca_per_film_a2 = _per_film_errors(pca_rebuilt * held_out_thickness[:, np.newaxis], held_out_a)

    def _slice_records(per_film_nm2: np.ndarray, per_film_a2: np.ndarray) -> dict[str, object]:
        return {
            "slices_q_space": thickness_slice_errors(
                per_film_nm2, held_out_thickness, evaluation_edges
            ),
            "slices_a_space": thickness_slice_errors(
                per_film_a2, held_out_thickness, evaluation_edges
            ),
        }

    reconstruction = {
        "pca": {
            "heldout_reconstruction_nm_squared": float(pca_per_film_nm2.mean()),
            "heldout_reconstruction_a_space_nm_squared": float(pca_per_film_a2.mean()),
            **_slice_records(pca_per_film_nm2, pca_per_film_a2),
        }
    }
    for name, trained, variational_flag in (
        ("autoencoder", autoencoder, False),
        (variational_name, variational, True),
    ):
        standardized_mse, nm2, per_film_nm2, per_film_a2 = _curve_reconstruction(
            trained.model,
            trained.standardizer,
            standardized_held_out,
            held_out,
            held_out_thickness,
            variational=variational_flag,
        )
        entry = {"heldout_reconstruction_standardized": standardized_mse}
        if variational_flag:
            entry["implied_noise_variance"] = trained.implied_noise_variance
            entry["latent_kl"] = [float(value) for value in trained.latent_kl]
        reconstruction[name] = {
            **entry,
            "heldout_reconstruction_nm_squared": nm2,
            "heldout_reconstruction_a_space_nm_squared": float(per_film_a2.mean()),
            **_slice_records(per_film_nm2, per_film_a2),
        }
    for name, trained, variational_flag in (
        ("conditioned_autoencoder", conditioned_autoencoder, False),
        (conditioned_variational_name, conditioned_variational, True),
    ):
        standardized_mse, nm2, per_film_nm2, per_film_a2 = _conditioned_reconstruction(
            trained.model,
            trained.standardizer,
            trained.thickness_standardizer,
            held_out,
            held_out_thickness,
            variational=variational_flag,
        )
        entry = {
            "heldout_reconstruction_standardized": standardized_mse,
            "conditioning_sensitivity": _conditioning_sensitivity(
                trained,
                train_thickness,
                sampling_seeds["conditioning_sensitivity"],
                held_out,
                wavelengths_nm,
            ),
        }
        if variational_flag:
            entry["implied_noise_variance"] = trained.implied_noise_variance
            entry["latent_kl"] = [float(value) for value in trained.latent_kl]
        reconstruction[name] = {
            **entry,
            "heldout_reconstruction_nm_squared": nm2,
            "heldout_reconstruction_a_space_nm_squared": float(per_film_a2.mean()),
            **_slice_records(per_film_nm2, per_film_a2),
        }

    robustness_config = config.get("robustness", {})
    robustness_ae_seeds = [int(seed) for seed in robustness_config.get("autoencoder_seeds", [])]
    robustness_vae_seeds = [int(seed) for seed in robustness_config.get("variational_seeds", [])]

    def _conditioned_ae_entry(seed: int) -> dict[str, object]:
        if seed == int(config["autoencoder"]["seed"]):
            headline = reconstruction["conditioned_autoencoder"]
            drawn_curves, drawn_thickness = conditioned_ae_curves, conditioned_ae_thickness
            entry = {
                "sampling_seed": sampling_seeds["conditioned_autoencoder_codes"],
                "heldout_reconstruction_standardized": headline[
                    "heldout_reconstruction_standardized"
                ],
                "heldout_reconstruction_nm_squared": headline["heldout_reconstruction_nm_squared"],
                "heldout_reconstruction_a_space_nm_squared": headline[
                    "heldout_reconstruction_a_space_nm_squared"
                ],
            }
        else:
            print(
                f"robustness: training the conditioned autoencoder at seed = {seed} ...",
                flush=True,
            )
            try:
                trained = train_thickness_conditioned_autoencoder(
                    train,
                    train_thickness,
                    AutoEncoderConfig(**{**config["autoencoder"], "seed": seed}),
                )
            except TrainingDivergenceError as error:
                return {"seed": seed, "diverged": True, "detail": str(error)}
            standardized_mse, nm2, _per_film_nm2, per_film_a2 = _conditioned_reconstruction(
                trained.model,
                trained.standardizer,
                trained.thickness_standardizer,
                held_out,
                held_out_thickness,
                variational=False,
            )
            sample_seed = base_seed + 100 + seed
            drawn_curves, drawn_thickness = trained.sample(n_samples, sample_seed)
            entry = {
                "sampling_seed": sample_seed,
                "heldout_reconstruction_standardized": standardized_mse,
                "heldout_reconstruction_nm_squared": nm2,
                "heldout_reconstruction_a_space_nm_squared": float(per_film_a2.mean()),
            }
        entry["seed"] = seed
        entry["coupling_pearson_r"] = coupling_record(
            _integrated_q(drawn_curves, wavelengths_nm), drawn_thickness
        )["pearson_r"]
        entry["memorization_median_ratio"] = memorization_record(
            nearest_training_distances(drawn_curves, train), heldout_distances
        )["median_ratio"]
        entry["covariance_relative_frobenius_q_space"] = covariance_record(
            held_out, drawn_curves
        )["covariance_relative_frobenius"]
        return entry

    def _conditioned_vae_entry(seed: int) -> dict[str, object]:
        if seed == int(config["variational"]["seed"]):
            headline = reconstruction[conditioned_variational_name]
            drawn_curves, drawn_thickness = conditioned_vae_curves, conditioned_vae_thickness
            entry = {
                "sampling_seed": sampling_seeds[conditioned_variational_name],
                "heldout_reconstruction_standardized": headline[
                    "heldout_reconstruction_standardized"
                ],
                "heldout_reconstruction_nm_squared": headline["heldout_reconstruction_nm_squared"],
                "heldout_reconstruction_a_space_nm_squared": headline[
                    "heldout_reconstruction_a_space_nm_squared"
                ],
                "latent_kl": headline["latent_kl"],
            }
        else:
            print(
                f"robustness: training the conditioned variational model at seed = {seed} ...",
                flush=True,
            )
            try:
                trained = train_thickness_conditioned_variational_autoencoder(
                    train,
                    train_thickness,
                    VariationalAutoEncoderConfig(**{**config["variational"], "seed": seed}),
                )
            except TrainingDivergenceError as error:
                return {"seed": seed, "diverged": True, "detail": str(error)}
            standardized_mse, nm2, _per_film_nm2, per_film_a2 = _conditioned_reconstruction(
                trained.model,
                trained.standardizer,
                trained.thickness_standardizer,
                held_out,
                held_out_thickness,
                variational=True,
            )
            sample_seed = base_seed + 100 + seed
            drawn_curves, drawn_thickness = trained.sample(n_samples, sample_seed)
            entry = {
                "sampling_seed": sample_seed,
                "heldout_reconstruction_standardized": standardized_mse,
                "heldout_reconstruction_nm_squared": nm2,
                "heldout_reconstruction_a_space_nm_squared": float(per_film_a2.mean()),
                "latent_kl": [float(value) for value in trained.latent_kl],
            }
        entry["seed"] = seed
        entry["coupling_pearson_r"] = coupling_record(
            _integrated_q(drawn_curves, wavelengths_nm), drawn_thickness
        )["pearson_r"]
        entry["memorization_median_ratio"] = memorization_record(
            nearest_training_distances(drawn_curves, train), heldout_distances
        )["median_ratio"]
        entry["covariance_relative_frobenius_q_space"] = covariance_record(
            held_out, drawn_curves
        )["covariance_relative_frobenius"]
        return entry

    robustness_ae_entries = [_conditioned_ae_entry(seed) for seed in robustness_ae_seeds]
    robustness_vae_entries = [_conditioned_vae_entry(seed) for seed in robustness_vae_seeds]

    def _spread(values: list[float]) -> dict[str, float]:
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    def _converged(entries: list[dict[str, object]], key: str) -> list[float]:
        """A metric's values over the entries whose training converged."""
        return [entry[key] for entry in entries if not entry.get("diverged")]

    robustness = {
        "autoencoder_seeds": robustness_ae_seeds,
        "variational_seeds": robustness_vae_seeds,
        "autoencoder": robustness_ae_entries,
        "variational": robustness_vae_entries,
        "summary": {
            "n_diverged_seeds": {
                "autoencoder": sum(
                    1 for entry in robustness_ae_entries if entry.get("diverged")
                ),
                "variational": sum(
                    1 for entry in robustness_vae_entries if entry.get("diverged")
                ),
            },
            "autoencoder_reconstruction_nm_squared": _spread(
                _converged(robustness_ae_entries, "heldout_reconstruction_nm_squared")
            ),
            "autoencoder_coupling_pearson_r": _spread(
                _converged(robustness_ae_entries, "coupling_pearson_r")
            ),
            "variational_reconstruction_nm_squared": _spread(
                _converged(robustness_vae_entries, "heldout_reconstruction_nm_squared")
            ),
            "variational_coupling_pearson_r": _spread(
                _converged(robustness_vae_entries, "coupling_pearson_r")
            ),
            "variational_memorization_median_ratio": _spread(
                _converged(robustness_vae_entries, "memorization_median_ratio")
            ),
        },
    }

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
            "training_thickness_nm": {
                "min": float(train_thickness.min()),
                "median": float(np.median(train_thickness)),
                "max": float(train_thickness.max()),
            },
            "heldout_thickness_nm": {
                "min": float(held_out_thickness.min()),
                "median": float(np.median(held_out_thickness)),
                "max": float(held_out_thickness.max()),
            },
            "heldout_coupling_pearson_r": heldout_coupling,
            "absorbance_scaling": absorbance_scaling,
        },
        "thickness_slices": {
            "n_slices": slice_generator.n_slices,
            "n_components": slice_generator.n_components,
            "film_counts": list(slice_generator.film_counts),
            "slice_edges_nm": [float(edge) for edge in slice_generator.slice_edges_nm],
            "evaluation_partition": {
                "method": "held-out thickness quantiles",
                "edges_nm": [float(edge) for edge in evaluation_edges],
            },
        },
        "coupling": coupling,
        "generation": generation,
        "reconstruction": reconstruction,
        "robustness": robustness,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.report_dir / "thickness_experiments.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _coupling_figure(
        [
            (
                "held-out measured",
                held_out_thickness,
                _integrated_q(held_out, wavelengths_nm),
                "tab:blue",
                ".",
                0.8,
            ),
            (
                "joint resampling floor",
                floor_thickness,
                _integrated_q(floor_curves, wavelengths_nm),
                "0.5",
                ".",
                0.25,
            ),
            (
                "thickness-slice generator",
                slice_thickness,
                _integrated_q(slice_curves, wavelengths_nm),
                "tab:green",
                ".",
                0.25,
            ),
            (
                r"arithmetic coupling ($\int A \perp t$)",
                reference_t,
                reference_q,
                "tab:purple",
                ".",
                0.15,
            ),
            (
                f"conditioned VAE, kl = {kl_weight:.4g}",
                conditioned_vae_thickness,
                _integrated_q(conditioned_vae_curves, wavelengths_nm),
                "tab:orange",
                ".",
                0.25,
            ),
            (
                "PCA curves + independent thickness",
                independent_thickness,
                _integrated_q(pca_curves, wavelengths_nm),
                "tab:red",
                "x",
                0.5,
            ),
        ],
        args.figure_dir / "thickness_coupling.png",
    )
    _slice_error_figure(
        evaluation_edges,
        [
            (
                "PCA",
                [entry["mean_error"] for entry in reconstruction["pca"]["slices_q_space"]],
                "0.4",
                "",
            ),
            (
                "autoencoder",
                [entry["mean_error"] for entry in reconstruction["autoencoder"]["slices_q_space"]],
                "tab:purple",
                "",
            ),
            (
                "conditioned autoencoder",
                [
                    entry["mean_error"]
                    for entry in reconstruction["conditioned_autoencoder"]["slices_q_space"]
                ],
                "tab:purple",
                "//",
            ),
            (
                f"VAE, kl = {kl_weight:.4g}",
                [entry["mean_error"] for entry in reconstruction[variational_name]["slices_q_space"]],
                "tab:orange",
                "",
            ),
            (
                f"conditioned VAE, kl = {kl_weight:.4g}",
                [
                    entry["mean_error"]
                    for entry in reconstruction[conditioned_variational_name]["slices_q_space"]
                ],
                "tab:orange",
                "//",
            ),
        ],
        args.figure_dir / "thickness_slice_errors.png",
    )
    _absorbance_curves_figure(
        wavelengths_nm,
        held_out_a,
        [
            (
                "joint resampling floor",
                absorbance_from_q(floor_curves, floor_thickness),
                "0.4",
            ),
            (
                f"conditioned VAE, kl = {kl_weight:.4g}",
                absorbance_from_q(conditioned_vae_curves, conditioned_vae_thickness),
                "tab:orange",
            ),
            (
                "PCA curves + independent thickness",
                independent_a,
                "tab:red",
            ),
        ],
        args.figure_dir / "thickness_absorbance_curves.png",
    )

    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"record: {record_path}")


if __name__ == "__main__":
    main()
