"""Device selection logic for the recommender.

Given the benchmark outcome of each candidate device, apply the latency
constraint as a hard filter and rank the survivors by the chosen optimization
factor (cost / energy / latency). Pure functions — no I/O — so they are
unit-testable with synthetic ``BenchmarkJob`` objects.

Latency note: ``Latency.average``/``percentiles`` are per-inference-batch times
in **seconds** (``time.perf_counter`` deltas in edge-inference-clients), so the
constraint compares against ``latency_threshold_ms`` after a seconds→ms
conversion. With the common ``batch_size=1`` this is effectively per-sample
latency.
"""

from dataclasses import dataclass
from typing import Optional

from edge_benchmarking_types.edge_device.models import BenchmarkJob
from edge_benchmarking_types.edge_farm.enums import (
    OptimizationFactor,
    LatencyPercentile,
)
from edge_benchmarking_types.edge_farm.models import (
    DeviceCatalogEntry,
    DeviceCandidateResult,
    DeviceRecommendation,
)

from edge_benchmarking_client.energy import compute_energy_joules

_SECONDS_TO_MS = 1000.0
_PERCENTILE_OF_METRIC = {
    LatencyPercentile.P95: 95,
    LatencyPercentile.P99: 99,
}


@dataclass
class CandidateInput:
    """A single device's benchmark outcome fed into ranking.

    ``error`` is set when the device was offline or the benchmark raised, in
    which case ``benchmark_job`` may be ``None``.
    """

    hostname: str
    benchmark_job: Optional[BenchmarkJob] = None
    benchmark_job_id: Optional[str] = None
    catalog_entry: Optional[DeviceCatalogEntry] = None
    error: Optional[str] = None


def extract_latency_ms(
    benchmark_job: BenchmarkJob, latency_metric: LatencyPercentile
) -> Optional[float]:
    """Pull the chosen inference-latency statistic (in ms) from a job."""
    if benchmark_job.inference_results is None:
        return None
    latency = benchmark_job.inference_results.performance.inference.latency

    if latency_metric == LatencyPercentile.AVG:
        seconds = latency.average
    else:
        seconds = latency.percentiles.get(_PERCENTILE_OF_METRIC[latency_metric])

    return None if seconds is None else seconds * _SECONDS_TO_MS


def extract_accuracy(
    benchmark_job: BenchmarkJob, accuracy_metric: str = "accuracy"
) -> Optional[float]:
    """Pull the accuracy metric from a job's ``metrics`` dict, if present.

    ``metrics`` is populated by the Edge-Farm API only when CVAT ground-truth
    annotations reached the job bucket, so it may be ``None`` (no ground truth)
    or missing the requested key — both return ``None`` here.
    """
    if benchmark_job.inference_results is None:
        return None
    metrics = benchmark_job.inference_results.metrics
    if not metrics:
        return None
    return metrics.get(accuracy_metric)


def rank_candidates(
    candidates: list[CandidateInput],
    *,
    factor: OptimizationFactor,
    latency_metric: LatencyPercentile,
    latency_threshold_ms: float,
    min_accuracy: Optional[float] = None,
    accuracy_metric: str = "accuracy",
) -> DeviceRecommendation:
    """Filter candidates by the latency and accuracy constraints, rank by factor.

    A device survives only if it satisfies *every* hard gate: its chosen latency
    statistic stays within ``latency_threshold_ms`` **and** — when
    ``min_accuracy`` is set — its accuracy is at least ``min_accuracy``. When
    ``min_accuracy`` is ``None`` the accuracy gate is skipped entirely.

    Returns a :class:`DeviceRecommendation` whose ``candidates`` lists surviving
    devices first (best→worst by factor) followed by excluded ones, each
    carrying an ``excluded_reason`` (multiple failed gates are joined).
    ``winner_hostname`` is the top survivor, or ``None`` if none qualified.
    """
    results: list[DeviceCandidateResult] = []

    for candidate in candidates:
        result = DeviceCandidateResult(
            hostname=candidate.hostname,
            benchmark_job_id=candidate.benchmark_job_id,
        )
        if candidate.catalog_entry is not None:
            result.cost_eur = candidate.catalog_entry.cost_eur
            result.tier_rank = candidate.catalog_entry.tier_rank

        if candidate.error is not None:
            result.excluded_reason = candidate.error
            results.append(result)
            continue

        if (
            candidate.benchmark_job is None
            or candidate.benchmark_job.inference_results is None
        ):
            result.excluded_reason = "benchmark produced no inference results"
            results.append(result)
            continue

        result.latency_ms = extract_latency_ms(candidate.benchmark_job, latency_metric)
        result.energy_joules = compute_energy_joules(
            candidate.benchmark_job.benchmark_results
        )
        result.accuracy = extract_accuracy(candidate.benchmark_job, accuracy_metric)

        # Latency AND accuracy are co-equal hard gates: a device survives only if
        # it clears every gate. Collect all failures so a device that misses more
        # than one is explained fully.
        reasons: list[str] = []

        if result.latency_ms is None:
            reasons.append(f"no {latency_metric.value} latency available")
        elif result.latency_ms > latency_threshold_ms:
            reasons.append(
                f"{latency_metric.value} latency {result.latency_ms:.1f}ms "
                f"exceeds threshold {latency_threshold_ms:.1f}ms"
            )

        if min_accuracy is not None:
            if result.accuracy is None:
                reasons.append(
                    f"no accuracy metric reported; floor is {min_accuracy:.3f}"
                )
            elif result.accuracy < min_accuracy:
                reasons.append(
                    f"accuracy {result.accuracy:.3f} below floor {min_accuracy:.3f}"
                )

        if factor == OptimizationFactor.COST and result.cost_eur is None:
            reasons.append("no catalog cost for device; cannot rank by cost")
        elif factor == OptimizationFactor.ENERGY and result.energy_joules is None:
            reasons.append("no power telemetry; cannot rank by energy")

        result.meets_constraint = not reasons
        if reasons:
            result.excluded_reason = "; ".join(reasons)

        results.append(result)

    survivors = [r for r in results if r.meets_constraint]
    excluded = [r for r in results if not r.meets_constraint]

    def sort_key(r: DeviceCandidateResult):
        if factor == OptimizationFactor.COST:
            return (r.cost_eur, r.tier_rank or 0, r.latency_ms)
        if factor == OptimizationFactor.ENERGY:
            return (r.energy_joules, r.latency_ms)
        return (r.latency_ms,)

    survivors.sort(key=sort_key)

    return DeviceRecommendation(
        factor=factor,
        latency_metric=latency_metric,
        latency_threshold_ms=latency_threshold_ms,
        min_accuracy=min_accuracy,
        accuracy_metric=accuracy_metric,
        winner_hostname=survivors[0].hostname if survivors else None,
        candidates=survivors + excluded,
    )
