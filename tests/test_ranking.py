#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for candidate filtering + ranking (pure, no devices/server)."""

from edge_benchmarking_types.edge_device.enums import JobStatus
from edge_benchmarking_types.edge_device.models import BenchmarkJob
from edge_benchmarking_types.edge_farm.enums import (
    OptimizationFactor,
    LatencyPercentile,
)
from edge_benchmarking_types.edge_farm.models import (
    DeviceCatalogEntry,
    BenchmarkInferResult,
    InferPerformance,
    PerformanceResult,
    Latency,
)
from edge_benchmarking_client.ranking import CandidateInput, rank_candidates


def _perf(latency_s: float) -> PerformanceResult:
    return PerformanceResult(
        total_time=latency_s,
        sample_count=1,
        samples_per_second=1.0 / latency_s,
        latency=Latency(average=latency_s, percentiles={95: latency_s, 99: latency_s}),
    )


def _job(latency_s: float, power_mw: int = 1000, n: int = 4) -> BenchmarkJob:
    # Constant-power telemetry so energy = power_w * duration is deterministic.
    times = [f"2026-06-29T12:00:{s:02d}" for s in range(n)]
    return BenchmarkJob(
        id=f"job-{latency_s}",
        benchmark_results={"time": times, "Power_TOT": [power_mw] * n},
        inference_results=BenchmarkInferResult(
            performance=InferPerformance(
                preprocess=_perf(latency_s),
                inference=_perf(latency_s),
                postprocess=_perf(latency_s),
            ),
            results={},
        ),
        status=JobStatus.SUCCESS,
    )


def _entry(gpu_model: str, tier: int, cost: float) -> DeviceCatalogEntry:
    return DeviceCatalogEntry(gpu_model=gpu_model, tier_rank=tier, cost_eur=cost)


def test_cost_factor_picks_cheapest_compliant_device():
    # Both meet the 50 ms budget (0.01 s = 10 ms); the cheaper should win.
    candidates = [
        CandidateInput("agx", _job(0.010), "j1", _entry("AGX Orin", 5, 1229)),
        CandidateInput("nano", _job(0.010), "j2", _entry("Orin Nano", 3, 330)),
    ]
    rec = rank_candidates(
        candidates,
        factor=OptimizationFactor.COST,
        latency_metric=LatencyPercentile.P95,
        latency_threshold_ms=50,
    )
    assert rec.winner_hostname == "nano"
    assert rec.candidates[0].hostname == "nano"  # survivors ranked first


def test_latency_constraint_excludes_too_slow_device():
    # cheap-but-slow (200 ms) is excluded; expensive-but-fast (10 ms) wins.
    candidates = [
        CandidateInput("cheap_slow", _job(0.200), "j1", _entry("Orin Nano", 3, 330)),
        CandidateInput("pricey_fast", _job(0.010), "j2", _entry("AGX Orin", 5, 1229)),
    ]
    rec = rank_candidates(
        candidates,
        factor=OptimizationFactor.COST,
        latency_metric=LatencyPercentile.P95,
        latency_threshold_ms=50,
    )
    assert rec.winner_hostname == "pricey_fast"
    excluded = {c.hostname: c for c in rec.candidates if not c.meets_constraint}
    assert "cheap_slow" in excluded
    assert "exceeds threshold" in excluded["cheap_slow"].excluded_reason


def test_energy_factor_picks_lowest_energy():
    # Same latency, different constant power => lower-power device wins on energy.
    candidates = [
        CandidateInput(
            "hungry", _job(0.010, power_mw=4000), "j1", _entry("AGX Orin", 5, 1229)
        ),
        CandidateInput(
            "frugal", _job(0.010, power_mw=1000), "j2", _entry("Orin Nano", 3, 330)
        ),
    ]
    rec = rank_candidates(
        candidates,
        factor=OptimizationFactor.ENERGY,
        latency_metric=LatencyPercentile.P95,
        latency_threshold_ms=50,
    )
    assert rec.winner_hostname == "frugal"
    winner = rec.candidates[0]
    assert winner.energy_joules is not None and winner.energy_joules > 0


def test_latency_factor_picks_fastest():
    candidates = [
        CandidateInput("slow", _job(0.030), "j1", _entry("Orin Nano", 3, 330)),
        CandidateInput("fast", _job(0.005), "j2", _entry("AGX Orin", 5, 1229)),
    ]
    rec = rank_candidates(
        candidates,
        factor=OptimizationFactor.LATENCY,
        latency_metric=LatencyPercentile.AVG,
        latency_threshold_ms=50,
    )
    assert rec.winner_hostname == "fast"


def test_cost_factor_excludes_device_without_catalog_entry():
    candidates = [
        CandidateInput("unknown", _job(0.010), "j1", catalog_entry=None),
    ]
    rec = rank_candidates(
        candidates,
        factor=OptimizationFactor.COST,
        latency_metric=LatencyPercentile.P95,
        latency_threshold_ms=50,
    )
    assert rec.winner_hostname is None
    assert "no catalog cost" in rec.candidates[0].excluded_reason


def test_failed_device_listed_with_reason():
    candidates = [
        CandidateInput("good", _job(0.010), "j1", _entry("Orin Nano", 3, 330)),
        CandidateInput("broken", benchmark_job=None, error="benchmark failed: boom"),
    ]
    rec = rank_candidates(
        candidates,
        factor=OptimizationFactor.COST,
        latency_metric=LatencyPercentile.P95,
        latency_threshold_ms=50,
    )
    assert rec.winner_hostname == "good"
    broken = next(c for c in rec.candidates if c.hostname == "broken")
    assert broken.meets_constraint is False
    assert broken.excluded_reason == "benchmark failed: boom"


def test_no_survivor_returns_none_winner_but_lists_all():
    candidates = [
        CandidateInput("a", _job(0.200), "j1", _entry("Orin Nano", 3, 330)),
        CandidateInput("b", _job(0.300), "j2", _entry("AGX Orin", 5, 1229)),
    ]
    rec = rank_candidates(
        candidates,
        factor=OptimizationFactor.COST,
        latency_metric=LatencyPercentile.P95,
        latency_threshold_ms=50,
    )
    assert rec.winner_hostname is None
    assert len(rec.candidates) == 2
    assert all(not c.meets_constraint for c in rec.candidates)
