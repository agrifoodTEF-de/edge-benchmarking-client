#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for EdgeBenchmarkingClient.recommend_device with the network
methods mocked (no live Edge-Farm)."""

import pytest

from edge_benchmarking_client.client import EdgeBenchmarkingClient
from edge_benchmarking_types.edge_device.enums import JobStatus
from edge_benchmarking_types.edge_device.models import BenchmarkJob
from edge_benchmarking_types.edge_farm.enums import (
    OptimizationFactor,
    LatencyPercentile,
)
from edge_benchmarking_types.edge_farm.models import (
    DeviceCatalogEntry,
    TritonDenseNetClient,
    BenchmarkInferResult,
    InferPerformance,
    PerformanceResult,
    Latency,
)


def _job(job_id: str, latency_s: float) -> BenchmarkJob:
    perf = PerformanceResult(
        total_time=latency_s,
        sample_count=1,
        samples_per_second=1.0 / latency_s,
        latency=Latency(average=latency_s, percentiles={95: latency_s, 99: latency_s}),
    )
    return BenchmarkJob(
        id=job_id,
        benchmark_results={
            "time": ["2026-06-29T12:00:00", "2026-06-29T12:00:01"],
            "Power_TOT": [1000, 1000],
        },
        inference_results=BenchmarkInferResult(
            performance=InferPerformance(
                preprocess=perf, inference=perf, postprocess=perf
            ),
            results={},
        ),
        status=JobStatus.SUCCESS,
    )


@pytest.fixture
def client(monkeypatch):
    # Bypass __init__ (which performs a live connection test).
    c = EdgeBenchmarkingClient.__new__(EdgeBenchmarkingClient)

    catalog = [
        DeviceCatalogEntry(gpu_model="Orin Nano", tier_rank=3, cost_eur=330),
        DeviceCatalogEntry(gpu_model="AGX Orin", tier_rank=5, cost_eur=1229),
    ]
    monkeypatch.setattr(c, "get_device_catalog", lambda: catalog)

    gpu_by_host = {"nano": "NVIDIA Jetson Orin Nano", "agx": "NVIDIA Jetson AGX Orin"}
    monkeypatch.setattr(c, "_device_gpu_model", lambda h: gpu_by_host.get(h))

    # Each host benchmarks with a fixed latency; agx faster but pricier.
    latency_by_host = {"nano": 0.010, "agx": 0.005}

    def fake_benchmark(*, edge_device, **kwargs):
        return _job(f"job-{edge_device}", latency_by_host[edge_device])

    monkeypatch.setattr(c, "benchmark", fake_benchmark)
    return c


def _inference_client():
    return TritonDenseNetClient(host="placeholder", model_name="densenet")


def test_recommend_cost_prefers_cheaper_when_both_compliant(client):
    rec = client.recommend_device(
        model=("m.onnx", None),
        dataset=[],
        inference_client=_inference_client(),
        candidate_devices=["nano", "agx"],
        factor=OptimizationFactor.COST,
        latency_threshold_ms=50,
        latency_metric=LatencyPercentile.P95,
    )
    assert rec.winner_hostname == "nano"
    assert {c.hostname for c in rec.candidates} == {"nano", "agx"}


def test_recommend_latency_prefers_faster(client):
    rec = client.recommend_device(
        model=("m.onnx", None),
        dataset=[],
        inference_client=_inference_client(),
        candidate_devices=["nano", "agx"],
        factor=OptimizationFactor.LATENCY,
        latency_threshold_ms=50,
    )
    assert rec.winner_hostname == "agx"


def test_recommend_handles_device_failure(client, monkeypatch):
    def flaky_benchmark(*, edge_device, **kwargs):
        if edge_device == "agx":
            raise RuntimeError("Triton unreachable")
        return _job("job-nano", 0.010)

    monkeypatch.setattr(client, "benchmark", flaky_benchmark)
    rec = client.recommend_device(
        model=("m.onnx", None),
        dataset=[],
        inference_client=_inference_client(),
        candidate_devices=["nano", "agx"],
        factor=OptimizationFactor.COST,
        latency_threshold_ms=50,
    )
    assert rec.winner_hostname == "nano"
    agx = next(c for c in rec.candidates if c.hostname == "agx")
    assert not agx.meets_constraint
    assert "benchmark failed" in agx.excluded_reason
