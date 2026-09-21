#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Failure reporting for benchmark jobs.

Unlike ``test_client.py`` these need no edge device: the two HTTP calls involved are
stubbed, so the assertions are about what the client does with a failed job rather than
about any particular hardware.

The behaviour under test matters because the runner container is removed as soon as a job
finishes. ``BenchmarkJob.error`` is the only surviving copy of its traceback, so a client
that discards it leaves every edge-benchmark failure unexplained.
"""

import pytest

from edge_benchmarking_client.client import (
    BenchmarkJobFailedError,
    EdgeBenchmarkingClient,
)
from edge_benchmarking_types.edge_device.models import BenchmarkJob, BenchmarkJobError

JOB_ID = "89f0fecf-3467-4857-b875-184330fc63d0"

DEVICE_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/code/edge_inference_clients/runner/main.py", line 271, in run\n'
    "    inference_results = client()\n"
    '  File "/code/edge_inference_clients/triton/densenet/client.py", line 131, in parse_model\n'
    "    assert len(input_metadata.shape) == expected_input_dims\n"
    "AssertionError: Expecting input to have 3 dimensions, model 'x' input has 4."
)


class _Client(EdgeBenchmarkingClient):
    """Bypasses __init__ so no network or credentials are needed."""

    def __init__(
        self, status: str, job: BenchmarkJob | None, lookup_error=None
    ) -> None:
        self._status = status
        self._job = job
        self._lookup_error = lookup_error
        self.job_lookups = 0

    def get_benchmark_job_status(self, job_id: str) -> dict:
        return {"status": self._status}

    def get_benchmark_job(self, job_id: str) -> BenchmarkJob:
        self.job_lookups += 1
        if self._lookup_error is not None:
            raise self._lookup_error
        return self._job


def _job(error: BenchmarkJobError | None) -> BenchmarkJob:
    return BenchmarkJob(
        id=JOB_ID,
        benchmark_results={},
        inference_results=None,
        status="failed",
        error=error,
    )


def test_failure_surfaces_device_traceback() -> None:
    client = _Client(
        "failed",
        _job(
            BenchmarkJobError(
                message="Runner exited with code 1.", traceback=DEVICE_TRACEBACK
            )
        ),
    )

    with pytest.raises(BenchmarkJobFailedError) as caught:
        client.get_benchmark_job_results(job_id=JOB_ID)

    text = str(caught.value)
    assert JOB_ID in text
    assert "failed" in text
    assert "Runner exited with code 1." in text
    assert (
        "Expecting input to have 3 dimensions" in text
    ), "device traceback was dropped"
    assert client.job_lookups == 1


def test_failure_exposes_structured_error() -> None:
    error = BenchmarkJobError(message="boom", traceback=DEVICE_TRACEBACK)
    client = _Client("failed", _job(error))

    with pytest.raises(BenchmarkJobFailedError) as caught:
        client.get_benchmark_job_results(job_id=JOB_ID)

    assert caught.value.job_id == JOB_ID
    assert caught.value.status == "failed"
    assert caught.value.error.traceback == DEVICE_TRACEBACK


def test_failure_without_detail_is_still_clear() -> None:
    client = _Client("failed", _job(None))

    with pytest.raises(BenchmarkJobFailedError) as caught:
        client.get_benchmark_job_results(job_id=JOB_ID)

    assert "no error detail" in str(caught.value)


def test_job_lookup_failure_does_not_mask_the_original() -> None:
    """A deleted job, or a network fault, must not turn into a confusing second error."""
    client = _Client("failed", None, lookup_error=ConnectionError("connection refused"))

    with pytest.raises(BenchmarkJobFailedError) as caught:
        client.get_benchmark_job_results(job_id=JOB_ID)

    text = str(caught.value)
    assert "returned unexpected status 'failed'" in text
    assert "could not be retrieved" in text
    assert "connection refused" in text


def test_still_a_runtime_error_for_existing_handlers() -> None:
    client = _Client("failed", _job(None))
    with pytest.raises(RuntimeError):
        client.get_benchmark_job_results(job_id=JOB_ID)


def test_success_is_unaffected() -> None:
    job = BenchmarkJob(
        id=JOB_ID,
        benchmark_results={"latency": [1.0]},
        inference_results=None,
        status="success",
        error=None,
    )
    client = _Client("success", job)

    assert client.get_benchmark_job_results(job_id=JOB_ID) is job
