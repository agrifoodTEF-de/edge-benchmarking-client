#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for energy estimation from power telemetry (no devices/server)."""

from edge_benchmarking_client.energy import (
    compute_energy_joules,
    average_power_watts,
    total_power_watts,
)


def _timeline(seconds: list[int]) -> list[str]:
    # Build ISO timestamps spaced by the given second offsets.
    return [f"2026-06-29T12:00:{s:02d}" for s in seconds]


def test_constant_power_trapezoidal_energy():
    # Two rails at 1000 mW each => 2 W total, held for 3 s => 6 J.
    results = {
        "time": _timeline([0, 1, 2, 3]),
        "Power_CPU": [1000, 1000, 1000, 1000],
        "Power_GPU": [1000, 1000, 1000, 1000],
    }
    assert compute_energy_joules(results) == 6.0
    assert average_power_watts(results) == 2.0


def test_milliwatt_to_watt_conversion():
    results = {
        "time": _timeline([0, 2]),
        "Power_TOT": [5000, 5000],  # 5 W
    }
    # 5 W over 2 s = 10 J
    assert compute_energy_joules(results) == 10.0


def test_total_power_sums_only_power_keys():
    results = {
        "time": _timeline([0, 1]),
        "Power_CPU": [1000, 1000],
        "Power_GPU": [2000, 2000],
        "Temp_CPU": [40, 41],  # must be ignored
        "GPU": [50, 60],  # utilization, not power
    }
    assert total_power_watts(results) == [3.0, 3.0]


def test_negative_readings_replaced_with_mean():
    # Mean of the non-negative samples (1000, 3000) is 2000 mW -> replaces -1.
    results = {
        "time": _timeline([0, 1, 2]),
        "Power_CPU": [1000, -1, 3000],
    }
    assert total_power_watts(results) == [1.0, 2.0, 3.0]


def test_none_when_no_power_rails():
    results = {"time": _timeline([0, 1]), "Temp_CPU": [40, 41]}
    assert compute_energy_joules(results) is None
    assert average_power_watts(results) is None


def test_none_when_insufficient_samples():
    results = {"time": _timeline([0]), "Power_CPU": [1000]}
    assert compute_energy_joules(results) is None


def test_none_when_time_unparsable():
    results = {"time": ["not-a-date", "also-bad"], "Power_CPU": [1000, 1000]}
    assert compute_energy_joules(results) is None
