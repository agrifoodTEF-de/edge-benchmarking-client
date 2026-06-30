"""Energy estimation from a benchmark job's raw power telemetry.

A benchmark run streams per-rail power readings into
``BenchmarkJob.benchmark_results`` as keys prefixed ``Power`` (values in
**milliwatts**) alongside a parallel ``time`` series of ISO-8601 timestamps —
the same contract the platform UI charts. Total instantaneous power is the sum
of all power rails; energy is that power integrated over the telemetry window.

These helpers are pure (dict in, float out) so they can be unit-tested and
reused without contacting any device or storage.
"""

import logging

from datetime import datetime
from typing import Any, Optional

POWER_KEY_PREFIX = "Power"
TIME_KEY = "time"
_MILLIWATTS_PER_WATT = 1000.0


def _clean_series(series: list[Any]) -> list[float]:
    """Coerce a power series to floats, replacing negative readings (sensor
    glitches) with the mean of the non-negative samples — mirroring how the
    platform UI post-processes these series before charting."""
    values: list[Optional[float]] = []
    for v in series:
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            values.append(None)

    valid = [v for v in values if v is not None and v >= 0]
    mean = sum(valid) / len(valid) if valid else 0.0
    return [mean if (v is None or v < 0) else v for v in values]


def _time_offsets_seconds(timestamps: list[Any]) -> Optional[list[float]]:
    parsed: list[datetime] = []
    for ts in timestamps:
        try:
            parsed.append(datetime.fromisoformat(str(ts)))
        except ValueError:
            return None
    if len(parsed) < 2:
        return None
    start = parsed[0]
    return [(t - start).total_seconds() for t in parsed]


def total_power_watts(benchmark_results: dict) -> Optional[list[float]]:
    """Per-timestep total power draw (watts), summed across all ``Power*`` rails.

    Returns ``None`` if no power rails are present.
    """
    power_keys = [
        k
        for k, v in benchmark_results.items()
        if k.startswith(POWER_KEY_PREFIX) and isinstance(v, list) and v
    ]
    if not power_keys:
        return None

    length = min(len(benchmark_results[k]) for k in power_keys)
    rails = [_clean_series(benchmark_results[k][:length]) for k in power_keys]
    return [
        sum(rail[i] for rail in rails) / _MILLIWATTS_PER_WATT for i in range(length)
    ]


def compute_energy_joules(benchmark_results: dict) -> Optional[float]:
    """Estimate total energy (joules) consumed during a benchmark run.

    Trapezoidal integration of total power (W) over the ``time`` series.
    Returns ``None`` when telemetry is insufficient (no ``Power*`` rails, no
    parsable ``time`` series, or fewer than two samples) so callers can fall
    back instead of reporting a misleading zero.
    """
    power_w = total_power_watts(benchmark_results)
    if power_w is None:
        return None

    offsets = _time_offsets_seconds(benchmark_results.get(TIME_KEY, []))
    if offsets is None:
        logging.warning("No parsable 'time' series; cannot integrate energy.")
        return None

    n = min(len(power_w), len(offsets))
    if n < 2:
        return None

    energy = 0.0
    for i in range(1, n):
        dt = offsets[i] - offsets[i - 1]
        if dt <= 0:
            continue
        energy += 0.5 * (power_w[i] + power_w[i - 1]) * dt
    return energy


def average_power_watts(benchmark_results: dict) -> Optional[float]:
    """Mean total power draw (watts) across the telemetry window."""
    power_w = total_power_watts(benchmark_results)
    if not power_w:
        return None
    return sum(power_w) / len(power_w)
