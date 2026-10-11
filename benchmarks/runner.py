import statistics
import time


def percentile(values: list[float], p: float) -> float:
    """Calculate a percentile using linear interpolation."""
    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def benchmark(
    operation,
    warmup_runs: int = 2,
    measured_runs: int = 10,
) -> dict:
    """Measure repeated executions and return timings in milliseconds."""
    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("Invalid run counts")

    for _ in range(warmup_runs):
        operation()

    timings = []

    for _ in range(measured_runs):
        start = time.perf_counter()
        operation()
        elapsed = time.perf_counter() - start
        timings.append(elapsed * 1000)

    return {
        "runs": measured_runs,
        "mean_ms": statistics.mean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "p50_ms": percentile(timings, 0.50),
        "p95_ms": percentile(timings, 0.95),
        "p99_ms": percentile(timings, 0.99),
    }
