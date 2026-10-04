import copy

from leanlean import Benchmark
from leanlean.benchmarks.leanlean import LeanLeanBenchmark


def get_benchmark_class(spec: str) -> type[Benchmark]:
    if spec != "leanlean":
        raise ValueError(
            f"Unknown benchmark type: {spec!r} (available: leanlean)"
        )
    return LeanLeanBenchmark


def get_benchmark(benchmark_config: dict) -> Benchmark:
    config = copy.deepcopy(benchmark_config)
    benchmark_class = config.pop("benchmark_class", None)
    assert benchmark_class is not None, "benchmark_class must be specified"
    return get_benchmark_class(benchmark_class)(**config)
