PLACEHOLDER = ""
LEANLEAN_CONFIG = {
    "benchmark_class": "leanlean",
    "filter_spec": PLACEHOLDER,
    "slice_spec": PLACEHOLDER,
    "shuffle": False,
    "compile_gate_enabled": False,
    "compile_gate_command": "lake build",
    "compile_gate_max_retries": 1,
    "compile_gate_output_chars": 12000,
    "task_metadata_enabled": True,
    "build_jobs": -1,
}

ALL_BENCHMARK_CONFIGS = {
    "leanlean": LEANLEAN_CONFIG,
}
