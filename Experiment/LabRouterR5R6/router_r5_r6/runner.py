"""Hardware execution, raw counts and resumable per-circuit records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BINARY_KEYS = tuple(f"P{index:04b}" for index in range(16))


def batch_options(job: dict[str, Any], bits: list[str], shots: int) -> dict[str, Any]:
    name = "RouterR5R6Circuit"
    return {name: {
        "meta": {"exp_class_name": name},
        "context_options": {"name": "quantum_circuit", "readout_type": "01",
                            "physical_unit": bits[0]},
        "options_for_regular_exec": {
            "experiment_options": {"bits": bits, "operations": job["operations"],
                                   "job_id": job["id"], "repeat": shots, "loop": 1,
                                   "plot_iq": False, "save_result": False},
            "analysis_options": {"is_plot": False},
        },
    }}


def extract_binary_counts(batch: Any, shots: int) -> tuple[dict[str, int], dict[str, float]]:
    records = list(batch.batch_records.values())
    if not records:
        raise RuntimeError("BatchRunner returned no record.")
    record = records[-1]
    if not record.get("is_pass"):
        raise RuntimeError(f"Hardware batch failed: {record.get('fail_reason', 'unknown')}.")
    candidates = [value.get("experiment_data", {})
                  for value in record.get("analysis_data", {}).values() if isinstance(value, dict)]
    raw = next((item for item in candidates if all(key in item for key in BINARY_KEYS)), None)
    if raw is None:
        raise RuntimeError("Expected joint P0000..P1111 probabilities in batch analysis.")
    probabilities = {key: float(raw[key][0]) for key in BINARY_KEYS}
    if abs(sum(probabilities.values()) - 1) > 1e-4:
        raise RuntimeError("Binary probabilities do not normalize.")
    counts = {key: round(shots * value) for key, value in probabilities.items()}
    counts[max(counts, key=lambda key: probabilities[key])] += shots - sum(counts.values())
    if any(value < 0 for value in counts.values()) or sum(counts.values()) != shots:
        raise RuntimeError("Invalid reconstructed count vector.")
    return counts, probabilities


def execute_job(job: dict[str, Any], bits: list[str], shots: int,
                directory: Path, backend: Any) -> Path:
    from app.batch_test.dynamic_injection import injection_experiment_libs
    from pyQCat.experiments.batch import BatchRunner

    from .experiment import RouterR5R6Circuit

    injection_experiment_libs(RouterR5R6Circuit)
    params, records = directory / "params", directory / "records"
    params.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)
    parameter_file = params / f"{job['id']}.json"
    parameter_file.write_text(json.dumps(batch_options(job, bits, shots), ensure_ascii=False,
                                         indent=2), encoding="utf-8")
    batch = BatchRunner(backend)
    batch.set_experiment_options(
        param_path=str(parameter_file), flows=["RouterR5R6Circuit"], bits=bits,
        exp_retry=0, record_batch=True, record_experiment_data=True,
        quality_filter=False, use_config_unit=True, unified_dir=False,
        refresh_context=False, use_simulator=False,
    )
    batch.run()
    counts, probabilities = extract_binary_counts(batch, shots)
    payload = {"job": job, "bits": bits, "shots": shots, "counts": counts,
               "probabilities": probabilities,
               "batch_record_path": str(batch.run_options.record_path)}
    target = records / f"{job['id']}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target
