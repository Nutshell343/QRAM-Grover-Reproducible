"""Plan, run, resume and offline-analyze R5/R6 down/up router experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import analysis, protocol
from .runner import BINARY_KEYS, execute_job

PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_FILES = ("protocol.py", "analysis.py", "experiment.py", "runner.py",
                "workflow.py", "plots.py", "diagram.py")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((PACKAGE_DIR / name).read_bytes())
    return digest.hexdigest()


def root_of(output_root: str, scheme: str, direction: str, run_name: str) -> Path:
    return Path(output_root).resolve() / scheme / direction / run_name


def stages(scheme: str, direction: str, mode: str, reuse_readout: bool,
           reuse_paired: bool, have_down_preparation: bool, query: bool) -> list[str]:
    result = [] if reuse_readout else ["readout_local"]
    if scheme == "r5" and not reuse_paired:
        result += ["paired"]
    result += ["truth_pilot", "ramsey_raw", "ramsey_dither", "ramsey_corrected", "truth"]
    if mode in ("bound", "both"):
        if direction == "down":
            result += ["fidelity_bound"]
        elif have_down_preparation:
            result += ["roundtrip_bound"]
    if mode in ("direct", "both"):
        if direction == "down":
            result += ["fidelity_direct"]
        elif have_down_preparation:
            result += ["roundtrip_direct"]
    if direction == "up" and have_down_preparation and query and mode != "pilot":
        result += ["query"]
    return result


def stage_count(stage: str, direction: str) -> int:
    if stage == "readout_local":
        return 36
    if stage == "paired":
        return 972
    if stage in ("truth_pilot", "truth"):
        return 324
    if stage in ("ramsey_raw", "ramsey_dither", "ramsey_corrected"):
        return 48
    if stage == "query":
        return 1296
    if stage.endswith(("bound", "direct")):
        return 81 * protocol.fidelity_core_count(stage, direction)
    raise ValueError(stage)


def plan(scheme: str, direction: str, mode: str = "bound", reuse_readout: bool = False,
         reuse_paired: bool = False, have_down_preparation: bool = False,
         query: bool = True) -> dict[str, Any]:
    if scheme not in protocol.SCHEMES or direction not in protocol.DIRECTIONS:
        raise ValueError("Choose --scheme r5|r6 and --direction down|up.")
    selected = stages(scheme, direction, mode, reuse_readout, reuse_paired,
                      have_down_preparation, query)
    counts = {stage: stage_count(stage, direction) for stage in selected}
    return {"scheme": scheme, "direction": direction, "mode": mode,
            "stages": selected, "jobs_per_stage": counts,
            "total_circuits": sum(counts.values()), "shots_per_circuit_default": 1000,
            "up_coherence_scope": "down-prepared roundtrip, not isolated up process"
            if direction == "up" else None}


def stage_jobs(stage: str, scheme: str, direction: str,
               paired: Mapping[str, Any] | None, angles: Mapping[str, float],
               down_preparation: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if stage == "readout_local":
        return list(protocol.readout_jobs())
    if stage == "paired":
        return list(protocol.paired_jobs(direction))
    if stage in ("truth_pilot", "truth"):
        return list(protocol.truth_jobs(stage, scheme, direction, paired,
                                        {} if stage == "truth_pilot" else angles))
    if stage in ("ramsey_raw", "ramsey_dither", "ramsey_corrected"):
        return list(protocol.ramsey_jobs(stage, scheme, direction, paired,
                                         angles if stage == "ramsey_corrected" else {}))
    if stage.endswith(("bound", "direct")):
        return list(protocol.fidelity_jobs(stage, scheme, direction, paired, angles,
                                           down_preparation))
    if stage == "query":
        if down_preparation is None:
            raise ValueError("Query requires calibrated down parameters.")
        return list(protocol.query_jobs(stage, scheme, down_preparation,
                                        {"paired": paired, "angles": angles}))
    raise ValueError(stage)


def check_configuration(path: Path, expected: dict[str, Any]) -> None:
    if path.exists():
        if read_json(path) != expected:
            raise RuntimeError(f"Immutable run configuration changed: {path}; use a new --run-name.")
    elif path.parent.exists() and any(path.parent.iterdir()):
        raise RuntimeError(f"Nonempty run directory has no configuration: {path.parent}.")
    else:
        write_json(path, expected)


def check_record(path: Path, job: dict[str, Any], bits: list[str], shots: int) -> None:
    value = read_json(path)
    if value.get("job") != job or value.get("bits") != bits or value.get("shots") != shots:
        raise RuntimeError(f"Existing record differs from current manifest: {path}")
    counts = value.get("counts", {})
    if set(counts) != set(BINARY_KEYS) or sum(counts.values()) != shots:
        raise RuntimeError(f"Incomplete existing record: {path}")


def acquire(stage: str, jobs: list[dict[str, Any]], output: Path,
            bits: list[str], shots: int, seed: int, backend: Any) -> None:
    ordered = list(jobs)
    random.Random(f"{seed}:{stage}").shuffle(ordered)
    manifest = {"stage": stage, "count": len(jobs), "seed": seed,
                "ordered_job_ids": [item["id"] for item in ordered]}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise RuntimeError(f"Stage manifest changed: {manifest_path}")
    write_json(manifest_path, manifest)
    print(f"{stage}: {len(ordered)} circuits, {shots} shots each", flush=True)
    for index, job in enumerate(ordered, start=1):
        path = output / "records" / f"{job['id']}.json"
        if path.exists():
            check_record(path, job, bits, shots)
        else:
            execute_job(job, bits, shots, output, backend)
        if index % 25 == 0 or index == len(ordered):
            print(f"{stage}: {index}/{len(ordered)}", flush=True)


def report_stage(stage: str, directory: Path, calibration: dict[str, Any],
                 direction: str) -> dict[str, Any]:
    if stage == "readout_local":
        return analysis.fit_readout(directory)
    if stage == "paired":
        return analysis.paired_report(directory, calibration, direction)
    if stage in ("truth_pilot", "truth"):
        return analysis.truth_report(directory, calibration, stage, direction)
    if stage in ("ramsey_raw", "ramsey_dither", "ramsey_corrected"):
        return analysis.ramsey_report(directory, stage, direction)
    if stage.endswith(("bound", "direct")):
        return analysis.fidelity_report(directory, calibration, stage, direction)
    if stage == "query":
        return analysis.query_report(directory, calibration)
    raise ValueError(stage)


def validate_down_preparation(path: Path, scheme: str, bits: list[str]) -> dict[str, Any]:
    value = read_json(path)
    if value.get("scheme") != scheme or value.get("direction") != "down" or value.get("bits") != bits:
        raise ValueError("Down preparation must be the matched scheme, direction and c,i,l,r mapping.")
    if scheme == "r5" and value.get("paired") is None:
        raise ValueError("R5 down preparation is missing calibrated V.")
    return value


def run(args: argparse.Namespace, backend: Any) -> Path:
    if args.scheme not in protocol.SCHEMES or args.direction not in protocol.DIRECTIONS:
        raise ValueError("Invalid scheme/direction.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise ValueError("--run-name accepts letters, digits, underscore and hyphen only.")
    if args.shots <= 0 or len(args.bits) != 4 or len(set(args.bits)) != 4:
        raise ValueError("Supply four distinct --bits and positive --shots.")
    root = root_of(args.output_root, args.scheme, args.direction, args.run_name)
    readout_path = Path(args.readout_calibration).resolve() if args.readout_calibration else None
    paired_path = Path(args.paired_parameters).resolve() if args.paired_parameters else None
    down_path = Path(args.down_parameters).resolve() if args.down_parameters else None
    if args.direction == "down" and down_path is not None:
        raise ValueError("--down-parameters is only for an up run.")
    if args.scheme == "r6" and paired_path is not None:
        raise ValueError("R6 must not import a paired V.")
    if args.direction == "up" and down_path is None:
        print("WARNING: no down preparation; up basis/Ramsey only, no roundtrip/query fidelity.")
    down = validate_down_preparation(down_path, args.scheme, args.bits) if down_path else None
    config = {"task": "router_r5_r6", "scheme": args.scheme, "direction": args.direction,
              "mode": args.mode, "bits": list(args.bits), "shots": args.shots, "seed": args.seed,
              "run_name": args.run_name, "query": args.query,
              "readout_file": str(readout_path) if readout_path else None,
              "readout_sha256": sha256(readout_path) if readout_path else None,
              "paired_file": str(paired_path) if paired_path else None,
              "paired_sha256": sha256(paired_path) if paired_path else None,
              "down_file": str(down_path) if down_path else None,
              "down_sha256": sha256(down_path) if down_path else None,
              "source_sha256": source_hash()}
    check_configuration(root / "run_config.json", config)
    if readout_path:
        calibration = read_json(readout_path)
        if calibration.get("bits") != args.bits:
            raise ValueError("Readout calibration belongs to different physical bits.")
        write_json(root / "frozen" / "readout_calibration.json", calibration)
    else:
        calibration = {}
    if paired_path:
        paired = read_json(paired_path).get("paired")
        protocol.paired_matrix(paired)
        write_json(root / "frozen" / "paired_parameters.json", read_json(paired_path))
    else:
        paired = None
    if down_path:
        write_json(root / "frozen" / "down_parameters.json", down)
    selected = stages(args.scheme, args.direction, args.mode, bool(readout_path),
                      bool(paired_path), bool(down_path), args.query)
    angles: dict[str, float] = {}
    status: dict[str, Any] = {"state": "running", "stages": selected,
                              "completed": [], "warnings": {}, "figures": {}}
    try:
        from .diagram import export_circuit
        export_circuit(root / "circuits", args.scheme, args.direction,
                       paired if paired is not None else None)
        for stage in selected:
            status["current_stage"] = stage
            write_json(root / "run_status.json", status)
            jobs = stage_jobs(stage, args.scheme, args.direction, paired, angles, down)
            if len(jobs) != stage_count(stage, args.direction):
                raise AssertionError(f"{stage} generator made {len(jobs)}; expected {stage_count(stage, args.direction)}.")
            stage_dir = root / "stages" / stage
            acquire(stage, jobs, stage_dir, args.bits, args.shots, args.seed, backend)
            if stage != "readout_local" and not calibration:
                raise RuntimeError("No readout calibration available for analysis.")
            report = report_stage(stage, stage_dir / "records", calibration, args.direction)
            write_json(root / "reports" / f"{stage}.json", report)
            if stage == "readout_local":
                calibration = report
                write_json(root / "readout_calibration.json", report)
                if any(x > 100 for x in report["local_condition"]):
                    status["warnings"][stage] = "Readout condition >100; corrected estimates may be noisy."
            elif stage == "paired":
                paired = report["paired"]
                write_json(root / "paired_parameters.json", {"scheme": args.scheme,
                           "direction": args.direction, "bits": args.bits,
                           "paired": paired, "diagnostics": report})
                export_circuit(root / "circuits", args.scheme, args.direction, paired)
                if (report["conditional_survival_linear"] < 0.7
                        or report["bloch_length"] < 0.8
                        or report["maximum_unitary_bloch_residual"] > 0.2):
                    status["warnings"][stage] = "Auxiliary transfer has loss or mixing; V cannot repair it."
            elif stage == "truth_pilot" and report["p_min"] < 0.5:
                status["warnings"][stage] = "Low pilot population; phase correction alone will not fix it."
            elif stage == "ramsey_dither":
                raw = read_json(root / "reports" / "ramsey_raw.json")
                try:
                    fitted = analysis.endpoint_calibration(raw, report)
                    angles = fitted["angles"]
                except ValueError as error:
                    angles = {"c": 0.0, "l": 0.0, "r": 0.0}
                    fitted = {"angles": angles, "fit_error": str(error)}
                    status["warnings"][stage] = str(error)
                write_json(root / "endpoint_phase_calibration.json", fitted)
            elif stage == "ramsey_corrected" and any(
                abs(math.remainder(fit["phase_rad"], 2 * math.pi)) > 0.3
                for fit in report["pairs"]
            ):
                status["warnings"][stage] = "Residual phase >0.3 rad; keep all data for diagnosis."
            try:
                from .plots import render_stage
                paths = render_stage(stage, report, root / "figures")
                status["figures"][stage] = [str(path) for path in paths]
            except Exception as error:
                status["warnings"][f"{stage}_plot"] = str(error)
            status["completed"].append(stage)
            write_json(root / "run_status.json", status)
        parameters = {"scheme": args.scheme, "direction": args.direction,
                      "bits": list(args.bits), "paired": paired, "angles": angles,
                      "frame_position": "output c/l/r" if args.direction == "down" else "input c/l/r",
                      "readout_file": str(root / "readout_calibration.json") if not readout_path
                      else str(root / "frozen" / "readout_calibration.json"),
                      "run_config_sha256": sha256(root / "run_config.json"),
                      "completed_utc": datetime.now(timezone.utc).isoformat(),
                      "up_coherence_scope": "matched down-prepared roundtrip, not isolated up fidelity"
                      if args.direction == "up" and down else None}
        write_json(root / "parameters.json", parameters)
        status["state"] = "complete"
        status.pop("current_stage", None)
        write_json(root / "run_status.json", status)
        return root
    except Exception as error:
        status["state"] = "stopped"
        status["reason"] = str(error)
        write_json(root / "run_status.json", status)
        raise


def analyze_run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(root / "run_config.json")
    calibration_path = root / "readout_calibration.json"
    if not calibration_path.exists():
        calibration_path = root / "frozen" / "readout_calibration.json"
    calibration = read_json(calibration_path) if calibration_path.exists() else {}
    selected = stages(config["scheme"], config["direction"], config["mode"],
                      bool(config.get("readout_file")), bool(config.get("paired_file")),
                      bool(config.get("down_file")), config.get("query", True))
    completed, skipped = [], {}
    for stage in selected:
        record_dir = root / "stages" / stage / "records"
        count = len(list(record_dir.glob(f"{stage}__r5r6__*.json"))) if record_dir.exists() else 0
        expected = stage_count(stage, config["direction"])
        if count != expected:
            skipped[stage] = f"{count}/{expected} records"
            continue
        if stage != "readout_local" and not calibration:
            skipped[stage] = "readout calibration unavailable"
            continue
        report = report_stage(stage, record_dir, calibration, config["direction"])
        if stage == "readout_local":
            calibration = report
            write_json(root / "readout_calibration.json", calibration)
        write_json(root / "reports" / f"{stage}.json", report)
        try:
            from .plots import render_stage
            render_stage(stage, report, root / "figures")
        except Exception as error:
            skipped[f"{stage}_figure"] = str(error)
        completed.append(stage)
    result = {"run": str(root), "completed": completed, "skipped": skipped,
              "analyzed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(root / "reanalysis_status.json", result)
    return result


def summarize_suite(run_roots: list[Path], output_root: Path, run_name: str) -> Path:
    """Compare four independent runs without conflating up and down fidelity."""
    rows = []
    for root in run_roots:
        parameters = read_json(root / "parameters.json")
        status = read_json(root / "run_status.json")
        row: dict[str, Any] = {"scheme": parameters["scheme"],
                               "direction": parameters["direction"],
                               "run_directory": str(root), "warnings": status.get("warnings", {})}
        for stage in ("paired", "truth", "fidelity_bound", "fidelity_direct",
                      "roundtrip_bound", "roundtrip_direct", "query"):
            path = root / "reports" / f"{stage}.json"
            if not path.exists():
                continue
            report = read_json(path)
            row[stage] = {key: report[key] for key in (
                "F_Z", "p_min", "L", "F_e_lower_linear", "F_e_upper_linear",
                "F_e_linear", "F_avg_linear", "mean_correct", "minimum_correct",
                "conditional_survival_linear", "maximum_unitary_bloch_residual")
                if key in report}
        rows.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / f"{run_name}_suite_summary.json"
    write_json(target, {"scope": "R5/R6 down and physically implemented up",
                        "up_note": "up coherent fidelity uses matched down preparation",
                        "runs": rows, "completed_utc": datetime.now(timezone.utc).isoformat()})
    try:
        from .plots import suite_plot
        suite_plot(rows, output_root / f"{run_name}_suite_comparison.png")
    except Exception as error:
        print(f"Suite comparison figure not generated: {error}", flush=True)
    return target


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("plan", "run", "run-suite", "analyze", "diagram"))
    cli.add_argument("--scheme", choices=protocol.SCHEMES, default="r6")
    cli.add_argument("--direction", choices=protocol.DIRECTIONS, default="down")
    cli.add_argument("--bits", nargs=4, metavar=("C", "I", "L", "R"))
    cli.add_argument("--mode", choices=("pilot", "bound", "direct", "both"), default="bound")
    cli.add_argument("--shots", type=int, default=1000, help="pyQCat repeat (shots) per circuit")
    cli.add_argument("--run-name", default="router-r5-r6-v1")
    cli.add_argument("--output-root", default=str(PACKAGE_DIR / "router_runs"))
    cli.add_argument("--run-directory", help="For offline analyze")
    cli.add_argument("--readout-calibration", help="Reuse matching four-qutrit readout JSON")
    cli.add_argument("--paired-parameters", help="Reuse direction-specific R5 paired_parameters.json")
    cli.add_argument("--down-parameters", help="Matched down parameters.json for up roundtrip/query")
    cli.add_argument("--seed", type=int, default=20261002)
    cli.add_argument("--no-query", dest="query", action="store_false")
    cli.set_defaults(query=True)
    return cli


def main() -> None:
    args = parser().parse_args()
    if args.action == "plan":
        value = plan(args.scheme, args.direction, args.mode,
                     bool(args.readout_calibration), bool(args.paired_parameters),
                     bool(args.down_parameters), args.query)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return
    if args.action == "analyze":
        if not args.run_directory:
            raise ValueError("Offline analyze needs --run-directory.")
        print(json.dumps(analyze_run(Path(args.run_directory)), ensure_ascii=False, indent=2))
        return
    if args.action == "diagram":
        from .diagram import export_circuit
        target = Path(args.output_root).resolve() / "diagrams"
        paths = [export_circuit(target, scheme, direction)
                 for scheme in protocol.SCHEMES for direction in protocol.DIRECTIONS]
        print(json.dumps(paths, ensure_ascii=False, indent=2))
        return
    if not args.bits:
        raise ValueError("Hardware run requires --bits C I L R.")
    from app.config import init_backend
    backend = init_backend()
    if args.action == "run":
        print(run(args, backend))
        return
    shared_readout = args.readout_calibration
    completed = []
    for scheme in ("r6", "r5"):
        down_args = argparse.Namespace(**vars(args))
        down_args.scheme, down_args.direction = scheme, "down"
        down_args.readout_calibration = shared_readout
        down_args.down_parameters = None
        down_args.paired_parameters = None
        down_root = run(down_args, backend)
        completed.append(str(down_root))
        if shared_readout is None:
            shared_readout = str(down_root / "readout_calibration.json")
        up_args = argparse.Namespace(**vars(args))
        up_args.scheme, up_args.direction = scheme, "up"
        up_args.readout_calibration = shared_readout
        up_args.down_parameters = str(down_root / "parameters.json")
        up_args.paired_parameters = None
        up_root = run(up_args, backend)
        completed.append(str(up_root))
    summary = summarize_suite([Path(path) for path in completed],
                              Path(args.output_root).resolve(), args.run_name)
    print(json.dumps({"completed_runs": completed, "suite_summary": str(summary)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
