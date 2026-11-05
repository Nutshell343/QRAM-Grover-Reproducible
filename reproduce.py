"""Portable, simulation-only entry point. Paths are relative to this file."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "Experiment"
STUDY = EXP / "DistributedGroverNoDecoherence20261102"
PULSE = EXP / "RouterPulseCalibration20261002"
BASELINE = STUDY / "feasibility"
READS: set[Path] = set()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path, root=ROOT):
    return Path(path).resolve().is_relative_to(root.resolve())


def audit_open(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    try:
        path = Path(os.fsdecode(args[0])).resolve()
    except (OSError, ValueError):
        return
    if path.suffix.lower() in (".json", ".npz"):
        if inside(path):
            READS.add(path)
        elif path.name in ("parameters.json", "native_parameters.json", "device.json"):
            raise RuntimeError(f"A frozen numerical input escaped the bundle: {path}")


def software_inputs():
    paths = [ROOT / "reproduce.py", ROOT / "requirements.txt"]
    paths += sorted(EXP.rglob("*.py"))
    paths += [PULSE / "device.json", PULSE / "results/native/native_parameters.json"]
    paths += sorted((PULSE / "results/lab_equivalent_runs").rglob("parameters.json"))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths if "__pycache__" not in p.parts}


def runtime_info():
    result = {"python": sys.version, "executable": sys.executable, "platform": sys.platform}
    for name in ("numpy", "scipy", "matplotlib", "threadpoolctl", "Pillow", "pypdf"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def verify_bundle():
    manifest = read(ROOT / "bundle_manifest.json")
    failures = []
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError("Bundle missing or modified files: " + ", ".join(failures[:20]))
    return {"checked_files": len(manifest["sha256"]), "all_hashes_match": True,
            "environment": runtime_info(), "root": str(ROOT)}


def new_run(args):
    name = args.run_name or (args.command + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or ".." in name:
        raise ValueError("run-name must use letters, digits, hyphens, dots or underscores, without '..'.")
    target = ROOT / "runs" / name
    if not inside(target, ROOT / "runs"):
        raise ValueError("Output must stay in this bundle's runs directory.")
    target.mkdir(parents=True, exist_ok=True)
    manifest = {"scope": "Frozen coherent feasibility baseline; no added noise or readout model",
                "input_sha256": software_inputs(), "environment": runtime_info()}
    old = target / "run_manifest.json"
    if old.exists():
        previous = read(old)
        if previous["input_sha256"] != manifest["input_sha256"]:
            raise RuntimeError("Source or calibration changed; use a new run-name.")
        for key in ("numpy", "scipy"):
            if previous["environment"].get(key) != manifest["environment"].get(key):
                raise RuntimeError(f"{key} version changed; use a new run-name.")
    else:
        write(old, manifest)
    print("OUTPUT", target, flush=True)
    return target


def load_model(output=None):
    for folder in (STUDY, EXP / "DistributedGroverSimulation20261102"):
        sys.path.insert(0, str(folder))
    fd = importlib.import_module("feasibility_data")
    if output is not None:
        fd.OUT = output / "feasibility"
        fd.RESULTS = output / "results"
        fd.OUT.mkdir(parents=True, exist_ok=True)
        fd.RESULTS.mkdir(parents=True, exist_ok=True)
        for name in ("arrays", "cases", "figures"):
            (fd.OUT / name).mkdir(exist_ok=True)
        fd.archive_report = lambda: None
    return fd


def check_imports():
    names = ("feasibility_data", "model", "circuit", "mps", "runner", "pulse_model", "bootstrap", "validate_extra")
    imported = {}
    for name in names:
        path = Path(sys.modules[name].__file__).resolve()
        if not inside(path):
            raise RuntimeError(f"Project import escaped this bundle: {name} -> {path}")
        imported[name] = path.relative_to(ROOT).as_posix()
    for name in sys.modules:
        if name.startswith(("router_r5_r6", "partial_cswap_vz4")):
            file = getattr(sys.modules[name], "__file__", None)
            if file and not inside(file):
                raise RuntimeError(f"Protocol import escaped this bundle: {name}")
    return imported


def export_circuits(fd, output):
    import numpy as np
    library = fd.lib()
    directory = output / "circuits"
    directory.mkdir(exist_ok=True)
    from router_r5_r6.diagram import export_circuit
    from partial_cswap_vz4.timing_plan import physical_layers
    operator_data = {}
    operator_catalog = {}
    resources = {}
    for scheme in ("r5", "r6"):
        for direction in ("down", "up"):
            parameter = read(PULSE / f"results/lab_equivalent_runs/{scheme}/{direction}/pulse-L4-shots1000/parameters.json")
            operations = fd.p.router(scheme, direction, parameter["paired"], parameter["angles"])
            export_circuit(directory / "nodes", scheme, direction, parameter["paired"])
            write(directory / "nodes" / f"{scheme}_{direction}_full.json",
                  {"wire_order": ["C", "I", "L", "R"], "parameters": parameter,
                   "chronological_operations": operations, "layers_with_frames": physical_layers(operations),
                   "SVG_scope": "core physical gates; input/output endpoint frames are in this full JSON"})
        for depth in (1, 2, 3):
            circuit = fd.NativeCircuit(library, depth, scheme)
            row = circuit.resource()
            row["CZ_depth"] = sum(layer["kind"] == "cz" for layer in circuit.stages)
            row["physical_layers"] = len(circuit.stages)
            resources[f"{depth}_{scheme}"] = row
            write(directory / f"schedule_{depth}_{scheme}.json", circuit.serializable())
            for layer in circuit.stages:
                active = [op for op in layer["ops"] if op["name"] != "frame"]
                kinds = {op["name"] in ("cz", "memory_cz") for op in active}
                wires = [q for op in active for q in op["wires"]]
                assert len(kinds) <= 1 and len(wires) == len(set(wires))
                for op in layer["ops"]:
                    if "u" not in op:
                        continue
                    array = np.ascontiguousarray(op["u"])
                    digest = hashlib.sha256(array.tobytes()).hexdigest()
                    operator_data[digest] = array
                    operator_catalog[digest] = {"npz_key": digest, "shape": list(array.shape), "dtype": str(array.dtype)}
    # Explicit 4-level ideal storage CZ for reconstructing the physical gate plan.
    operator_data["ideal_memory_cz"] = library.cz
    np.savez_compressed(directory / "operators.npz", **operator_data)
    write(directory / "operator_catalog.json", {"effective_operators": operator_catalog,
          "memory_cz": {"physical_matrix_key": "ideal_memory_cz", "conditional_apply": "I4 if m=0; diag(1,-1,1,1) if m=1"}})
    write(directory / "resources.json", resources)
    # Rebuild the reference node/circuit/architecture/waveform figures, not just copy them.
    plot = importlib.import_module("feasibility_plots")
    plot.OUT = directory
    plot.FIG = directory / "figures"
    plot.FIG.mkdir(exist_ok=True)
    plot.CAT = []
    plot.METRICS = {}
    plot.structural()
    write(directory / "figure_catalog.json", plot.CAT)
    (directory / "README.md").write_text(
        "# Rebuilt circuits\n\nFour down/up node SVGs and full JSONs are in nodes/. "
        "SVGs show physical core gates; full JSONs include endpoint frames. "
        "schedule_*.json contains the full 1/2/3-level query; n=2 also includes diffusion. "
        "All effective matrices are in operators.npz, keyed by the schedule SHA256. "
        "memory_cz uses the separate ideal storage matrix/conditional rule. "
        "figures/ contains freshly generated PNG/PDF circuit and waveform plots.\n", encoding="utf8")
    print("CIRCUITS", len(operator_catalog), "effective operators; four node SVGs; six full schedules", flush=True)
    return resources


def ideal_check(fd, output):
    import itertools
    import numpy as np
    checks = importlib.import_module("feasibility_checks")
    library, solutions = checks.ideal_library()
    rows = []
    for scheme in ("r5", "r6"):
        circuit = fd.NativeCircuit(library, 2, scheme)
        for word in itertools.product("01", repeat=4):
            memory = "".join(word)
            mem = np.array(list(map(int, memory)))
            state = fd.MPS.choi(circuit.quantum, circuit.inputs, max_bond=64, tolerance=1e-12)
            rng = np.random.default_rng(9)
            row = {"scheme": scheme, "memory": memory}
            for j, layer in enumerate(circuit.stages):
                mem = fd.evolve_layer(state, mem, layer, library, fd.ZERO, rng, circuit.quantum)
                if j + 1 == circuit.query_stages:
                    _, selected, actual = fd.select_choi(state, circuit.inputs)
                    ideal = np.zeros((8, 8))
                    for a, d in itertools.product(range(4), range(2)):
                        ideal[2*a + (d ^ int(memory[a])), 2*a+d] = 1
                    phase = np.angle(np.trace(ideal.conj().T @ actual))
                    error = float(np.max(abs(actual * np.exp(-1j*phase) - ideal)))
                    row.update(matrix_error=error, p_aux0=float(np.trace(selected).real))
                    assert error < 1e-10 and abs(row["p_aux0"] - 1) < 1e-10
            if memory.count("1") == 1:
                vector = np.kron(np.ones(4)/2, fd.STATES["-"])
                rho = state.copy().reduce_inputs(circuit.inputs, [vector], np.array([1.,0,0,0]))[0][0]
                target = memory.index("1")
                indices = [np.ravel_multi_index((target//2,target%2,d),(4,4,4)) for d in range(4)]
                row["grover_success"] = float(np.trace(rho[np.ix_(indices,indices)]).real)
                assert abs(row["grover_success"] - 1) < 1e-10
            rows.append(row)
        print("IDEAL", scheme, "all 16 databases passed", flush=True)
    result = {"ideal_native_solutions": solutions, "ideal_network": rows,
              "ideal_max_matrix_error": max(r["matrix_error"] for r in rows), "logical": fd.logical()}
    write(output / "ideal_checks.json", result)
    return result


def compare_case(value, arrays, tolerance=1e-6):
    import numpy as np
    key = value["scheme"] + "_" + value["memory"]
    expected = read(BASELINE / "cases" / (key + ".json"))
    metrics = {name: abs(value["process"][name] - expected["process"][name])
               for name in ("Fe", "Fe_aux0_joint", "p_aux0_average", "Favg")}
    for stage in ("query", "grover"):
        for j, row in enumerate(value["rows"][stage]):
            for name in ("F", "F_aux0", "p_aux0", "success", "success_aux0", "joint_correct_aux0"):
                if name in row:
                    metrics[f"{stage}_{j}_{name}"] = abs(row[name] - expected["rows"][stage][j][name])
    with np.load(BASELINE / "arrays" / (key + ".npz")) as baseline:
        matrix_error = float(np.max(abs(arrays["choi"] - baseline["choi"])))
    maximum = max(metrics.values())
    result = {"case": key, "metric_max_abs_error": maximum, "Choi_max_abs_error": matrix_error,
              "tolerance": tolerance, "passed": maximum < tolerance and matrix_error < tolerance}
    if not result["passed"]:
        raise RuntimeError("Frozen baseline comparison failed: " + json.dumps(result))
    return result


def run_case(fd, output, scheme, memory, bond):
    import numpy as np
    if len(memory) != 4 or set(memory) - set("01"):
        raise ValueError("memory must have four binary digits in M00,M01,M10,M11 order")
    job = (scheme, memory, bond)
    started = time.perf_counter()
    value = fd.network(job)
    arrays = value.pop("arrays")
    key = scheme + "_" + memory
    np.savez_compressed(fd.OUT / "arrays" / (key + ".npz"), **arrays)
    write(fd.OUT / "cases" / (key + ".json"), value)
    comparison = compare_case(value, arrays) if bond == 128 else {"case": key, "note": "non-reference bond; no frozen128 equality claim"}
    comparison["elapsed_seconds"] = time.perf_counter() - started
    write(output / (key + "_comparison.json"), comparison)
    print("CASE", key, "Fe", value["process"]["Fe"], comparison, flush=True)
    return comparison


def full_run(fd, output):
    # This destination starts empty. The shipped reference cases are never
    # copied here, so a first full run genuinely recomputes all 32 channels.
    fd.main()
    comparisons = []
    import numpy as np
    for path in sorted((fd.OUT / "cases").glob("*.json")):
        with np.load(fd.OUT / "arrays" / (path.stem + ".npz")) as a:
            comparisons.append(compare_case(read(path), a))
    write(output / "all_case_comparisons.json", comparisons)
    checks = importlib.import_module("feasibility_checks")
    checks.OUT = fd.OUT
    checks.main()
    rebuild_plots(fd.OUT)
    export_circuits(fd, output)
    return {"cases": len(comparisons), "all_baseline_comparisons_passed": True}


def rebuild_plots(data):
    plot = importlib.import_module("feasibility_plots")
    plot.OUT = data
    plot.FIG = data / "figures"
    plot.CAT = []
    plot.METRICS = {}
    plot.main()


def copy_analysis_data(output, data_run):
    source = BASELINE if not data_run else ROOT / "runs" / data_run / "feasibility"
    if not inside(source) or not (source / "SUMMARY.json").exists():
        raise ValueError("A complete data-run with SUMMARY.json is required")
    target = output / "feasibility"
    if source.resolve() != target.resolve():
        shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def report(output, args):
    target = output / "latex_report"
    shutil.copytree(STUDY / "latex_report", target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.aux", "*.log", "*.toc", "*.xdv", "__pycache__", "main.pdf"))
    if args.data_run:
        raise ValueError("Report source is the frozen reference edition. For new results update tables/figures explicitly; do not mix report versions.")
    compiler = args.compiler or shutil.which("tectonic") or shutil.which("xelatex")
    if args.prepare_only:
        return {"TeX_project": str(target), "compiled": False, "reason": "prepare-only requested"}
    if not compiler:
        raise RuntimeError("TeX project prepared. Install XeLaTeX/Tectonic or pass --compiler; numeric simulation does not need TeX.")
    command = [compiler, "--keep-logs", "--keep-intermediates", "main.tex"] if "tectonic" in Path(compiler).stem.lower() else [compiler,"-interaction=nonstopmode","-halt-on-error","main.tex"]
    for _ in range(1 if "tectonic" in Path(compiler).stem.lower() else 2):
        subprocess.run(command, cwd=target, check=True)
    subprocess.run([sys.executable,"-X","utf8",str(target / "finalize_pdf.py")],cwd=target,check=True)
    return {"TeX_project": str(target), "compiled": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify","circuits","ideal","pulse","case","quick","full","plots","report","calibrate"))
    parser.add_argument("--run-name")
    parser.add_argument("--scheme", choices=("r5","r6"), default="r5")
    parser.add_argument("--memory", default="0010")
    parser.add_argument("--bond", type=int, default=128)
    parser.add_argument("--data-run", help="Use an existing completed full run as the plotting source")
    parser.add_argument("--compiler", help="Installed XeLaTeX/Tectonic path, for report only")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare report files without requiring TeX")
    parser.add_argument("--calibrate-only", action="store_true", help="calibrate: only native waveform optimization")
    parser.add_argument("--shots", type=int, default=1000, help="only used by optional shot-based recalibration")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/.matplotlib"))
    sys.addaudithook(audit_open)
    verified = verify_bundle()
    if args.command == "verify":
        write(ROOT / "runs/package_verification.json", verified)
        print(json.dumps(verified, ensure_ascii=False, indent=2))
        return
    output = new_run(args)
    started = time.perf_counter()
    if args.command == "report":
        result = report(output, args)
    elif args.command == "calibrate":
        command = [sys.executable,"-X","utf8",str(PULSE/"run.py"),"--output",str(output/"calibration"),"--shots",str(args.shots),"--mode","bound"]
        if args.calibrate_only:
            command.append("--calibrate-only")
        subprocess.run(command, cwd=ROOT, check=True)
        result = {"new_calibration": str(output/"calibration"), "frozen_library_replaced": False}
    else:
        fd = load_model(output)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            if args.command == "circuits":
                result = export_circuits(fd, output)
            elif args.command == "ideal":
                result = ideal_check(fd, output)
            elif args.command == "pulse":
                fd.local()
                fd.pulse_checks()
                result = {"local": str(fd.OUT/"local.json"), "convergence": str(fd.OUT/"pulse_convergence.json")}
            elif args.command == "case":
                result = run_case(fd, output, args.scheme, args.memory, args.bond)
            elif args.command == "quick":
                export_circuits(fd, output)
                ideal = ideal_check(fd, output)
                fd.local()
                cases = [run_case(fd, output, scheme, "0010", 128) for scheme in ("r5","r6")]
                result = {"ideal_max_error": ideal["ideal_max_matrix_error"], "cases": cases,
                          "local": str(fd.OUT/"local.json")}
            elif args.command == "full":
                result = full_run(fd, output)
            else:
                data = copy_analysis_data(output, args.data_run)
                rebuild_plots(data)
                result = {"figures": len(read(data/"figure_catalog.json")), "data_source": args.data_run or "shipped frozen baseline"}
        result = {"result": result, "project_imports": check_imports()}
    write(output / f"{args.command}_result.json", {"elapsed_seconds": time.perf_counter()-started, **result})
    write(output / "runtime_input_audit.json", {"project_json_npz_accesses": sorted(p.relative_to(ROOT).as_posix() for p in READS),
          "note": "Frozen device/native/direction inputs are forbidden outside this extracted bundle."})
    print("DONE", args.command, "seconds", round(time.perf_counter()-started,2), flush=True)


if __name__ == "__main__":
    main()
