"""Run complete lab-equivalent R5/R6 experiments with a pulse backend.

No hardware entry point or hardware file is modified. Each saved circuit is
executed with full four-transmon waveforms, including preparation and readout
mapping gates, and multinomial binary shots are fed to the ORIGINAL analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from bootstrap import ROOT, protocol as p, workflow, analysis
from pulse_model import PulseModel, save_json, occupation_index, encode
from calibrate import calibrate_native


class PulseExperimentBackend:
    def __init__(self, model, seed, identity):
        self.model, self.seed, self.identity = model, seed, identity
        self.executed = 0

    def execute(self, job, bits, shots, directory, backend):
        state, diagnostics = self.model.evolve(job["operations"])
        probabilities = self.model.binary_probabilities(state)
        parts = Path(directory).parts
        anchor = parts.index("lab_equivalent_runs")
        relative_directory = Path(*parts[anchor:]).as_posix()
        digest = hashlib.sha256(f"{self.seed}:{relative_directory}:{job['id']}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        counts = rng.multinomial(shots, probabilities)
        payload = {"job": job, "bits": bits, "shots": shots,
                   "counts": {f"P{k:04b}": int(v) for k, v in enumerate(counts)},
                   "probabilities": {f"P{k:04b}": float(v) for k, v in enumerate(probabilities)},
                   "pulse_simulation": {"model_sha256": self.identity,
                                        "levels": self.model.levels, "dt_ns": self.model.dt,
                                        "duration_ns": diagnostics["duration_ns"],
                                        "norm_error": diagnostics["norm_error"],
                                        "p_any_level3plus_after_readout_mapping": float(np.sum(abs(state[np.any(self.model.words>2,axis=1)])**2)),
                                        "source": "Hamiltonian pulse simulation; synthetic binary shots"}}
        target = directory / "records" / f"{job['id']}.json"
        save_json(target, payload)
        self.executed += 1
        return target


def exact_metrics(model, operations, sources, targets):
    basis = np.eye(model.dimension, dtype=complex)[:, [occupation_index(x, model.levels) for x in sources]]
    final, trace = model.evolve(operations, basis, trace=True, use_cache=False)
    target_indices = [occupation_index(x, model.levels) for x in targets]
    block = final[target_indices]
    correct = abs(np.diag(block))**2
    survival = np.sum(abs(block)**2, axis=0)
    comp = np.all(model.words < 2, axis=1)
    level3 = np.any(model.words > 2, axis=1)
    leakage = np.sum(abs(final[~comp])**2, axis=0)
    above2 = np.sum(abs(final[level3])**2, axis=0)
    fe = abs(np.trace(block))**2/len(sources)**2
    return {"Fe": float(fe), "Favg_restricted_unconditional": float((len(sources)*fe+np.mean(survival))/(len(sources)+1)),
            "FZ": float(np.mean(correct)), "L": float(np.mean(leakage)),
            "L3plus": float(np.mean(above2)), "per_input": [
                {"source": s, "target": t, "correct": float(c), "leakage": float(l),
                 "level3plus": float(l3)} for s,t,c,l,l3 in zip(sources,targets,correct,leakage,above2)],
            "restricted_matrix": encode(block), "full_physical_population": (abs(final)**2).tolist(),
            "trace": trace}, final


def cached_inverse():
    original = analysis.response_inverse
    cache = {}
    def wrapped(calibration):
        key = json.dumps(calibration["local_A"])
        if key not in cache:
            cache[key] = original(calibration)
        return cache[key]
    return wrapped


def run_suite(model, output, mode="bound", shots=1000, seed=20261002):
    identity = hashlib.sha256(json.dumps(model.config,sort_keys=True).encode() +
                              (output / "native" / "native_parameters.json").read_bytes()).hexdigest()
    backend = PulseExperimentBackend(model, seed, identity)
    roots = []
    shared_readout = None
    run_name = f"pulse-L{model.levels}-shots{shots}"
    with patch.object(workflow, "execute_job", backend.execute), \
            patch.object(analysis, "response_inverse", cached_inverse()):
        for scheme in ("r6", "r5"):
            down = None
            for direction in ("down", "up"):
                args = argparse.Namespace(scheme=scheme, direction=direction, mode=mode,
                       bits=model.config["physical_labels"], shots=shots, seed=seed,
                       run_name=run_name, output_root=str(output / "lab_equivalent_runs"),
                       readout_calibration=shared_readout, paired_parameters=None,
                       down_parameters=str(down / "parameters.json") if down else None,
                       query=mode != "pilot")
                print(f"START {scheme}/{direction}",flush=True)
                root = workflow.run(args, backend)
                roots.append(root)
                if direction=="down":
                    down=root
                    if shared_readout is None:
                        shared_readout=str(root/"readout_calibration.json")
                parameters = workflow.read_json(root / "parameters.json")
                raw_ops = p.core(scheme, direction)
                paired_ops = p.core(scheme, direction, parameters["paired"])
                corrected = p.router(scheme, direction, parameters["paired"], parameters["angles"])
                exact = {}
                for label, ops in (("unpaired",raw_ops),("paired",paired_ops),("corrected",corrected)):
                    metrics, _ = exact_metrics(model,ops,p.input_words(direction),p.output_words(direction))
                    exact[label]=metrics
                save_json(root / "reports" / "pulse_ground_truth.json", exact)
                print(f"EXACT {scheme}/{direction}: FZ={exact['corrected']['FZ']:.6f}, Fe={exact['corrected']['Fe']:.6f}, L={exact['corrected']['L']:.6g}",flush=True)
        workflow.summarize_suite(roots, output / "lab_equivalent_runs", run_name)
    save_json(output / "suite_manifest.json", {"roots":[str(x) for x in roots],
              "synthetic_records_generated":backend.executed,"shots":shots,
              "mode":mode,"model_sha256":identity,"maximum_norm_error":model.last_normalization_error})
    return roots


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config",type=Path,default=ROOT/"device.json")
    cli.add_argument("--output",type=Path,default=ROOT/"results")
    cli.add_argument("--levels",type=int,default=4)
    cli.add_argument("--dt",type=float,default=.1)
    cli.add_argument("--shots",type=int,default=1000)
    cli.add_argument("--mode",choices=("pilot","bound","direct","both"),default="bound")
    cli.add_argument("--calibrate-only",action="store_true")
    cli.add_argument("--seed",type=int,default=20261002)
    args=cli.parse_args()
    config=json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True,exist_ok=True)
    metadata={"device":config,"levels":args.levels,"dt":args.dt,"seed":args.seed,"shots":args.shots,"mode":args.mode}
    path=args.output/"simulation_config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=metadata:
        raise RuntimeError("Simulation settings changed; choose a new --output directory.")
    save_json(path,metadata)
    sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
               for name in ("bootstrap.py","pulse_model.py","calibrate.py","run.py")}
    sources["lab_workflow_sha256"] = workflow.source_hash()
    source_path = args.output/"source_manifest.json"
    if source_path.exists() and json.loads(source_path.read_text(encoding="utf-8"))!=sources:
        raise RuntimeError("Pulse simulation source changed; choose a new --output directory.")
    save_json(source_path,sources)
    model=PulseModel(config,args.levels,args.dt)
    calibrate_native(model,args.output/"native")
    if args.calibrate_only:
        return
    run_suite(model,args.output,args.mode,args.shots,args.seed)
    print(f"Saved {args.output}",flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
