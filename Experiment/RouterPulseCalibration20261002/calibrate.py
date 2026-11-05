"""Calibrate native waveforms, save scans and distinguish auxiliary loss.

Waveform optimization uses simulator propagators (an oracle), explicitly
recorded as such. Ramsey/population scans are also generated for inspection;
they are not presented as independent hardware measurements or as the sole
source of these pulse parameters. Router V/endframes are fitted from the
same shot-based experiments as the existing lab workflow.
"""
from dataclasses import replace, asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from pulse_model import CZPulse, SinglePulse, PulseModel, target_rotation, save_json, encode


def single_score(model, pulse):
    u = model.single_interaction(pulse)
    target = target_rotation(model.levels, int(pulse.gate == "12_pi"),
                             np.pi / 2 if pulse.gate == "01_half" else np.pi)
    # Correct only phases on 0/1/2; no inaccessible 2/3 virtual correction.
    k, t = u[:3, :3], target[:3, :3]
    overlaps = np.sum(t.conj() * k, axis=1)
    phases = np.angle(overlaps) - np.angle(overlaps[0])
    phases = np.angle(np.exp(1j * phases))
    corrected = np.exp(-1j * phases)[:, None] * k
    fe = abs(np.trace(t.conj().T @ corrected)) ** 2 / 9
    return float(1 - fe), (float(phases[1]), float(phases[2] - phases[1])), u


def cz_score(model, pulse):
    u = model.cz_interaction(pulse)
    inds = [0, 1, model.levels, model.levels + 1]
    k = u[np.ix_(inds, inds)]
    phase = np.angle(np.diag(k))
    angles = (float(phase[2] - phase[0]), float(phase[1] - phase[0]))
    corr = np.exp(-1j * np.array([0., angles[1], angles[0], sum(angles)]))
    fixed = corr[:, None] * k
    target = np.diag([1., 1., 1., -1.])
    fe = abs(np.trace(target @ fixed)) ** 2 / 16
    survival = np.linalg.norm(k) ** 2 / 4
    return float(1 - (4 * fe + survival) / 5), angles, u


def calibrate_native(model: PulseModel, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "native_parameters.json").exists():
        model.load_calibration(output / "native_parameters.json")
        print("Native pulse calibration resumed from saved parameters", flush=True)
        return
    summary = {"method": "propagator-oracle pulse optimization; hardware scans shown for guidance",
               "single": [], "cz": [], "scans": {}}
    for q in range(4):
        for name in ("01_half", "01_pi", "12_pi"):
            duration = model.config["x12_duration_ns"] if name == "12_pi" else model.config["single_duration_ns"]
            theta = np.pi / 2 if name == "01_half" else np.pi
            lower = int(name == "12_pi")
            guess = theta / (np.pi * duration * np.sqrt(lower + 1))
            def pulse(x):
                return SinglePulse(q, name, x[0] * guess, x[1] * .01, x[2], duration)
            cache = {}
            def objective(x):
                key = tuple(x)
                if key not in cache:
                    cache[key] = single_score(model, pulse(x))[0]
                return cache[key]
            result = minimize(objective, [1., 0., .15], method="L-BFGS-B",
                              bounds=[(.65, 1.35), (-1.2, 1.2), (0., 1.2)],
                              options={"maxiter": 75, "ftol": 1e-14, "gtol": 1e-8, "maxls": 30})
            best = pulse(min(cache, key=cache.get))
            cost, correction, u = single_score(model, best)
            best = replace(best, corrections=correction)
            model.single[(q, name)] = best
            row = {**asdict(best), "trace_fidelity_qutrit_phase_corrected": 1 - cost,
                   "unconditional_leak_to_level3plus": float(np.sum(abs(u[3:, :3]) ** 2) / 3),
                   "optimizer_success": bool(result.success), "evaluations": len(cache),
                   "interaction_matrix": encode(u)}
            summary["single"].append(row)
            print(f"single {q}/{name}: Fe3={1-cost:.9f}; leakage3+={row['unconditional_leak_to_level3plus']:.3g}", flush=True)
            values = []
            amplitudes = np.linspace(.5 * best.amplitude, 1.5 * best.amplitude, 25)
            for amplitude in amplitudes:
                scan = model.single_interaction(replace(best, amplitude=float(amplitude)))
                values.append({"amplitude_ghz": float(amplitude),
                               "transfer": float(abs(scan[lower + 1, lower]) ** 2),
                               "off_transition_probability": float(1 - np.sum(abs(scan[[lower, lower + 1], lower]) ** 2))})
            summary["scans"][f"single_{q}_{name}"] = values
            save_json(output / "native_calibration_in_progress.json", summary)
    # Computational CZ objective depends only on excitation sectors <=2;
    # a 3-level pair already contains those sectors exactly. Full 4-level
    # propagators are recomputed afterward to expose 21->30/12 channels.
    cz_model = PulseModel(model.config, levels=3, dt=model.dt, cache_limit=0)
    for seed in model.config["cz_seeds"]:
        edge = tuple(seed["edge"])
        base = CZPulse(edge, seed["tuning_wire"], seed["frequency_ghz"],
                       seed["duration_ns"], seed["ramp_ns"])
        bounds = [(-1., 1.), (.6, 1.55), (.55, 1.5)]
        def pulse(x):
            return replace(base, frequency=base.frequency + .06 * x[0],
                           duration=base.duration * x[1], ramp=base.ramp * x[2])
        cache = {}
        def objective(x):
            key = tuple(x)
            if key not in cache:
                cache[key] = cz_score(cz_model, pulse(x))[0]
            return cache[key]
        seeds = [np.array([f, t, 1.]) for f in np.linspace(-.6, .6, 5)
                 for t in np.linspace(.7, 1.4, 9)]
        seeds += [np.array([0., 1., 1.])]
        ranked = sorted(seeds, key=objective)
        results = [minimize(objective, x, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 90, "ftol": 1e-14, "gtol": 1e-8, "maxls": 30})
                   for x in (ranked[0], np.array([0., 1., 1.]))]
        best = pulse(min(cache, key=cache.get))
        cost, correction, u = cz_score(model, best)
        best = replace(best, corrections=correction)
        model.cz[edge] = best
        phases = np.angle(np.diag(u)).reshape(model.levels, model.levels)
        conditional = phases - phases[:, :1] - phases[:1, :] + phases[0, 0]
        conditional = np.angle(np.exp(1j * conditional))
        n = model.levels
        channels = {}
        for a, b in ((0, 1), (1, 0), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)):
            index = n * a + b
            channels[f"{a}{b}"] = {f"{c}{d}": float(abs(u[n*c+d, index]) ** 2)
                                    for c, d in np.ndindex(n, n)
                                    if abs(u[n*c+d, index]) ** 2 > 1e-9}
        row = {**asdict(best), "average_fidelity_computational_phase_corrected": 1 - cost,
               "evaluations": len(cache), "conditional_phase_rad": conditional.tolist(),
               "population_channels": channels, "interaction_matrix": encode(u)}
        summary["cz"].append(row)
        print(f"CZ {edge}: Favg01={1-cost:.9f}; duration={best.duration:.3f}; aux21 stay={channels['21'].get('21',0):.6f}", flush=True)
        grid = []
        for duration in np.linspace(.8 * best.duration, 1.2 * best.duration, 25):
            scan_pulse = replace(best, duration=float(duration))
            c, _, scan = cz_score(model, scan_pulse)
            diag = np.angle(np.diag(scan)).reshape(n, n)
            conditional11 = np.angle(np.exp(1j*(diag[1, 1]-diag[1, 0]-diag[0, 1]+diag[0, 0])))
            grid.append({"duration_ns": float(duration), "fidelity01": 1-c,
                         "conditional_phase11_rad": float(conditional11),
                         "return11": float(abs(scan[n+1, n+1])**2),
                         "return21": float(abs(scan[2*n+1, 2*n+1])**2),
                         "return12": float(abs(scan[n+2, n+2])**2),
                         "p21_to30": float(abs(scan[3*n, 2*n+1])**2) if n>3 else 0.})
        summary["scans"]["cz_" + "".join(map(str, edge))] = grid
        save_json(output / "native_calibration_in_progress.json", summary)
    save_json(output / "native_calibration.json", summary)
    model.save_calibration(output / "native_parameters.json")
    model.clear_caches()
