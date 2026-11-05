"""Binary-readout inversion and direction-aware router estimators.

All reported fidelities include computational loss; a postselected number is
never substituted for the unconditional result.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from . import protocol as p
from .runner import BINARY_KEYS


def records(directory: Path, stage: str) -> list[dict[str, Any]]:
    paths = sorted(directory.glob(f"{stage}__r5r6__*.json"))
    if not paths:
        raise FileNotFoundError(f"No {stage} records in {directory}.")
    values = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len({item["job"]["id"] for item in values}) != len(values):
        raise ValueError(f"Duplicate {stage} job identifiers.")
    if len({tuple(item["bits"]) for item in values}) != 1:
        raise ValueError(f"Mixed physical bit mappings in {stage}.")
    for item in values:
        if item["job"]["stage"] != stage or item["job"]["case"] != "r5r6":
            raise ValueError(f"Mismatched job record in {stage}.")
        if set(item["counts"]) != set(BINARY_KEYS) or sum(item["counts"].values()) != item["shots"]:
            raise ValueError(f"Incomplete binary histogram in {item['job']['id']}.")
    return values


def count_vector(item: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([item["counts"][key] for key in BINARY_KEYS], dtype=float)


def fit_readout(directory: Path) -> dict[str, Any]:
    data = records(directory, "readout_local")
    if len(data) != 36:
        raise ValueError(f"Expected 36 local readout references, found {len(data)}.")
    local = np.full((4, 3, 3), np.nan)
    nzero = np.zeros((4, 3, 3), dtype=int)
    shots = np.zeros((4, 3, 3), dtype=int)
    for item in data:
        meta = item["job"]["metadata"]
        wire, measured, prepared = meta["qubit"], meta["measured"], meta["prepared"]
        zero = sum(item["counts"][f"P{bits:04b}"] for bits in range(16)
                   if f"{bits:04b}"[wire] == "0")
        local[wire, measured, prepared] = zero / item["shots"]
        nzero[wire, measured, prepared] = zero
        shots[wire, measured, prepared] = item["shots"]
    if np.isnan(local).any():
        raise ValueError("Local readout references are incomplete.")
    return {"bits": data[0]["bits"], "local_A": local.tolist(),
            "local_zero_counts": nzero.tolist(), "local_shots": shots.tolist(),
            "local_rank": [int(np.linalg.matrix_rank(local[q])) for q in range(4)],
            "local_condition": [float(np.linalg.cond(local[q])) for q in range(4)],
            "source": "36 binary references; local-product readout approximation"}


def response_matrix(calibration: Mapping[str, Any]) -> np.ndarray:
    local = np.asarray(calibration["local_A"], dtype=float)
    if local.shape != (4, 3, 3) or not np.all(np.isfinite(local)):
        raise ValueError("Expected finite local_A shape (4,3,3).")
    result = np.empty((1296, 81), dtype=float)
    for mode_index, mode in enumerate(p.modes()):
        for state_index, state in enumerate(p.QUTRIT_WORDS):
            for binary_index in range(16):
                bits = f"{binary_index:04b}"
                result[16 * mode_index + binary_index, state_index] = math.prod(
                    local[q, mode[q], int(state[q])] if bits[q] == "0"
                    else 1 - local[q, mode[q], int(state[q])]
                    for q in range(4))
    if not np.allclose(result.reshape(81, 16, 81).sum(axis=1), 1, atol=1e-6):
        raise ValueError("Readout response does not normalize.")
    return result


def response_inverse(calibration: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = response_matrix(calibration)
    rank, condition = int(np.linalg.matrix_rank(matrix)), float(np.linalg.cond(matrix))
    if rank != 81 or not math.isfinite(condition):
        raise ValueError(f"Readout model not identifiable: rank={rank}, condition={condition}.")
    return np.linalg.pinv(matrix, rcond=1e-10), {"rank": rank, "condition": condition}


def grouped_histograms(data: list[dict[str, Any]],
                       key: Callable[[dict[str, Any]], str]) -> dict[str, np.ndarray]:
    mode_index = {mode: index for index, mode in enumerate(p.modes())}
    groups: dict[str, np.ndarray] = {}
    seen: dict[str, set[tuple[int, ...]]] = {}
    for item in data:
        name = key(item)
        mode = tuple(item["job"]["metadata"]["mode"])
        if mode in seen.setdefault(name, set()):
            raise ValueError(f"Repeated mapping {mode} for {name}.")
        seen[name].add(mode)
        vector = groups.setdefault(name, np.zeros(1296))
        index = mode_index[mode]
        vector[16 * index:16 * (index + 1)] = count_vector(item)
    if any(len(value) != 81 for value in seen.values()):
        raise ValueError("A setting is missing one or more of its 81 readout mappings.")
    return groups


def linear_population(histogram: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    counts = histogram.reshape(81, 16)
    shots = counts.sum(axis=1)
    if np.any(shots <= 0):
        raise ValueError("Each readout mapping needs positive shot count.")
    return inverse @ (counts / shots[:, None]).reshape(1296)


def project_simplex(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values)[::-1]
    sums = np.cumsum(ordered)
    valid = ordered - (sums - 1) / np.arange(1, len(values) + 1) > 0
    rho = np.nonzero(valid)[0][-1]
    return np.maximum(values - (sums[rho] - 1) / (rho + 1), 0)


def population_metrics(populations: Mapping[str, np.ndarray],
                       sources: tuple[str, ...], targets: tuple[str, ...]) -> dict[str, Any]:
    details = {}
    for source, target in zip(sources, targets):
        raw = populations[source]
        physical = project_simplex(raw)
        p_correct = float(raw[p.QUTRIT_WORDS.index(target)])
        target_survival = float(sum(raw[p.QUTRIT_WORDS.index(word)] for word in targets))
        comp_survival = float(sum(value for word, value in zip(p.QUTRIT_WORDS, raw)
                                  if "2" not in word))
        details[source] = {
            "target": target, "correct_probability_linear": p_correct,
            "target_survival_linear": target_survival,
            "computational_survival_linear": comp_survival,
            "leakage_linear": 1 - comp_survival,
            "correct_probability_projected": float(physical[p.QUTRIT_WORDS.index(target)]),
            "top_outputs_projected": [
                {"state": p.QUTRIT_WORDS[index], "probability": float(physical[index])}
                for index in np.argsort(physical)[-5:][::-1]],
        }
    correct = [item["correct_probability_linear"] for item in details.values()]
    survival = [item["target_survival_linear"] for item in details.values()]
    computational = [item["computational_survival_linear"] for item in details.values()]
    fz, s, comp = map(lambda a: float(np.mean(a)), (correct, survival, computational))
    return {"F_Z": fz, "p_min": float(min(correct)), "s": s, "L": 1 - comp,
            "q_in": s - fz, "q_off": comp - s, "inputs": details}


def truth_report(directory: Path, calibration: Mapping[str, Any], stage: str,
                 direction: str) -> dict[str, Any]:
    data = records(directory, stage)
    if len(data) != 324:
        raise ValueError(f"{stage} requires 324 records; found {len(data)}.")
    if data[0]["bits"] != calibration["bits"]:
        raise ValueError("Readout reference uses a different four-wire mapping.")
    inverse, check = response_inverse(calibration)
    grouped = grouped_histograms(data, lambda item: item["job"]["metadata"]["input"])
    populations = {word: linear_population(grouped[word], inverse)
                   for word in p.input_words(direction)}
    return {"stage": stage, "direction": direction, "response": check,
            "unconditional": True, "population_81_linear":
            {word: values.tolist() for word, values in populations.items()},
            **population_metrics(populations, p.input_words(direction), p.output_words(direction))}


def paired_report(directory: Path, calibration: Mapping[str, Any], direction: str) -> dict[str, Any]:
    data = records(directory, "paired")
    if len(data) != 972:
        raise ValueError(f"Paired auxiliary block needs 972 records; found {len(data)}.")
    inverse, check = response_inverse(calibration)
    grouped = grouped_histograms(data, lambda item:
                                 item["job"]["metadata"]["input"] + ":"
                                 + item["job"]["metadata"]["axis"])
    states = {}
    for state in ("0", "1", "+", "+i"):
        weights, differences = [], []
        for axis in "XYZ":
            pop = linear_population(grouped[f"{state}:{axis}"], inverse)
            p0 = sum(value for word, value in zip(p.QUTRIT_WORDS, pop)
                     if word[1] == "2" and word[2] == "0")
            p1 = sum(value for word, value in zip(p.QUTRIT_WORDS, pop)
                     if word[1] == "2" and word[2] == "1")
            weights.append(float(p0 + p1))
            differences.append(float(p0 - p1))
        weight = float(np.mean(weights))
        if weight < 0.05:
            raise ValueError(f"Auxiliary state {state} survival is too low to identify V.")
        bloch = np.array(differences) / weight
        states[state] = {"conditional_survival_linear": weight,
                         "survival_by_axis": weights, "bloch_linear": bloch.tolist(),
                         "bloch_length": float(np.linalg.norm(bloch))}
    bloch0 = np.asarray(states["0"]["bloch_linear"])
    norm = float(np.linalg.norm(bloch0))
    if norm < 0.05:
        raise ValueError("Receiver Bloch vector for input 0 is too short to identify V.")
    pure = bloch0 / norm
    a = math.sqrt(max(0, (1 + pure[2]) / 2))
    b = (pure[0] + 1j * pure[1]) / (2 * a) if a > 1e-7 else 1 + 0j
    column = np.array([a, b], dtype=complex)
    column /= np.linalg.norm(column)
    a, b = column
    canonical = np.array([[a, -np.conj(b)], [b, np.conj(a)]], dtype=complex)
    plus = np.asarray(states["+"]["bloch_linear"])
    rho_plus = np.array([[1 + plus[2], plus[0] - 1j * plus[1]],
                         [plus[0] + 1j * plus[1], 1 - plus[2]]], dtype=complex) / 2
    rho_rotated = canonical.conj().T @ rho_plus @ canonical
    coherence = rho_rotated[1, 0]
    if abs(coherence) < 0.05:
        raise ValueError("The |+> input has insufficient output coherence to identify V.")
    relative = float(np.angle(coherence))
    v = canonical @ np.diag([1, np.exp(1j * relative)])
    predicted = {}
    for state in ("0", "1", "+", "+i"):
        vector = v @ p.STATES[state]
        rho = np.outer(vector, vector.conj())
        prediction = np.array([float(np.trace(rho @ sigma).real)
                               for sigma in (p.X, p.Y, p.Z)])
        measured = np.asarray(states[state]["bloch_linear"])
        predicted[state] = {"predicted_bloch": prediction.tolist(),
                            "bloch_residual_norm": float(np.linalg.norm(measured - prediction))}
    average_weight = float(np.mean([item["conditional_survival_linear"]
                                     for item in states.values()]))
    return {"stage": "paired", "direction": direction, "response": check,
            "conditional_survival_linear": average_weight,
            "conditional_bloch_linear": states["0"]["bloch_linear"],
            "bloch_length": norm, "input_states": states,
            "relative_column_phase_rad": relative, "unitary_residual": predicted,
            "maximum_unitary_bloch_residual": max(item["bloch_residual_norm"]
                                                 for item in predicted.values()),
            "paired": {"matrix": np.stack([v.real, v.imag], axis=-1).tolist(),
                       "source": "conditional i=2 four-input 01 process tomography; U(2) fit"},
            "warning": "V corrects coherent rotation only, not conditional loss or mixing"}


def phase_scan(points: list[tuple[float, float, float]]) -> dict[str, Any]:
    if len(points) != p.PHASE_POINTS:
        raise ValueError("Expected 16 Ramsey phase samples.")
    phase = np.array([x[0] for x in points])
    contrast = np.array([x[1] for x in points])
    variance = np.maximum(np.array([x[2] for x in points]), 1e-9)
    design = np.column_stack((np.ones(len(phase)), np.cos(phase), np.sin(phase)))
    weighted = design / np.sqrt(variance[:, None])
    fit = np.linalg.lstsq(weighted, contrast / np.sqrt(variance), rcond=None)[0]
    predicted = design @ fit
    chi2 = float(np.sum((contrast - predicted) ** 2 / variance))
    offset, cosine, sine = fit
    return {"phase_rad": float(math.atan2(cosine, -sine)),
            "visibility": float(math.hypot(cosine, sine)), "offset": float(offset),
            "reduced_chi2": chi2 / 13, "scan_phase_rad": phase.tolist(),
            "scan_contrast": contrast.tolist(), "scan_standard_error": np.sqrt(variance).tolist(),
            "fit_contrast": predicted.tolist()}


def ramsey_report(directory: Path, stage: str, direction: str) -> dict[str, Any]:
    data = records(directory, stage)
    if len(data) != 48:
        raise ValueError(f"{stage} requires 48 records; found {len(data)}.")
    grouped: dict[int, list[tuple[float, float, float]]] = {i: [] for i in range(3)}
    for item in data:
        meta = item["job"]["metadata"]
        left, right = meta["output_pair"]
        p_left = item["counts"][f"P{left}"] / item["shots"]
        p_right = item["counts"][f"P{right}"] / item["shots"]
        contrast = p_left - p_right
        variance = max((p_left + p_right - contrast**2) / item["shots"], 1e-9)
        grouped[meta["pair"]].append((meta["phase"], contrast, variance))
    fits = [phase_scan(sorted(grouped[index])) for index in range(3)]
    return {"stage": stage, "direction": direction, "pairs": fits,
            "minimum_visibility": min(fit["visibility"] for fit in fits),
            "fit_phase_rad": [fit["phase_rad"] for fit in fits]}


def endpoint_calibration(raw: Mapping[str, Any], dither: Mapping[str, Any]) -> dict[str, Any]:
    angles, response = {}, {}
    for key, first, second in zip(("c", "l", "r"), raw["pairs"], dither["pairs"]):
        delta = math.remainder(second["phase_rad"] - first["phase_rad"], 2 * p.PI)
        gain = delta / (p.PI / 2)
        if min(first["visibility"], second["visibility"]) < 0.03 or abs(gain) < 0.5:
            raise ValueError(f"Cannot identify endpoint phase response on {key}.")
        angles[key] = float(math.remainder(-first["phase_rad"] / gain, 2 * p.PI))
        response[key] = {"raw_phase_rad": first["phase_rad"], "dither_delta_rad": delta,
                         "phase_gain": gain}
    return {"angles": angles, "response": response,
            "method": "measured +pi/2 virtual-frame dither; independent c/l/r paths"}


def pauli_expectation(population: np.ndarray, word: str) -> float:
    value = 0.0
    for state, probability in zip(p.QUTRIT_WORDS, population):
        if "2" in state:
            continue
        sign = (-1) ** sum(int(level) for level, axis in zip(state, word) if axis != "I")
        value += sign * probability
    return float(value)


def fidelity_report(directory: Path, calibration: Mapping[str, Any], stage: str,
                    direction: str) -> dict[str, Any]:
    down = direction == "down"
    if direction == "up" and not stage.startswith("roundtrip_"):
        raise ValueError("Do not report composite state preparation as isolated up fidelity.")
    target_words = p.DOWN_OUT if down else p.DOWN_IN
    design = p.measurement_design(target_words)
    expected = 81 * p.fidelity_core_count(stage, direction)
    data = records(directory, stage)
    if len(data) != expected:
        raise ValueError(f"{stage} needs {expected} records; found {len(data)}.")
    if data[0]["bits"] != calibration["bits"]:
        raise ValueError("Readout calibration has a different wire mapping.")
    inverse, check = response_inverse(calibration)
    grouped = grouped_histograms(data, lambda item:
                                  "_".join(item["job"]["metadata"]["logical_input"])
                                  + "__" + item["job"]["metadata"]["axes"])
    cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for identifier, histogram in grouped.items():
        labels, axes = identifier.split("__")
        cache.setdefault(tuple(labels.split("_")), {})[axes] = linear_population(histogram, inverse)
    populations = {word: cache[tuple(word[:2])]["ZZZZ"] for word in p.DOWN_IN}
    metrics = population_metrics(populations, p.DOWN_IN, target_words)
    if stage.endswith("bound"):
        fx = 0.0
        for item in design["bound_x"]:
            populations_by_axis = cache[tuple(item["logical_input"])]
            for group in item["groups"]:
                fx += sum(term["coefficient"] * pauli_expectation(
                    populations_by_axis[group["basis"]], term["word"])
                    for term in group["terms"])
        metrics.update({"F_X": float(fx),
                        "F_e_lower_linear": float(max(0, metrics["F_Z"] + fx - metrics["s"])),
                        "F_e_upper_linear": float(min(metrics["F_Z"], fx))})
    else:
        fe = 0.0
        for item in design["direct"]:
            populations_by_axis = cache[tuple(item["logical_input"])]
            for group in item["groups"]:
                fe += sum(term["coefficient"] * pauli_expectation(
                    populations_by_axis[group["basis"]], term["word"])
                    for term in group["terms"])
        metrics.update({"F_e_linear": float(fe),
                        "F_avg_linear": float((4 * fe + metrics["s"]) / 5)})
    return {"stage": stage, "direction": direction, "response": check,
            "scope": "isolated down isometry" if down else "down-prepared up roundtrip; NOT isolated up fidelity",
            "method": "raw linear inversion; no clipping or postselection in fidelity",
            **metrics}


def query_report(directory: Path, calibration: Mapping[str, Any]) -> dict[str, Any]:
    data = records(directory, "query")
    if len(data) != 1296:
        raise ValueError(f"Query stage needs 1296 circuits; found {len(data)}.")
    inverse, check = response_inverse(calibration)
    grouped = grouped_histograms(data, lambda item:
                                  item["job"]["metadata"]["table"] + ":"
                                  + item["job"]["metadata"]["input"])
    results = {}
    for table, source in itertools.product(("00", "01", "10", "11"), p.DOWN_IN):
        population = linear_population(grouped[f"{table}:{source}"], inverse)
        target = f"{source[0]}{int(source[1]) ^ int(table[int(source[0])])}00"
        results[f"{table}:{source}"] = {
            "target": target,
            "correct_probability_linear": float(population[p.QUTRIT_WORDS.index(target)]),
            "computational_survival_linear": float(sum(value for word, value in zip(p.QUTRIT_WORDS, population)
                                                        if "2" not in word))}
    return {"stage": "query", "response": check, "inputs": results,
            "mean_correct": float(np.mean([x["correct_probability_linear"] for x in results.values()])),
            "minimum_correct": float(min(x["correct_probability_linear"] for x in results.values())),
            "mean_leakage": float(np.mean([1 - x["computational_survival_linear"]
                                            for x in results.values()]))}
