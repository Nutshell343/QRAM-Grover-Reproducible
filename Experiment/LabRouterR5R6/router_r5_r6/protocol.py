"""Native-gate circuits and measurement designs for star-topology R5/R6.

Wire order is always c,i,l,r.  Every list of operations is chronological.
No simulator-fitted drive phase is uploaded as a hardware calibration.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterator, Mapping

import numpy as np

try:
    from app.temp_need.TCG_new.partial_cswap_vz4.protocol import single_01_unitary
except ModuleNotFoundError:  # offline workspace
    from partial_cswap_vz4.protocol import single_01_unitary  # type: ignore

PI = math.pi
SCHEMES = ("r5", "r6")
DIRECTIONS = ("down", "up")
DOWN_IN = ("0000", "0100", "1000", "1100")
DOWN_OUT = ("0000", "0010", "1000", "1001")
PHASE_POINTS = 16
QUTRIT_WORDS = tuple("".join(x) for x in itertools.product("012", repeat=4))
RAMSEY = {
    "down": (("0000", "1000", 0, 0), ("0000", "0100", 2, 2),
             ("1000", "1100", 3, 3)),
    "up": (("0000", "1000", 0, 0), ("0000", "0010", 1, 2),
           ("1000", "1001", 1, 3)),
}
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def gate(name: str, *wires: int, phase: float = 0.0) -> dict[str, Any]:
    if name not in ("01_half", "01_pi", "12_pi", "cz", "frame", "frame_01", "frame_12"):
        raise ValueError(name)
    if len(wires) != (2 if name == "cz" else 1):
        raise ValueError(f"Wrong gate arity: {name} {wires}")
    if any(w not in range(4) for w in wires) or not math.isfinite(float(phase)):
        raise ValueError(f"Invalid gate: {name} {wires} {phase}")
    return {"gate": name, "qubits": list(wires), "phase": float(phase)}


def x01(wire: int, phase: float = 0.0) -> list[dict[str, Any]]:
    return [gate("01_pi", wire, phase=phase)]


def x12(wire: int, phase: float = 0.0) -> list[dict[str, Any]]:
    return [gate("12_pi", wire, phase=phase)]


def cnot(control: int, target: int) -> list[dict[str, Any]]:
    return [gate("01_half", target, phase=-PI / 2), gate("cz", control, target),
            gate("01_half", target, phase=PI / 2)]


def transfer(source: int, target: int) -> list[dict[str, Any]]:
    return cnot(source, target) + cnot(target, source)


def reverse_same_cz(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reverse chronology; use the same physical CZ, not its mathematical dagger."""
    result = []
    for item in reversed(operations):
        copy = dict(item)
        copy["qubits"] = list(item["qubits"])
        if item["gate"] == "01_half":
            copy["phase"] = float(math.remainder(item["phase"] + PI, 2 * PI))
        elif item["gate"].startswith("frame"):
            copy["phase"] = -item["phase"]
        result.append(copy)
    return result


def paired_matrix(value: Mapping[str, Any] | None) -> np.ndarray:
    if value is None:
        return np.eye(2, dtype=complex)
    raw = np.asarray(value.get("matrix"), dtype=float)
    if raw.shape != (2, 2, 2) or not np.all(np.isfinite(raw)):
        raise ValueError("Paired correction requires finite 2x2 complex matrix pairs.")
    matrix = raw[..., 0] + 1j * raw[..., 1]
    if not np.allclose(matrix.conj().T @ matrix, np.eye(2), atol=1e-5):
        raise ValueError("Paired correction is not unitary.")
    return matrix


def encoder_left() -> list[dict[str, Any]]:
    return x01(1) + x12(1) + cnot(0, 1) + x12(1) + x01(1)


def r6_down() -> list[dict[str, Any]]:
    # The compact six-CZ sequence from RouterDepthStudy20260929/native_path_search.json.
    return (cnot(1, 2) + x01(1) + x12(1) + cnot(0, 1) + x12(1)
            + x01(1) + cnot(2, 1) + x12(1) + cnot(1, 2)
            + cnot(1, 3) + cnot(3, 1))


def core(scheme: str, direction: str, paired: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if scheme not in SCHEMES or direction not in DIRECTIONS:
        raise ValueError(f"Unknown router {scheme}/{direction}.")
    if scheme == "r6":
        if paired is not None:
            raise ValueError("R6 has no paired V calibration.")
        raw = r6_down()
        return raw if direction == "down" else reverse_same_cz(raw)
    v = paired_matrix(paired)
    before = [] if np.allclose(v, np.eye(2), atol=1e-12) else single_01_unitary(1, v)
    after = [] if np.allclose(v, np.eye(2), atol=1e-12) else single_01_unitary(2, v.conj().T)
    if direction == "down":
        return (encoder_left() + before + transfer(1, 2)
                + after + x12(1) + transfer(1, 3))
    return (transfer(3, 1) + x12(1) + after
            + transfer(2, 1) + before
            + reverse_same_cz(encoder_left()))


def endpoint_frame(direction: str, angles: Mapping[str, float] | None) -> list[dict[str, Any]]:
    values = {} if angles is None else dict(angles)
    if set(values) - {"c", "l", "r"}:
        raise ValueError("Only c/l/r endpoint phase parameters are allowed.")
    return [gate("frame", {"c": 0, "l": 2, "r": 3}[name], phase=float(values[name]))
            for name in ("c", "l", "r") if float(values.get(name, 0.0)) != 0.0]


def router(scheme: str, direction: str, paired: Mapping[str, Any] | None = None,
           angles: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    body = core(scheme, direction, paired)
    correction = endpoint_frame(direction, angles)
    return correction + body if direction == "up" else body + correction


def input_words(direction: str) -> tuple[str, ...]:
    return DOWN_IN if direction == "down" else DOWN_OUT


def output_words(direction: str) -> tuple[str, ...]:
    return DOWN_OUT if direction == "down" else DOWN_IN


def prepare_basis(word: str) -> list[dict[str, Any]]:
    if len(word) != 4 or any(x not in "012" for x in word):
        raise ValueError(f"Invalid qutrit word {word!r}.")
    result = []
    for wire, level in enumerate(word):
        if level in "12":
            result += x01(wire)
        if level == "2":
            result += x12(wire)
    return result


def prepare_01(wire: int, label: str) -> list[dict[str, Any]]:
    if label == "0":
        return []
    if label == "1":
        return x01(wire)
    phases = {"+": PI / 2, "-": -PI / 2, "+i": PI}
    if label not in phases:
        raise ValueError(label)
    return [gate("01_half", wire, phase=phases[label])]


def analyze_axis(wire: int, axis: str) -> list[dict[str, Any]]:
    if axis == "Z":
        return []
    if axis == "X":
        return [gate("01_half", wire, phase=-PI / 2)]
    if axis == "Y":
        return [gate("01_half", wire)]
    raise ValueError(axis)


def readout(mode: tuple[int, ...]) -> list[dict[str, Any]]:
    if len(mode) != 4:
        raise ValueError(mode)
    result = []
    for wire, setting in enumerate(mode):
        if setting == 1:
            result += x01(wire)
        elif setting == 2:
            result += x01(wire) + x12(wire) + x01(wire)
        elif setting != 0:
            raise ValueError(mode)
    return result


def modes() -> Iterator[tuple[int, int, int, int]]:
    yield from itertools.product(range(3), repeat=4)


def job(stage: str, tag: str, operations: list[dict[str, Any]],
        **metadata: Any) -> dict[str, Any]:
    return {"id": f"{stage}__r5r6__{tag}", "stage": stage,
            "case": "r5r6", "operations": operations, "metadata": metadata}


def readout_jobs() -> Iterator[dict[str, Any]]:
    for wire, prepared, measured in itertools.product(range(4), range(3), range(3)):
        state = ["0"] * 4
        state[wire] = str(prepared)
        mode = [0] * 4
        mode[wire] = measured
        yield job("readout_local", f"q{wire}_s{prepared}_m{measured}",
                  prepare_basis("".join(state)) + readout(tuple(mode)),
                  qubit=wire, prepared=prepared, measured=measured, mode=mode)


def paired_jobs(direction: str) -> Iterator[dict[str, Any]]:
    # Four receiver inputs identify the 2x2 auxiliary block up to global phase.
    # The full block is essential for up, where V† precedes the reverse block.
    block = transfer(1, 2) if direction == "down" else transfer(2, 1)
    for state, axis, mode in itertools.product(("0", "1", "+", "+i"), "XYZ", modes()):
        label = state.replace("+", "p")
        yield job("paired", f"{direction}_s{label}_a{axis}_m{''.join(map(str, mode))}",
                  prepare_basis("0200") + prepare_01(2, state) + block
                  + analyze_axis(2, axis) + readout(mode),
                  direction=direction, input=state, axis=axis, mode=list(mode),
                  auxiliary_wire=1, receiver_wire=2)


def truth_jobs(stage: str, scheme: str, direction: str, paired: Mapping[str, Any] | None,
               angles: Mapping[str, float] | None) -> Iterator[dict[str, Any]]:
    for source, target in zip(input_words(direction), output_words(direction)):
        for mode in modes():
            yield job(stage, f"{scheme}_{direction}_s{source}_m{''.join(map(str, mode))}",
                      prepare_basis(source) + router(scheme, direction, paired, angles) + readout(mode),
                      input=source, target=target, mode=list(mode))


def ramsey_jobs(stage: str, scheme: str, direction: str, paired: Mapping[str, Any] | None,
                angles: Mapping[str, float] | None) -> Iterator[dict[str, Any]]:
    for index, (left, right, output_wire, frame_wire) in enumerate(RAMSEY[direction]):
        input_wire = next(w for w in range(4) if left[w] != right[w])
        applied = dict(angles or {})
        if stage == "ramsey_dither":
            applied[{0: "c", 2: "l", 3: "r"}[frame_wire]] = PI / 2
        for step in range(PHASE_POINTS):
            phase = 2 * PI * step / PHASE_POINTS
            operations = (prepare_basis(left) + [gate("01_half", input_wire, phase=PI / 2)]
                          + router(scheme, direction, paired, applied)
                          + [gate("01_half", output_wire, phase=phase)])
            target_left = output_words(direction)[input_words(direction).index(left)]
            target_right = output_words(direction)[input_words(direction).index(right)]
            yield job(stage, f"{scheme}_{direction}_pair{index}_p{step:02d}", operations,
                      pair=index, phase=phase, output_pair=[target_left, target_right],
                      frame_wire=frame_wire)


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
PAULI = dict(zip("IXYZ", (I2, X, Y, Z)))
STATES = {"0": np.array([1, 0]), "1": np.array([0, 1]),
          "+": np.array([1, 1]) / math.sqrt(2),
          "-": np.array([1, -1]) / math.sqrt(2),
          "+i": np.array([1, 1j]) / math.sqrt(2)}
DUALS = {"0": (I2 + Z - X - Y) / 2, "1": (I2 - Z - X - Y) / 2,
         "+": X, "+i": Y}


def _kron(matrices: Any) -> np.ndarray:
    result = np.ones((1, 1), dtype=complex)
    for item in matrices:
        result = np.kron(result, item)
    return result


def _pauli_terms(observable: np.ndarray) -> dict[str, float]:
    result = {}
    for labels in itertools.product("IXYZ", repeat=4):
        word = "".join(labels)
        coefficient = np.trace(_kron(PAULI[x] for x in word) @ observable) / 16
        if abs(coefficient.imag) > 1e-10:
            raise AssertionError("Non-Hermitian target observable.")
        if abs(coefficient.real) > 1e-11:
            result[word] = float(coefficient.real)
    return result


def _groups(terms: dict[str, float]) -> list[dict[str, Any]]:
    pending = dict(terms)
    candidates = ["".join(x) for x in itertools.product("XYZ", repeat=4)]
    groups = []
    while pending:
        basis = min(candidates, key=lambda b: (-sum(all(p == "I" or p == q for p, q in zip(w, b))
                                                   for w in pending), b))
        words = [word for word in pending if all(p == "I" or p == q for p, q in zip(word, basis))]
        groups.append({"basis": basis,
                       "terms": [{"word": word, "coefficient": pending.pop(word)} for word in words]})
    return groups


def measurement_design(target_words: tuple[str, ...]) -> dict[str, Any]:
    """Dimension-four unconditional isometry estimators, including leakage."""
    w = np.eye(16, dtype=complex)[:, [int(word, 2) for word in target_words]]
    direct = []
    for a, b in itertools.product(("0", "1", "+", "+i"), repeat=2):
        target = w @ np.kron(DUALS[a], DUALS[b]) @ w.conj().T / 16
        direct.append({"logical_input": (a, b), "groups": _groups(_pauli_terms(target))})
    bound_x = []
    for a, b in itertools.product(("+", "-"), repeat=2):
        state = w @ np.kron(STATES[a], STATES[b])
        bound_x.append({"logical_input": (a, b),
                        "groups": _groups(_pauli_terms(np.outer(state, state.conj()) / 4))})
    return {"direct": direct, "bound_x": bound_x,
            "counts": {"bound_core": 4 + sum(len(x["groups"]) for x in bound_x),
                       "direct_core": sum(len(x["groups"]) for x in direct)}}


def fidelity_core_count(stage: str, direction: str) -> int:
    target = DOWN_OUT if direction == "down" else DOWN_IN
    design = measurement_design(target)
    selected = design["bound_x" if stage.endswith("bound") else "direct"]
    settings = {(a, b, "ZZZZ") for a, b in itertools.product("01", repeat=2)}
    settings.update((item["logical_input"][0], item["logical_input"][1], group["basis"])
                    for item in selected for group in item["groups"])
    return len(settings)


def fidelity_jobs(stage: str, scheme: str, direction: str, paired: Mapping[str, Any] | None,
                  angles: Mapping[str, float] | None,
                  down_preparation: Mapping[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    if direction == "up" and down_preparation is None:
        raise ValueError("Up coherence needs a separately calibrated down preparation.")
    if direction == "down" and down_preparation is not None:
        raise ValueError("Down fidelity must not include a down preparation block.")
    def circuit(logical: tuple[str, str], axes: str) -> list[dict[str, Any]]:
        operations = prepare_01(0, logical[0]) + prepare_01(1, logical[1])
        if down_preparation is not None:
            operations += router(str(down_preparation["scheme"]), "down",
                                 down_preparation.get("paired"), down_preparation.get("angles"))
        operations += router(scheme, direction, paired, angles)
        for wire, axis in enumerate(axes):
            operations += analyze_axis(wire, axis)
        return operations
    if stage.endswith("bound") or stage.endswith("direct"):
        for logical in itertools.product("01", repeat=2):
            for mode in modes():
                yield job(stage, f"{scheme}_{direction}_z{''.join(logical)}_m{''.join(map(str, mode))}",
                          circuit(logical, "ZZZZ") + readout(mode), logical_input=list(logical),
                          axes="ZZZZ", mode=list(mode), design_role="Z_population")
    if stage.endswith("bound"):
        design = measurement_design(DOWN_OUT if direction == "down" else DOWN_IN)["bound_x"]
    elif stage.endswith("direct"):
        design = measurement_design(DOWN_OUT if direction == "down" else DOWN_IN)["direct"]
    else:
        raise ValueError(stage)
    seen = {(a, b, "ZZZZ") for a, b in itertools.product("01", repeat=2)}
    for item in design:
        logical = tuple(item["logical_input"])
        for group_index, group in enumerate(item["groups"]):
            axes = group["basis"]
            setting_key = (logical[0], logical[1], axes)
            if setting_key in seen:
                continue
            seen.add(setting_key)
            for mode in modes():
                label = "_".join(logical).replace("+", "p").replace("-", "m")
                yield job(stage, f"{scheme}_{direction}_{label}_g{group_index:02d}_m{''.join(map(str, mode))}",
                          circuit(logical, axes) + readout(mode), logical_input=list(logical),
                          axes=axes, group=group_index, mode=list(mode), design_role="projector")


def query_jobs(stage: str, scheme: str, down: Mapping[str, Any], up: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    for table, source, mode in itertools.product(("00", "01", "10", "11"), DOWN_IN, modes()):
        target_bit = int(source[1]) ^ int(table[int(source[0])])
        target = f"{source[0]}{target_bit}00"
        operations = prepare_basis(source) + single_01_unitary(1, H)
        operations += router(scheme, "down", down.get("paired"), down.get("angles"))
        for leaf, bit in ((2, table[0]), (3, table[1])):
            if bit == "1":
                operations += [gate("frame", leaf, phase=PI)]
        operations += router(scheme, "up", up.get("paired"), up.get("angles"))
        operations += single_01_unitary(1, H) + readout(mode)
        yield job(stage, f"{scheme}_t{table}_s{source}_m{''.join(map(str, mode))}",
                  operations, table=table, input=source, target=target, mode=list(mode))
