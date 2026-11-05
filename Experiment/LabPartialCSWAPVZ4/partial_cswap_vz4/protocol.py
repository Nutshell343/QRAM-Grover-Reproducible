"""Generate laboratory circuits for the three partial-CSWAP schemes."""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterator, Mapping

import numpy as np

from .timing_plan import FRAME_GATES, physical_layers

PI = math.pi
PHASE_POINTS = 16
READOUT_MODES = (0, 1, 2)
CASES = ("a_110_to_101", "b_101_to_110", "c_i1_3cz_110_to_101")
TARGETS = {
    CASES[0]: {"000": "000", "010": "010", "100": "100", "110": "101"},
    CASES[1]: {"000": "000", "001": "001", "100": "100", "101": "110"},
    CASES[2]: {"000": "000", "010": "010", "100": "100", "110": "101"},
}
RAMSEY_PAIRS = {
    CASES[0]: (("000", "100", 0), ("000", "010", 1), ("100", "110", 2)),
    CASES[1]: (("000", "100", 0), ("100", "101", 1), ("000", "001", 2)),
    CASES[2]: (("000", "100", 0), ("000", "010", 1), ("100", "110", 2)),
}
CORE_PHASE_KEYS = {
    CASES[0]: (
        "x01_q2_1", "x01_q2_2", "x12_q2_1", "x12_q2_2", "x12_q2_3", "x12_q2_4",
    ),
    CASES[1]: (
        "x01_q1_pre", "x01_q1_post", "x12_q2_1", "x12_q2_2", "x12_q2_3",
    ),
    CASES[2]: (
        "x01_q2_encode", "x12_q2_encode", "x01_q2_xa_1", "x12_q2_xa",
        "x01_q2_xa_2", "x12_q2_decode",
    ),
}
OPTIMIZED_PHASE_KEYS = {
    CASES[0]: ("x12_q2_4", "x12_q2_3", "x12_q2_2", "x01_q2_2"),
    CASES[1]: ("x12_q2_3", "x12_q2_2", "x01_q1_post"),
    CASES[2]: ("x12_q2_decode", "x01_q2_xa_2", "x12_q2_xa", "x01_q2_xa_1"),
}
DD_QUBITS = (0, 2)


def gate(name: str, *qubits: int, phase: float = 0.0) -> dict[str, Any]:
    """Create one chronological circuit instruction."""
    return {"gate": name, "qubits": list(qubits), "phase": float(phase)}


def x01(qubit: int, phase: float = 0.0) -> list[dict[str, Any]]:
    """Implement a 01 pi rotation using two native 01 pi/2 pulses."""
    return [gate("01_half", qubit, phase=phase), gate("01_half", qubit, phase=phase)]


def x12(qubit: int, phase: float = 0.0) -> list[dict[str, Any]]:
    """Implement the native ef pi rotation through the Ref compiler path."""
    return [gate("12_pi", qubit, phase=phase)]


def native_x01(qubit: int) -> dict[str, Any]:
    """Return one calibrated native 01 pi pulse for dynamical decoupling."""
    return gate("01_pi", qubit)


def cnot(control: int, target: int) -> list[dict[str, Any]]:
    """Use chronological -Y/2, CZ, +Y/2 on the target."""
    return [
        gate("01_half", target, phase=-PI / 2),
        gate("cz", control, target),
        gate("01_half", target, phase=PI / 2),
    ]


def _complex_matrix(value: Any) -> np.ndarray:
    """Decode a 2x2 complex matrix from JSON-safe [real, imag] entries."""
    raw = np.asarray(value)
    if raw.shape == (2, 2, 2):
        matrix = raw[..., 0].astype(float) + 1j * raw[..., 1].astype(float)
    else:
        matrix = np.asarray(value, dtype=complex)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("The paired-block correction must contain a finite 2x2 matrix.")
    return matrix


def encode_complex_matrix(matrix: np.ndarray) -> list[list[list[float]]]:
    """Encode a complex 2x2 matrix without relying on nonstandard JSON types."""
    value = np.asarray(matrix, dtype=complex)
    if value.shape != (2, 2):
        raise ValueError("Only 2x2 matrices can be encoded as a paired correction.")
    return [[[float(entry.real), float(entry.imag)] for entry in row] for row in value]


def paired_rotation_parameters(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate Scheme C's calibrated 01 unitary and its phase relative to |2>."""
    supplied = {} if value is None else dict(value)
    unknown = set(supplied) - {"su2_matrix", "common_phase_rad", "source"}
    if unknown:
        raise ValueError(f"Unknown paired-rotation keys: {sorted(unknown)}.")
    matrix = _complex_matrix(supplied.get("su2_matrix", np.eye(2, dtype=complex)))
    left, _, right = np.linalg.svd(matrix)
    matrix = left @ right
    determinant = np.linalg.det(matrix)
    matrix *= np.exp(-0.5j * np.angle(determinant))
    if not np.allclose(matrix.conj().T @ matrix, np.eye(2), atol=1e-10):
        raise ValueError("The paired-block correction could not be projected to a unitary.")
    common = float(supplied.get("common_phase_rad", 0.0))
    if not math.isfinite(common):
        raise ValueError("The paired-block common phase must be finite.")
    return {
        "su2_matrix": encode_complex_matrix(matrix),
        "common_phase_rad": common,
        "source": str(supplied.get("source", "identity-default")),
    }


def _unitary_zyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Return alpha,beta,theta,gamma for U=e^(i alpha)Rz(beta)Ry(theta)Rz(gamma)."""
    unitary = np.asarray(matrix, dtype=complex)
    alpha = float(np.angle(np.linalg.det(unitary)) / 2)
    special = unitary * np.exp(-1j * alpha)
    theta = float(2 * math.atan2(abs(special[1, 0]), abs(special[0, 0])))
    if abs(math.sin(theta / 2)) < 1e-12:
        beta = 0.0
        gamma = float(-2 * np.angle(special[0, 0]))
    elif abs(math.cos(theta / 2)) < 1e-12:
        beta = float(2 * np.angle(special[1, 0]))
        gamma = 0.0
    else:
        beta = float(np.angle(special[1, 0]) - np.angle(special[0, 0]))
        gamma = float(-np.angle(special[1, 0]) - np.angle(special[0, 0]))
    return alpha, beta, theta, gamma


def embedded_rz(qubit: int, angle: float) -> list[dict[str, Any]]:
    """Implement Rz(angle) on 01 while leaving level 2 fixed, up to global phase."""
    return [
        gate("frame_01", qubit, phase=-angle),
        gate("frame_12", qubit, phase=angle / 2),
    ]


def single_01_unitary(qubit: int, matrix: Any) -> list[dict[str, Any]]:
    """Compile an arbitrary embedded 01 unitary to two X/2 pulses and virtual frames."""
    unitary = _complex_matrix(matrix)
    left, _, right = np.linalg.svd(unitary)
    unitary = left @ right
    alpha, beta, theta, gamma = _unitary_zyz(unitary)
    return (
        embedded_rz(qubit, gamma)
        + embedded_rz(qubit, PI / 2)
        + [gate("01_half", qubit, phase=PI / 2)]
        + embedded_rz(qubit, theta)
        + [gate("01_half", qubit, phase=-PI / 2)]
        + embedded_rz(qubit, -PI / 2)
        + embedded_rz(qubit, beta)
        + [gate("frame_12", qubit, phase=alpha)]
    )


def core_phase_parameters(
    case: str, phases: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return a complete, finite phase table while fixing the two gauge slots."""
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; choose one of {CASES}.")
    supplied = {} if phases is None else dict(phases)
    unknown = set(supplied) - set(CORE_PHASE_KEYS[case])
    if unknown:
        raise ValueError(f"Unknown core phase keys for {case}: {sorted(unknown)}.")
    result = {key: float(supplied.get(key, 0.0)) for key in CORE_PHASE_KEYS[case]}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("Every core phase must be finite.")
    if case == CASES[0]:
        fixed = ("x01_q2_1", "x12_q2_1")
    elif case == CASES[1]:
        fixed = ("x01_q1_pre", "x12_q2_1")
    else:
        fixed = ("x01_q2_encode", "x12_q2_encode")
    if any(abs(result[key]) > 1e-12 for key in fixed):
        raise ValueError(f"Gauge phase slots must remain zero for {case}: {fixed}.")
    return result


def _base_core_operations(
    case: str,
    phases: Mapping[str, float] | None = None,
    paired_rotation: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the chronological native expansion before layer optimization."""
    phase = core_phase_parameters(case, phases)
    if case == CASES[0]:
        return (
            cnot(1, 2)
            + x01(1, phase["x01_q2_1"])
            + x12(1, phase["x12_q2_1"])
            + cnot(0, 1)
            + x12(1, phase["x12_q2_2"])
            + x01(1, phase["x01_q2_2"])
            + x12(1, phase["x12_q2_3"])
            + cnot(2, 1)
            + x12(1, phase["x12_q2_4"])
            + cnot(1, 2)
        )
    if case == CASES[1]:
        # Disjoint Q1/Q2/Q3 rotations are intentionally moved left or right so
        # the existing layerizer realizes the verified 15-layer schedule.
        return (
            [
                gate("01_half", 0, phase=phase["x01_q1_pre"]),
                gate("01_half", 1, phase=-PI / 2),
                gate("01_half", 0, phase=phase["x01_q1_pre"]),
                gate("cz", 0, 1),
                gate("01_half", 1, phase=PI / 2),
            ]
            + x12(1, phase["x12_q2_1"])
            + cnot(2, 1)
            + x12(1, phase["x12_q2_2"])
            + [
                gate("01_half", 1, phase=-PI / 2),
                gate("cz", 0, 1),
                gate("01_half", 0, phase=phase["x01_q1_post"]),
                gate("01_half", 1, phase=PI / 2),
                gate("01_half", 0, phase=phase["x01_q1_post"]),
                gate("12_pi", 1, phase=phase["x12_q2_3"]),
                gate("01_half", 2, phase=-PI / 2),
                gate("cz", 1, 2),
                gate("01_half", 2, phase=PI / 2),
            ]
        )
    if case == CASES[2]:
        paired = paired_rotation_parameters(paired_rotation)
        unitary = _complex_matrix(paired["su2_matrix"])
        unitary *= np.exp(1j * paired["common_phase_rad"])
        return (
            x01(1, phase["x01_q2_encode"])
            + x12(1, phase["x12_q2_encode"])
            + cnot(0, 1)
            + x01(1, phase["x01_q2_xa_1"])
            + x12(1, phase["x12_q2_xa"])
            + x01(1, phase["x01_q2_xa_2"])
            + single_01_unitary(1, unitary)
            + cnot(1, 2)
            + cnot(2, 1)
            + single_01_unitary(2, unitary.conj().T)
            + x12(1, phase["x12_q2_decode"])
        )
    raise ValueError(f"Unknown case {case!r}; choose one of {CASES}.")


def _embedded_01_matrix(operation: Mapping[str, Any]) -> np.ndarray:
    """Represent one embedded-01 pulse/frame while leaving level 2 explicit."""
    name = str(operation["gate"])
    phase = float(operation["phase"])
    if name == "01_half":
        cosine = math.cos(PI / 4)
        sine = math.sin(PI / 4)
        return np.array(
            [
                [cosine, -1j * sine * np.exp(-1j * phase), 0.0],
                [-1j * sine * np.exp(1j * phase), cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )
    if name == "frame":
        return np.diag(np.exp(-1j * phase * np.arange(3)))
    if name == "frame_01":
        return np.diag([1.0, np.exp(-1j * phase), np.exp(-1j * phase)])
    if name == "frame_12":
        return np.diag([1.0, 1.0, np.exp(-1j * phase)])
    raise ValueError(f"Operation {name!r} is not an embedded-01 instruction.")


def _synthesise_embedded_01(qubit: int, matrix: np.ndarray) -> list[dict[str, Any]]:
    """Synthesize an embedded 01 block with one X/2 when algebraically possible."""
    relative = matrix[:2, :2] / matrix[2, 2]
    alpha, beta, theta, gamma = _unitary_zyz(relative)
    if abs(theta - PI / 2) < 1e-10:
        return (
            embedded_rz(qubit, gamma)
            + [gate("01_half", qubit, phase=PI / 2)]
            + embedded_rz(qubit, beta)
            + [gate("frame_12", qubit, phase=alpha)]
        )
    return single_01_unitary(qubit, relative)


def _compress_adjacent_01_blocks(
    operations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine adjacent 01/frame blocks without changing their qutrit unitary."""
    result: list[dict[str, Any]] = []
    allowed = {"01_half", *FRAME_GATES}
    index = 0
    while index < len(operations):
        operation = operations[index]
        if operation["gate"] not in allowed:
            result.append(operation)
            index += 1
            continue
        end = index + 1
        while (
            end < len(operations)
            and operations[end]["gate"] in allowed
            and operations[end]["qubits"] == operation["qubits"]
        ):
            end += 1
        block = operations[index:end]
        if sum(item["gate"] == "01_half" for item in block) > 2:
            matrix = np.eye(3, dtype=complex)
            for item in block:
                matrix = _embedded_01_matrix(item) @ matrix
            result.extend(_synthesise_embedded_01(operation["qubits"][0], matrix))
        else:
            result.extend(block)
        index = end
    return result


def _idle_windows(
    layers: list[list[dict[str, Any]]], qubit: int,
) -> list[tuple[int, int]]:
    """Find inclusive non-CZ layer windows where one qutrit has no instruction."""
    windows: list[tuple[int, int]] = []
    start: int | None = None
    for layer_index, layer in enumerate(layers):
        blocked = any(item["gate"] == "cz" for item in layer) or any(
            qubit in item["qubits"] for item in layer
        )
        if not blocked and start is None:
            start = layer_index
        elif blocked and start is not None:
            if layer_index - start >= 2:
                windows.append((start, layer_index - 1))
            start = None
    if start is not None and len(layers) - start >= 2:
        windows.append((start, len(layers) - 1))
    return windows


def _allocate_pairs(capacities: list[int], pairs: int) -> list[int]:
    """Spread a per-qutrit pair budget chronologically across idle windows."""
    if pairs > sum(capacities):
        raise ValueError(
            f"Requested {pairs} DD pairs, but the reduced schedule can hold only "
            f"{sum(capacities)} without increasing circuit depth."
        )
    allocation = [0] * len(capacities)
    remaining = pairs
    while remaining:
        progressed = False
        for index, capacity in enumerate(capacities):
            if allocation[index] >= capacity:
                continue
            allocation[index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise AssertionError("DD allocation stopped before exhausting the pair budget.")
    return allocation


def _pair_positions(first: int, last: int, pairs: int) -> list[int]:
    """Place 2N native X pulses as evenly as possible within one idle window."""
    if pairs == 0:
        return []
    length = last - first + 1
    pulse_count = 2 * pairs
    if pulse_count > length:
        raise ValueError("A DD window cannot contain the requested native X pulses.")
    positions = [
        first + math.floor((index + 0.5) * length / pulse_count)
        for index in range(pulse_count)
    ]
    if len(set(positions)) != pulse_count:
        raise AssertionError("DD pulse placement produced a duplicate physical layer.")
    return positions


def _insert_dd_pairs(
    layers: list[list[dict[str, Any]]], pairs: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Insert native XX plus a virtual 12-phase closure on Q1 and Q3."""
    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 0:
        raise ValueError("dd_pairs must be a non-negative integer.")
    result = [[dict(item) for item in layer] for layer in layers]
    summary: dict[str, Any] = {"pairs_per_protected_qutrit": pairs, "qubits": {}}
    for qubit in DD_QUBITS:
        windows = _idle_windows(result, qubit)
        capacities = [(last - first + 1) // 2 for first, last in windows]
        allocation = _allocate_pairs(capacities, pairs)
        placements = []
        pair_number = 0
        for (first, last), count, capacity in zip(windows, allocation, capacities):
            positions = _pair_positions(first, last, count)
            for pulse_index, layer_index in enumerate(positions):
                operation = native_x01(qubit)
                operation.update({
                    "dd": True,
                    "dd_pair": pair_number + pulse_index // 2,
                    "dd_pulse": pulse_index % 2,
                })
                result[layer_index].append(operation)
                if pulse_index % 2 == 1:
                    closure = gate("frame_12", qubit, phase=PI)
                    closure.update({"dd": True, "dd_phase_closure": True})
                    result[layer_index].append(closure)
            pair_number += count
            placements.append({
                "layers_1_based": [first + 1, last + 1],
                "capacity_pairs": capacity,
                "inserted_pairs": count,
                "x_layers_1_based": [index + 1 for index in positions],
            })
        summary["qubits"][f"Q{qubit + 1}"] = placements
    return result, summary


def _mark_explicit_layers(
    layers: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Flatten layers while retaining exact barrier locations as instruction metadata."""
    result: list[dict[str, Any]] = []
    for layer in layers:
        if not layer:
            continue
        marked = [dict(item) for item in layer]
        marked[0]["layer_start"] = True
        marked[-1]["layer_end"] = True
        result.extend(marked)
    return result


def dd_schedule_summary(
    case: str,
    dd_pairs: int,
    phases: Mapping[str, float] | None = None,
    paired_rotation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact DD allocation without creating hardware jobs."""
    operations = _base_core_operations(case, phases, paired_rotation)
    if case == CASES[2]:
        operations = _compress_adjacent_01_blocks(operations)
    layers = physical_layers(operations)
    _, summary = _insert_dd_pairs(layers, dd_pairs)
    summary.update({"scheme": case, "base_physical_layers": len(layers)})
    return summary


def core(
    case: str,
    phases: Mapping[str, float] | None = None,
    paired_rotation: Mapping[str, Any] | None = None,
    dd_pairs: int = 0,
) -> list[dict[str, Any]]:
    """Return the optimized native schedule with optional depth-neutral XX DD pairs."""
    operations = _base_core_operations(case, phases, paired_rotation)
    if case == CASES[2]:
        operations = _compress_adjacent_01_blocks(operations)
    layers = physical_layers(operations)
    layers, _ = _insert_dd_pairs(layers, dd_pairs)
    return _mark_explicit_layers(layers)


def basis(state: str) -> list[dict[str, Any]]:
    """Prepare a 000..222 product state from 000."""
    if len(state) != 3 or any(level not in "012" for level in state):
        raise ValueError(f"Invalid qutrit basis state {state!r}.")
    operations: list[dict[str, Any]] = []
    for qubit, level in enumerate(state):
        if level in "12":
            operations += x01(qubit)
        if level == "2":
            operations += x12(qubit)
    return operations


def readout(mode: tuple[int, int, int]) -> list[dict[str, Any]]:
    """Pre-permute levels before the platform's binary measurement."""
    operations: list[dict[str, Any]] = []
    for qubit, setting in enumerate(mode):
        if setting == 1:
            operations += x01(qubit)
        elif setting == 2:
            operations += x01(qubit) + x12(qubit) + x01(qubit)
        elif setting != 0:
            raise ValueError(f"Invalid readout mode {setting}.")
    return operations


def frame(angles: tuple[float, float, float]) -> list[dict[str, Any]]:
    """Record a qutrit harmonic virtual-Z frame without a zero-amplitude pulse."""
    return [gate("frame", qubit, phase=angle) for qubit, angle in enumerate(angles) if angle]


def paired_transfer_jobs() -> Iterator[dict[str, Any]]:
    """Tomograph the C23-C32 block on Q3 while Q2 occupies physical |2>."""
    preparations = {
        "0": [],
        "1": x01(2),
        "+": [gate("01_half", 2, phase=PI / 2)],
        "-": [gate("01_half", 2, phase=-PI / 2)],
        "+i": [gate("01_half", 2, phase=PI)],
        "-i": [gate("01_half", 2, phase=0.0)],
    }
    for spectator, state, axis, mode in itertools.product(
        (0, 1), preparations, "XYZ", READOUT_MODES
    ):
        initial = [str(spectator), "2", "0"]
        operations = (
            basis("".join(initial))
            + preparations[state]
            + cnot(1, 2)
            + cnot(2, 1)
            + analyze_axis(2, axis)
            + readout((mode, mode, mode))
        )
        safe_state = state.replace("+", "p").replace("-", "m")
        yield job(
            "paired_transfer", CASES[2],
            f"o{spectator}_s{safe_state}_a{axis}_m{mode}", operations,
            spectator=spectator, input=state, axis=axis, mode=mode,
        )


def job(
    stage: str,
    case: str,
    tag: str,
    operations: list[dict[str, Any]],
    **metadata: Any,
) -> dict[str, Any]:
    """Create a manifest entry with a stable, descriptive identifier."""
    return {
        "id": f"{stage}__{case}__{tag}",
        "stage": stage,
        "case": case,
        "operations": operations,
        "metadata": metadata,
    }


def all_modes() -> Iterator[tuple[int, int, int]]:
    """Enumerate the 27 pre-measurement level mappings."""
    yield from itertools.product(READOUT_MODES, repeat=3)


def readout_local_jobs() -> Iterator[dict[str, Any]]:
    """Calibrate three local 3x3 binary response matrices (27 settings)."""
    for qubit, prepared, measured in itertools.product(range(3), READOUT_MODES, READOUT_MODES):
        state = ["0"] * 3
        state[qubit] = str(prepared)
        mode = [0] * 3
        mode[qubit] = measured
        yield job(
            "readout_local",
            "shared",
            f"q{qubit}_s{prepared}_m{measured}",
            basis("".join(state)) + readout(tuple(mode)),
            qubit=qubit,
            prepared=prepared,
            measured=measured,
        )


def readout_joint_jobs() -> Iterator[dict[str, Any]]:
    """Optional 27x27 full response check when joint readout is correlated."""
    for state in itertools.product("012", repeat=3):
        state_str = "".join(state)
        for mode in all_modes():
            tag = f"s{state_str}_m{''.join(map(str, mode))}"
            yield job("readout_joint", "shared", tag, basis(state_str) + readout(mode))


def readout_povm_jobs() -> Iterator[dict[str, Any]]:
    """Check whether binary mapping depends on 01/02/12 coherence phase."""
    states = {"plus": PI / 2, "minus": -PI / 2, "plus_i": PI, "minus_i": 0.0}
    for qubit, subspace, label, mapping in itertools.product(
        range(3), ("01", "02", "12"), states, READOUT_MODES
    ):
        operations = [gate("01_half", qubit, phase=states[label])]
        if subspace in ("02", "12"):
            operations += x12(qubit)
        if subspace == "12":
            operations += x01(qubit)
        mode = [0] * 3
        mode[qubit] = mapping
        tag = f"q{qubit}_{subspace}_{label}_m{mapping}"
        yield job(
            "readout_povm", "shared", tag, operations + readout(tuple(mode)),
            qubit=qubit, subspace=subspace, label=label, mapping=mapping,
        )


def single_gate_jobs() -> Iterator[dict[str, Any]]:
    """Population/return diagnostics for 01, 12, and XA operations."""
    for qubit in range(3):
        for name, repetitions in (("01_half", (0, 1, 2, 4)), ("12_pi", (0, 1, 2, 3))):
            initial = ["0"] * 3
            if name == "12_pi":
                initial[qubit] = "1"
            for repetitions_count in repetitions:
                pulse = (
                    [gate("01_half", qubit)]
                    if name == "01_half"
                    else x12(qubit)
                )
                for mode in READOUT_MODES:
                    mapping = [0] * 3
                    mapping[qubit] = mode
                    operations = basis("".join(initial)) + pulse * repetitions_count
                    yield job(
                        "single_gate",
                        "shared",
                        f"q{qubit}_{name}_n{repetitions_count}_m{mode}",
                        operations + readout(tuple(mapping)),
                    )
        for prepared in (0, 2):
            initial = ["0"] * 3
            initial[qubit] = str(prepared)
            for mode in READOUT_MODES:
                mapping = [0] * 3
                mapping[qubit] = mode
                yield job(
                    "single_gate",
                    "shared",
                    f"q{qubit}_xa_s{prepared}_m{mode}",
                    basis("".join(initial))
                    + x01(qubit)
                    + x12(qubit)
                    + x01(qubit)
                    + readout(tuple(mapping)),
                )


def frame_jobs() -> Iterator[dict[str, Any]]:
    """Test 01, 02, and 12 interference with known virtual frame updates."""
    for qubit, pair, angle, phase_index in itertools.product(
        range(3), ("01", "02", "12"), (0.0, PI / 2), range(PHASE_POINTS)
    ):
        operations = [gate("01_half", qubit, phase=PI / 2)]
        if pair in ("02", "12"):
            operations += x12(qubit)
        if pair == "12":
            operations += x01(qubit)
        operations += [gate("frame", qubit, phase=angle)]
        if pair == "12":
            operations += x01(qubit)
        if pair in ("02", "12"):
            operations += x12(qubit)
        analysis_phase = 2 * PI * phase_index / PHASE_POINTS
        operations += [gate("01_half", qubit, phase=analysis_phase)]
        yield job(
            "frame",
            "shared",
            f"q{qubit}_{pair}_z{int(angle > 0)}_p{phase_index:02d}",
            operations,
            qubit=qubit,
            pair=pair,
            angle=angle,
            phase=analysis_phase,
        )


def independent_frame_jobs() -> Iterator[dict[str, Any]]:
    """Verify the independent 01/12 frame accumulators required by Scheme C."""
    for qubit, pair, frame_gate, phase_index in itertools.product(
        (1, 2), ("01", "02", "12"), ("identity", "frame_01", "frame_12"),
        range(PHASE_POINTS),
    ):
        operations = [gate("01_half", qubit, phase=PI / 2)]
        if pair in ("02", "12"):
            operations += x12(qubit)
        if pair == "12":
            operations += x01(qubit)
        if frame_gate != "identity":
            operations += [gate(frame_gate, qubit, phase=PI / 2)]
        if pair == "12":
            operations += x01(qubit)
        if pair in ("02", "12"):
            operations += x12(qubit)
        analysis_phase = 2 * PI * phase_index / PHASE_POINTS
        operations += [gate("01_half", qubit, phase=analysis_phase)]
        yield job(
            "frame_independent", CASES[2],
            f"q{qubit}_{pair}_{frame_gate}_p{phase_index:02d}", operations,
            qubit=qubit, pair=pair, frame_gate=frame_gate, phase=analysis_phase,
        )


def cz_jobs() -> Iterator[dict[str, Any]]:
    """Conditional 01 Ramsey with both endpoint roles and spectator 0/1."""
    for pair, target, control_level, spectator_level, phase_index in itertools.product(
        ((0, 1), (1, 2)), (0, 1), (0, 1), (0, 1), range(PHASE_POINTS)
    ):
        target_qubit = pair[target]
        control_qubit = pair[1 - target]
        spectator = ({0, 1, 2} - set(pair)).pop()
        initial = ["0"] * 3
        initial[control_qubit] = str(control_level)
        initial[spectator] = str(spectator_level)
        phase = 2 * PI * phase_index / PHASE_POINTS
        operations = (
            basis("".join(initial))
            + [gate("01_half", target_qubit, phase=PI / 2)]
            + [gate("cz", *pair)]
            + [gate("01_half", target_qubit, phase=phase)]
        )
        tag = (
            f"q{pair[0]}q{pair[1]}_t{target_qubit}_c{control_level}"
            f"_o{spectator_level}_p{phase_index:02d}"
        )
        yield job(
            "cz",
            "shared",
            tag,
            operations,
            pair=list(pair),
            target=target_qubit,
            control=control_qubit,
            control_level=control_level,
            spectator=spectator,
            spectator_level=spectator_level,
            phase=phase,
        )


def cz_aux_jobs() -> Iterator[dict[str, Any]]:
    """Check CZ population return with a physical level-2 participant."""
    for pair, level, spectator_level in itertools.product(
        ((0, 1), (1, 2)), (0, 1), (0, 1)
    ):
        other = pair[0] if pair[1] == 1 else pair[1]
        spectator = ({0, 1, 2} - set(pair)).pop()
        initial = ["0"] * 3
        initial[1] = "2"
        initial[other] = str(level)
        initial[spectator] = str(spectator_level)
        for mode in all_modes():
            tag = (
                f"q{pair[0]}q{pair[1]}_n{level}_o{spectator_level}"
                f"_m{''.join(map(str, mode))}"
            )
            yield job(
                "cz_aux", "shared", tag,
                basis("".join(initial)) + [gate("cz", *pair)] + readout(mode),
                initial="".join(initial), pair=list(pair), mode=list(mode),
            )


def cz_aux_phase_jobs() -> Iterator[dict[str, Any]]:
    """Characterize 02/12 phase paths after CZ using the corrected Ref branch."""
    for pair, subspace, neighbor_level, spectator_level, phase_index in itertools.product(
        ((0, 1), (1, 2)), ("02", "12"), (0, 1), (0, 1), range(PHASE_POINTS)
    ):
        neighbor = pair[0] if pair[1] == 1 else pair[1]
        spectator = ({0, 1, 2} - set(pair)).pop()
        initial = ["0"] * 3
        initial[neighbor] = str(neighbor_level)
        initial[spectator] = str(spectator_level)
        operations = basis("".join(initial)) + [gate("01_half", 1, phase=PI / 2)] + x12(1)
        if subspace == "12":
            operations += x01(1)
        operations += [gate("cz", *pair)]
        if subspace == "12":
            operations += x01(1)
        phase = 2 * PI * phase_index / PHASE_POINTS
        operations += x12(1) + [gate("01_half", 1, phase=phase)]
        tag = (
            f"q{pair[0]}q{pair[1]}_{subspace}_n{neighbor_level}"
            f"_o{spectator_level}_p{phase_index:02d}"
        )
        yield job(
            "cz_aux_phase", "shared", tag, operations,
            pair=list(pair), subspace=subspace, neighbor=neighbor,
            neighbor_level=neighbor_level, spectator=spectator,
            spectator_level=spectator_level, phase=phase,
        )


def truth_jobs(
    case: str, angles: tuple[float, float, float],
    core_phases: Mapping[str, float] | None = None,
    paired_rotation: Mapping[str, Any] | None = None,
    dd_pairs: int = 0,
) -> Iterator[dict[str, Any]]:
    """Four required truth-table inputs with complete joint leakage mappings."""
    for state in TARGETS[case]:
        for mode in all_modes():
            tag = f"s{state}_m{''.join(map(str, mode))}"
            operations = (
                basis(state) + core(case, core_phases, paired_rotation, dd_pairs)
                + frame(angles) + readout(mode)
            )
            yield job(
                "truth", case, tag, operations,
                input=state, target=TARGETS[case][state], mode=list(mode),
                dd_pairs=dd_pairs,
            )


def ramsey_jobs(
    case: str, angles: tuple[float, float, float], corrected: bool,
    core_phases: Mapping[str, float] | None = None,
    paired_rotation: Mapping[str, Any] | None = None,
    dd_pairs: int = 0,
) -> Iterator[dict[str, Any]]:
    """Three output path-phase scans, each prepared without extra CZ."""
    applied_angles = angles if corrected else (0.0, 0.0, 0.0)
    for pair_index, (left, right, target_qubit) in enumerate(RAMSEY_PAIRS[case]):
        input_qubit = next(index for index in range(3) if left[index] != right[index])
        if left[input_qubit] != "0" or right[input_qubit] != "1":
            raise ValueError(
                f"Ramsey pair {left}/{right} cannot use the specified plus preparation."
            )
        for phase_index in range(PHASE_POINTS):
            phase = 2 * PI * phase_index / PHASE_POINTS
            operations = (
                basis(left)
                + [gate("01_half", input_qubit, phase=PI / 2)]
                + core(case, core_phases, paired_rotation, dd_pairs)
                + frame(applied_angles)
                + [gate("01_half", target_qubit, phase=phase)]
            )
            yield job(
                "ramsey_corrected" if corrected else "ramsey_raw",
                case,
                f"pair{pair_index}_p{phase_index:02d}",
                operations,
                pair=pair_index,
                input_pair=[left, right],
                output_pair=[TARGETS[case][left], TARGETS[case][right]],
                target_qubit=target_qubit,
                phase=phase,
                dd_pairs=dd_pairs,
            )


def prepare_tomo_state(qubit: int, state: str) -> list[dict[str, Any]]:
    """Prepare 0, 1, +, or +i with the project's rotation convention."""
    if state == "0":
        return []
    if state == "1":
        return x01(qubit)
    if state == "+":
        return [gate("01_half", qubit, phase=PI / 2)]
    if state == "+i":
        return [gate("01_half", qubit, phase=PI)]
    raise ValueError(f"Unsupported tomography preparation {state!r}.")


def analyze_axis(qubit: int, axis: str) -> list[dict[str, Any]]:
    """Rotate an X/Y eigenbasis onto the binary Z readout basis."""
    if axis == "X":
        return [gate("01_half", qubit, phase=-PI / 2)]
    if axis == "Y":
        return [gate("01_half", qubit, phase=0)]
    if axis == "Z":
        return []
    raise ValueError(f"Unsupported measurement axis {axis!r}.")


def tomo_jobs(
    case: str, angles: tuple[float, float, float],
    core_phases: Mapping[str, float] | None = None,
    paired_rotation: Mapping[str, Any] | None = None,
    dd_pairs: int = 0,
) -> Iterator[dict[str, Any]]:
    """Optional restricted 4-to-8 process tomography, including leakage."""
    active = (0, 2) if case == CASES[1] else (0, 1)
    for preparation, axes, mode in itertools.product(
        itertools.product(("0", "1", "+", "+i"), repeat=2),
        itertools.product("XYZ", repeat=3),
        all_modes(),
    ):
        operations: list[dict[str, Any]] = []
        for qubit, state in zip(active, preparation):
            operations += prepare_tomo_state(qubit, state)
        operations += core(case, core_phases, paired_rotation, dd_pairs) + frame(angles)
        for qubit, axis in enumerate(axes):
            operations += analyze_axis(qubit, axis)
        operations += readout(mode)
        tag = (
            f"s{preparation[0].replace('+', 'p')}{preparation[1].replace('+', 'p')}"
            f"_a{''.join(axes)}_m{''.join(map(str, mode))}"
        )
        yield job(
            "tomo", case, tag, operations,
            preparation=list(preparation), axes=list(axes), mode=list(mode),
            dd_pairs=dd_pairs,
        )


def coherence_jobs(
    case: str, angles: tuple[float, float, float],
    core_phases: Mapping[str, float] | None = None,
    paired_rotation: Mapping[str, Any] | None = None,
    dd_pairs: int = 0,
) -> Iterator[dict[str, Any]]:
    """Characterize the equal four-path superposition with 729 binary circuits."""
    active = (0, 2) if case == CASES[1] else (0, 1)
    for axes, mode in itertools.product(itertools.product("XYZ", repeat=3), all_modes()):
        operations = (
            prepare_tomo_state(active[0], "+")
            + prepare_tomo_state(active[1], "+")
            + core(case, core_phases, paired_rotation, dd_pairs)
            + frame(angles)
        )
        for qubit, axis in enumerate(axes):
            operations += analyze_axis(qubit, axis)
        operations += readout(mode)
        tag = f"a{''.join(axes)}_m{''.join(map(str, mode))}"
        yield job(
            "coherence", case, tag, operations, axes=list(axes), mode=list(mode),
            dd_pairs=dd_pairs,
        )


def stage_jobs(
    stage: str,
    case: str,
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    core_phases: Mapping[str, float] | None = None,
    paired_rotation: Mapping[str, Any] | None = None,
    dd_pairs: int = 0,
) -> Iterator[dict[str, Any]]:
    """Select a complete stage or one independently runnable experiment."""
    shared = {
        "readout_local": readout_local_jobs,
        "readout_joint": readout_joint_jobs,
        "readout_povm": readout_povm_jobs,
        "single_gate": single_gate_jobs,
        "frame": frame_jobs,
        "frame_independent": independent_frame_jobs,
        "cz": cz_jobs,
        "cz_aux": cz_aux_jobs,
        "cz_aux_phase": cz_aux_phase_jobs,
        "paired_transfer": paired_transfer_jobs,
    }
    if stage in shared:
        if core_phases:
            raise ValueError(f"Stage {stage} does not use circuit-core phase parameters.")
        yield from shared[stage]()
    elif stage == "truth":
        yield from truth_jobs(case, angles, core_phases, paired_rotation, dd_pairs)
    elif stage == "ramsey_raw":
        yield from ramsey_jobs(
            case, angles, corrected=False, core_phases=core_phases,
            paired_rotation=paired_rotation, dd_pairs=dd_pairs,
        )
    elif stage == "ramsey_corrected":
        yield from ramsey_jobs(
            case, angles, corrected=True, core_phases=core_phases,
            paired_rotation=paired_rotation, dd_pairs=dd_pairs,
        )
    elif stage == "tomo":
        yield from tomo_jobs(case, angles, core_phases, paired_rotation, dd_pairs)
    elif stage == "coherence":
        yield from coherence_jobs(case, angles, core_phases, paired_rotation, dd_pairs)
    else:
        raise ValueError(f"Unknown stage {stage!r}.")
