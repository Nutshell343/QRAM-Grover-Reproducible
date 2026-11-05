"""Separate single- and two-qubit physical gate layers with clock barriers."""

from __future__ import annotations

from typing import Any, Iterable

SINGLE_GATES = frozenset(("01_half", "01_pi", "12_pi"))
TWO_QUBIT_GATES = frozenset(("cz",))
BARRIER = {"gate": "barrier", "qubits": [], "phase": 0.0}
FRAME_GATES = frozenset(("frame", "frame_01", "frame_12"))


def _barrier(schedule: list[dict[str, Any]]) -> None:
    """Append one clock barrier without creating duplicate empty layers."""
    if schedule and schedule[-1]["gate"] != "barrier":
        schedule.append(dict(BARRIER))


def layerize(operations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve instruction order while packing disjoint gates into clock layers.

    Frames are software-only updates. A physical layer uses each qubit at most
    once and contains either single-qubit gates or two-qubit gates, never both.
    """
    schedule: list[dict[str, Any]] = []
    kind: str | None = None
    occupied: set[int] = set()
    for operation in operations:
        if operation.get("layer_start") and kind is not None:
            _barrier(schedule)
            kind = None
            occupied.clear()
        name = operation["gate"]
        if name in FRAME_GATES:
            schedule.append(operation)
            if operation.get("layer_end") and kind is not None:
                _barrier(schedule)
                kind = None
                occupied.clear()
            continue
        if name in SINGLE_GATES:
            next_kind = "single"
        elif name in TWO_QUBIT_GATES:
            next_kind = "two_qubit"
        else:
            raise ValueError(f"Unsupported physical gate for layer scheduling: {name!r}.")
        qubits = set(operation["qubits"])
        if not qubits:
            raise ValueError(f"Physical operation {name!r} has no qubits.")
        if kind is not None and (kind != next_kind or occupied.intersection(qubits)):
            _barrier(schedule)
            occupied.clear()
        schedule.append(operation)
        kind = next_kind
        occupied.update(qubits)
        if operation.get("layer_end"):
            _barrier(schedule)
            kind = None
            occupied.clear()
    if kind is not None:
        _barrier(schedule)
    return schedule


def physical_layers(operations: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Return the clock layers produced by :func:`layerize`, excluding barriers."""
    layers: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for operation in layerize(operations):
        if operation["gate"] == "barrier":
            if current:
                layers.append(current)
                current = []
        else:
            current.append(operation)
    if current:
        layers.append(current)
    return layers
