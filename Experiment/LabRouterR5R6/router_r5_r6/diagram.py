"""Export readable native-gate schedule diagrams and machine-readable layers."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Mapping

try:
    from app.temp_need.TCG_new.partial_cswap_vz4.timing_plan import physical_layers
except ModuleNotFoundError:  # offline workspace
    from partial_cswap_vz4.timing_plan import physical_layers  # type: ignore

from . import protocol as p
from .workflow import write_json

PHYSICAL = {"01_half", "01_pi", "12_pi", "cz"}


def validate_layers(operations: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Assert every physical layer is disjoint and homogeneous by arity."""
    layers = [[item for item in layer if item["gate"] in PHYSICAL]
              for layer in physical_layers(operations)]
    layers = [layer for layer in layers if layer]
    for index, layer in enumerate(layers, start=1):
        cz = [item for item in layer if item["gate"] == "cz"]
        if cz and len(cz) != len(layer):
            raise AssertionError(f"Layer {index} mixes single-qubit gates and CZ.")
        wires = [wire for item in layer for wire in item["qubits"]]
        if len(set(wires)) != len(wires):
            raise AssertionError(f"Layer {index} uses a qutrit twice.")
    return layers


def svg(layers: list[list[dict[str, Any]]], title: str) -> str:
    step, left, top, gap = 70, 92, 60, 82
    width, height = left + step * len(layers) + 36, top + gap * 3 + 58
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="16" y="25" font-size="18" font-family="Arial">{html.escape(title)}</text>']
    for wire, name in enumerate(("c", "i", "l", "r")):
        y = top + wire * gap
        parts += [f'<text x="30" y="{y + 6}" font-size="20" font-family="Arial">{name}</text>',
                  f'<line x1="65" y1="{y}" x2="{width - 16}" y2="{y}" '
                  'stroke="#222" stroke-width="2"/>']
    labels = {"01_half": "X01/2", "01_pi": "X01", "12_pi": "X12"}
    for column, layer in enumerate(layers):
        x = left + step * column
        for item in layer:
            wires = item["qubits"]
            if item["gate"] == "cz":
                y1, y2 = (top + wire * gap for wire in wires)
                parts += [f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" '
                          'stroke="#17429e" stroke-width="3"/>']
                parts += [f'<circle cx="{x}" cy="{y}" r="6" fill="#17429e"/>'
                          for y in (y1, y2)]
            else:
                y = top + wires[0] * gap
                color = "#17429e" if item["gate"] != "12_pi" else "#9d1642"
                parts += [f'<rect x="{x - 29}" y="{y - 18}" width="58" height="36" '
                          f'fill="{color}"/>',
                          f'<text x="{x}" y="{y + 5}" text-anchor="middle" '
                          f'font-size="13" fill="white" font-family="Arial">'
                          f'{labels[item["gate"]]}</text>']
        parts.append(f'<text x="{x}" y="{height - 14}" text-anchor="middle" '
                     f'font-size="11" fill="#555">{column + 1}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def export_circuit(directory: Path, scheme: str, direction: str,
                   paired: Mapping[str, Any] | None = None) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    operations = p.core(scheme, direction, paired)
    layers = validate_layers(operations)
    cz_count = sum(item["gate"] == "cz" for item in operations)
    required = 5 if scheme == "r5" else 6
    if cz_count != required:
        raise AssertionError(f"{scheme} expected {required} CZ, found {cz_count}.")
    payload = {"scheme": scheme, "direction": direction, "wires": ["c", "i", "l", "r"],
               "paired_calibrated": paired is not None if scheme == "r5" else None,
               "operations": operations, "layers": layers, "cz_count": cz_count,
               "physical_layers": len(layers),
               "constraint": "same-layer singles allowed; CZ never overlaps any single gate"}
    stem = f"{scheme}_{direction}"
    write_json(directory / f"{stem}_schedule.json", payload)
    path = directory / f"{stem}.svg"
    suffix = " | V pending (identity placeholder)" if scheme == "r5" and paired is None else ""
    path.write_text(svg(layers, f"{scheme.upper()} {direction} | {cz_count} CZ{suffix}"),
                    encoding="utf-8")
    return {"schedule": str(directory / f"{stem}_schedule.json"), "diagram": str(path),
            "cz_count": cz_count, "physical_layers": len(layers)}
