"""Compile R5/R6 through the existing pyQCat four-qutrit circuit compiler.

The inherited compiler uses timing_plan.layerize: disjoint one-qubit gates
share a layer; every transition between one- and two-qubit layers has a
barrier.  The 01 pi gate stays one native pulse.
"""

from __future__ import annotations

from app.temp_need.TCG_new.router_fidelity.experiment import RouterFidelityCircuit


class RouterR5R6Circuit(RouterFidelityCircuit):
    """Distinct registered experiment name; same checked native gate backend."""

