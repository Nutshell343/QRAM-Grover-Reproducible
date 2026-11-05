"""Pulse-resolved four-Duffing model and compiler-faithful virtual frames.

Four-body states are evolved at every physical layer. With all unused
couplings zero the Hamiltonian factorizes EXACTLY, so locally integrated
pulse propagators can be applied by tensor contractions. This is not a
three-qubit bank clone. A separate direct full-Hamiltonian solver checks it.
Units: GHz, ns, Hamiltonian angular frequencies rad/ns.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from bootstrap import physical_layers

TAU = 2 * np.pi


def encode(value):
    return np.stack([np.asarray(value).real, np.asarray(value).imag], axis=-1).tolist()


def decode(value):
    a = np.asarray(value)
    return a[..., 0] + 1j * a[..., 1]


def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def occupation_index(word, levels):
    return int(np.ravel_multi_index(tuple(map(int, word)), (levels,) * len(word)))


def ramp_weight(t, duration, ramp):
    ramp = min(ramp, duration / 2)
    if ramp <= 0:
        return 1.
    if t < ramp:
        return .5 * (1 - np.cos(np.pi * t / ramp))
    if t > duration - ramp:
        return .5 * (1 - np.cos(np.pi * (duration - t) / ramp))
    return 1.


def apply_axes(state, operator, axes, levels, n=4):
    """Works for one state or a matrix of propagated input columns."""
    matrix = np.asarray(state)
    columns = 1 if matrix.ndim == 1 else matrix.shape[1]
    axes = tuple(axes)
    rest = tuple(q for q in range(n) if q not in axes)
    order = axes + rest + (n,)
    tensor = matrix.reshape((levels,) * n + (columns,)).transpose(order)
    result = operator @ tensor.reshape(levels ** len(axes), -1)
    result = result.reshape((levels,) * n + (columns,)).transpose(np.argsort(order))
    return result.reshape(matrix.shape)


def target_rotation(levels, lower, theta, phase=0.):
    u = np.eye(levels, dtype=complex)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    u[lower, lower] = u[lower + 1, lower + 1] = c
    u[lower, lower + 1] = -1j * s * np.exp(-1j * phase)
    u[lower + 1, lower] = -1j * s * np.exp(1j * phase)
    return u


@dataclass
class SinglePulse:
    wire: int
    gate: str
    amplitude: float
    detuning: float
    drag: float
    duration: float
    corrections: tuple[float, float] = (0., 0.)


@dataclass
class CZPulse:
    edge: tuple[int, int]
    tuning_wire: int
    frequency: float
    duration: float
    ramp: float
    peak_fraction: float = 1.
    corrections: tuple[float, float] = (0., 0.)


class PulseModel:
    def __init__(self, config, levels=4, dt=.1, cache_limit=24000):
        self.config = config
        self.levels, self.dt = levels, dt
        self.frequencies = np.asarray(config["frequencies_ghz"])
        self.alpha = np.asarray(config["anharmonicities_ghz"])
        self.rotating = config["rotating_frequency_ghz"]
        self.edges = [tuple(edge) for edge in config["edges"]]
        self.rho = dict(zip(self.edges, config["rho"]))
        self.n = np.arange(levels)
        self.a = np.diag(np.sqrt(np.arange(1, levels)), 1).astype(complex)
        self.ad = self.a.conj().T
        self.energy = TAU * (self.frequencies[:, None] - self.rotating) * self.n
        self.energy += np.pi * self.alpha[:, None] * self.n * (self.n - 1)
        self.words = np.array(list(np.ndindex(*(levels,) * 4)))
        self.total_energy = sum(self.energy[q, self.words[:, q]] for q in range(4))
        self.single: dict[tuple[int, str], SinglePulse] = {}
        self.cz: dict[tuple[int, int], CZPulse] = {}
        self.primitive_cache = {}
        self.prefix_cache = OrderedDict()
        self.cache_limit = cache_limit
        self.dimension = levels ** 4
        self.last_normalization_error = 0.

    def clear_caches(self):
        self.primitive_cache.clear()
        self.prefix_cache.clear()

    def singles_raw(self, pulse: SinglePulse, dt=None):
        """Integrate in carrier frame; restore common rotating frame."""
        q = pulse.wire
        lower = int(pulse.gate == "12_pi")
        carrier = self.frequencies[q] + lower * self.alpha[q] + pulse.detuning
        energy = TAU * (self.frequencies[q] - carrier) * self.n
        energy += np.pi * self.alpha[q] * self.n * (self.n - 1)
        steps = int(np.ceil(pulse.duration / (dt or self.dt)))
        step = pulse.duration / steps
        u = np.eye(self.levels, dtype=complex)
        for j in range(steps):
            t = (j + .5) * step
            envelope = .5 * (1 - np.cos(TAU * t / pulse.duration))
            derivative = np.pi / pulse.duration * np.sin(TAU * t / pulse.duration)
            drive = pulse.amplitude * (envelope - 1j * pulse.drag * derivative / self.alpha[q])
            h = np.diag(energy).astype(complex) + np.pi * (drive * self.ad + drive.conjugate() * self.a)
            eig, vectors = np.linalg.eigh(h)
            u = (vectors * np.exp(-1j * step * eig)) @ vectors.conj().T @ u
        return np.exp(-1j * TAU * (carrier - self.rotating) * pulse.duration * self.n)[:, None] * u

    def single_interaction(self, pulse, dt=None):
        return np.exp(1j * self.energy[pulse.wire] * pulse.duration)[:, None] * self.singles_raw(pulse, dt)

    def cz_raw(self, pulse: CZPulse, dt=None):
        """Exchange conserves total excitation; integrate every block, incl. 30."""
        a, b = pulse.edge
        number_a = np.repeat(self.n, self.levels)
        number_b = np.tile(self.n, self.levels)
        na, nb = np.diag(number_a), np.diag(number_b)
        exchange = np.kron(self.ad, self.a) + np.kron(self.a, self.ad)
        idle = self.energy[a, number_a] + self.energy[b, number_b]
        tuning_number = number_a if pulse.tuning_wire == a else number_b
        groups = [np.nonzero(number_a + number_b == total)[0]
                  for total in range(2 * self.levels - 1)]
        blocks = [np.eye(len(group), dtype=complex) for group in groups]
        steps = int(np.ceil(pulse.duration / (dt or self.dt)))
        step = pulse.duration / steps
        for j in range(steps):
            t = (j + .5) * step
            weight = ramp_weight(t, pulse.duration, pulse.ramp)
            f = self.frequencies.copy()
            f[pulse.tuning_wire] += weight * (pulse.frequency - f[pulse.tuning_wire])
            diag = idle + TAU * weight * (pulse.frequency - self.frequencies[pulse.tuning_wire]) * tuning_number
            strength = self.rho[pulse.edge] * np.sqrt(f[a] * f[b]) * weight * pulse.peak_fraction
            h = np.diag(diag) + TAU * strength * exchange
            for k, indices in enumerate(groups):
                eig, vectors = np.linalg.eigh(h[np.ix_(indices, indices)])
                blocks[k] = (vectors * np.exp(-1j * step * eig)) @ vectors.T @ blocks[k]
        u = np.zeros((self.levels ** 2, self.levels ** 2), dtype=complex)
        for indices, block in zip(groups, blocks):
            u[np.ix_(indices, indices)] = block
        return u

    def cz_interaction(self, pulse, dt=None):
        a, b = pulse.edge
        idle = (self.energy[a, :, None] + self.energy[b, None, :]).reshape(-1)
        return np.exp(1j * idle * pulse.duration)[:, None] * self.cz_raw(pulse, dt)

    def save_calibration(self, path):
        save_json(path, {"levels": self.levels, "dt_ns": self.dt,
                        "single": [asdict(v) for v in self.single.values()],
                        "cz": [asdict(v) for v in self.cz.values()]})

    def load_calibration(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in data["single"]:
            row["corrections"] = tuple(row["corrections"])
            pulse = SinglePulse(**row)
            self.single[(pulse.wire, pulse.gate)] = pulse
        for row in data["cz"]:
            row["edge"] = tuple(row["edge"])
            row["corrections"] = tuple(row["corrections"])
            pulse = CZPulse(**row)
            self.cz[pulse.edge] = pulse
        self.clear_caches()

    def compile_layer(self, layer, time, frames):
        """Match hardware: only carrier phases change at a virtual-frame update."""
        frames = frames.copy()
        operations = []
        for item in layer:
            name, wires, phase = item["gate"], item["qubits"], item["phase"]
            q = wires[0]
            if name == "frame":
                frames[q] += phase
                continue
            if name == "frame_01":
                frames[q, 0] += phase
                continue
            if name == "frame_12":
                frames[q, 1] += phase
                continue
            if name == "cz":
                edge = tuple(sorted(wires))
                pulse = self.cz[edge]
                operations.append((name, edge, pulse, 0.))
                for wire, correction in zip(edge, pulse.corrections):
                    frames[wire] += correction
            else:
                pulse = self.single[(q, name)]
                lower = int(name == "12_pi")
                effective = phase + frames[q, lower]
                operations.append((name, (q,), pulse, effective))
                frames[q] += np.asarray(pulse.corrections)
        duration = max([op[2].duration for op in operations] or [0.])
        if any(op[0] == "cz" for op in operations) and any(op[0] != "cz" for op in operations):
            raise AssertionError("Mixed single/CZ layer")
        return operations, duration, frames

    def advance(self, state, operations, time, duration):
        touched = set()
        for name, axes, pulse, phase in operations:
            key = json.dumps(asdict(pulse), sort_keys=True)
            if key not in self.primitive_cache:
                self.primitive_cache[key] = self.cz_raw(pulse) if name == "cz" else self.singles_raw(pulse)
            u = self.primitive_cache[key]
            if name != "cz":
                q = axes[0]
                lower = int(name == "12_pi")
                # The calibrated detuning is a pulse-local modulation. Its
                # phase resets at each gate; only the nominal f01/f12 carrier
                # follows the absolute clock. Treating detuning as absolute
                # would spuriously rotate every later gate by detuning*t_start.
                carrier_reference = self.frequencies[q] + lower * self.alpha[q]
                angle = phase - TAU * (carrier_reference - self.rotating) * time
                diagonal = np.exp(1j * angle * self.n)
                u = diagonal[:, None] * u * diagonal.conj()[None, :]
            state = apply_axes(state, u, axes, self.levels)
            for q in axes:
                if duration > pulse.duration:
                    state = apply_axes(state, np.diag(np.exp(-1j * self.energy[q] * (duration - pulse.duration))),
                                       (q,), self.levels)
            touched.update(axes)
        for q in set(range(4)) - touched:
            state = apply_axes(state, np.diag(np.exp(-1j * self.energy[q] * duration)), (q,), self.levels)
        return state

    def final_frame(self, state, time, frames):
        # Known nonlinear idle rotation defines the f01/f12 reference frames.
        diag = np.zeros((4, self.levels))
        for q in range(4):
            diag[q, 1:] = frames[q, 0] + np.arange(self.levels - 1) * frames[q, 1]
        angle = sum(diag[q, self.words[:, q]] for q in range(4))
        factors = np.exp(1j * self.total_energy * time - 1j * angle)
        return factors * state if state.ndim == 1 else factors[:, None] * state

    def evolve(self, operations, initial=None, trace=False, use_cache=True):
        if initial is None:
            state = np.eye(1, self.dimension, 0, dtype=complex).reshape(self.dimension)
        else:
            state = np.asarray(initial, dtype=complex).copy()
        cacheable = initial is None and use_cache and not trace
        frames = np.zeros((4, 2))
        time = 0.
        key = b"initial"
        history = []
        for index, layer in enumerate(physical_layers(operations)):
            key = hashlib.sha256(key + json.dumps(layer, sort_keys=True).encode()).digest()
            if cacheable and key in self.prefix_cache:
                state, time, frames = self.prefix_cache[key]
                state, frames = state.copy(), frames.copy()
                self.prefix_cache.move_to_end(key)
            else:
                pulses, duration, updated = self.compile_layer(layer, time, frames)
                state = self.advance(state, pulses, time, duration)
                time += duration
                frames = updated
                if cacheable:
                    self.prefix_cache[key] = (state.copy(), time, frames.copy())
                    if len(self.prefix_cache) > self.cache_limit:
                        self.prefix_cache.popitem(last=False)
            if trace:
                pop = abs(state) ** 2
                if pop.ndim == 1:
                    pop = pop[:, None]
                comp = np.all(self.words < 2, axis=1)
                above2 = np.any(self.words > 2, axis=1)
                history.append({"layer": index + 1, "time_ns": time,
                                "operations": layer, "p_outside_01": pop[~comp].sum(axis=0).tolist(),
                                "p_any_level3plus": pop[above2].sum(axis=0).tolist(),
                                "mean_level_population": [[float(pop[self.words[:, q] == k].mean(axis=1).sum())
                                                          for k in range(self.levels)] for q in range(4)]})
        error = float(np.max(abs(np.sum(abs(state) ** 2, axis=0) - 1)))
        self.last_normalization_error = max(self.last_normalization_error, error)
        return self.final_frame(state, time, frames), {"duration_ns": time, "layers": history,
                                                       "frames": frames.tolist(), "norm_error": error}

    def binary_probabilities(self, state):
        local = np.asarray(self.config["readout_p0_given_level"])[:, :self.levels]
        if local.shape[1] < self.levels:
            local = np.pad(local, ((0, 0), (0, self.levels - local.shape[1])), mode="edge")
        population = abs(state) ** 2
        response = np.ones((16, self.dimension))
        for b in range(16):
            for q, bit in enumerate(f"{b:04b}"):
                p0 = local[q, self.words[:, q]]
                response[b] *= p0 if bit == "0" else 1 - p0
        result = response @ population
        return np.maximum(result, 0) / result.sum()

    def direct_full_evolve(self, operations, initial, residual_fraction=0., rtol=2e-9):
        """Independent sparse-free full Hamiltonian RHS, including spectators.

        This deliberately bypasses the primitive cache and factorized advances.
        Constant residual couplings are an optional model stress test, not a
        measured hardware estimate. Drive phases use the absolute pulse clock.
        """
        state = np.asarray(initial, dtype=complex).copy()
        shape = state.shape
        time, frames = 0., np.zeros((4, 2))
        tensor_number = [self.words[:, q].astype(float) for q in range(4)]
        for layer in physical_layers(operations):
            compiled, duration, updated = self.compile_layer(layer, time, frames)
            if duration == 0:
                frames = updated
                continue
            start = time
            def rhs(t, flattened):
                psi = flattened.reshape(shape)
                local_t = t - start
                frequencies = self.frequencies.copy()
                diag = self.total_energy.copy()
                fractions = {edge: residual_fraction for edge in self.edges}
                drives = []
                for name, axes, pulse, phase in compiled:
                    if local_t > pulse.duration:
                        continue
                    if name == "cz":
                        weight = ramp_weight(local_t, pulse.duration, pulse.ramp)
                        q = pulse.tuning_wire
                        frequencies[q] += weight * (pulse.frequency - self.frequencies[q])
                        diag += TAU * (frequencies[q] - self.frequencies[q]) * tensor_number[q]
                        fractions[pulse.edge] = residual_fraction + weight * pulse.peak_fraction
                    else:
                        q = axes[0]
                        lower = int(name == "12_pi")
                        carrier = self.frequencies[q] + lower * self.alpha[q]
                        env = .5 * (1 - np.cos(TAU * local_t / pulse.duration))
                        deriv = np.pi / pulse.duration * np.sin(TAU * local_t / pulse.duration)
                        drive = pulse.amplitude * (env - 1j * pulse.drag * deriv / self.alpha[q])
                        drive *= np.exp(1j * phase - 1j * TAU * (carrier - self.rotating) * t
                                        - 1j * TAU * pulse.detuning * local_t)
                        drives.append((q, drive))
                result = diag * psi if psi.ndim == 1 else diag[:, None] * psi
                for edge, fraction in fractions.items():
                    if fraction == 0:
                        continue
                    a, b = edge
                    strength = TAU * self.rho[edge] * np.sqrt(frequencies[a] * frequencies[b]) * fraction
                    exchange = np.kron(self.ad, self.a) + np.kron(self.a, self.ad)
                    result += strength * apply_axes(psi, exchange, edge, self.levels)
                for q, drive in drives:
                    result += np.pi * apply_axes(psi, drive * self.ad + drive.conjugate() * self.a,
                                               (q,), self.levels)
                return (-1j * result).reshape(-1)
            solution = solve_ivp(rhs, (time, time + duration), state.reshape(-1), method="DOP853",
                                 rtol=rtol, atol=rtol / 20, max_step=.25)
            if not solution.success:
                raise RuntimeError(solution.message)
            state = solution.y[:, -1].reshape(shape)
            time += duration
            frames = updated
        return self.final_frame(state, time, frames)
