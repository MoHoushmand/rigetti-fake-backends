"""
FakeCepheus - Qiskit Noise Model
================================

Creates Qiskit noise models matching Cepheus-1-36Q hardware characteristics.

KEY FEATURE: Heterogeneous noise model - intra-chip edges use higher fidelity
(F_2Q = 99.5%) while inter-chip edges use lower fidelity (F_2Q = 99.0%)
to reflect intermodule coupler overhead.

Author: Daniel Mo Houshmand (QDaria AS)
"""

from typing import Dict, List, Optional, Any
import numpy as np

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import RZGate, SXGate, XGate, CZGate, IGate
from qiskit.circuit import Parameter, Reset, Measure, Delay

from ..specs import (
    CEPHEUS_SPECS,
    CEPHEUS_INTRA_CHIP_EDGES,
    CEPHEUS_INTER_CHIP_EDGES,
    CEPHEUS_EDGES,
    CEPHEUS_QUBITS,
    CepheusSpecs,
)


def create_cepheus_noise_model(
    specs: CepheusSpecs = CEPHEUS_SPECS,
    include_readout: bool = True,
    include_thermal: bool = True,
) -> NoiseModel:
    """
    Create a Qiskit NoiseModel for FakeCepheus with heterogeneous fidelities.

    Includes:
    - Depolarizing errors on 1Q gates (from F_1Q)
    - T1/T2 thermal relaxation (optional)
    - HETEROGENEOUS 2Q errors: intra-chip vs inter-chip fidelities
    - Asymmetric readout errors (optional)

    Args:
        specs: Cepheus hardware specifications
        include_readout: Include readout errors
        include_thermal: Include T1/T2 relaxation

    Returns:
        Configured NoiseModel
    """
    noise_model = NoiseModel()

    # Convert units
    t1 = specs.t1_us * 1e-6
    t2 = specs.t2_us * 1e-6
    gate_time_1q = specs.gate_time_1q_ns * 1e-9
    gate_time_2q = specs.gate_time_2q_ns * 1e-9

    # ----- Single-qubit gate error -----
    p_error_1q = 1 - specs.fidelity_1q

    if include_thermal and t1 > 0 and t2 > 0:
        error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        if p_error_1q > 0:
            depol_1q = depolarizing_error(p_error_1q, 1)
            error_1q = error_1q.compose(depol_1q)
    else:
        error_1q = (
            depolarizing_error(p_error_1q, 1) if p_error_1q > 0 else None
        )

    if error_1q is not None:
        noise_model.add_all_qubit_quantum_error(
            error_1q, ['rz', 'sx', 'x', 'id']
        )

    # ----- Two-qubit gate errors (HETEROGENEOUS) -----
    def _make_2q_error(fidelity_2q: float):
        """Create a 2Q error channel for the given fidelity."""
        p_error = 1 - fidelity_2q
        if include_thermal and t1 > 0 and t2 > 0:
            error_single = thermal_relaxation_error(t1, t2, gate_time_2q)
            error = error_single.tensor(error_single)
            if p_error > 0:
                depol = depolarizing_error(p_error, 2)
                error = error.compose(depol)
            return error
        else:
            return depolarizing_error(p_error, 2) if p_error > 0 else None

    # Intra-chip edges: higher fidelity
    error_2q_intra = _make_2q_error(specs.fidelity_2q_intra)
    if error_2q_intra is not None:
        for edge in CEPHEUS_INTRA_CHIP_EDGES:
            noise_model.add_quantum_error(
                error_2q_intra, ['cz'], [edge[0], edge[1]]
            )
            noise_model.add_quantum_error(
                error_2q_intra, ['cz'], [edge[1], edge[0]]
            )

    # Inter-chip edges: lower fidelity
    error_2q_inter = _make_2q_error(specs.fidelity_2q_inter)
    if error_2q_inter is not None:
        for edge in CEPHEUS_INTER_CHIP_EDGES:
            noise_model.add_quantum_error(
                error_2q_inter, ['cz'], [edge[0], edge[1]]
            )
            noise_model.add_quantum_error(
                error_2q_inter, ['cz'], [edge[1], edge[0]]
            )

    # ----- Readout errors (asymmetric) -----
    if include_readout:
        p_error = 1 - specs.readout_fidelity
        p_1_given_0 = p_error * 0.4
        p_0_given_1 = p_error * 0.6

        readout_error = ReadoutError([
            [1 - p_1_given_0, p_1_given_0],
            [p_0_given_1, 1 - p_0_given_1],
        ])
        noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


def create_cepheus_target(specs: CepheusSpecs = CEPHEUS_SPECS) -> Target:
    """
    Create a Qiskit Target for FakeCepheus.

    Defines basis gates, coupling map, gate times, and error rates.
    Uses heterogeneous error rates for intra-chip vs inter-chip CZ gates.

    Args:
        specs: Cepheus hardware specifications

    Returns:
        Configured Target
    """
    target = Target(
        description="Rigetti Cepheus-1-36Q simulated target",
        num_qubits=specs.num_qubits,
        dt=1e-9,
    )

    duration_1q = specs.gate_time_1q_ns * 1e-9
    duration_2q = specs.gate_time_2q_ns * 1e-9
    error_1q = 1 - specs.fidelity_1q
    error_2q_intra = 1 - specs.fidelity_2q_intra
    error_2q_inter = 1 - specs.fidelity_2q_inter

    # Single-qubit gates
    single_qubit_gates = [
        (IGate(), "id"),
        (RZGate(Parameter("theta")), "rz"),
        (SXGate(), "sx"),
        (XGate(), "x"),
    ]

    for gate, name in single_qubit_gates:
        props = {
            (q,): InstructionProperties(duration=duration_1q, error=error_1q)
            for q in CEPHEUS_QUBITS
        }
        target.add_instruction(gate, props, name=name)

    # CZ gates with heterogeneous error rates
    cz_props = {}
    for edge in CEPHEUS_INTRA_CHIP_EDGES:
        cz_props[(edge[0], edge[1])] = InstructionProperties(
            duration=duration_2q, error=error_2q_intra
        )
        cz_props[(edge[1], edge[0])] = InstructionProperties(
            duration=duration_2q, error=error_2q_intra
        )
    for edge in CEPHEUS_INTER_CHIP_EDGES:
        cz_props[(edge[0], edge[1])] = InstructionProperties(
            duration=duration_2q, error=error_2q_inter
        )
        cz_props[(edge[1], edge[0])] = InstructionProperties(
            duration=duration_2q, error=error_2q_inter
        )
    target.add_instruction(CZGate(), cz_props)

    # Measurement
    measure_props = {
        (q,): InstructionProperties(
            duration=1e-6, error=1 - specs.readout_fidelity
        )
        for q in CEPHEUS_QUBITS
    }
    target.add_instruction(Measure(), measure_props)

    # Reset
    reset_props = {
        (q,): InstructionProperties(duration=1e-6) for q in CEPHEUS_QUBITS
    }
    target.add_instruction(Reset(), reset_props)

    # Delay
    delay_props = {(q,): None for q in CEPHEUS_QUBITS}
    target.add_instruction(Delay(Parameter("t")), delay_props)

    return target
