"""
FakeNovera - PyQuil Backend Implementation
==========================================

Native Rigetti implementation using PyQuil's noise modeling and QVM.

This provides authentic Rigetti ecosystem support with:
- Native Quil assembly
- QVM simulation with realistic noise
- quilc compilation
- Compatible with Rigetti QCS for real hardware

Prerequisites:
    docker run -d -p 5555:5555 rigetti/quilc -R
    docker run -d -p 5000:5000 rigetti/qvm -S

Author: Daniel Mo Houshmand (QDaria AS)
License: MIT
"""

from typing import List, Optional, Dict, Any
import numpy as np

# Check for PyQuil availability
try:
    from pyquil import Program
    from pyquil.quilbase import Declare
    from pyquil.gates import RX, RY, RZ, CZ, MEASURE, I
    from pyquil.noise import (
        NoiseModel,
        KrausModel,
        damping_kraus_map,
        dephasing_kraus_map,
        combine_kraus_maps,
        add_decoherence_noise,
    )
    from pyquil.quantum_processor import NxQuantumProcessor
    PYQUIL_AVAILABLE = True
except ImportError:
    PYQUIL_AVAILABLE = False

from ..specs import (
    NoveraSpecs,
    NOVERA_SPECS,
    NOVERA_EDGES,
    NOVERA_QUBITS,
    get_topology,
)


def _check_pyquil():
    """Raise ImportError if PyQuil is not available."""
    if not PYQUIL_AVAILABLE:
        raise ImportError(
            "PyQuil is required for native Rigetti support.\n"
            "Install with: pip install pyquil\n"
            "For Qiskit compatibility, use: from fakenovera.qiskit import FakeNovera"
        )


def _get_bidirectional_edges() -> List[tuple]:
    """Get edges in both directions for symmetric 2Q gates."""
    edges = []
    for a, b in NOVERA_EDGES:
        edges.append((a, b))
        edges.append((b, a))
    return edges


def _compute_kraus_ops(
    t1: float,
    t2: float,
    gate_time: float,
) -> List[np.ndarray]:
    """
    Compute combined Kraus operators for T1/T2 decoherence noise.

    Combines amplitude damping (T1) and pure dephasing (T2) channels.
    Depolarizing noise from gate infidelity is captured separately via
    the ``fidelity`` parameter in ``KrausModel``, which PyQuil uses to
    scale the overall gate error budget.

    Args:
        t1: T1 relaxation time in seconds.
        t2: T2 dephasing time in seconds.
        gate_time: Gate duration in seconds.

    Returns:
        List of 2x2 Kraus operator matrices.
    """
    _check_pyquil()

    # Amplitude damping probability
    p_decay = 1 - np.exp(-gate_time / t1) if t1 > 0 else 0

    # Pure dephasing probability
    if t2 < 2 * t1 and t2 > 0:
        t_phi = 1 / (1/t2 - 1/(2*t1))
        p_dephase = 1 - np.exp(-gate_time / t_phi)
    else:
        p_dephase = 0.0

    # Get Kraus maps
    damping = damping_kraus_map(p_decay)
    dephasing = dephasing_kraus_map(p_dephase)

    return combine_kraus_maps(damping, dephasing)


def create_noise_model(specs: NoveraSpecs = NOVERA_SPECS) -> "NoiseModel":
    """
    Create a PyQuil NoiseModel for FakeNovera.

    Includes:
    - T1/T2 decoherence on all gates
    - Depolarizing noise from gate infidelity
    - Asymmetric readout errors

    Args:
        specs: Hardware specifications

    Returns:
        Configured PyQuil NoiseModel
    """
    _check_pyquil()

    # Convert units to seconds
    t1 = specs.t1_us * 1e-6
    t2 = specs.t2_us * 1e-6
    gate_time_1q = specs.gate_time_1q_ns * 1e-9
    gate_time_2q = specs.gate_time_2q_ns * 1e-9

    # Gate noise definitions
    gate_noise: List[KrausModel] = []

    # Single-qubit gates (T1/T2 decoherence; gate infidelity via KrausModel fidelity)
    kraus_1q = _compute_kraus_ops(t1, t2, gate_time_1q)

    # Native 1Q gates: RX(k*pi/2) and RZ(theta)
    # RY is NOT native on Rigetti hardware but quilc decomposes it into
    # RX/RZ; we include it here so user programs work without compilation.
    for q in NOVERA_QUBITS:
        for gate_name in ["RX", "RZ", "RY"]:
            gate_noise.append(KrausModel(
                gate=gate_name,
                params=tuple(),
                targets=(q,),
                kraus_ops=kraus_1q,
                fidelity=specs.fidelity_1q
            ))

    # Two-qubit CZ gates (T1/T2 decoherence; gate infidelity via KrausModel fidelity)
    kraus_2q = _compute_kraus_ops(t1, t2, gate_time_2q)

    for edge in _get_bidirectional_edges():
        gate_noise.append(KrausModel(
            gate="CZ",
            params=tuple(),
            targets=edge,
            kraus_ops=kraus_2q,
            fidelity=specs.fidelity_2q
        ))

    # Readout errors (asymmetric)
    p_error = 1 - specs.readout_fidelity
    p_1_given_0 = p_error * 0.4  # False positive
    p_0_given_1 = p_error * 0.6  # False negative

    assignment_probs: Dict[int, np.ndarray] = {}
    for q in NOVERA_QUBITS:
        assignment_probs[q] = np.array([
            [1 - p_1_given_0, p_0_given_1],
            [p_1_given_0, 1 - p_0_given_1]
        ])

    return NoiseModel(gates=gate_noise, assignment_probs=assignment_probs)


def create_quantum_processor(
    specs: NoveraSpecs = NOVERA_SPECS,
    noisy: bool = True
) -> "NxQuantumProcessor":
    """
    Create an NxQuantumProcessor for FakeNovera.

    Args:
        specs: Hardware specifications
        noisy: Whether to include noise model

    Returns:
        Configured NxQuantumProcessor
    """
    _check_pyquil()

    topology = get_topology()
    processor = NxQuantumProcessor(topology=topology)

    if noisy:
        processor.noise_model = create_noise_model(specs)

    return processor


def get_fakenovera_qc(
    name: str = "FakeNovera-9Q",
    specs: NoveraSpecs = NOVERA_SPECS,
    noisy: bool = True,
    qvm_url: str = "http://127.0.0.1:5000",
    quilc_url: str = "tcp://127.0.0.1:5555",
) -> Any:
    """
    Get a QuantumComputer configured for FakeNovera simulation.

    Creates a complete quantum computing environment with:
    - QVM backend with Novera noise model
    - quilc compiler configured for Novera topology
    - Native gate set: RX(k*pi/2), RZ(theta), CZ

    Prerequisites:
        docker run -d -p 5555:5555 rigetti/quilc -R
        docker run -d -p 5000:5000 rigetti/qvm -S

    Args:
        name: Name for the QuantumComputer
        specs: Hardware specifications
        noisy: Whether to include noise model
        qvm_url: URL for QVM service
        quilc_url: URL for quilc compiler

    Returns:
        Configured QuantumComputer

    Example:
        >>> qc = get_fakenovera_qc()
        >>> p = Program()
        >>> p += H(4)
        >>> p += CZ(4, 1)
        >>> p.measure_all()
        >>> result = qc.run(qc.compile(p))
    """
    _check_pyquil()

    from pyquil import get_qc

    processor = create_quantum_processor(specs, noisy)

    try:
        qc = get_qc("9q-square-qvm", noisy=noisy)

        if noisy and processor.noise_model is not None:
            qc.qam.noise_model = processor.noise_model

        return qc

    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to QVM/quilc. Ensure they are running:\n"
            f"  docker run -d -p 5555:5555 rigetti/quilc -R\n"
            f"  docker run -d -p 5000:5000 rigetti/qvm -S\n"
            f"Original error: {e}"
        )


# Alias for convenience
get_qc = get_fakenovera_qc


def add_novera_noise(
    program: "Program",
    specs: NoveraSpecs = NOVERA_SPECS
) -> "Program":
    """
    Add Novera decoherence noise to a program.

    Uses PyQuil's built-in add_decoherence_noise function.

    Args:
        program: Input Quil program
        specs: Hardware specifications

    Returns:
        Program with noise added
    """
    _check_pyquil()

    return add_decoherence_noise(
        program,
        T1=specs.t1_us * 1e-6,
        T2=specs.t2_us * 1e-6,
        gate_time_1q=specs.gate_time_1q_ns * 1e-9,
        gate_time_2q=specs.gate_time_2q_ns * 1e-9,
    )


def get_noise_pragmas(specs: NoveraSpecs = NOVERA_SPECS) -> str:
    """
    Generate Quil PRAGMA statements for noise simulation.

    Args:
        specs: Hardware specifications

    Returns:
        Quil PRAGMA string for prepending to programs
    """
    lines = [
        "# FakeNovera Noise Model",
        f"# T1 = {specs.t1_us} μs, T2 = {specs.t2_us} μs",
        f"# F_1Q = {specs.fidelity_1q*100}%, F_2Q = {specs.fidelity_2q*100}%",
        "",
    ]

    # Readout POVMs
    p_error = 1 - specs.readout_fidelity
    p00 = 1 - p_error * 0.4
    p01 = p_error * 0.4
    p10 = p_error * 0.6
    p11 = 1 - p_error * 0.6

    for q in NOVERA_QUBITS:
        lines.append(f'PRAGMA READOUT-POVM {q} "({p00:.6f} {p01:.6f} {p10:.6f} {p11:.6f})"')

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("FakeNovera PyQuil Backend")
    print("=" * 50)

    if not PYQUIL_AVAILABLE:
        print("PyQuil not installed. Install with: pip install pyquil")
    else:
        print("PyQuil available!")
        print("\nNoise model configuration:")
        noise_model = create_noise_model()
        print(f"  Gate noise definitions: {len(noise_model.gates)}")
        print(f"  Qubits with readout error: {len(noise_model.assignment_probs)}")

        print("\nQuil PRAGMA statements:")
        print(get_noise_pragmas())
