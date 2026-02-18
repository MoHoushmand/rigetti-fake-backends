"""
FakeNovera - Qiskit Compatibility Layer
=======================================

Qiskit-compatible implementation for users without QVM/quilc access.

This provides a BackendV2-compatible simulated backend using Qiskit's
AerSimulator with Novera-inspired noise parameters.

NOTE: For authentic Rigetti ecosystem support, use fakenovera.pyquil instead.

Example:
    from fakenovera.qiskit import FakeNovera

    backend = FakeNovera()
    qc = QuantumCircuit(9)
    qc.h(4)
    qc.cz(4, 1)
    qc.measure_all()
    job = backend.run(qc)

Author: Daniel Mo Houshmand (QDaria AS)
"""

from .backend import FakeNovera, FakeNoveraBackend
from .noise import create_novera_noise_model, create_novera_target

from ..specs import (
    NoveraSpecs,
    NOVERA_SPECS,
    NOVERA_SPECS_OFFICIAL,
    NOVERA_SPECS_ZURICH,
    NOVERA_EDGES,
    NOVERA_COUPLING_MAP,
    cumulative_fidelity,
    optimal_depth,
    print_specs,
)

__all__ = [
    # Backend
    "FakeNovera",
    "FakeNoveraBackend",
    # Noise
    "create_novera_noise_model",
    "create_novera_target",
    # Specifications
    "NoveraSpecs",
    "NOVERA_SPECS",
    "NOVERA_SPECS_OFFICIAL",
    "NOVERA_SPECS_ZURICH",
    "NOVERA_EDGES",
    "NOVERA_COUPLING_MAP",
    # Analysis
    "cumulative_fidelity",
    "optimal_depth",
    "print_specs",
]
