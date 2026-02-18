"""
FakeCepheus - Qiskit Compatibility Layer
========================================

Qiskit-compatible implementation for users without QVM/quilc access.

This provides a BackendV2-compatible simulated backend using Qiskit's
AerSimulator with Cepheus-inspired noise parameters, including
heterogeneous intra-chip vs inter-chip fidelities.

NOTE: For authentic Rigetti ecosystem support, use fakecepheus.pyquil instead.

Example:
    from fakecepheus.qiskit import FakeCepheus

    backend = FakeCepheus()
    qc = QuantumCircuit(36)
    qc.h(4)
    qc.cz(4, 1)
    qc.measure_all()
    job = backend.run(qc)

Author: Daniel Mo Houshmand (QDaria AS)
"""

from .backend import FakeCepheus, FakeCepheusBackend
from .noise import create_cepheus_noise_model, create_cepheus_target

from ..specs import (
    CepheusSpecs,
    CEPHEUS_SPECS,
    CEPHEUS_SPECS_CONSERVATIVE,
    CEPHEUS_EDGES,
    CEPHEUS_COUPLING_MAP,
    cumulative_fidelity,
    optimal_depth,
    print_specs,
)

__all__ = [
    # Backend
    "FakeCepheus",
    "FakeCepheusBackend",
    # Noise
    "create_cepheus_noise_model",
    "create_cepheus_target",
    # Specifications
    "CepheusSpecs",
    "CEPHEUS_SPECS",
    "CEPHEUS_SPECS_CONSERVATIVE",
    "CEPHEUS_EDGES",
    "CEPHEUS_COUPLING_MAP",
    # Analysis
    "cumulative_fidelity",
    "optimal_depth",
    "print_specs",
]
