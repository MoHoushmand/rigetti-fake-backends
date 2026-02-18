"""
FakeCepheus - Native PyQuil Implementation
===========================================

Native Rigetti implementation using PyQuil and QVM.

This is the PRIMARY implementation for authentic Rigetti ecosystem support.
Requires QVM and quilc to be running (Docker recommended).

Quick Start:
    # Start QVM/quilc
    docker run -d -p 5555:5555 rigetti/quilc -R
    docker run -d -p 5000:5000 rigetti/qvm -S

    # Use FakeCepheus
    from fakecepheus.pyquil import get_qc, CEPHEUS_SPECS
    qc = get_qc()

Author: Daniel Mo Houshmand (QDaria AS)
"""

from .backend import (
    get_fakecepheus_qc,
    get_qc,
    create_noise_model,
    create_quantum_processor,
    add_cepheus_noise,
    get_noise_pragmas,
)

from ..specs import (
    CepheusSpecs,
    CEPHEUS_SPECS,
    CEPHEUS_SPECS_CONSERVATIVE,
    CEPHEUS_EDGES,
    CEPHEUS_QUBITS,
    cumulative_fidelity,
    optimal_depth,
    print_specs,
)

__all__ = [
    # Backend factory
    "get_fakecepheus_qc",
    "get_qc",
    # Noise model
    "create_noise_model",
    "create_quantum_processor",
    "add_cepheus_noise",
    "get_noise_pragmas",
    # Specifications
    "CepheusSpecs",
    "CEPHEUS_SPECS",
    "CEPHEUS_SPECS_CONSERVATIVE",
    "CEPHEUS_EDGES",
    "CEPHEUS_QUBITS",
    # Analysis
    "cumulative_fidelity",
    "optimal_depth",
    "print_specs",
]
