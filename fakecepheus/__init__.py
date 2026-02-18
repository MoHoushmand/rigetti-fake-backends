"""
FakeCepheus - Dual Implementation Noise Model for Rigetti Cepheus-1-36Q
========================================================================

A complete noise model package for simulating Rigetti's Cepheus-1-36Q QPU.

Cepheus-1-36Q is a 36-qubit multi-chip processor built from four 9-qubit
Novera-class chiplets with intermodule couplers. This package models the
HETEROGENEOUS fidelity: intra-chip gates have higher fidelity than
inter-chip gates.

TWO IMPLEMENTATIONS AVAILABLE:

1. **PyQuil (Native Rigetti)** - PRIMARY
   - Uses Rigetti's native QVM and quilc
   - Authentic Rigetti ecosystem support
   - Requires Docker: quilc + qvm

   ```python
   from fakecepheus.pyquil import get_qc, CEPHEUS_SPECS
   qc = get_qc()  # Returns configured QuantumComputer
   ```

2. **Qiskit (Compatibility)** - SECONDARY
   - Uses Qiskit AerSimulator
   - No external dependencies
   - Good for quick prototyping

   ```python
   from fakecepheus.qiskit import FakeCepheus
   backend = FakeCepheus()  # Returns BackendV2
   ```

SPECIFICATION PROFILES:
- CEPHEUS_SPECS (default): F_2Q_intra=99.5%, F_2Q_inter=99.0%
- CEPHEUS_SPECS_CONSERVATIVE: F_2Q_intra=99.4%, F_2Q_inter=98.5%

Quick Start (PyQuil - recommended):
    # Start QVM/quilc
    docker run -d -p 5555:5555 rigetti/quilc -R
    docker run -d -p 5000:5000 rigetti/qvm -S

    # Use
    from fakecepheus.pyquil import get_qc
    qc = get_qc()

Quick Start (Qiskit - no Docker needed):
    from fakecepheus.qiskit import FakeCepheus
    backend = FakeCepheus()

For QRC depth analysis:
    from fakecepheus import optimal_depth, cumulative_fidelity, print_specs
    print(f"Optimal depth: {optimal_depth(0.70):.1f}")
    print_specs()

Author: Daniel Mo Houshmand (QDaria AS)
License: MIT
Version: 1.0.0
"""

# Shared specifications (always available)
from .specs import (
    CepheusSpecs,
    CEPHEUS_SPECS,
    CEPHEUS_SPECS_CONSERVATIVE,
    CEPHEUS_EDGES,
    CEPHEUS_INTRA_CHIP_EDGES,
    CEPHEUS_INTER_CHIP_EDGES,
    CEPHEUS_QUBITS,
    CEPHEUS_COUPLING_MAP,
    CEPHEUS_CHIPS,
    is_inter_chip_edge,
    get_chip_id,
    get_topology,
    cumulative_fidelity,
    optimal_depth,
    print_specs,
)

# Version info
__version__ = "1.0.0"
__author__ = "Daniel Mo Houshmand"
__email__ = "mo@qdaria.com"

__all__ = [
    # Specifications
    "CepheusSpecs",
    "CEPHEUS_SPECS",
    "CEPHEUS_SPECS_CONSERVATIVE",
    "CEPHEUS_EDGES",
    "CEPHEUS_INTRA_CHIP_EDGES",
    "CEPHEUS_INTER_CHIP_EDGES",
    "CEPHEUS_QUBITS",
    "CEPHEUS_COUPLING_MAP",
    "CEPHEUS_CHIPS",
    # Topology helpers
    "is_inter_chip_edge",
    "get_chip_id",
    "get_topology",
    # Analysis
    "cumulative_fidelity",
    "optimal_depth",
    "print_specs",
    # Submodules (import as needed)
    "pyquil",
    "qiskit",
]


def get_backend(framework: str = "auto", **kwargs):
    """
    Get a FakeCepheus backend for the specified framework.

    Args:
        framework: "pyquil", "qiskit", or "auto" (tries pyquil first)
        **kwargs: Passed to backend constructor
            - For pyquil: noisy (bool), qvm_url (str), quilc_url (str)
            - For qiskit: seed (int), noise (bool)
            - Common: specs (CepheusSpecs)

    Returns:
        Configured backend (QuantumComputer or BackendV2)

    Example:
        >>> backend = get_backend("qiskit", seed=42)
        >>> backend = get_backend("pyquil", noisy=True)
        >>> backend = get_backend("auto")  # Tries pyquil, falls back to qiskit
    """
    pyquil_params = {'noisy', 'qvm_url', 'quilc_url', 'name', 'specs'}
    qiskit_params = {'seed', 'noise', 'specs'}

    def filter_kwargs(valid_params):
        return {k: v for k, v in kwargs.items() if k in valid_params}

    if framework == "auto":
        try:
            from .pyquil import get_qc
            return get_qc(**filter_kwargs(pyquil_params))
        except (ImportError, RuntimeError):
            from .qiskit import FakeCepheus
            return FakeCepheus(**filter_kwargs(qiskit_params))

    elif framework == "pyquil":
        from .pyquil import get_qc
        return get_qc(**filter_kwargs(pyquil_params))

    elif framework == "qiskit":
        from .qiskit import FakeCepheus
        return FakeCepheus(**filter_kwargs(qiskit_params))

    else:
        raise ValueError(
            f"Unknown framework: {framework}. Use 'pyquil', 'qiskit', or 'auto'"
        )


def _show_status():
    """Check available backends."""
    status = []

    try:
        import pyquil
        status.append("PyQuil available")
    except ImportError:
        status.append("PyQuil not installed (pip install pyquil)")

    try:
        import qiskit
        import qiskit_aer
        status.append("Qiskit available")
    except ImportError:
        status.append("Qiskit not installed (pip install qiskit qiskit-aer)")

    return status
