"""
FakeNovera - Dual Implementation Noise Model for Rigetti Novera 9Q
==================================================================

A complete noise model package for simulating Rigetti's Novera 9Q QPU.

TWO IMPLEMENTATIONS AVAILABLE:

1. **PyQuil (Native Rigetti)** - PRIMARY
   - Uses Rigetti's native QVM and quilc
   - Authentic Rigetti ecosystem support
   - Requires Docker: quilc + qvm

   ```python
   from fakenovera.pyquil import get_qc, NOVERA_SPECS
   qc = get_qc()  # Returns configured QuantumComputer
   ```

2. **Qiskit (Compatibility)** - SECONDARY
   - Uses Qiskit AerSimulator
   - No external dependencies
   - Good for quick prototyping

   ```python
   from fakenovera.qiskit import FakeNovera
   backend = FakeNovera()  # Returns BackendV2
   ```

SPECIFICATION PROFILES:
- NOVERA_SPECS_OFFICIAL: T1=27μs, T2=27μs, F_1Q=99.9% (Rigetti website)
- NOVERA_SPECS_ZURICH: T1=45.9μs, T2=25.5μs, F_1Q=99.51% (Zurich benchmarks)

Both share: 9 qubits, 12 edges, F_2Q=99.4%

Quick Start (PyQuil - recommended):
    # Start QVM/quilc
    docker run -d -p 5555:5555 rigetti/quilc -R
    docker run -d -p 5000:5000 rigetti/qvm -S

    # Use
    from fakenovera.pyquil import get_qc
    qc = get_qc()

Quick Start (Qiskit - no Docker needed):
    from fakenovera.qiskit import FakeNovera
    backend = FakeNovera()

For QRC depth analysis:
    from fakenovera import optimal_depth, cumulative_fidelity, print_specs
    print(f"Optimal depth: {optimal_depth(0.70):.1f}")  # ~5 layers
    print_specs()

Author: Daniel Mo Houshmand (QDaria AS)
License: MIT
Version: 3.0.0
"""

# Shared specifications (always available)
from .specs import (
    NoveraSpecs,
    NOVERA_SPECS,
    NOVERA_SPECS_OFFICIAL,
    NOVERA_SPECS_ZURICH,
    NOVERA_EDGES,
    NOVERA_QUBITS,
    NOVERA_COUPLING_MAP,
    get_topology,
    cumulative_fidelity,
    optimal_depth,
    print_specs,
)

# Version info
__version__ = "3.0.0"
__author__ = "Daniel Mo Houshmand"
__email__ = "mo@qdaria.com"

__all__ = [
    # Specifications
    "NoveraSpecs",
    "NOVERA_SPECS",
    "NOVERA_SPECS_OFFICIAL",
    "NOVERA_SPECS_ZURICH",
    "NOVERA_EDGES",
    "NOVERA_QUBITS",
    "NOVERA_COUPLING_MAP",
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
    Get a FakeNovera backend for the specified framework.

    Args:
        framework: "pyquil", "qiskit", or "auto" (tries pyquil first)
        **kwargs: Passed to backend constructor
            - For pyquil: noisy (bool), qvm_url (str), quilc_url (str)
            - For qiskit: seed (int), noise (bool)
            - Common: specs (NoveraSpecs)

    Returns:
        Configured backend (QuantumComputer or BackendV2)

    Example:
        >>> backend = get_backend("qiskit", seed=42)
        >>> backend = get_backend("pyquil", noisy=True)
        >>> backend = get_backend("auto")  # Tries pyquil, falls back to qiskit
    """
    # Framework-specific parameter filtering
    pyquil_params = {'noisy', 'qvm_url', 'quilc_url', 'name', 'specs'}
    qiskit_params = {'seed', 'noise', 'specs'}

    def filter_kwargs(valid_params):
        return {k: v for k, v in kwargs.items() if k in valid_params}

    if framework == "auto":
        try:
            from .pyquil import get_qc
            return get_qc(**filter_kwargs(pyquil_params))
        except (ImportError, RuntimeError):
            from .qiskit import FakeNovera
            return FakeNovera(**filter_kwargs(qiskit_params))

    elif framework == "pyquil":
        from .pyquil import get_qc
        return get_qc(**filter_kwargs(pyquil_params))

    elif framework == "qiskit":
        from .qiskit import FakeNovera
        return FakeNovera(**filter_kwargs(qiskit_params))

    else:
        raise ValueError(f"Unknown framework: {framework}. Use 'pyquil', 'qiskit', or 'auto'")


# Print helpful info on import
def _show_status():
    """Check available backends."""
    status = []

    # Check PyQuil
    try:
        import pyquil
        status.append("✓ PyQuil available")
    except ImportError:
        status.append("✗ PyQuil not installed (pip install pyquil)")

    # Check Qiskit
    try:
        import qiskit
        import qiskit_aer
        status.append("✓ Qiskit available")
    except ImportError:
        status.append("✗ Qiskit not installed (pip install qiskit qiskit-aer)")

    return status
