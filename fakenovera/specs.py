"""
FakeNovera - Shared Hardware Specifications
============================================

Rigetti Novera 9Q QPU specifications from multiple verified sources.

TWO SPECIFICATION PROFILES:
1. OFFICIAL (Rigetti website): T1=27μs, T2=27μs, F_1Q=99.9%
2. BENCHMARKED (Zurich Instruments): T1=45.9μs, T2=25.5μs, F_1Q=99.51%

Both profiles share F_2Q=99.4% (iSWAP/CZ) and topology.

Sources:
- https://www.rigetti.com/novera (Official specs)
- Zurich Instruments Medium blog (March 2024 benchmarks)

Author: Daniel Mo Houshmand (QDaria AS)
License: MIT
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# =============================================================================
# TOPOLOGY (Shared across all implementations)
# =============================================================================

# 3×3 square lattice:
#   0 — 1 — 2
#   |   |   |
#   3 — 4 — 5
#   |   |   |
#   6 — 7 — 8

NOVERA_EDGES: List[Tuple[int, int]] = [
    # Horizontal
    (0, 1), (1, 2),
    (3, 4), (4, 5),
    (6, 7), (7, 8),
    # Vertical
    (0, 3), (1, 4), (2, 5),
    (3, 6), (4, 7), (5, 8),
]

NOVERA_QUBITS: List[int] = list(range(9))

# Bidirectional for Qiskit coupling map
NOVERA_COUPLING_MAP: List[List[int]] = [
    [a, b] for a, b in NOVERA_EDGES
] + [
    [b, a] for a, b in NOVERA_EDGES
]


# =============================================================================
# SPECIFICATION DATACLASS
# =============================================================================

@dataclass(frozen=True)
class NoveraSpecs:
    """
    Rigetti Novera 9Q hardware specifications.

    Attributes:
        name: Profile name
        num_qubits: Number of data qubits (9)
        num_couplers: Number of tunable couplers (12)
        t1_us: T1 relaxation time (microseconds)
        t2_us: T2 dephasing time (microseconds)
        fidelity_1q: Single-qubit gate fidelity
        fidelity_2q: Two-qubit gate fidelity (iSWAP/CZ)
        readout_fidelity: Measurement assignment fidelity
        gate_time_1q_ns: Single-qubit gate duration (nanoseconds)
        gate_time_2q_ns: Two-qubit gate duration (nanoseconds)
    """
    name: str = "novera-9q"

    # Topology
    num_qubits: int = 9
    num_couplers: int = 12

    # Coherence times
    t1_us: float = 27.0
    t2_us: float = 27.0

    # Gate fidelities
    fidelity_1q: float = 0.999
    fidelity_2q: float = 0.994

    # Readout
    readout_fidelity: float = 0.9796

    # Gate times
    gate_time_1q_ns: float = 40.0
    gate_time_2q_ns: float = 200.0

    def __post_init__(self):
        """Validate physical constraints."""
        if self.t2_us > 2 * self.t1_us:
            raise ValueError(f"T2 ({self.t2_us}) cannot exceed 2*T1 ({2*self.t1_us})")
        if not (0 < self.fidelity_1q <= 1):
            raise ValueError("Fidelity must be in (0, 1]")
        if not (0 < self.fidelity_2q <= 1):
            raise ValueError("Fidelity must be in (0, 1]")

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """Get topology edges."""
        return NOVERA_EDGES

    @property
    def n_edges(self) -> int:
        """Number of two-qubit gate connections per layer."""
        return len(NOVERA_EDGES)


# =============================================================================
# SPECIFICATION PROFILES
# =============================================================================

# Official Rigetti specifications (from rigetti.com/novera, Ankaa-9Q-3)
# NOTE: The Rigetti website lists "Median Fidelity (per op.)" = 99.9% and
# "Single-qubit gates" = 99.4%.  Different calibration runs and control
# systems yield different numbers (e.g. Quantum Machines AI calibration
# achieved 99.9% 1Q / 98.5% 2Q).  We use F_1Q=99.9% and F_2Q=99.4% here
# as representative best-case values consistent with the product page.
NOVERA_SPECS_OFFICIAL = NoveraSpecs(
    name="novera-9q-official",
    t1_us=27.0,
    t2_us=27.0,
    fidelity_1q=0.999,
    fidelity_2q=0.994,
    readout_fidelity=0.9796,
)

# Zurich Instruments benchmark specifications (March 2024)
NOVERA_SPECS_ZURICH = NoveraSpecs(
    name="novera-9q-zurich",
    t1_us=45.9,
    t2_us=25.5,
    fidelity_1q=0.9951,
    fidelity_2q=0.994,
    readout_fidelity=0.9796,
)

# Default: Official Rigetti specs (most conservative)
NOVERA_SPECS = NOVERA_SPECS_OFFICIAL


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def cumulative_fidelity(
    depth: int,
    specs: NoveraSpecs = NOVERA_SPECS
) -> float:
    """
    Calculate cumulative circuit fidelity for given depth.

    Formula: F_cum = F_2Q^(n_edges × depth)

    Args:
        depth: Circuit depth (number of variational layers)
        specs: Hardware specifications

    Returns:
        Cumulative fidelity as a fraction [0, 1]
    """
    n_cz_gates = specs.n_edges * depth
    return specs.fidelity_2q ** n_cz_gates


def optimal_depth(
    fidelity_threshold: float = 0.70,
    specs: NoveraSpecs = NOVERA_SPECS
) -> float:
    """
    Calculate optimal circuit depth from cumulative fidelity threshold.

    Formula: d_opt = ln(F_threshold) / (n_edges × ln(F_2Q))

    For QRC, the 70% threshold corresponds to the quantum-to-classical
    phase transition where prediction performance degrades.

    Args:
        fidelity_threshold: Target cumulative fidelity (default 0.70)
        specs: Hardware specifications

    Returns:
        Optimal depth (may be fractional; round for practical use)
    """
    return np.log(fidelity_threshold) / (specs.n_edges * np.log(specs.fidelity_2q))


def get_topology():
    """
    Get Novera topology as NetworkX graph.

    Returns:
        NetworkX Graph with 9 nodes and 12 edges

    Raises:
        ImportError: If networkx is not installed
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required: pip install networkx")

    G = nx.Graph()
    G.add_nodes_from(NOVERA_QUBITS)
    G.add_edges_from(NOVERA_EDGES)
    return G


def print_specs(specs: NoveraSpecs = NOVERA_SPECS):
    """Print hardware specifications and depth analysis."""
    print("=" * 60)
    print(f"NOVERA 9Q SPECIFICATIONS ({specs.name})")
    print("=" * 60)
    print(f"  Qubits: {specs.num_qubits}")
    print(f"  Couplers: {specs.num_couplers}")
    print(f"  Edges: {specs.n_edges}")
    print(f"  T1: {specs.t1_us} μs")
    print(f"  T2: {specs.t2_us} μs")
    print(f"  F_1Q: {specs.fidelity_1q * 100:.2f}%")
    print(f"  F_2Q: {specs.fidelity_2q * 100:.2f}%")
    print(f"  Readout: {specs.readout_fidelity * 100:.2f}%")
    print()
    print("DEPTH ANALYSIS:")
    print("-" * 60)
    print(f"{'Depth':<8} {'CZ Gates':<10} {'F_cum':<12} {'Regime'}")
    print("-" * 60)

    for d in range(1, 10):
        f_cum = cumulative_fidelity(d, specs)
        n_gates = d * specs.n_edges

        if f_cum > 0.80:
            regime = "Underfit"
        elif f_cum > 0.70:
            regime = "Improving"
        elif f_cum > 0.60:
            regime = "OPTIMAL"
        elif f_cum > 0.50:
            regime = "Degrading"
        else:
            regime = "Collapsed"

        marker = " <<<" if 0.65 <= f_cum <= 0.75 else ""
        print(f"{d:<8} {n_gates:<10} {f_cum*100:>6.1f}%      {regime}{marker}")

    print("-" * 60)
    d_opt = optimal_depth(0.70, specs)
    print(f"Optimal depth (70% threshold): d = {d_opt:.2f} ≈ {round(d_opt)}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n=== OFFICIAL RIGETTI SPECS ===")
    print_specs(NOVERA_SPECS_OFFICIAL)

    print("\n=== ZURICH INSTRUMENTS BENCHMARKS ===")
    print_specs(NOVERA_SPECS_ZURICH)
