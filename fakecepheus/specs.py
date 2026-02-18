"""
FakeCepheus - Shared Hardware Specifications
=============================================

Rigetti Cepheus-1-36Q QPU specifications from verified sources.

Cepheus-1-36Q is a 36-qubit multi-chip processor composed of FOUR 9-qubit
Novera-class chiplets tiled in a 2x2 square arrangement with intermodule
couplers connecting adjacent chiplets.

KEY NOVELTY: Heterogeneous fidelity model - intra-chip gates have higher
fidelity than inter-chip gates due to intermodule coupler overhead.

Sources:
- Rigetti press release (2024): Ankaa-3 84Q median F_2Q = 99.5%
  https://www.rigetti.com/newsroom/press-releases/rigetti-announces-ankaa-3
- Rigetti Cepheus-1-36Q Azure Quantum listing (2025):
  36-qubit multi-chip, 4 x 9-qubit chiplets
  https://learn.microsoft.com/en-us/azure/quantum/provider-rigetti
- AWS Braket Rigetti docs: gate times ~40ns 1Q, iSWAP native gate set
  https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices-rigetti.html
- SQMS Center collaboration: improved T1 coherence times
  https://sqms.fnal.gov/

Inferred / conservative estimates (clearly marked):
- T1 = 34 us (Ankaa-3 median, improved from Novera 27us via SQMS)
- T2 = 20 us (Ankaa-3 median)
- F_1Q = 99.91% (Ankaa-3 median)
- 2Q gate time = 72 ns (Ankaa-3 iSWAP, vs 200ns Novera CZ)
- Readout = 97.96% (Novera value, conservative for Cepheus)
- Inter-chip F_2Q = 99.0% (conservative estimate for intermodule coupler)

Author: Daniel Mo Houshmand (QDaria AS)
License: MIT
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple
import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# =============================================================================
# TOPOLOGY
# =============================================================================

# 4 chiplets in a 2x2 arrangement, each a 3x3 square lattice:
#
# Chip 0 (qubits 0-8):       Chip 1 (qubits 9-17):
#   0 -- 1 -- 2                 9 -- 10 -- 11
#   |    |    |                 |     |     |
#   3 -- 4 -- 5  ~~interchip~~ 12 -- 13 -- 14
#   |    |    |                 |     |     |
#   6 -- 7 -- 8                15 -- 16 -- 17
#        |                           |
#      inter                       inter
#        |                           |
# Chip 2 (qubits 18-26):     Chip 3 (qubits 27-35):
#  18 -- 19 -- 20               27 -- 28 -- 29
#   |     |    |                  |     |     |
#  21 -- 22 -- 23  ~~interchip~~ 30 -- 31 -- 32
#   |     |    |                  |     |     |
#  24 -- 25 -- 26               33 -- 34 -- 35

# Chiplet qubit assignments
CEPHEUS_CHIPS: Dict[int, List[int]] = {
    0: list(range(0, 9)),     # Chip 0: qubits 0-8
    1: list(range(9, 18)),    # Chip 1: qubits 9-17
    2: list(range(18, 27)),   # Chip 2: qubits 18-26
    3: list(range(27, 36)),   # Chip 3: qubits 27-35
}

# Intra-chip edges: 12 per chip (3x3 square lattice), 48 total
# Each chip has the same internal topology as Novera, offset by chip_id * 9
_INTRA_CHIP_OFFSETS: List[Tuple[int, int]] = [
    # Horizontal
    (0, 1), (1, 2),
    (3, 4), (4, 5),
    (6, 7), (7, 8),
    # Vertical
    (0, 3), (1, 4), (2, 5),
    (3, 6), (4, 7), (5, 8),
]

CEPHEUS_INTRA_CHIP_EDGES: List[Tuple[int, int]] = []
for chip_id in range(4):
    offset = chip_id * 9
    for a, b in _INTRA_CHIP_OFFSETS:
        CEPHEUS_INTRA_CHIP_EDGES.append((a + offset, b + offset))

# Inter-chip edges: connections between adjacent chiplets
# Chip 0 right column (2,5,8) <-> Chip 1 left column (9,12,15)
# Chip 2 right column (20,23,26) <-> Chip 3 left column (27,30,33)
# Chip 0 bottom row (6,7,8) <-> Chip 2 top row (18,19,20)
# Chip 1 bottom row (15,16,17) <-> Chip 3 top row (27,28,29)
CEPHEUS_INTER_CHIP_EDGES: List[Tuple[int, int]] = [
    # Chip 0 right <-> Chip 1 left
    (2, 9),    # row 0
    (5, 12),   # row 1
    (8, 15),   # row 2
    # Chip 2 right <-> Chip 3 left
    (20, 27),  # row 0
    (23, 30),  # row 1
    (26, 33),  # row 2
    # Chip 0 bottom <-> Chip 2 top
    (6, 18),   # col 0
    (7, 19),   # col 1
    (8, 20),   # col 2
    # Chip 1 bottom <-> Chip 3 top
    (15, 27),  # col 0
    (16, 28),  # col 1
    (17, 29),  # col 2
]

# All edges combined
CEPHEUS_EDGES: List[Tuple[int, int]] = (
    CEPHEUS_INTRA_CHIP_EDGES + CEPHEUS_INTER_CHIP_EDGES
)

CEPHEUS_QUBITS: List[int] = list(range(36))

# Bidirectional for Qiskit coupling map
CEPHEUS_COUPLING_MAP: List[List[int]] = [
    [a, b] for a, b in CEPHEUS_EDGES
] + [
    [b, a] for a, b in CEPHEUS_EDGES
]

# Sets for fast lookup
_INTER_CHIP_SET: FrozenSet[Tuple[int, int]] = frozenset(
    CEPHEUS_INTER_CHIP_EDGES
    + [(b, a) for a, b in CEPHEUS_INTER_CHIP_EDGES]
)


def is_inter_chip_edge(q0: int, q1: int) -> bool:
    """Check if an edge crosses chiplet boundaries.

    Args:
        q0: First qubit index
        q1: Second qubit index

    Returns:
        True if the edge is an inter-chip connection
    """
    return (q0, q1) in _INTER_CHIP_SET


def get_chip_id(qubit: int) -> int:
    """Return the chip ID (0-3) for a given qubit index.

    Args:
        qubit: Qubit index (0-35)

    Returns:
        Chip ID (0, 1, 2, or 3)

    Raises:
        ValueError: If qubit index is out of range
    """
    if not (0 <= qubit < 36):
        raise ValueError(f"Qubit {qubit} out of range [0, 35]")
    return qubit // 9


# =============================================================================
# SPECIFICATION DATACLASS
# =============================================================================

@dataclass(frozen=True)
class CepheusSpecs:
    """
    Rigetti Cepheus-1-36Q hardware specifications.

    Supports HETEROGENEOUS fidelity: intra-chip gates have higher fidelity
    than inter-chip gates due to intermodule coupler overhead.

    Attributes:
        name: Profile name
        num_qubits: Number of data qubits (36)
        num_chips: Number of chiplets (4)
        num_qubits_per_chip: Qubits per chiplet (9)
        num_intra_edges: Intra-chip two-qubit connections (48)
        num_inter_edges: Inter-chip two-qubit connections (12)
        t1_us: T1 relaxation time (microseconds)
        t2_us: T2 dephasing time (microseconds)
        fidelity_1q: Single-qubit gate fidelity
        fidelity_2q_intra: Intra-chip two-qubit gate fidelity
        fidelity_2q_inter: Inter-chip two-qubit gate fidelity
        readout_fidelity: Measurement assignment fidelity
        gate_time_1q_ns: Single-qubit gate duration (nanoseconds)
        gate_time_2q_ns: Two-qubit gate duration (nanoseconds)
    """
    name: str = "cepheus-1-36q"

    # Topology
    num_qubits: int = 36
    num_chips: int = 4
    num_qubits_per_chip: int = 9
    num_intra_edges: int = 48    # 12 per chip x 4 chips
    num_inter_edges: int = 12    # 3 per chiplet boundary x 4 boundaries

    # Coherence times
    # Source: Ankaa-3 median values, improved via SQMS collaboration
    # Novera = 27 us T1; Ankaa-3 = 34 us median T1
    t1_us: float = 34.0     # Ankaa-3 median (SQMS-improved)
    t2_us: float = 20.0     # Ankaa-3 median

    # Gate fidelities - HETEROGENEOUS
    # Source: Rigetti press release - Ankaa-3 median F_2Q = 99.5%
    # Inter-chip estimated at 99.0% (conservative for intermodule coupler)
    fidelity_1q: float = 0.9991         # Ankaa-3 median F_1Q
    fidelity_2q_intra: float = 0.995    # On-chip F_2Q (Ankaa-3 median)
    fidelity_2q_inter: float = 0.990    # Cross-chip F_2Q (conservative)

    # Readout
    # Source: Novera value (conservative; Cepheus has "enhanced readout")
    readout_fidelity: float = 0.9796

    # Gate times
    # Source: AWS Braket docs (1Q), Ankaa-3 benchmarks (2Q iSWAP = 72ns)
    gate_time_1q_ns: float = 40.0    # Same as Novera
    gate_time_2q_ns: float = 72.0    # Ankaa-3 iSWAP (vs 200ns Novera CZ)

    def __post_init__(self):
        """Validate physical constraints."""
        if self.t2_us > 2 * self.t1_us:
            raise ValueError(
                f"T2 ({self.t2_us}) cannot exceed 2*T1 ({2 * self.t1_us})"
            )
        if not (0 < self.fidelity_1q <= 1):
            raise ValueError("F_1Q must be in (0, 1]")
        if not (0 < self.fidelity_2q_intra <= 1):
            raise ValueError("F_2Q_intra must be in (0, 1]")
        if not (0 < self.fidelity_2q_inter <= 1):
            raise ValueError("F_2Q_inter must be in (0, 1]")

    @property
    def fidelity_2q(self) -> float:
        """Weighted average F_2Q across all edges.

        Used for simplified (homogeneous) fidelity calculations.
        Weight: 48 intra-chip edges at F_intra + 12 inter-chip at F_inter.
        """
        total = self.num_intra_edges + self.num_inter_edges
        return (
            self.num_intra_edges * self.fidelity_2q_intra
            + self.num_inter_edges * self.fidelity_2q_inter
        ) / total

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """Get all topology edges."""
        return CEPHEUS_EDGES

    @property
    def n_edges(self) -> int:
        """Total number of two-qubit gate connections."""
        return len(CEPHEUS_EDGES)

    @property
    def intra_edges(self) -> List[Tuple[int, int]]:
        """Get intra-chip topology edges."""
        return CEPHEUS_INTRA_CHIP_EDGES

    @property
    def inter_edges(self) -> List[Tuple[int, int]]:
        """Get inter-chip topology edges."""
        return CEPHEUS_INTER_CHIP_EDGES


# =============================================================================
# SPECIFICATION PROFILES
# =============================================================================

# Default Cepheus-1-36Q specs (Ankaa-3 generation, multi-chip)
# Sources cited inline in the CepheusSpecs docstring above.
CEPHEUS_SPECS = CepheusSpecs(
    name="cepheus-1-36q",
)

# Pessimistic profile: lower inter-chip fidelity
CEPHEUS_SPECS_CONSERVATIVE = CepheusSpecs(
    name="cepheus-1-36q-conservative",
    fidelity_2q_intra=0.994,   # Same as Novera
    fidelity_2q_inter=0.985,   # Pessimistic intermodule coupler
)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def cumulative_fidelity(
    depth: int,
    specs: CepheusSpecs = CEPHEUS_SPECS,
    heterogeneous: bool = True,
) -> float:
    """
    Calculate cumulative circuit fidelity for given depth.

    With heterogeneous=True (default), accounts for different intra-chip
    and inter-chip gate fidelities:
        F_cum = F_intra^(n_intra * depth) * F_inter^(n_inter * depth)

    With heterogeneous=False, uses the weighted average F_2Q:
        F_cum = F_2Q_avg^(n_edges * depth)

    Args:
        depth: Circuit depth (number of variational layers)
        specs: Hardware specifications
        heterogeneous: Use per-edge fidelities (default True)

    Returns:
        Cumulative fidelity as a fraction [0, 1]
    """
    if heterogeneous:
        n_intra = specs.num_intra_edges * depth
        n_inter = specs.num_inter_edges * depth
        return (
            specs.fidelity_2q_intra ** n_intra
            * specs.fidelity_2q_inter ** n_inter
        )
    else:
        n_total = specs.n_edges * depth
        return specs.fidelity_2q ** n_total


def optimal_depth(
    fidelity_threshold: float = 0.70,
    specs: CepheusSpecs = CEPHEUS_SPECS,
    heterogeneous: bool = True,
) -> float:
    """
    Calculate optimal circuit depth from cumulative fidelity threshold.

    With heterogeneous=True (default):
        d_opt = ln(F_threshold) / (n_intra * ln(F_intra) + n_inter * ln(F_inter))

    With heterogeneous=False:
        d_opt = ln(F_threshold) / (n_edges * ln(F_2Q_avg))

    For QRC, the 70% threshold corresponds to the quantum-to-classical
    phase transition where prediction performance degrades.

    Args:
        fidelity_threshold: Target cumulative fidelity (default 0.70)
        specs: Hardware specifications
        heterogeneous: Use per-edge fidelities (default True)

    Returns:
        Optimal depth (may be fractional; round for practical use)
    """
    if heterogeneous:
        log_per_layer = (
            specs.num_intra_edges * np.log(specs.fidelity_2q_intra)
            + specs.num_inter_edges * np.log(specs.fidelity_2q_inter)
        )
    else:
        log_per_layer = specs.n_edges * np.log(specs.fidelity_2q)

    return np.log(fidelity_threshold) / log_per_layer


def get_topology():
    """
    Get Cepheus topology as NetworkX graph.

    Nodes have a 'chip' attribute indicating which chiplet they belong to.
    Edges have a 'type' attribute: 'intra' or 'inter'.

    Returns:
        NetworkX Graph with 36 nodes and 60 edges

    Raises:
        ImportError: If networkx is not installed
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required: pip install networkx")

    G = nx.Graph()

    # Add nodes with chip attribute
    for chip_id, qubits in CEPHEUS_CHIPS.items():
        for q in qubits:
            G.add_node(q, chip=chip_id)

    # Add intra-chip edges
    for a, b in CEPHEUS_INTRA_CHIP_EDGES:
        G.add_edge(a, b, type="intra")

    # Add inter-chip edges
    for a, b in CEPHEUS_INTER_CHIP_EDGES:
        G.add_edge(a, b, type="inter")

    return G


def print_specs(specs: CepheusSpecs = CEPHEUS_SPECS):
    """Print hardware specifications and depth analysis."""
    print("=" * 65)
    print(f"CEPHEUS-1-36Q SPECIFICATIONS ({specs.name})")
    print("=" * 65)
    print(f"  Qubits: {specs.num_qubits} ({specs.num_chips} chips x "
          f"{specs.num_qubits_per_chip} qubits)")
    print(f"  Intra-chip edges: {specs.num_intra_edges}")
    print(f"  Inter-chip edges: {specs.num_inter_edges}")
    print(f"  Total edges: {specs.n_edges}")
    print(f"  T1: {specs.t1_us} us")
    print(f"  T2: {specs.t2_us} us")
    print(f"  F_1Q: {specs.fidelity_1q * 100:.2f}%")
    print(f"  F_2Q (intra-chip): {specs.fidelity_2q_intra * 100:.2f}%")
    print(f"  F_2Q (inter-chip): {specs.fidelity_2q_inter * 100:.2f}%")
    print(f"  F_2Q (weighted avg): {specs.fidelity_2q * 100:.3f}%")
    print(f"  Readout: {specs.readout_fidelity * 100:.2f}%")
    print(f"  1Q gate time: {specs.gate_time_1q_ns} ns")
    print(f"  2Q gate time: {specs.gate_time_2q_ns} ns")
    print()
    print("DEPTH ANALYSIS (heterogeneous fidelity):")
    print("-" * 65)
    print(f"{'Depth':<8} {'2Q Gates':<10} {'F_cum':<12} {'Regime'}")
    print("-" * 65)

    for d in range(1, 10):
        f_cum = cumulative_fidelity(d, specs, heterogeneous=True)
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
        print(f"{d:<8} {n_gates:<10} {f_cum * 100:>6.1f}%      {regime}{marker}")

    print("-" * 65)
    d_opt = optimal_depth(0.70, specs, heterogeneous=True)
    print(f"Optimal depth (70% threshold): d = {d_opt:.2f} ~ {round(d_opt)}")
    print("=" * 65)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n=== DEFAULT CEPHEUS SPECS ===")
    print_specs(CEPHEUS_SPECS)

    print("\n=== CONSERVATIVE CEPHEUS SPECS ===")
    print_specs(CEPHEUS_SPECS_CONSERVATIVE)

    # Compare with Novera
    from fakenovera.specs import NOVERA_SPECS, optimal_depth as novera_optimal_depth
    d_novera = novera_optimal_depth(0.70, NOVERA_SPECS)
    d_cepheus = optimal_depth(0.70, CEPHEUS_SPECS)
    print(f"\nNovera d_opt(70%): {d_novera:.2f} ~ {round(d_novera)}")
    print(f"Cepheus d_opt(70%): {d_cepheus:.2f} ~ {round(d_cepheus)}")
