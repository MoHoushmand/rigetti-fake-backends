"""
Tests for FakeCepheus Dual Implementation Package
==================================================

Tests both PyQuil and Qiskit implementations for Rigetti Cepheus-1-36Q.

Verifies:
- 36 qubits, 60 edges (48 intra + 12 inter)
- 4 chiplets x 9 qubits
- Heterogeneous noise model (intra vs inter chip fidelities)
- Cumulative fidelity with heterogeneous edges
- Optimal depth predictions
- Comparison with FakeNovera

Author: Daniel Mo Houshmand (QDaria AS)
"""

import sys
import pytest
import numpy as np

from fakecepheus import (
    get_backend,
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
    cumulative_fidelity,
    optimal_depth,
    print_specs,
)


# =============================================================================
# TOPOLOGY TESTS
# =============================================================================

class TestCepheusTopology:
    """Test the 4-chiplet topology."""

    def test_qubit_count(self):
        assert CEPHEUS_SPECS.num_qubits == 36

    def test_qubits_list(self):
        assert CEPHEUS_QUBITS == list(range(36))

    def test_total_edge_count(self):
        """60 edges total: 48 intra + 12 inter."""
        assert len(CEPHEUS_EDGES) == 60

    def test_intra_chip_edge_count(self):
        """48 intra-chip edges: 12 per chip x 4 chips."""
        assert len(CEPHEUS_INTRA_CHIP_EDGES) == 48

    def test_inter_chip_edge_count(self):
        """12 inter-chip edges: 3 per boundary x 4 boundaries."""
        assert len(CEPHEUS_INTER_CHIP_EDGES) == 12

    def test_edges_sum(self):
        """Intra + inter should equal total."""
        assert (
            len(CEPHEUS_INTRA_CHIP_EDGES) + len(CEPHEUS_INTER_CHIP_EDGES)
            == len(CEPHEUS_EDGES)
        )

    def test_no_duplicate_edges(self):
        """All edges should be unique."""
        edge_set = set(CEPHEUS_EDGES)
        assert len(edge_set) == len(CEPHEUS_EDGES)

    def test_coupling_map_bidirectional(self):
        """Should have 120 entries (60 edges x 2 directions)."""
        assert len(CEPHEUS_COUPLING_MAP) == 120

    def test_all_qubits_in_edges(self):
        """Every qubit should appear in at least one edge."""
        qubits_in_edges = set()
        for a, b in CEPHEUS_EDGES:
            qubits_in_edges.add(a)
            qubits_in_edges.add(b)
        assert qubits_in_edges == set(range(36))


class TestChipletStructure:
    """Test the 4-chiplet layout."""

    def test_four_chips(self):
        assert len(CEPHEUS_CHIPS) == 4

    def test_nine_qubits_per_chip(self):
        for chip_id, qubits in CEPHEUS_CHIPS.items():
            assert len(qubits) == 9, f"Chip {chip_id} has {len(qubits)} qubits"

    def test_no_qubit_overlap(self):
        """Qubits should not belong to multiple chips."""
        all_qubits = []
        for qubits in CEPHEUS_CHIPS.values():
            all_qubits.extend(qubits)
        assert len(set(all_qubits)) == 36

    def test_chip_id_function(self):
        assert get_chip_id(0) == 0
        assert get_chip_id(8) == 0
        assert get_chip_id(9) == 1
        assert get_chip_id(17) == 1
        assert get_chip_id(18) == 2
        assert get_chip_id(26) == 2
        assert get_chip_id(27) == 3
        assert get_chip_id(35) == 3

    def test_chip_id_out_of_range(self):
        with pytest.raises(ValueError):
            get_chip_id(36)
        with pytest.raises(ValueError):
            get_chip_id(-1)

    def test_intra_edges_same_chip(self):
        """All intra-chip edges should connect qubits on the same chip."""
        for a, b in CEPHEUS_INTRA_CHIP_EDGES:
            assert get_chip_id(a) == get_chip_id(b), (
                f"Intra-chip edge ({a}, {b}) crosses chips: "
                f"chip {get_chip_id(a)} != chip {get_chip_id(b)}"
            )

    def test_inter_edges_different_chips(self):
        """All inter-chip edges should connect qubits on different chips."""
        for a, b in CEPHEUS_INTER_CHIP_EDGES:
            assert get_chip_id(a) != get_chip_id(b), (
                f"Inter-chip edge ({a}, {b}) is on same chip {get_chip_id(a)}"
            )

    def test_is_inter_chip_edge_function(self):
        """Test the inter-chip edge detection function."""
        # Known inter-chip edges
        assert is_inter_chip_edge(2, 9)
        assert is_inter_chip_edge(9, 2)   # Bidirectional
        assert is_inter_chip_edge(8, 15)
        assert is_inter_chip_edge(6, 18)
        assert is_inter_chip_edge(15, 27)

        # Known intra-chip edges
        assert not is_inter_chip_edge(0, 1)
        assert not is_inter_chip_edge(4, 5)
        assert not is_inter_chip_edge(9, 10)

    def test_per_chip_internal_topology(self):
        """Each chip should have 12 internal edges (3x3 square lattice)."""
        for chip_id in range(4):
            chip_edges = [
                (a, b) for a, b in CEPHEUS_INTRA_CHIP_EDGES
                if a // 9 == chip_id
            ]
            assert len(chip_edges) == 12, (
                f"Chip {chip_id} has {len(chip_edges)} edges, expected 12"
            )


# =============================================================================
# SPECIFICATION TESTS
# =============================================================================

class TestCepheusSpecs:
    """Test hardware specifications dataclass."""

    def test_default_values(self):
        assert CEPHEUS_SPECS.num_qubits == 36
        assert CEPHEUS_SPECS.num_chips == 4
        assert CEPHEUS_SPECS.num_qubits_per_chip == 9
        assert CEPHEUS_SPECS.num_intra_edges == 48
        assert CEPHEUS_SPECS.num_inter_edges == 12

    def test_fidelities_valid(self):
        assert 0 < CEPHEUS_SPECS.fidelity_1q <= 1
        assert 0 < CEPHEUS_SPECS.fidelity_2q_intra <= 1
        assert 0 < CEPHEUS_SPECS.fidelity_2q_inter <= 1
        assert 0 < CEPHEUS_SPECS.readout_fidelity <= 1

    def test_intra_higher_than_inter(self):
        """Intra-chip fidelity should exceed inter-chip fidelity."""
        assert CEPHEUS_SPECS.fidelity_2q_intra > CEPHEUS_SPECS.fidelity_2q_inter

    def test_t2_constraint(self):
        """T2 cannot exceed 2*T1 (physical constraint)."""
        assert CEPHEUS_SPECS.t2_us <= 2 * CEPHEUS_SPECS.t1_us

    def test_weighted_average_fidelity(self):
        """Test the weighted average F_2Q property."""
        expected = (
            48 * 0.995 + 12 * 0.990
        ) / 60
        assert abs(CEPHEUS_SPECS.fidelity_2q - expected) < 1e-10

    def test_invalid_t2_raises(self):
        """T2 > 2*T1 should raise ValueError."""
        with pytest.raises(ValueError):
            CepheusSpecs(t1_us=10.0, t2_us=30.0)

    def test_invalid_fidelity_raises(self):
        with pytest.raises(ValueError):
            CepheusSpecs(fidelity_1q=1.5)
        with pytest.raises(ValueError):
            CepheusSpecs(fidelity_2q_intra=0.0)

    def test_n_edges_property(self):
        assert CEPHEUS_SPECS.n_edges == 60


class TestSpecProfiles:
    """Test different specification profiles."""

    def test_default_profile(self):
        assert CEPHEUS_SPECS.name == "cepheus-1-36q"
        assert CEPHEUS_SPECS.fidelity_2q_intra == 0.995
        assert CEPHEUS_SPECS.fidelity_2q_inter == 0.990

    def test_conservative_profile(self):
        assert CEPHEUS_SPECS_CONSERVATIVE.name == "cepheus-1-36q-conservative"
        assert CEPHEUS_SPECS_CONSERVATIVE.fidelity_2q_intra == 0.994
        assert CEPHEUS_SPECS_CONSERVATIVE.fidelity_2q_inter == 0.985

    def test_conservative_lower_fidelity(self):
        """Conservative should have lower fidelities than default."""
        assert (
            CEPHEUS_SPECS_CONSERVATIVE.fidelity_2q_intra
            <= CEPHEUS_SPECS.fidelity_2q_intra
        )
        assert (
            CEPHEUS_SPECS_CONSERVATIVE.fidelity_2q_inter
            <= CEPHEUS_SPECS.fidelity_2q_inter
        )


# =============================================================================
# CUMULATIVE FIDELITY TESTS
# =============================================================================

class TestCumulativeFidelity:
    """Test cumulative fidelity calculations."""

    def test_single_layer_heterogeneous(self):
        f = cumulative_fidelity(1, heterogeneous=True)
        expected = (0.995 ** 48) * (0.990 ** 12)
        assert abs(f - expected) < 1e-10

    def test_single_layer_homogeneous(self):
        f = cumulative_fidelity(1, heterogeneous=False)
        avg_f2q = CEPHEUS_SPECS.fidelity_2q
        expected = avg_f2q ** 60
        assert abs(f - expected) < 1e-10

    def test_heterogeneous_lower_than_all_intra(self):
        """Heterogeneous fidelity should be lower than if all edges were intra."""
        f_het = cumulative_fidelity(3, heterogeneous=True)
        f_all_intra = CEPHEUS_SPECS.fidelity_2q_intra ** (60 * 3)
        assert f_het < f_all_intra

    def test_monotonic_decay(self):
        """Fidelity must decrease with depth."""
        fidelities = [cumulative_fidelity(d) for d in range(1, 10)]
        for i in range(len(fidelities) - 1):
            assert fidelities[i] > fidelities[i + 1]

    def test_depth_zero(self):
        """Depth 0 should give fidelity 1.0."""
        assert cumulative_fidelity(0) == 1.0

    def test_with_conservative_specs(self):
        """Conservative specs should give lower fidelity."""
        f_default = cumulative_fidelity(3, CEPHEUS_SPECS)
        f_conservative = cumulative_fidelity(3, CEPHEUS_SPECS_CONSERVATIVE)
        assert f_conservative < f_default


# =============================================================================
# OPTIMAL DEPTH TESTS
# =============================================================================

class TestOptimalDepth:
    """Test optimal depth formula."""

    def test_default_threshold(self):
        d = optimal_depth(0.70)
        # Should be reasonable for 60-edge processor
        assert 0.5 < d < 10.0

    def test_consistency_heterogeneous(self):
        """F_cum(d_opt) should equal threshold."""
        threshold = 0.70
        d_opt = optimal_depth(threshold, heterogeneous=True)
        f_at_opt = cumulative_fidelity(d_opt, heterogeneous=True)
        assert abs(f_at_opt - threshold) < 0.01

    def test_consistency_homogeneous(self):
        """F_cum(d_opt) should equal threshold (homogeneous mode)."""
        threshold = 0.70
        d_opt = optimal_depth(threshold, heterogeneous=False)
        f_at_opt = cumulative_fidelity(d_opt, heterogeneous=False)
        assert abs(f_at_opt - threshold) < 0.01

    def test_higher_threshold_means_shallower(self):
        """Higher fidelity threshold requires shallower circuits."""
        d_high = optimal_depth(0.80)
        d_low = optimal_depth(0.60)
        assert d_high < d_low

    def test_cepheus_shallower_than_novera(self):
        """Cepheus (60 edges) should have shallower optimal depth than Novera (12 edges).

        Despite higher per-edge fidelity, more edges means cumulative
        fidelity decays faster, requiring shallower circuits.
        """
        from fakenovera.specs import (
            NOVERA_SPECS,
            optimal_depth as novera_optimal_depth,
        )
        d_novera = novera_optimal_depth(0.70, NOVERA_SPECS)
        d_cepheus = optimal_depth(0.70, CEPHEUS_SPECS)
        assert d_cepheus < d_novera


# =============================================================================
# NETWORKX TOPOLOGY TESTS
# =============================================================================

class TestNetworkXTopology:
    """Test NetworkX graph generation."""

    def test_topology_creation(self):
        try:
            from fakecepheus import get_topology
            G = get_topology()
            assert G.number_of_nodes() == 36
            assert G.number_of_edges() == 60
        except ImportError:
            pytest.skip("networkx not installed")

    def test_node_chip_attribute(self):
        try:
            from fakecepheus import get_topology
            G = get_topology()
            for q in range(36):
                assert "chip" in G.nodes[q]
                assert G.nodes[q]["chip"] == q // 9
        except ImportError:
            pytest.skip("networkx not installed")

    def test_edge_type_attribute(self):
        try:
            from fakecepheus import get_topology
            G = get_topology()
            intra_count = sum(
                1 for _, _, d in G.edges(data=True) if d["type"] == "intra"
            )
            inter_count = sum(
                1 for _, _, d in G.edges(data=True) if d["type"] == "inter"
            )
            assert intra_count == 48
            assert inter_count == 12
        except ImportError:
            pytest.skip("networkx not installed")


# =============================================================================
# QISKIT BACKEND TESTS
# =============================================================================

class TestQiskitBackend:
    """Test Qiskit-based FakeCepheus backend."""

    def test_creation(self):
        from fakecepheus.qiskit import FakeCepheus
        backend = FakeCepheus()
        assert backend.num_qubits == 36

    def test_with_seed(self):
        from fakecepheus.qiskit import FakeCepheus
        backend = FakeCepheus(seed=42)
        assert backend is not None

    def test_noisy_vs_ideal(self):
        from fakecepheus.qiskit import FakeCepheus
        noisy = FakeCepheus(noise=True)
        ideal = FakeCepheus(noise=False)
        assert noisy.noise_model is not None
        assert ideal.noise_model is None

    def test_specs_property(self):
        from fakecepheus.qiskit import FakeCepheus
        backend = FakeCepheus(specs=CEPHEUS_SPECS_CONSERVATIVE)
        assert backend.specs.name == "cepheus-1-36q-conservative"

    def test_run_circuit(self):
        from qiskit import QuantumCircuit
        from fakecepheus.qiskit import FakeCepheus

        backend = FakeCepheus(seed=42)
        qc = QuantumCircuit(36)
        qc.h(4)
        qc.measure_all()

        job = backend.run(qc, shots=100)
        counts = job.result().get_counts()
        assert sum(counts.values()) == 100

    def test_cz_on_intra_chip_edge(self):
        from qiskit import QuantumCircuit
        from fakecepheus.qiskit import FakeCepheus

        backend = FakeCepheus(seed=42)
        qc = QuantumCircuit(36)
        qc.cz(0, 1)   # Intra-chip edge
        qc.measure_all()

        job = backend.run(qc, shots=10)
        result = job.result()
        assert result.success

    def test_cz_on_inter_chip_edge(self):
        from qiskit import QuantumCircuit
        from fakecepheus.qiskit import FakeCepheus

        backend = FakeCepheus(seed=42)
        qc = QuantumCircuit(36)
        qc.cz(2, 9)   # Inter-chip edge (chip 0 -> chip 1)
        qc.measure_all()

        job = backend.run(qc, shots=10)
        result = job.result()
        assert result.success

    def test_optimal_depth_method(self):
        from fakecepheus.qiskit import FakeCepheus
        backend = FakeCepheus()
        d = backend.optimal_depth()
        assert isinstance(d, int)
        assert d > 0
        # Use wider thresholds to avoid rounding to same integer (60 edges
        # means very small fractional depths)
        assert backend.optimal_depth(0.90) <= backend.optimal_depth(0.30)

    def test_cumulative_fidelity_method(self):
        from fakecepheus.qiskit import FakeCepheus
        backend = FakeCepheus()
        f1 = backend.cumulative_fidelity(1)
        f5 = backend.cumulative_fidelity(5)
        assert 0 < f5 < f1 < 1.0

    def test_repr(self):
        from fakecepheus.qiskit import FakeCepheus
        backend = FakeCepheus()
        assert "FakeCepheus" in repr(backend)
        assert "Qiskit" in repr(backend)
        assert "36q" in repr(backend)


# =============================================================================
# NOISE MODEL TESTS
# =============================================================================

class TestQiskitNoiseModel:
    """Test Qiskit noise model creation."""

    def test_noise_model_creation(self):
        from fakecepheus.qiskit import create_cepheus_noise_model
        noise = create_cepheus_noise_model()
        assert noise is not None

    def test_target_creation(self):
        from fakecepheus.qiskit import create_cepheus_target
        target = create_cepheus_target()
        assert target.num_qubits == 36

    def test_noise_model_with_conservative_specs(self):
        from fakecepheus.qiskit import create_cepheus_noise_model
        noise = create_cepheus_noise_model(specs=CEPHEUS_SPECS_CONSERVATIVE)
        assert noise is not None


# =============================================================================
# UNIFIED API TESTS
# =============================================================================

class TestUnifiedAPI:
    """Test the unified get_backend() factory function."""

    def test_get_qiskit_backend(self):
        backend = get_backend("qiskit", seed=42)
        assert backend.num_qubits == 36
        assert "Qiskit" in repr(backend)

    def test_auto_fallback(self):
        """Auto should fall back to Qiskit when PyQuil unavailable."""
        backend = get_backend("auto")
        assert backend.num_qubits == 36

    def test_invalid_framework_raises(self):
        with pytest.raises(ValueError):
            get_backend("invalid_framework")

    def test_specs_passed_through(self):
        backend = get_backend("qiskit", specs=CEPHEUS_SPECS_CONSERVATIVE)
        assert backend.specs.name == "cepheus-1-36q-conservative"


# =============================================================================
# PYQUIL AVAILABILITY TESTS
# =============================================================================

class TestPyQuilAvailability:
    """Test PyQuil implementation availability checks."""

    def test_pyquil_import_check(self):
        try:
            from fakecepheus.pyquil import backend
            has_pyquil = backend.PYQUIL_AVAILABLE
            assert isinstance(has_pyquil, bool)
        except ImportError:
            pass

    def test_pyquil_noise_model_requires_qvm(self):
        try:
            backend = get_backend("pyquil")
            pytest.skip("QVM is available")
        except RuntimeError as e:
            assert "QVM" in str(e) or "quilc" in str(e)
        except ImportError:
            pytest.skip("PyQuil not installed")


# =============================================================================
# COMPARISON WITH NOVERA
# =============================================================================

class TestNoveraComparison:
    """Compare FakeCepheus with FakeNovera."""

    def test_cepheus_more_qubits(self):
        from fakenovera.specs import NOVERA_SPECS
        assert CEPHEUS_SPECS.num_qubits > NOVERA_SPECS.num_qubits
        assert CEPHEUS_SPECS.num_qubits == 4 * NOVERA_SPECS.num_qubits

    def test_cepheus_more_edges(self):
        from fakenovera.specs import NOVERA_EDGES
        assert len(CEPHEUS_EDGES) > len(NOVERA_EDGES)
        # 48 intra (4 x 12 Novera) + 12 inter = 60
        assert len(CEPHEUS_EDGES) == 4 * len(NOVERA_EDGES) + 12

    def test_cepheus_higher_per_edge_fidelity(self):
        """Cepheus intra-chip F_2Q should be >= Novera F_2Q."""
        from fakenovera.specs import NOVERA_SPECS
        assert (
            CEPHEUS_SPECS.fidelity_2q_intra >= NOVERA_SPECS.fidelity_2q
        )

    def test_faster_gate_time(self):
        """Cepheus (Ankaa-3 gen) has faster 2Q gates than Novera."""
        from fakenovera.specs import NOVERA_SPECS
        assert CEPHEUS_SPECS.gate_time_2q_ns < NOVERA_SPECS.gate_time_2q_ns

    def test_optimal_depth_comparison(self):
        """Compare optimal depths - more edges means shallower depth."""
        from fakenovera.specs import (
            NOVERA_SPECS,
            optimal_depth as novera_optimal_depth,
        )
        d_novera = round(novera_optimal_depth(0.70, NOVERA_SPECS))
        d_cepheus = round(optimal_depth(0.70, CEPHEUS_SPECS))

        # Novera: d_opt ~ 5, Cepheus should be much smaller
        assert d_cepheus < d_novera


# =============================================================================
# QRC INTEGRATION TESTS
# =============================================================================

class TestQRCIntegration:
    """Test QRC-specific functionality."""

    def test_70_percent_threshold(self):
        """70% is the QRC phase transition threshold."""
        d_opt = optimal_depth(0.70)
        f_at_opt = cumulative_fidelity(round(d_opt))
        assert 0.50 < f_at_opt < 0.90

    def test_reservoir_encoding_layers(self):
        """Test fidelity at typical reservoir depths."""
        for d in range(1, 6):
            f = cumulative_fidelity(d)
            assert 0 < f < 1.0

    def test_noise_impact_on_qrc(self):
        """Test that noise model affects circuit outcomes."""
        from qiskit import QuantumCircuit
        from fakecepheus.qiskit import FakeCepheus

        noisy = FakeCepheus(seed=42, noise=True)
        ideal = FakeCepheus(seed=42, noise=False)

        qc = QuantumCircuit(36)
        qc.h(4)
        # Use intra-chip edges
        qc.cz(4, 1)
        qc.cz(4, 3)
        qc.cz(4, 5)
        qc.cz(4, 7)
        qc.measure_all()

        noisy_counts = noisy.run(qc, shots=1000).result().get_counts()
        ideal_counts = ideal.run(qc, shots=1000).result().get_counts()

        # Noisy should have more entropy (more states)
        assert len(noisy_counts) >= len(ideal_counts)

    def test_print_specs_runs(self, capsys):
        """Test that print_specs runs without errors."""
        print_specs(CEPHEUS_SPECS)
        captured = capsys.readouterr()
        assert "CEPHEUS" in captured.out
        assert "36" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
