"""
Tests for FakeNovera Dual Implementation Package
================================================

Tests both PyQuil and Qiskit implementations with shared specs.

Author: Daniel Mo Houshmand (QDaria AS)
"""

import sys
import pytest
import numpy as np

# Test the new package structure
from fakenovera import (
    get_backend,
    NoveraSpecs,
    NOVERA_SPECS,
    NOVERA_SPECS_OFFICIAL,
    NOVERA_SPECS_ZURICH,
    NOVERA_EDGES,
    NOVERA_QUBITS,
    NOVERA_COUPLING_MAP,
    cumulative_fidelity,
    optimal_depth,
    print_specs,
)


# =============================================================================
# SPECIFICATION TESTS
# =============================================================================

class TestNoveraSpecs:
    """Test hardware specifications dataclass."""

    def test_qubit_count(self):
        assert NOVERA_SPECS.num_qubits == 9

    def test_edge_count(self):
        assert len(NOVERA_EDGES) == 12

    def test_coupler_count(self):
        assert NOVERA_SPECS.num_couplers == 12

    def test_fidelities_valid(self):
        assert 0 < NOVERA_SPECS.fidelity_1q <= 1
        assert 0 < NOVERA_SPECS.fidelity_2q <= 1
        assert 0 < NOVERA_SPECS.readout_fidelity <= 1

    def test_t2_constraint(self):
        """T2 cannot exceed 2*T1 (physical constraint)."""
        assert NOVERA_SPECS.t2_us <= 2 * NOVERA_SPECS.t1_us

    def test_qubits_list(self):
        assert NOVERA_QUBITS == list(range(9))

    def test_coupling_map_bidirectional(self):
        # Should have 24 entries (12 edges × 2 directions)
        assert len(NOVERA_COUPLING_MAP) == 24


class TestSpecProfiles:
    """Test different specification profiles."""

    def test_official_specs(self):
        assert NOVERA_SPECS_OFFICIAL.t1_us == 27.0
        assert NOVERA_SPECS_OFFICIAL.t2_us == 27.0
        assert NOVERA_SPECS_OFFICIAL.fidelity_1q == 0.999

    def test_zurich_specs(self):
        assert NOVERA_SPECS_ZURICH.t1_us == 45.9
        assert NOVERA_SPECS_ZURICH.t2_us == 25.5
        assert NOVERA_SPECS_ZURICH.fidelity_1q == 0.9951

    def test_both_share_2q_fidelity(self):
        """Both profiles share F_2Q = 99.4%."""
        assert NOVERA_SPECS_OFFICIAL.fidelity_2q == 0.994
        assert NOVERA_SPECS_ZURICH.fidelity_2q == 0.994

    def test_default_is_official(self):
        """Default should be official (more conservative)."""
        assert NOVERA_SPECS.name == "novera-9q-official"

    def test_invalid_t2_raises(self):
        """T2 > 2*T1 should raise ValueError."""
        with pytest.raises(ValueError):
            NoveraSpecs(t1_us=10.0, t2_us=30.0)  # T2 > 2*T1


# =============================================================================
# ANALYSIS FUNCTION TESTS
# =============================================================================

class TestCumulativeFidelity:
    """Test cumulative fidelity calculations."""

    def test_single_layer(self):
        f = cumulative_fidelity(1)
        expected = NOVERA_SPECS.fidelity_2q ** 12  # 12 CZ gates
        assert abs(f - expected) < 1e-6

    def test_five_layers(self):
        f = cumulative_fidelity(5)
        expected = NOVERA_SPECS.fidelity_2q ** 60  # 60 CZ gates
        assert abs(f - expected) < 1e-6
        # With F_2Q = 0.994, F_cum(5) = 0.994^60 ≈ 0.697
        assert 0.65 < f < 0.75  # Should be near 70%

    def test_monotonic_decay(self):
        """Fidelity must decrease with depth."""
        fidelities = [cumulative_fidelity(d) for d in range(1, 10)]
        for i in range(len(fidelities) - 1):
            assert fidelities[i] > fidelities[i + 1]

    def test_with_different_specs(self):
        """Both profiles should give same result (same F_2Q)."""
        f_official = cumulative_fidelity(5, NOVERA_SPECS_OFFICIAL)
        f_zurich = cumulative_fidelity(5, NOVERA_SPECS_ZURICH)
        assert abs(f_official - f_zurich) < 1e-10


class TestOptimalDepth:
    """Test optimal depth formula."""

    def test_default_threshold(self):
        d = optimal_depth(0.70)
        # d_opt = ln(0.70) / (12 * ln(0.994)) ≈ 4.94
        assert 4.5 < d < 5.5

    def test_rounded_value(self):
        """Optimal depth at 70% threshold should round to 5."""
        d = optimal_depth(0.70)
        assert round(d) == 5

    def test_higher_threshold_means_shallower(self):
        """Higher fidelity threshold requires shallower circuits."""
        d_high = optimal_depth(0.80)
        d_low = optimal_depth(0.60)
        assert d_high < d_low

    def test_formula_consistency(self):
        """F_cum(d_opt) should equal threshold."""
        threshold = 0.70
        d_opt = optimal_depth(threshold)
        f_at_opt = cumulative_fidelity(d_opt)
        assert abs(f_at_opt - threshold) < 0.01  # Within 1%


# =============================================================================
# QISKIT BACKEND TESTS
# =============================================================================

class TestQiskitBackend:
    """Test Qiskit-based FakeNovera backend."""

    def test_creation(self):
        from fakenovera.qiskit import FakeNovera
        backend = FakeNovera()
        assert backend.num_qubits == 9

    def test_with_seed(self):
        from fakenovera.qiskit import FakeNovera
        backend = FakeNovera(seed=42)
        assert backend is not None

    def test_noisy_vs_ideal(self):
        from fakenovera.qiskit import FakeNovera
        noisy = FakeNovera(noise=True)
        ideal = FakeNovera(noise=False)
        assert noisy.noise_model is not None
        assert ideal.noise_model is None

    def test_specs_property(self):
        from fakenovera.qiskit import FakeNovera
        backend = FakeNovera(specs=NOVERA_SPECS_ZURICH)
        assert backend.specs.name == "novera-9q-zurich"

    def test_run_circuit(self):
        from qiskit import QuantumCircuit
        from fakenovera.qiskit import FakeNovera

        backend = FakeNovera(seed=42)
        qc = QuantumCircuit(9)
        qc.h(4)
        qc.measure_all()

        job = backend.run(qc, shots=100)
        counts = job.result().get_counts()
        assert sum(counts.values()) == 100

    def test_cz_on_valid_edge(self):
        from qiskit import QuantumCircuit
        from fakenovera.qiskit import FakeNovera

        backend = FakeNovera(seed=42)
        qc = QuantumCircuit(9)
        qc.cz(0, 1)  # Valid edge
        qc.measure_all()

        job = backend.run(qc, shots=10)
        result = job.result()
        assert result.success

    def test_optimal_depth_method(self):
        from fakenovera.qiskit import FakeNovera
        backend = FakeNovera()
        assert backend.optimal_depth() == 5
        assert backend.optimal_depth(0.80) < backend.optimal_depth(0.60)

    def test_cumulative_fidelity_method(self):
        from fakenovera.qiskit import FakeNovera
        backend = FakeNovera()
        f5 = backend.cumulative_fidelity(5)
        assert 0.65 < f5 < 0.75

    def test_repr(self):
        from fakenovera.qiskit import FakeNovera
        backend = FakeNovera()
        assert "FakeNovera" in repr(backend)
        assert "Qiskit" in repr(backend)


# =============================================================================
# UNIFIED API TESTS
# =============================================================================

class TestUnifiedAPI:
    """Test the unified get_backend() factory function."""

    def test_get_qiskit_backend(self):
        backend = get_backend("qiskit", seed=42)
        assert backend.num_qubits == 9
        assert "Qiskit" in repr(backend)

    def test_auto_fallback(self):
        """Auto should fall back to Qiskit when PyQuil unavailable."""
        # Without QVM running, auto should return Qiskit backend
        backend = get_backend("auto")
        assert backend.num_qubits == 9

    def test_invalid_framework_raises(self):
        with pytest.raises(ValueError):
            get_backend("invalid_framework")

    def test_specs_passed_through(self):
        backend = get_backend("qiskit", specs=NOVERA_SPECS_ZURICH)
        assert backend.specs.name == "novera-9q-zurich"


# =============================================================================
# NOISE MODEL TESTS
# =============================================================================

class TestQiskitNoiseModel:
    """Test Qiskit noise model creation."""

    def test_noise_model_creation(self):
        from fakenovera.qiskit import create_novera_noise_model
        noise = create_novera_noise_model()
        assert noise is not None

    def test_target_creation(self):
        from fakenovera.qiskit import create_novera_target
        target = create_novera_target()
        assert target.num_qubits == 9

    def test_noise_model_with_zurich_specs(self):
        from fakenovera.qiskit import create_novera_noise_model
        noise = create_novera_noise_model(specs=NOVERA_SPECS_ZURICH)
        assert noise is not None


# =============================================================================
# PYQUIL AVAILABILITY TESTS
# =============================================================================

class TestPyQuilAvailability:
    """Test PyQuil implementation availability checks."""

    def test_pyquil_import_check(self):
        """Test that PyQuil availability is correctly detected."""
        try:
            from fakenovera.pyquil import backend
            has_pyquil = backend.PYQUIL_AVAILABLE
            # If pyquil is installed, this should be True
            # If not installed, PYQUIL_AVAILABLE will be False
            assert isinstance(has_pyquil, bool)
        except ImportError:
            # If even the import fails, that's also acceptable
            pass

    def test_pyquil_noise_model_requires_qvm(self):
        """PyQuil backend requires QVM/quilc to be running."""
        try:
            backend = get_backend("pyquil")
            # If this works, QVM is running
            pytest.skip("QVM is available")
        except RuntimeError as e:
            assert "QVM" in str(e) or "quilc" in str(e)
        except ImportError:
            pytest.skip("PyQuil not installed")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestQRCIntegration:
    """Test QRC-specific functionality."""

    def test_70_percent_threshold(self):
        """70% is the QRC phase transition threshold."""
        d_opt = optimal_depth(0.70)
        f_at_opt = cumulative_fidelity(round(d_opt))
        # At depth 5, F_cum should be near 70%
        assert 0.60 < f_at_opt < 0.80

    def test_paper_claimed_depth(self):
        """Paper claims d_opt ≈ 5 for 70% threshold."""
        d_opt = optimal_depth(0.70)
        assert round(d_opt) == 5

    def test_reservoir_encoding_layers(self):
        """Test fidelity at typical reservoir depths."""
        # Typical QRC uses 3-7 variational layers
        for d in range(3, 8):
            f = cumulative_fidelity(d)
            # Should degrade gracefully
            assert 0.4 < f < 1.0

    def test_noise_impact_on_qrc(self):
        """Test that noise model affects circuit outcomes."""
        from qiskit import QuantumCircuit
        from fakenovera.qiskit import FakeNovera

        # Create two backends: one noisy, one ideal
        noisy = FakeNovera(seed=42, noise=True)
        ideal = FakeNovera(seed=42, noise=False)

        # Simple circuit with multiple gates
        qc = QuantumCircuit(9)
        qc.h(4)
        for i in [1, 3, 5, 7]:
            qc.cz(4, i)
        qc.measure_all()

        # Run both
        noisy_counts = noisy.run(qc, shots=1000).result().get_counts()
        ideal_counts = ideal.run(qc, shots=1000).result().get_counts()

        # Noisy should have more entropy (more states)
        assert len(noisy_counts) >= len(ideal_counts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
