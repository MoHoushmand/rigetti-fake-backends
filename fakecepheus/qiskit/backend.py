"""
FakeCepheus - Qiskit Backend
============================

Qiskit-compatible BackendV2 implementation with Cepheus-1-36Q noise model.

This provides compatibility for users without QVM/quilc infrastructure.
For native Rigetti support, use fakecepheus.pyquil instead.

KEY FEATURE: Heterogeneous noise - intra-chip gates use F_2Q = 99.5%,
inter-chip gates use F_2Q = 99.0%.

Author: Daniel Mo Houshmand (QDaria AS)
License: MIT
"""

from typing import Optional, List, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target
from qiskit_aer import AerSimulator

from ..specs import (
    CEPHEUS_SPECS,
    CEPHEUS_SPECS_CONSERVATIVE,
    CEPHEUS_COUPLING_MAP,
    CepheusSpecs,
    cumulative_fidelity,
    optimal_depth,
)
from .noise import create_cepheus_noise_model, create_cepheus_target


class FakeCepheus(BackendV2):
    """
    Qiskit-based noise model for Rigetti Cepheus-1-36Q quantum processor.

    Simulates Cepheus QPU with realistic heterogeneous noise matching
    verified specs. Intra-chip gates have higher fidelity than inter-chip
    gates due to intermodule coupler overhead.

    Specification profiles:
    - CEPHEUS_SPECS (default): F_2Q_intra=99.5%, F_2Q_inter=99.0%
    - CEPHEUS_SPECS_CONSERVATIVE: F_2Q_intra=99.4%, F_2Q_inter=98.5%

    Note: This uses Qiskit/AerSimulator. For native Rigetti simulation,
    use fakecepheus.pyquil with QVM.

    Example:
        >>> backend = FakeCepheus()
        >>> qc = QuantumCircuit(36)
        >>> qc.h(4)
        >>> qc.cz(4, 1)
        >>> qc.measure_all()
        >>> job = backend.run(qc, shots=1000)
        >>> counts = job.result().get_counts()
    """

    def __init__(
        self,
        specs: CepheusSpecs = CEPHEUS_SPECS,
        seed: Optional[int] = None,
        noise: bool = True,
    ):
        """
        Initialize FakeCepheus backend.

        Args:
            specs: Hardware specifications (CEPHEUS_SPECS or CEPHEUS_SPECS_CONSERVATIVE)
            seed: Random seed for reproducibility
            noise: Whether to include noise model (default: True)
        """
        super().__init__(
            name="fake_cepheus_qiskit",
            description=f"Qiskit-based Cepheus-1-36Q ({specs.name})",
            backend_version="1.0.0",
        )

        self._specs = specs
        self._seed = seed
        self._noise_enabled = noise

        self._target = create_cepheus_target(specs)
        self._noise_model = (
            create_cepheus_noise_model(specs) if noise else None
        )

        self._simulator = AerSimulator(
            noise_model=self._noise_model,
            seed_simulator=seed,
        )

        self._options = Options()
        self._options.update_options(shots=2000, seed_simulator=seed)

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> Optional[int]:
        return None

    @property
    def num_qubits(self) -> int:
        return self._specs.num_qubits

    @property
    def specs(self) -> CepheusSpecs:
        return self._specs

    @property
    def coupling_map(self) -> List[List[int]]:
        return CEPHEUS_COUPLING_MAP

    @property
    def noise_model(self):
        return self._noise_model

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=2000)

    def run(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        **kwargs
    ):
        """Run circuits on the simulated backend."""
        options = dict(self._options.__dict__)
        options.update(kwargs)
        return self._simulator.run(circuits, **options)

    def cumulative_fidelity(self, depth: int, heterogeneous: bool = True) -> float:
        """Calculate cumulative fidelity for given depth.

        Args:
            depth: Circuit depth (number of variational layers)
            heterogeneous: Use per-edge fidelities (default True)

        Returns:
            Cumulative fidelity as a fraction [0, 1]
        """
        return cumulative_fidelity(depth, self._specs, heterogeneous)

    def optimal_depth(
        self, threshold: float = 0.70, heterogeneous: bool = True
    ) -> int:
        """Calculate optimal circuit depth for QRC.

        Args:
            threshold: Target cumulative fidelity (default 0.70)
            heterogeneous: Use per-edge fidelities (default True)

        Returns:
            Optimal depth (rounded to nearest integer)
        """
        return round(optimal_depth(threshold, self._specs, heterogeneous))

    def __repr__(self) -> str:
        noise_str = "noisy" if self._noise_enabled else "ideal"
        return (
            f"FakeCepheus[Qiskit]({noise_str}, {self.num_qubits}q, "
            f"{self._specs.name})"
        )


# Alias
FakeCepheusBackend = FakeCepheus


def get_fake_cepheus(
    specs: CepheusSpecs = CEPHEUS_SPECS,
    seed: Optional[int] = None,
    noise: bool = True,
) -> FakeCepheus:
    """Factory function to create FakeCepheus backend."""
    return FakeCepheus(specs=specs, seed=seed, noise=noise)
