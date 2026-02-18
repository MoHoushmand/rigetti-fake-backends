"""
FakeNovera - Qiskit Backend
===========================

Qiskit-compatible BackendV2 implementation with Novera noise model.

This provides compatibility for users without QVM/quilc infrastructure.
For native Rigetti support, use fakenovera.pyquil instead.

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
    NOVERA_SPECS,
    NOVERA_SPECS_ZURICH,
    NOVERA_COUPLING_MAP,
    NoveraSpecs,
    cumulative_fidelity,
    optimal_depth,
)
from .noise import create_novera_noise_model, create_novera_target


class FakeNovera(BackendV2):
    """
    Qiskit-based noise model for Rigetti Novera 9Q quantum processor.

    Simulates Novera QPU with realistic noise matching verified specs.

    Two specification profiles available:
    - NOVERA_SPECS (default): Official Rigetti specs (T1=27μs, F_1Q=99.9%)
    - NOVERA_SPECS_ZURICH: Zurich benchmarks (T1=45.9μs, F_1Q=99.51%)

    Note: This uses Qiskit/AerSimulator. For native Rigetti simulation,
    use fakenovera.pyquil with QVM.

    Example:
        >>> backend = FakeNovera()
        >>> qc = QuantumCircuit(9)
        >>> qc.h(4)
        >>> qc.cz(4, 1)
        >>> qc.measure_all()
        >>> job = backend.run(qc, shots=1000)
        >>> counts = job.result().get_counts()
    """

    def __init__(
        self,
        specs: NoveraSpecs = NOVERA_SPECS,
        seed: Optional[int] = None,
        noise: bool = True,
    ):
        """
        Initialize FakeNovera backend.

        Args:
            specs: Hardware specifications (NOVERA_SPECS or NOVERA_SPECS_ZURICH)
            seed: Random seed for reproducibility
            noise: Whether to include noise model (default: True)
        """
        super().__init__(
            name="fake_novera_qiskit",
            description=f"Qiskit-based Novera 9Q ({specs.name})",
            backend_version="2.2.0",
        )

        self._specs = specs
        self._seed = seed
        self._noise_enabled = noise

        self._target = create_novera_target(specs)
        self._noise_model = create_novera_noise_model(specs) if noise else None

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
    def specs(self) -> NoveraSpecs:
        return self._specs

    @property
    def coupling_map(self) -> List[List[int]]:
        return NOVERA_COUPLING_MAP

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

    def cumulative_fidelity(self, depth: int) -> float:
        """Calculate cumulative fidelity for given depth."""
        return cumulative_fidelity(depth, self._specs)

    def optimal_depth(self, threshold: float = 0.70) -> int:
        """Calculate optimal circuit depth for QRC."""
        return round(optimal_depth(threshold, self._specs))

    def __repr__(self) -> str:
        noise_str = "noisy" if self._noise_enabled else "ideal"
        return f"FakeNovera[Qiskit]({noise_str}, {self.num_qubits}q, {self._specs.name})"


# Alias
FakeNoveraBackend = FakeNovera


def get_fake_novera(
    specs: NoveraSpecs = NOVERA_SPECS,
    seed: Optional[int] = None,
    noise: bool = True,
) -> FakeNovera:
    """Factory function to create FakeNovera backend."""
    return FakeNovera(specs=specs, seed=seed, noise=noise)
