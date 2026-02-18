# FakeNovera & FakeCepheus

**Open-source noise models for Rigetti quantum processors**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/MoHoushmand/rigetti-fake-backends/actions/workflows/ci.yml/badge.svg)](https://github.com/MoHoushmand/rigetti-fake-backends/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

The first open-source "fake backend" simulators for Rigetti's **Novera 9Q** and **Cepheus-1 36Q** quantum processors. Dual PyQuil/Qiskit implementations with calibrated noise parameters from published hardware specifications.

> **Accompanying paper:** D. M. Houshmand, "FakeNovera and FakeCepheus: Open-Source Noise Models for Rigetti Quantum Processors with Application to Depth-Optimized Reservoir Computing," *IEEE Transactions on Quantum Engineering*, 2026. [[arXiv](https://arxiv.org/abs/XXXX.XXXXX)]

---

## Why?

IBM has [`FakeManila`](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/fake_provider), IQM has [`IQMFakeAdonis`](https://iqm-finland.github.io/qiskit-on-iqm/), but **Rigetti has no open-source fake backends**. Until now.

| Feature | FakeNovera | FakeCepheus |
|---------|-----------|-------------|
| Qubits | 9 | 36 (4 x 9 chiplets) |
| Topology | 3x3 square lattice | 2x2 tiled multi-chip |
| Edges | 12 | 60 (48 intra + 12 inter) |
| Noise model | Homogeneous | **Heterogeneous** (intra vs inter-chip) |
| Native gates | CZ, RZ, SX, X | CZ, RZ, SX, X |
| Frameworks | PyQuil + Qiskit | PyQuil + Qiskit |
| d_opt (QRC, 70%) | 5 layers | 1 layer |

---

## Architecture

```
                    FakeNovera 9Q                          FakeCepheus 36Q
                    3x3 Square Lattice                     2x2 Multi-Chip Tiling

                    0 --- 1 --- 2               Chip 0              Chip 1
                    |     |     |              0 -- 1 -- 2        9 -- 10 -- 11
                    3 --- 4 --- 5              |    |    |        |     |     |
                    |     |     |              3 -- 4 -- 5 ~~~~~ 12 -- 13 -- 14
                    6 --- 7 --- 8              |    |    |        |     |     |
                                               6 -- 7 -- 8       15 -- 16 -- 17
                    12 edges                        |                   |
                    F_2Q = 99.4%                  inter               inter
                                                    |                   |
                                              Chip 2              Chip 3
                                              18 -- 19 -- 20      27 -- 28 -- 29
                                               |     |    |        |     |     |
                                              21 -- 22 -- 23 ~~~~~ 30 -- 31 -- 32
                                               |     |    |        |     |     |
                                              24 -- 25 -- 26      33 -- 34 -- 35

                                               48 intra-chip edges (F_2Q = 99.5%)
                                               12 inter-chip edges (F_2Q = 99.0%)
                                               ~~~~ = intermodule coupler
```

---

## Installation

```bash
# Qiskit backend (recommended for most users)
pip install rigetti-fake-backends[qiskit]

# PyQuil backend (requires QVM/quilc)
pip install rigetti-fake-backends[pyquil]

# Both frameworks
pip install rigetti-fake-backends[all]

# Development
pip install rigetti-fake-backends[dev]
```

**From source:**

```bash
git clone https://github.com/MoHoushmand/rigetti-fake-backends.git
cd rigetti-fake-backends
pip install -e ".[all,dev]"
```

---

## Quick Start

### Qiskit (no external dependencies)

```python
from fakenovera.qiskit import FakeNovera
from qiskit import QuantumCircuit

# Create backend with realistic Novera noise
backend = FakeNovera(seed=42)

# Build and run a circuit
qc = QuantumCircuit(9)
qc.h(4)                    # Hadamard on center qubit
qc.cz(4, 1); qc.cz(4, 3)  # CZ on lattice edges
qc.cz(4, 5); qc.cz(4, 7)
qc.measure_all()

job = backend.run(qc, shots=2000)
counts = job.result().get_counts()
```

### FakeCepheus with heterogeneous noise

```python
from fakecepheus.qiskit import FakeCepheus

backend = FakeCepheus(seed=42)

# Intra-chip gates: F_2Q = 99.5%
# Inter-chip gates: F_2Q = 99.0%  (automatic per-edge assignment)
print(backend)  # FakeCepheus[Qiskit](noisy, 36q, cepheus-1-36q)
```

### PyQuil (native Rigetti)

```bash
# Start QVM and quilc first
docker run -d -p 5555:5555 rigetti/quilc -R
docker run -d -p 5000:5000 rigetti/qvm -S
```

```python
from fakenovera.pyquil import get_qc
from pyquil import Program
from pyquil.gates import H, CZ, MEASURE

qc = get_qc()  # Noisy 9Q-square-lattice QuantumComputer
p = Program(H(4), CZ(4, 1), MEASURE(4, 0))
result = qc.run(qc.compile(p))
```

### Depth Analysis (no framework needed)

```python
from fakenovera import optimal_depth, cumulative_fidelity, print_specs

# Optimal QRC depth at 70% fidelity threshold
d_opt = optimal_depth(0.70)
print(f"Novera d_opt = {d_opt:.2f} -> {round(d_opt)} layers")
# Novera d_opt = 4.94 -> 5 layers

# Cumulative fidelity at depth 5
f_cum = cumulative_fidelity(5)
print(f"F_cum(d=5) = {f_cum:.1%}")
# F_cum(d=5) = 69.7%

# Full analysis table
print_specs()
```

```python
from fakecepheus import optimal_depth, cumulative_fidelity

# Cepheus: 60 edges per layer -> fidelity drops fast
d_opt = optimal_depth(0.70)
print(f"Cepheus d_opt = {d_opt:.2f} -> {round(d_opt)} layer")
# Cepheus d_opt = 0.99 -> 1 layer  (curse of connectivity!)
```

---

## Hardware Specifications

### FakeNovera 9Q

| Parameter | Official | Zurich Benchmarks | Source |
|-----------|----------|-------------------|--------|
| Qubits | 9 | 9 | [rigetti.com/novera](https://www.rigetti.com/novera) |
| Topology | 3x3 square | 3x3 square | |
| T1 | 27 us | 45.9 us | Zurich Instruments (2024) |
| T2 | 27 us | 25.5 us | Zurich Instruments (2024) |
| F_1Q | 99.9% | 99.51% | |
| F_2Q (CZ) | 99.4% | 99.4% | |
| Readout | 97.96% | 97.96% | |
| 1Q gate time | 40 ns | 40 ns | |
| 2Q gate time | 200 ns | 200 ns | |

### FakeCepheus 36Q

| Parameter | Value | Source |
|-----------|-------|--------|
| Qubits | 36 (4 x 9 chiplets) | [Azure Quantum](https://learn.microsoft.com/en-us/azure/quantum/provider-rigetti) |
| Intra-chip edges | 48 (12 per chip) | |
| Inter-chip edges | 12 (3 per boundary) | |
| T1 | 34 us | Ankaa-3 / SQMS (improved) |
| T2 | 20 us | Ankaa-3 median |
| F_1Q | 99.91% | [Rigetti Ankaa-3](https://www.rigetti.com/newsroom/press-releases/rigetti-announces-ankaa-3) |
| F_2Q intra-chip | 99.5% | Ankaa-3 median |
| F_2Q inter-chip | 99.0% | Conservative estimate |
| Readout | 97.96% | Novera baseline |
| 2Q gate time | 72 ns (iSWAP) | [AWS Braket](https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices-rigetti.html) |

---

## API Reference

### Specification Profiles

```python
# FakeNovera
from fakenovera import NOVERA_SPECS            # Official Rigetti (default)
from fakenovera import NOVERA_SPECS_ZURICH      # Zurich benchmarks

# FakeCepheus
from fakecepheus import CEPHEUS_SPECS           # Default (F_2Q_intra=99.5%)
from fakecepheus import CEPHEUS_SPECS_CONSERVATIVE  # Pessimistic (F_2Q_intra=99.4%)
```

### Backend Factory

```python
from fakenovera import get_backend

# Auto-detect: tries PyQuil first, falls back to Qiskit
backend = get_backend("auto")

# Explicit framework selection
backend = get_backend("qiskit", seed=42, specs=NOVERA_SPECS_ZURICH)
backend = get_backend("pyquil", noisy=True)
```

### Topology

```python
from fakenovera import NOVERA_EDGES, NOVERA_COUPLING_MAP, get_topology
from fakecepheus import (
    CEPHEUS_EDGES,
    CEPHEUS_INTRA_CHIP_EDGES,
    CEPHEUS_INTER_CHIP_EDGES,
    is_inter_chip_edge,
    get_chip_id,
    get_topology,
)

# NetworkX graph with node/edge attributes
G = get_topology()  # nodes have 'chip' attr, edges have 'type' attr
```

### Noise Model (Qiskit)

```python
from fakenovera.qiskit.noise import create_novera_noise_model, create_novera_target
from fakecepheus.qiskit.noise import create_cepheus_noise_model, create_cepheus_target

# Standalone noise model (for use with your own AerSimulator)
noise = create_novera_noise_model(include_readout=True, include_thermal=True)
target = create_novera_target()

# Cepheus: heterogeneous noise (per-edge fidelity)
noise = create_cepheus_noise_model()  # Automatically assigns F_2Q per edge type
```

### Depth Analysis

```python
from fakenovera import cumulative_fidelity, optimal_depth

# Cumulative fidelity: F_cum = F_2Q^(n_edges * depth)
f = cumulative_fidelity(depth=5)  # 0.697

# Optimal depth: d_opt = ln(F_threshold) / (n_edges * ln(F_2Q))
d = optimal_depth(fidelity_threshold=0.70)  # 4.94

# Cepheus supports heterogeneous mode
from fakecepheus import cumulative_fidelity, optimal_depth
f = cumulative_fidelity(depth=1, heterogeneous=True)   # Per-edge fidelities
f = cumulative_fidelity(depth=1, heterogeneous=False)  # Weighted average
```

---

## Noise Model Details

Both simulators compose three noise channels:

1. **Depolarizing errors** -- derived from gate fidelity (1 - F)
2. **Thermal relaxation** -- T1/T2 decay during gate execution
3. **Asymmetric readout** -- P(1|0) = 0.4 * (1 - F_readout), P(0|1) = 0.6 * (1 - F_readout)

**FakeCepheus** additionally implements **heterogeneous 2Q noise**: intra-chip CZ gates use F_2Q = 99.5% while inter-chip CZ gates use F_2Q = 99.0%, reflecting the overhead of intermodule couplers.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific suite
pytest tests/test_fakenovera.py -v    # 41 tests
pytest tests/test_fakecepheus.py -v   # 70 tests

# With coverage
pytest tests/ --cov=fakenovera --cov=fakecepheus --cov-report=term-missing
```

---

## Project Structure

```
rigetti-fake-backends/
├── fakenovera/                 # FakeNovera 9Q package
│   ├── __init__.py             # Unified API + get_backend()
│   ├── specs.py                # Hardware specs, topology, analysis
│   ├── qiskit/                 # Qiskit BackendV2 implementation
│   │   ├── backend.py          # FakeNovera class
│   │   └── noise.py            # NoiseModel + Target creation
│   └── pyquil/                 # PyQuil QuantumComputer implementation
│       └── backend.py          # Noisy QVM configuration
├── fakecepheus/                # FakeCepheus 36Q package
│   ├── __init__.py             # Unified API + get_backend()
│   ├── specs.py                # Multi-chip specs, heterogeneous fidelity
│   ├── qiskit/                 # Qiskit BackendV2 implementation
│   │   ├── backend.py          # FakeCepheus class
│   │   └── noise.py            # Heterogeneous NoiseModel + Target
│   └── pyquil/                 # PyQuil QuantumComputer implementation
│       └── backend.py          # Noisy QVM configuration
├── tests/                      # 111 tests total
│   ├── test_fakenovera.py      # Specs, backend, noise, QRC integration
│   └── test_fakecepheus.py     # Topology, chiplets, heterogeneous noise
├── paper/                      # Accompanying IEEE paper
│   ├── main.tex                # LaTeX source
│   ├── references.bib          # 51 references
│   └── figures/                # 12 publication figures
├── pyproject.toml              # Package configuration
├── LICENSE                     # MIT
├── CITATION.cff                # Citation metadata
└── README.md                   # This file
```

---

## Citing This Work

If you use FakeNovera or FakeCepheus in your research, please cite:

```bibtex
@article{houshmand2026fakenovera,
  author  = {Houshmand, Daniel Mo},
  title   = {{FakeNovera} and {FakeCepheus}: Open-Source Noise Models for
             {Rigetti} Quantum Processors with Application to
             Depth-Optimized Reservoir Computing},
  journal = {IEEE Transactions on Quantum Engineering},
  year    = {2026},
  note    = {Preprint: arXiv:XXXX.XXXXX},
  doi     = {10.XXXX/XXXXXX}
}
```

For the software itself:

```bibtex
@software{houshmand2026fakenovera_software,
  author    = {Houshmand, Daniel Mo},
  title     = {rigetti-fake-backends: FakeNovera and FakeCepheus},
  version   = {0.1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/MoHoushmand/rigetti-fake-backends}
}
```

---

## Related Work

| Package | Vendor | Qubits | Source |
|---------|--------|--------|--------|
| `qiskit-ibm-runtime` (FakeManila, etc.) | IBM | 5-127 | [Qiskit Docs](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/fake_provider) |
| `qiskit-on-iqm` (IQMFakeAdonis) | IQM | 5-20 | [IQM Docs](https://iqm-finland.github.io/qiskit-on-iqm/) |
| **rigetti-fake-backends** | **Rigetti** | **9-36** | **This work** |

---

## Contributing

Contributions welcome! Areas of interest:

- **FakeAnkaa84**: Extend to the full 84-qubit Ankaa-3 processor
- **Calibration data**: Real device calibration snapshots
- **Benchmarks**: Comparative studies with real hardware
- **Additional frameworks**: Cirq, Pennylane adapters

```bash
# Development setup
git clone https://github.com/MoHoushmand/rigetti-fake-backends.git
cd rigetti-fake-backends
pip install -e ".[all,dev]"
pytest tests/ -v
```

---

## License

[MIT](LICENSE) -- Copyright (c) 2025-2026 Daniel Mo Houshmand / QDaria AS

---

## Acknowledgments

Hardware specifications sourced from:
- [Rigetti Computing](https://www.rigetti.com/novera) -- Novera QPU product page
- [Rigetti Ankaa-3 Press Release](https://www.rigetti.com/newsroom/press-releases/rigetti-announces-ankaa-3) -- 84Q median fidelities
- [Zurich Instruments](https://www.zhinst.com/) -- March 2024 Novera benchmarks
- [Azure Quantum](https://learn.microsoft.com/en-us/azure/quantum/provider-rigetti) -- Cepheus-1 listing
- [AWS Braket](https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices-rigetti.html) -- Gate times
- [SQMS Center (Fermilab)](https://sqms.fnal.gov/) -- Improved coherence times
