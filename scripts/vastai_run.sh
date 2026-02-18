#!/usr/bin/env bash
# =============================================================================
# vastai_run.sh - QRC Depth Sweep Experiment for Vast.ai GPU Instances
# =============================================================================
#
# Runs the QRC depth optimization experiment on both FakeNovera (9Q) and
# FakeCepheus (36Q) backends. Designed to be executed inside a Vast.ai
# container or any machine with a GPU and Python 3.9+.
#
# Usage:
#   bash vastai_run.sh                          # Full sweep (all depths)
#   bash vastai_run.sh --novera-depths 3,4,5    # Subset of Novera depths
#   bash vastai_run.sh --cepheus-depths 1,2     # Subset of Cepheus depths
#   bash vastai_run.sh --shots 2000             # Override shot count
#   bash vastai_run.sh --seed 42                # Set random seed
#   bash vastai_run.sh --output /tmp/results    # Custom output dir
#   bash vastai_run.sh --worker-id 2            # Worker ID for filenames
#
# Output:
#   results/novera_depth_sweep.json
#   results/cepheus_depth_sweep.json
#   (or results/novera_depth_sweep_worker_N.json if --worker-id is set)
#
# Author: Daniel Mo Houshmand (QDaria AS)
# =============================================================================

set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
NOVERA_DEPTHS="1,2,3,4,5,6,7,8,9,10"
CEPHEUS_DEPTHS="1,2,3,4,5"
SHOTS=1000
SEED=42
OUTPUT_DIR=""
WORKER_ID=""
SKIP_INSTALL=0

# ---- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --novera-depths)  NOVERA_DEPTHS="$2"; shift 2 ;;
        --cepheus-depths) CEPHEUS_DEPTHS="$2"; shift 2 ;;
        --shots)          SHOTS="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --output)         OUTPUT_DIR="$2"; shift 2 ;;
        --worker-id)      WORKER_ID="$2"; shift 2 ;;
        --skip-install)   SKIP_INSTALL=1; shift ;;
        -h|--help)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---- Locate project root ----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/results"
fi
mkdir -p "$OUTPUT_DIR"

# ---- File suffix for worker --------------------------------------------------
SUFFIX=""
if [[ -n "$WORKER_ID" ]]; then
    SUFFIX="_worker_${WORKER_ID}"
fi

# ---- Install -----------------------------------------------------------------
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    echo "================================================================"
    echo " Installing rigetti-fake-backends with Qiskit support..."
    echo "================================================================"
    pip install -e "$PROJECT_ROOT[qiskit]" 2>&1 | tail -5
    echo ""
fi

# ---- Verify imports ----------------------------------------------------------
echo "Verifying imports..."
python3 -c "
from fakenovera.qiskit import FakeNovera
from fakecepheus.qiskit import FakeCepheus
print('  FakeNovera:', FakeNovera())
print('  FakeCepheus:', FakeCepheus())
print('  Imports OK')
"
echo ""

# ---- Run the depth sweep experiment -----------------------------------------
echo "================================================================"
echo " QRC Depth Sweep Experiment"
echo " Novera depths: $NOVERA_DEPTHS"
echo " Cepheus depths: $CEPHEUS_DEPTHS"
echo " Shots per circuit: $SHOTS"
echo " Seed: $SEED"
echo " Output: $OUTPUT_DIR"
echo "================================================================"
echo ""

python3 - "$NOVERA_DEPTHS" "$CEPHEUS_DEPTHS" "$SHOTS" "$SEED" "$OUTPUT_DIR" "$SUFFIX" <<'PYEOF'
import sys
import json
import time
import numpy as np
from datetime import datetime

novera_depths = [int(d) for d in sys.argv[1].split(",")]
cepheus_depths = [int(d) for d in sys.argv[2].split(",")]
shots = int(sys.argv[3])
seed = int(sys.argv[4])
output_dir = sys.argv[5]
suffix = sys.argv[6]

# ---- Lorenz-63 target generator ---------------------------------------------

def lorenz63(x, y, z, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01):
    """Single Lorenz-63 integration step (Euler)."""
    dx = sigma * (y - x) * dt
    dy = (x * (rho - z) - y) * dt
    dz = (x * y - beta * z) * dt
    return x + dx, y + dy, z + dz

def generate_lorenz_series(n_steps, warmup=500, seed_val=0):
    """Generate Lorenz-63 x-component time series."""
    np.random.seed(seed_val)
    x, y, z = 1.0 + 0.01 * np.random.randn(), 1.0, 1.0
    # Warmup
    for _ in range(warmup):
        x, y, z = lorenz63(x, y, z)
    series = []
    for _ in range(n_steps):
        x, y, z = lorenz63(x, y, z)
        series.append(x)
    arr = np.array(series)
    # Normalize to [0, 1]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
    return arr

# ---- QRC reservoir readout (simplified) --------------------------------------

def build_qrc_circuit(backend, depth, input_val, n_qubits):
    """Build a QRC circuit: encode input -> variational layers -> measure."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Input encoding: RY rotations proportional to input
    for q in range(n_qubits):
        angle = input_val * np.pi * (1.0 + 0.1 * q)
        qc.ry(angle, q)

    # Variational layers (entangling + rotations)
    coupling = backend.coupling_map
    # Deduplicate to undirected edges
    edges_seen = set()
    edges = []
    for pair in coupling:
        a, b = pair[0], pair[1]
        key = (min(a, b), max(a, b))
        if key not in edges_seen:
            edges_seen.add(key)
            edges.append((a, b))

    for layer in range(depth):
        # Entangling: CZ on all edges
        for a, b in edges:
            qc.cz(a, b)
        # Single-qubit rotations
        for q in range(n_qubits):
            angle = (layer + 1) * 0.3 + input_val * 0.5
            qc.rx(angle, q)
            qc.rz(angle * 0.7, q)

    # Measure all
    qc.measure(range(n_qubits), range(n_qubits))
    return qc

def extract_features(counts, n_qubits, shots):
    """Extract expectation values from measurement counts."""
    features = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        # Qiskit returns bitstrings in reverse order
        bits = bitstring[::-1]
        for q in range(min(len(bits), n_qubits)):
            if bits[q] == '1':
                features[q] += count
    features /= shots
    return features

def run_qrc_experiment(backend, depths, shots, seed_val, backend_name):
    """Run QRC depth sweep for one backend."""
    n_qubits = backend.num_qubits
    n_train = 80
    n_test = 20
    n_total = n_train + n_test + 1  # +1 for target offset

    # Generate Lorenz time series
    lorenz = generate_lorenz_series(n_total + 100, seed_val=seed_val)

    results = []

    for depth in depths:
        t_start = time.time()
        f_cum = backend.cumulative_fidelity(depth)

        print(f"  [{backend_name}] depth={depth}, F_cum={f_cum:.4f} ... ", end="", flush=True)

        # Collect reservoir states
        all_features = []
        for t in range(n_total):
            input_val = lorenz[t]
            qc = build_qrc_circuit(backend, depth, input_val, n_qubits)
            job = backend.run(qc, shots=shots)
            counts = job.result().get_counts()
            feat = extract_features(counts, n_qubits, shots)
            all_features.append(feat)

        X = np.array(all_features[:-1])  # inputs: t=0..N-2
        y = lorenz[1:n_total]             # targets: t=1..N-1

        # Train/test split
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Ridge regression
        alpha = 1e-4
        XtX = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
        Xty = X_train.T @ y_train
        w = np.linalg.solve(XtX, Xty)

        y_pred = X_test @ w
        # R-squared
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)

        elapsed = time.time() - t_start

        print(f"R2={r2:.4f}, time={elapsed:.1f}s")

        results.append({
            "depth": depth,
            "r2": round(float(r2), 6),
            "cumulative_fidelity": round(float(f_cum), 6),
            "n_qubits": n_qubits,
            "shots": shots,
            "n_train": n_train,
            "n_test": n_test,
            "elapsed_seconds": round(elapsed, 2),
        })

    return results

# ==============================================================================
# MAIN
# ==============================================================================

print("=" * 70)
print(f" QRC Depth Sweep - {datetime.now().isoformat()}")
print("=" * 70)

# ---- Novera (9Q) ----
print(f"\n--- FakeNovera 9Q (depths {novera_depths}) ---")
from fakenovera.qiskit import FakeNovera
from fakenovera.specs import NOVERA_SPECS

novera_backend = FakeNovera(seed=seed)
novera_results = run_qrc_experiment(
    novera_backend, novera_depths, shots, seed, "Novera-9Q"
)

novera_output = {
    "backend": "FakeNovera",
    "num_qubits": 9,
    "specs": NOVERA_SPECS.name,
    "f_2q": NOVERA_SPECS.fidelity_2q,
    "optimal_depth_70pct": round(float(novera_backend.optimal_depth(0.70))),
    "shots": shots,
    "seed": seed,
    "timestamp": datetime.now().isoformat(),
    "results": novera_results,
}

novera_file = f"{output_dir}/novera_depth_sweep{suffix}.json"
with open(novera_file, "w") as f:
    json.dump(novera_output, f, indent=2)
print(f"  Saved: {novera_file}")

# ---- Cepheus (36Q) ----
print(f"\n--- FakeCepheus 36Q (depths {cepheus_depths}) ---")
from fakecepheus.qiskit import FakeCepheus
from fakecepheus.specs import CEPHEUS_SPECS

cepheus_backend = FakeCepheus(seed=seed)
cepheus_results = run_qrc_experiment(
    cepheus_backend, cepheus_depths, shots, seed, "Cepheus-36Q"
)

cepheus_output = {
    "backend": "FakeCepheus",
    "num_qubits": 36,
    "specs": CEPHEUS_SPECS.name,
    "f_2q_intra": CEPHEUS_SPECS.fidelity_2q_intra,
    "f_2q_inter": CEPHEUS_SPECS.fidelity_2q_inter,
    "optimal_depth_70pct": round(float(cepheus_backend.optimal_depth(0.70))),
    "shots": shots,
    "seed": seed,
    "timestamp": datetime.now().isoformat(),
    "results": cepheus_results,
}

cepheus_file = f"{output_dir}/cepheus_depth_sweep{suffix}.json"
with open(cepheus_file, "w") as f:
    json.dump(cepheus_output, f, indent=2)
print(f"  Saved: {cepheus_file}")

# ---- Summary Table -----------------------------------------------------------
print("\n" + "=" * 70)
print(" SUMMARY")
print("=" * 70)

print(f"\n{'Backend':<14} {'Depth':<7} {'F_cum':<10} {'R2':<10} {'Time (s)'}")
print("-" * 55)
for r in novera_results:
    print(f"{'Novera-9Q':<14} {r['depth']:<7} {r['cumulative_fidelity']:<10.4f} "
          f"{r['r2']:<10.4f} {r['elapsed_seconds']:.1f}")
print("-" * 55)
for r in cepheus_results:
    print(f"{'Cepheus-36Q':<14} {r['depth']:<7} {r['cumulative_fidelity']:<10.4f} "
          f"{r['r2']:<10.4f} {r['elapsed_seconds']:.1f}")
print("-" * 55)

# Find best depth per backend
best_novera = max(novera_results, key=lambda x: x["r2"])
best_cepheus = max(cepheus_results, key=lambda x: x["r2"])
print(f"\nBest Novera:  depth={best_novera['depth']}, R2={best_novera['r2']:.4f}, "
      f"F_cum={best_novera['cumulative_fidelity']:.4f}")
print(f"Best Cepheus: depth={best_cepheus['depth']}, R2={best_cepheus['r2']:.4f}, "
      f"F_cum={best_cepheus['cumulative_fidelity']:.4f}")

# Paper predictions
d_opt_novera = round(float(novera_backend.optimal_depth(0.70)))
d_opt_cepheus = round(float(cepheus_backend.optimal_depth(0.70)))
print(f"\nPaper predicted optimal depths (70% fidelity threshold):")
print(f"  Novera:  d_opt = {d_opt_novera}")
print(f"  Cepheus: d_opt = {d_opt_cepheus}")
print(f"\nSimulation best depths:")
print(f"  Novera:  d_best = {best_novera['depth']}  "
      f"({'MATCHES' if best_novera['depth'] == d_opt_novera else 'DIFFERS by ' + str(abs(best_novera['depth'] - d_opt_novera))})")
print(f"  Cepheus: d_best = {best_cepheus['depth']}  "
      f"({'MATCHES' if best_cepheus['depth'] == d_opt_cepheus else 'DIFFERS by ' + str(abs(best_cepheus['depth'] - d_opt_cepheus))})")

print("\n" + "=" * 70)
print(" DONE")
print("=" * 70)
PYEOF

echo ""
echo "Results saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*depth_sweep*.json 2>/dev/null || true
