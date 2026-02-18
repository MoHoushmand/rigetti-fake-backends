#!/usr/bin/env python3
"""
vastai_launch.py - Launch Parallel QRC Depth Sweep on Vast.ai GPU Instances
===========================================================================

Searches for cheap GPU instances on Vast.ai, launches N parallel workers,
each running a different depth range of the QRC experiment, and collects
the results when all workers finish.

Setup:
  1. pip install vastai
  2. vastai set api-key YOUR_KEY
  3. python scripts/vastai_launch.py --num-workers 4

Usage:
  python scripts/vastai_launch.py                           # 4 workers, $0.30/hr max
  python scripts/vastai_launch.py --num-workers 8           # 8 workers
  python scripts/vastai_launch.py --max-price 0.15          # Cheaper instances
  python scripts/vastai_launch.py --gpu-name "RTX 3090"     # Specific GPU
  python scripts/vastai_launch.py --dry-run                 # Show plan, don't launch
  python scripts/vastai_launch.py --collect-only 12345,12346  # Just collect results

The script will:
  1. Search for available GPU instances matching criteria
  2. Split the depth ranges across N workers:
       - Novera depths 1..10 and Cepheus depths 1..5 are divided evenly
  3. Launch each worker with vastai_run.sh and the assigned depth subset
  4. Poll for completion every 60 seconds
  5. Download and merge results into a single JSON per backend
  6. Destroy instances when done

Requirements:
  - vastai CLI installed and API key configured
  - This script must be run from the project root or scripts/ directory

Author: Daniel Mo Houshmand (QDaria AS)
License: MIT
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default depth ranges from the paper
NOVERA_DEPTHS = list(range(1, 11))   # d = 1..10
CEPHEUS_DEPTHS = list(range(1, 6))   # d = 1..5

# Docker image with Python 3.11 + CUDA
DEFAULT_IMAGE = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"

# Vast.ai search criteria
DEFAULT_GPU = "RTX_3090"
DEFAULT_MAX_PRICE = 0.30  # $/hr
DEFAULT_DISK_GB = 16
DEFAULT_MIN_RAM = 8  # GB

# Repo URL for cloning on remote instances
REPO_URL = "https://github.com/MoHoushmand/fakenovera.git"


# =============================================================================
# HELPERS
# =============================================================================

def run_cmd(cmd: str, check: bool = True, capture: bool = True) -> str:
    """Run a shell command and return stdout."""
    result = subprocess.run(
        cmd, shell=True, capture_output=capture, text=True
    )
    if check and result.returncode != 0:
        stderr = result.stderr if capture else ""
        raise RuntimeError(f"Command failed: {cmd}\n{stderr}")
    return result.stdout.strip() if capture else ""


def check_vastai_cli():
    """Verify vastai CLI is installed and configured."""
    if not shutil.which("vastai"):
        print("ERROR: vastai CLI not found.")
        print("Install it with: pip install vastai")
        print("Then configure: vastai set api-key YOUR_KEY")
        sys.exit(1)

    # Check API key is set
    try:
        output = run_cmd("vastai show user", check=False)
        if "error" in output.lower() or "unauthorized" in output.lower():
            print("ERROR: Vast.ai API key not configured or invalid.")
            print("Run: vastai set api-key YOUR_KEY")
            sys.exit(1)
    except Exception:
        pass  # Some versions don't have 'show user'


def split_depths(
    depths: List[int], n_workers: int
) -> List[List[int]]:
    """Split a list of depths evenly across workers."""
    if n_workers >= len(depths):
        # One depth per worker (some workers get nothing)
        return [[d] for d in depths] + [[] for _ in range(n_workers - len(depths))]

    chunk_size = math.ceil(len(depths) / n_workers)
    chunks = []
    for i in range(0, len(depths), chunk_size):
        chunks.append(depths[i : i + chunk_size])
    # Pad if needed
    while len(chunks) < n_workers:
        chunks.append([])
    return chunks


def format_depth_list(depths: List[int]) -> str:
    """Format depth list as comma-separated string."""
    return ",".join(str(d) for d in depths)


# =============================================================================
# VAST.AI OPERATIONS
# =============================================================================

def search_instances(
    gpu_name: str = DEFAULT_GPU,
    max_price: float = DEFAULT_MAX_PRICE,
    min_ram: int = DEFAULT_MIN_RAM,
    disk_gb: int = DEFAULT_DISK_GB,
    num_needed: int = 4,
) -> List[Dict]:
    """Search for available GPU instances on Vast.ai."""
    print(f"Searching for {num_needed} x {gpu_name} instances (<= ${max_price}/hr)...")

    # Build search query
    query = (
        f"vastai search offers "
        f"'gpu_name={gpu_name} "
        f"num_gpus=1 "
        f"inet_down>=100 "
        f"disk_space>={disk_gb} "
        f"cpu_ram>={min_ram * 1024} "  # MB
        f"dph<={max_price} "
        f"reliability>=0.95' "
        f"--order 'dph' "
        f"--limit {num_needed * 3} "
        f"--raw"
    )

    try:
        output = run_cmd(query)
        if not output:
            return []
        offers = json.loads(output)
    except (json.JSONDecodeError, RuntimeError) as e:
        print(f"Warning: Could not parse search results: {e}")
        # Fallback: try without --raw
        query_plain = (
            f"vastai search offers "
            f"'gpu_name={gpu_name} "
            f"num_gpus=1 "
            f"dph<={max_price} "
            f"reliability>=0.95' "
            f"--order 'dph' "
            f"--limit {num_needed * 3}"
        )
        output = run_cmd(query_plain, check=False)
        print(f"Plain search output:\n{output}")
        return []

    if not offers:
        print(f"No offers found for {gpu_name} at <= ${max_price}/hr")
        return []

    print(f"Found {len(offers)} matching offers")
    return offers[:num_needed]


def create_instance(
    offer_id: int,
    image: str = DEFAULT_IMAGE,
    disk_gb: int = DEFAULT_DISK_GB,
    onstart_cmd: str = "",
) -> Optional[int]:
    """Create (rent) a Vast.ai instance and return its ID."""
    cmd = (
        f"vastai create instance {offer_id} "
        f"--image {image} "
        f"--disk {disk_gb} "
        f"--raw"
    )
    if onstart_cmd:
        cmd += f' --onstart-cmd "{onstart_cmd}"'

    try:
        output = run_cmd(cmd)
        result = json.loads(output)
        instance_id = result.get("new_contract")
        if instance_id:
            print(f"  Created instance {instance_id} from offer {offer_id}")
            return instance_id
    except (json.JSONDecodeError, RuntimeError) as e:
        print(f"  Failed to create instance from offer {offer_id}: {e}")

    return None


def wait_for_instances(
    instance_ids: List[int],
    timeout: int = 600,
    poll_interval: int = 15,
) -> Dict[int, str]:
    """Wait for instances to be running and return their SSH info."""
    print(f"Waiting for {len(instance_ids)} instances to start (timeout={timeout}s)...")
    start = time.time()
    ssh_info = {}

    while time.time() - start < timeout:
        remaining = [iid for iid in instance_ids if iid not in ssh_info]
        if not remaining:
            break

        for iid in remaining:
            try:
                output = run_cmd(f"vastai show instance {iid} --raw", check=False)
                if output:
                    info = json.loads(output)
                    status = info.get("actual_status", "")
                    if status == "running":
                        ssh_host = info.get("ssh_host", "")
                        ssh_port = info.get("ssh_port", "")
                        if ssh_host and ssh_port:
                            ssh_info[iid] = f"ssh -p {ssh_port} root@{ssh_host}"
                            print(f"  Instance {iid}: RUNNING ({ssh_host}:{ssh_port})")
            except (json.JSONDecodeError, RuntimeError):
                pass

        if len(ssh_info) < len(instance_ids):
            elapsed = int(time.time() - start)
            print(f"  {len(ssh_info)}/{len(instance_ids)} running ({elapsed}s elapsed)...")
            time.sleep(poll_interval)

    if len(ssh_info) < len(instance_ids):
        missing = [iid for iid in instance_ids if iid not in ssh_info]
        print(f"WARNING: {len(missing)} instances did not start: {missing}")

    return ssh_info


def execute_on_instance(
    instance_id: int,
    command: str,
) -> str:
    """Execute a command on a running Vast.ai instance."""
    cmd = f"vastai execute {instance_id} '{command}'"
    return run_cmd(cmd, check=False)


def build_worker_script(
    worker_id: int,
    novera_depths: List[int],
    cepheus_depths: List[int],
    shots: int,
    seed: int,
) -> str:
    """Build the setup + run command for a single worker."""
    novera_str = format_depth_list(novera_depths)
    cepheus_str = format_depth_list(cepheus_depths)

    # The command sequence to run on the remote instance
    script = (
        f"cd /workspace && "
        f"git clone {REPO_URL} qrc-depth 2>/dev/null || "
        f"(cd qrc-depth && git pull) && "
        f"cd qrc-depth && "
        f"pip install -e '.[qiskit]' && "
        f"bash scripts/vastai_run.sh "
        f"--novera-depths {novera_str} "
        f"--cepheus-depths {cepheus_str} "
        f"--shots {shots} "
        f"--seed {seed} "
        f"--worker-id {worker_id} "
        f"--skip-install "
        f"--output /workspace/results"
    )
    return script


def download_results(
    instance_id: int,
    worker_id: int,
    local_dir: str,
) -> List[str]:
    """Download result JSON files from a worker instance."""
    files_downloaded = []

    for prefix in ["novera", "cepheus"]:
        remote_path = f"/workspace/results/{prefix}_depth_sweep_worker_{worker_id}.json"
        local_path = os.path.join(local_dir, f"{prefix}_depth_sweep_worker_{worker_id}.json")

        cmd = f"vastai copy {instance_id}:{remote_path} {local_path}"
        try:
            run_cmd(cmd, check=False)
            if os.path.exists(local_path):
                files_downloaded.append(local_path)
                print(f"  Downloaded: {local_path}")
        except RuntimeError:
            print(f"  Failed to download {remote_path}")

    return files_downloaded


def destroy_instances(instance_ids: List[int]):
    """Destroy (stop billing) all instances."""
    for iid in instance_ids:
        try:
            run_cmd(f"vastai destroy instance {iid}", check=False)
            print(f"  Destroyed instance {iid}")
        except RuntimeError:
            print(f"  Failed to destroy instance {iid}")


def merge_results(result_dir: str, prefix: str) -> Dict:
    """Merge per-worker result files into a single combined result."""
    all_results = []
    metadata = {}

    pattern = f"{prefix}_depth_sweep_worker_"
    for fname in sorted(os.listdir(result_dir)):
        if fname.startswith(pattern) and fname.endswith(".json"):
            fpath = os.path.join(result_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            if not metadata:
                metadata = {k: v for k, v in data.items() if k != "results"}
            all_results.extend(data.get("results", []))

    if not all_results:
        return {}

    # Sort by depth
    all_results.sort(key=lambda x: x["depth"])

    merged = {**metadata}
    merged["results"] = all_results
    merged["timestamp"] = datetime.now().isoformat()
    merged["merged_from_workers"] = True

    # Save merged file
    merged_path = os.path.join(result_dir, f"{prefix}_depth_sweep_merged.json")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  Merged {len(all_results)} results -> {merged_path}")

    return merged


def poll_for_completion(
    instance_ids: List[int],
    timeout: int = 3600,
    poll_interval: int = 60,
) -> List[int]:
    """Poll instances until their onstart script completes.

    Returns list of instance IDs that completed successfully.
    """
    print(f"\nPolling {len(instance_ids)} workers for completion "
          f"(timeout={timeout}s, poll every {poll_interval}s)...")

    start = time.time()
    completed = set()

    while time.time() - start < timeout:
        for iid in instance_ids:
            if iid in completed:
                continue

            # Check if the results file exists (indicates completion)
            try:
                output = execute_on_instance(
                    iid,
                    "ls /workspace/results/*_depth_sweep_worker_*.json 2>/dev/null | wc -l"
                )
                # We expect 2 files (novera + cepheus) per worker
                count = int(output.strip()) if output.strip().isdigit() else 0
                if count >= 2:
                    completed.add(iid)
                    print(f"  Instance {iid}: COMPLETED ({count} result files)")
            except (RuntimeError, ValueError):
                pass

        if len(completed) == len(instance_ids):
            print(f"All {len(instance_ids)} workers completed!")
            break

        elapsed = int(time.time() - start)
        print(f"  {len(completed)}/{len(instance_ids)} done ({elapsed}s elapsed)...")
        time.sleep(poll_interval)

    if len(completed) < len(instance_ids):
        incomplete = [iid for iid in instance_ids if iid not in completed]
        print(f"WARNING: {len(incomplete)} workers did not complete: {incomplete}")

    return list(completed)


def print_plan(
    workers: List[Dict],
    offers: List[Dict],
    shots: int,
    seed: int,
):
    """Print the execution plan without launching."""
    print("\n" + "=" * 70)
    print(" EXECUTION PLAN (dry run)")
    print("=" * 70)

    for w in workers:
        print(f"\n  Worker {w['id']}:")
        print(f"    Novera depths:  {format_depth_list(w['novera_depths'])}")
        print(f"    Cepheus depths: {format_depth_list(w['cepheus_depths'])}")
        total_circuits = (
            len(w["novera_depths"]) * 101 +   # 80 train + 20 test + 1
            len(w["cepheus_depths"]) * 101
        )
        print(f"    Total circuits: ~{total_circuits}")

    if offers:
        print(f"\n  Available offers ({len(offers)}):")
        for i, offer in enumerate(offers):
            gpu = offer.get("gpu_name", "?")
            price = offer.get("dph_total", offer.get("dph", "?"))
            ram = offer.get("gpu_ram", "?")
            print(f"    [{i}] {gpu} ({ram} GB VRAM) @ ${price}/hr")

    est_time_per_depth = 30  # rough seconds per depth point
    total_depths = len(NOVERA_DEPTHS) + len(CEPHEUS_DEPTHS)
    max_worker_depths = max(
        len(w["novera_depths"]) + len(w["cepheus_depths"]) for w in workers
    )
    est_wall_time = max_worker_depths * est_time_per_depth
    est_cost = (
        est_wall_time / 3600
        * len(workers)
        * (offers[0].get("dph_total", DEFAULT_MAX_PRICE) if offers else DEFAULT_MAX_PRICE)
    )
    print(f"\n  Estimated wall time: ~{est_wall_time // 60} min")
    print(f"  Estimated cost: ~${est_cost:.2f}")
    print(f"  Shots per circuit: {shots}")
    print(f"  Seed: {seed}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Launch parallel QRC depth sweep on Vast.ai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/vastai_launch.py --num-workers 4
  python scripts/vastai_launch.py --num-workers 8 --max-price 0.15
  python scripts/vastai_launch.py --dry-run
  python scripts/vastai_launch.py --collect-only 12345,12346,12347,12348
        """,
    )

    parser.add_argument(
        "--num-workers", "-n", type=int, default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--max-price", type=float, default=DEFAULT_MAX_PRICE,
        help=f"Maximum price per hour in USD (default: {DEFAULT_MAX_PRICE})",
    )
    parser.add_argument(
        "--gpu-name", type=str, default=DEFAULT_GPU,
        help=f"GPU type to search for (default: {DEFAULT_GPU})",
    )
    parser.add_argument(
        "--image", type=str, default=DEFAULT_IMAGE,
        help=f"Docker image (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--shots", type=int, default=1000,
        help="Shots per circuit (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for results (default: ./results)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without launching instances",
    )
    parser.add_argument(
        "--collect-only", type=str, default=None,
        help="Comma-separated instance IDs to collect results from (skip launch)",
    )
    parser.add_argument(
        "--no-destroy", action="store_true",
        help="Don't destroy instances after completion",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Timeout in seconds for waiting for results (default: 3600)",
    )

    args = parser.parse_args()

    # Determine project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    result_dir = args.output or str(project_root / "results")
    os.makedirs(result_dir, exist_ok=True)

    # ---- Split depths across workers ----
    novera_chunks = split_depths(NOVERA_DEPTHS, args.num_workers)
    cepheus_chunks = split_depths(CEPHEUS_DEPTHS, args.num_workers)

    workers = []
    for i in range(args.num_workers):
        workers.append({
            "id": i,
            "novera_depths": novera_chunks[i] if i < len(novera_chunks) else [],
            "cepheus_depths": cepheus_chunks[i] if i < len(cepheus_chunks) else [],
        })

    # Remove workers with no work
    workers = [w for w in workers if w["novera_depths"] or w["cepheus_depths"]]

    if not workers:
        print("ERROR: No work to distribute. Check depth ranges vs worker count.")
        sys.exit(1)

    # ---- Dry run (no CLI needed) ----
    if args.dry_run:
        print_plan(workers, [], args.shots, args.seed)
        return

    # ---- Check CLI ----
    check_vastai_cli()

    # ---- Collect-only mode ----
    if args.collect_only:
        instance_ids = [int(x) for x in args.collect_only.split(",")]
        print(f"Collecting results from instances: {instance_ids}")

        for i, iid in enumerate(instance_ids):
            download_results(iid, i, result_dir)

        # Merge
        for prefix in ["novera", "cepheus"]:
            merge_results(result_dir, prefix)

        if not args.no_destroy:
            destroy_instances(instance_ids)
        return

    # ---- Search for instances ----
    offers = search_instances(
        gpu_name=args.gpu_name,
        max_price=args.max_price,
        num_needed=len(workers),
    )

    if not offers:
        print("No suitable GPU instances found. Try:")
        print(f"  --max-price {args.max_price * 2}")
        print(f"  --gpu-name 'RTX_4090'")
        print(f"  --gpu-name 'RTX_A5000'")
        sys.exit(1)

    if len(offers) < len(workers):
        print(f"WARNING: Only {len(offers)} offers found for {len(workers)} workers.")
        print(f"Reducing to {len(offers)} workers.")
        workers = workers[: len(offers)]

    # ---- Launch instances ----
    print(f"\nLaunching {len(workers)} workers...")
    instance_ids = []

    for i, (worker, offer) in enumerate(zip(workers, offers)):
        offer_id = offer.get("id", offer.get("offer_id"))
        if offer_id is None:
            print(f"  Skipping worker {i}: no offer ID")
            continue

        worker_cmd = build_worker_script(
            worker_id=worker["id"],
            novera_depths=worker["novera_depths"],
            cepheus_depths=worker["cepheus_depths"],
            shots=args.shots,
            seed=args.seed,
        )

        instance_id = create_instance(
            offer_id=offer_id,
            image=args.image,
            onstart_cmd=worker_cmd,
        )
        if instance_id:
            instance_ids.append(instance_id)

    if not instance_ids:
        print("ERROR: Failed to create any instances.")
        sys.exit(1)

    print(f"\nCreated {len(instance_ids)} instances: {instance_ids}")

    # ---- Wait for startup ----
    ssh_info = wait_for_instances(instance_ids, timeout=600)

    # ---- Poll for completion ----
    completed = poll_for_completion(
        instance_ids, timeout=args.timeout
    )

    # ---- Download results ----
    print(f"\nDownloading results from {len(completed)} workers...")
    for i, iid in enumerate(completed):
        download_results(iid, i, result_dir)

    # ---- Merge results ----
    print("\nMerging worker results...")
    for prefix in ["novera", "cepheus"]:
        merged = merge_results(result_dir, prefix)
        if merged:
            results = merged.get("results", [])
            if results:
                best = max(results, key=lambda x: x["r2"])
                print(f"  {prefix}: best depth={best['depth']}, "
                      f"R2={best['r2']:.4f}")

    # ---- Cleanup ----
    if not args.no_destroy:
        print("\nDestroying instances...")
        destroy_instances(instance_ids)
    else:
        print(f"\nInstances left running (--no-destroy): {instance_ids}")
        print("Remember to destroy them manually: vastai destroy instance <id>")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print(" PARALLEL SWEEP COMPLETE")
    print("=" * 70)
    print(f"  Workers used: {len(completed)}/{len(instance_ids)}")
    print(f"  Results dir:  {result_dir}")
    print(f"  Merged files:")
    for prefix in ["novera", "cepheus"]:
        merged_path = os.path.join(result_dir, f"{prefix}_depth_sweep_merged.json")
        if os.path.exists(merged_path):
            print(f"    {merged_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
