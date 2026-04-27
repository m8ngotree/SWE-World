"""
crawl_js_repos.py — Data collection script for JavaScript / TypeScript repos.

This script is the JS/TS counterpart of crawl_repos.py (Python).  It:

  1. Downloads Multi-SWE-bench JS/TS instances from HuggingFace.
  2. For each instance, runs the multi-swe-bench Docker harness to collect
     real execution traces: (command, stdout, stderr, exit_code).
  3. Outputs JSONL in the same format as the Python training data so the
     existing SFT pipeline in OpenRLHF_SFT/ can consume it without changes.

Prerequisites
-------------
  pip install datasets huggingface_hub tqdm
  Docker must be running and the multi-swe-bench image must be available.
  Clone https://github.com/multi-swe-bench/multi-swe-bench alongside this
  repo and set MULTI_SWE_BENCH_DIR below (or via env var).

Usage
-----
  python crawl_js_repos.py \
      --output_dir /path/to/output \
      --languages javascript typescript \
      --max_instances 500 \
      --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Try importing HuggingFace datasets; give a clear error if missing.
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset
except ImportError:
    print(
        "ERROR: 'datasets' package not found.\n"
        "Install it with:  pip install datasets huggingface_hub",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MULTI_SWE_BENCH_DATASET = "ByteDance-Seed/Multi-SWE-bench"
MULTI_SWE_BENCH_DIR = os.environ.get(
    "MULTI_SWE_BENCH_DIR",
    str(Path(__file__).parent.parent / "multi-swe-bench"),
)

SUPPORTED_LANGUAGES = {"javascript", "typescript", "js", "ts"}

# Map language string → Language enum name (used in output metadata)
LANG_NAME_MAP = {
    "javascript": "JAVASCRIPT",
    "js": "JAVASCRIPT",
    "typescript": "TYPESCRIPT",
    "ts": "TYPESCRIPT",
}

# ---------------------------------------------------------------------------
# Helper: load Multi-SWE-bench instances
# ---------------------------------------------------------------------------

def load_instances(languages: List[str], max_instances: Optional[int] = None) -> List[Dict]:
    """
    Download and filter Multi-SWE-bench instances for the requested languages.

    Returns a list of raw HuggingFace dataset rows as plain dicts.
    """
    logger.info(f"Loading Multi-SWE-bench dataset ({MULTI_SWE_BENCH_DATASET}) …")
    ds = load_dataset(MULTI_SWE_BENCH_DATASET, split="test")

    lang_set = {l.lower() for l in languages}
    # Normalise: js → javascript, ts → typescript
    lang_set = {LANG_NAME_MAP.get(l, l) for l in lang_set}

    filtered = [
        row for row in ds
        if str(row.get("language", "")).upper() in lang_set
    ]

    logger.info(f"Found {len(filtered)} instances for languages: {lang_set}")

    if max_instances and len(filtered) > max_instances:
        filtered = filtered[:max_instances]
        logger.info(f"Truncated to {max_instances} instances.")

    return [dict(row) for row in filtered]


# ---------------------------------------------------------------------------
# Helper: run Docker harness for one instance
# ---------------------------------------------------------------------------

def run_docker_harness(instance: Dict, output_dir: Path, timeout: int = 600) -> Optional[Dict]:
    """
    Run the multi-swe-bench Docker evaluation harness for a single instance.

    Returns a dict with real execution traces, or None on failure.
    """
    instance_id = instance.get("instance_id", "unknown")
    instance_output_dir = output_dir / instance_id
    instance_output_dir.mkdir(parents=True, exist_ok=True)

    # Write instance to a temp JSON file for the harness
    instance_json = instance_output_dir / "instance.json"
    with open(instance_json, "w") as f:
        json.dump([instance], f)

    harness_script = Path(MULTI_SWE_BENCH_DIR) / "harness" / "run_evaluation.py"
    if not harness_script.exists():
        logger.error(
            f"multi-swe-bench harness not found at {harness_script}.\n"
            f"Clone https://github.com/multi-swe-bench/multi-swe-bench and set "
            f"MULTI_SWE_BENCH_DIR env var."
        )
        return None

    cmd = [
        sys.executable, str(harness_script),
        "--instances_path", str(instance_json),
        "--output_dir", str(instance_output_dir),
        "--collect_traces",  # flag that tells harness to dump execution traces
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"[{instance_id}] Docker harness timed out after {timeout}s")
        return None
    except Exception as e:
        logger.error(f"[{instance_id}] Harness subprocess error: {e}")
        return None

    if result.returncode != 0:
        logger.warning(
            f"[{instance_id}] Harness exited with code {result.returncode}.\n"
            f"STDERR: {result.stderr[:500]}"
        )

    # Read traces written by the harness
    traces_file = instance_output_dir / "traces.jsonl"
    if not traces_file.exists():
        logger.warning(f"[{instance_id}] No traces.jsonl produced by harness.")
        return None

    traces = []
    with open(traces_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not traces:
        logger.warning(f"[{instance_id}] traces.jsonl is empty.")
        return None

    return {
        "instance_id": instance_id,
        "repo": instance.get("repo", ""),
        "language": instance.get("language", ""),
        "traces": traces,
        "harness_exit_code": result.returncode,
    }


# ---------------------------------------------------------------------------
# Helper: convert harness traces → SWT/SWR training format
# ---------------------------------------------------------------------------

def traces_to_training_samples(
    instance: Dict,
    trace_data: Dict,
) -> List[Dict]:
    """
    Convert raw harness execution traces into the JSONL training format
    expected by the existing SFT pipeline.

    Each trace becomes one sample with:
      - context  (same keys as Python SWT data)
      - real_output
      - real_error_code
      - instance_id
    """
    samples = []
    instance_id = instance.get("instance_id", "")
    repo = instance.get("repo", "")
    language = instance.get("language", "")

    problem_statement = instance.get("problem_statement", "")
    gold_patch = instance.get("patch", "")
    test_patch = instance.get("test_patch", "")
    f2p = instance.get("FAIL_TO_PASS", [])
    p2p = instance.get("PASS_TO_PASS", [])

    for trace in trace_data.get("traces", []):
        cmd = trace.get("command", "")
        stdout = trace.get("stdout", "")
        stderr = trace.get("stderr", "")
        exit_code = trace.get("exit_code", -1)
        trace_type = trace.get("type", "step_execution")  # "step_execution" | "reward_calculation"

        real_output = f"[STDOUT]\n\n{stdout}\n\n[STDERR]\n\n{stderr}"

        context: Dict[str, Any] = {
            "type": trace_type,
            "repo": repo,
            "language": language,
            "problem_statement": problem_statement,
            "command_to_simulate": cmd,
            "agent_patch": trace.get("agent_patch", ""),
            "gold_patch": gold_patch,
            "test_patch": test_patch,
            "execution_code_content": trace.get("execution_code_content", {}),
            "original_files_content": trace.get("original_files_content", {}),
            "initial_analysis": trace.get("initial_analysis", ""),
            "FAIL_TO_PASS": f2p,
            "PASS_TO_PASS": p2p,
        }

        sample: Dict[str, Any] = {
            "instance_id": instance_id,
            "real_output": real_output,
            "real_error_code": exit_code,
            "real_reward": trace.get("reward", None),
            "context": context,
        }

        samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect JS/TS SWT/SWR training data from Multi-SWE-bench."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./js_ts_traces",
        help="Directory to write output JSONL files.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["javascript", "typescript"],
        choices=["javascript", "typescript", "js", "ts"],
        help="Languages to collect data for.",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Maximum number of instances to process (useful for quick runs).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel Docker workers.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-instance Docker timeout in seconds.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip instances whose output file already exists.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help=(
            "If set, merge all samples into this single JSONL file "
            "(in addition to per-instance files)."
        ),
    )
    return parser.parse_args()


def process_instance(
    instance: Dict,
    output_dir: Path,
    timeout: int,
    skip_existing: bool,
) -> Tuple[str, int]:
    """
    Process one instance: run harness, convert traces, write per-instance JSONL.

    Returns (instance_id, num_samples_written).
    """
    instance_id = instance.get("instance_id", "unknown")
    out_file = output_dir / f"{instance_id.replace('/', '__')}.jsonl"

    if skip_existing and out_file.exists():
        logger.debug(f"[{instance_id}] Skipping (already exists).")
        return instance_id, 0

    trace_data = run_docker_harness(instance, output_dir / "harness_runs", timeout=timeout)
    if trace_data is None:
        return instance_id, 0

    samples = traces_to_training_samples(instance, trace_data)
    if not samples:
        return instance_id, 0

    with open(out_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return instance_id, len(samples)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(args.languages, max_instances=args.max_instances)

    if not instances:
        logger.error("No instances found. Check --languages and dataset availability.")
        sys.exit(1)

    logger.info(f"Processing {len(instances)} instances with {args.workers} workers …")

    total_samples = 0
    merged_jsonl_handle = None
    if args.output_jsonl:
        merged_jsonl_handle = open(args.output_jsonl, "w", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_instance,
                inst,
                output_dir,
                args.timeout,
                args.skip_existing,
            ): inst.get("instance_id", "?")
            for inst in instances
        }

        with tqdm(total=len(futures), desc="Collecting traces") as pbar:
            for future in as_completed(futures):
                instance_id = futures[future]
                try:
                    iid, n = future.result()
                    total_samples += n
                    pbar.set_postfix({"samples": total_samples})

                    # Append to merged file if requested
                    if merged_jsonl_handle and n > 0:
                        per_file = output_dir / f"{iid.replace('/', '__')}.jsonl"
                        if per_file.exists():
                            with open(per_file) as pf:
                                for line in pf:
                                    merged_jsonl_handle.write(line)
                except Exception as exc:
                    logger.error(f"[{instance_id}] Unhandled exception: {exc}")
                finally:
                    pbar.update(1)

    if merged_jsonl_handle:
        merged_jsonl_handle.close()
        logger.info(f"Merged output written to: {args.output_jsonl}")

    logger.info(
        f"Done. Processed {len(instances)} instances, "
        f"collected {total_samples} training samples."
    )
    logger.info(f"Per-instance JSONL files: {output_dir}")


if __name__ == "__main__":
    main()
