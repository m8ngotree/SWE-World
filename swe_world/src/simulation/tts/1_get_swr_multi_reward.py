#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from r2egym.agenthub.agent.simulator import SimulatorAgent
from r2egym.agenthub.utils.log import get_logger


# =========================
# 全局参数（由 argparse 注入）
# =========================
NUM_WORKERS = 16
SKIP_ALREADY_DONE = True

INPUT_JSON_PATH = None
OUTPUT_JSONL_PATH = None

SIMULATOR_CONFIG_PATH = None
SIMULATOR_CONFIG: List[Dict[str, Any]] = []
TOKENIZER_PATH = None

SLEEP_BETWEEN_CALLS_SEC = 0.0
NUM_SCORES_DEFAULT = 3


# =========================
# IO：兼容 JSON array / JSONL
# =========================
def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file is not a list")
        else:
            data = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))

    # 兼容：real_reward -> eval_reward（如存在）
    for item in data:
        if "real_reward" in item and "eval_reward" not in item:
            item["eval_reward"] = item["real_reward"]

    return data


# =========================
# 断点续跑：基于 instance_id
# =========================
def get_instance_id(item: Dict[str, Any]) -> str:
    if "instance_id" in item and item["instance_id"] is not None:
        return str(item["instance_id"])
    ds = item.get("ds")
    if isinstance(ds, dict) and ds.get("instance_id") is not None:
        return str(ds["instance_id"])
    meta = item.get("metadata")
    if isinstance(meta, dict) and meta.get("instance_id") is not None:
        return str(meta["instance_id"])
    raise KeyError(
        "Cannot find instance_id in item (tried item['instance_id'], item['ds']['instance_id'], item['metadata']['instance_id'])"
    )


def load_done_instance_ids(output_jsonl_path: str) -> set:
    done_ids = set()
    if not os.path.exists(output_jsonl_path):
        return done_ids

    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                done_ids.add(get_instance_id(obj))
            except Exception:
                continue
    return done_ids


# =========================
# 每线程一个 agent
# =========================
_thread_local = threading.local()

def get_thread_agent() -> SimulatorAgent:
    agent = getattr(_thread_local, "agent", None)
    if agent is None:
        agent = SimulatorAgent(
            simulator_config=SIMULATOR_CONFIG,
            simulator_config_path=SIMULATOR_CONFIG_PATH,
            tokenizer_path=TOKENIZER_PATH,
            logger=get_logger("SimulatorAgent"),
        )
        _thread_local.agent = agent
    return agent


# =========================
# 过滤无效模拟
# =========================
def is_bad_simulation(sim_response_content: Optional[str]) -> bool:
    if not sim_response_content:
        return True
    if ("ERROR11111111" in sim_response_content) or ("Combined test-report & reward LLM failed" in sim_response_content):
        return True
    return False


# =========================
# 主流程
# =========================
def main():
    logger = get_logger("sim_reward_multi_score")
    logger.info(f"Input Path: {INPUT_JSON_PATH}")
    logger.info(f"Output Path: {OUTPUT_JSONL_PATH}")
    logger.info(f"Config Path: {SIMULATOR_CONFIG_PATH}")
    logger.info(f"Tokenizer Path: {TOKENIZER_PATH}")
    logger.info(f"NUM_SCORES_DEFAULT: {NUM_SCORES_DEFAULT}")

    data = load_json_or_jsonl(INPUT_JSON_PATH)
    logger.info(f"Loaded {len(data)} records from: {INPUT_JSON_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_JSONL_PATH), exist_ok=True)

    done_ids = set()
    if SKIP_ALREADY_DONE:
        done_ids = load_done_instance_ids(OUTPUT_JSONL_PATH)
        logger.info(f"[checkpoint] Found {len(done_ids)} done instance_ids in: {OUTPUT_JSONL_PATH}")

    # 待处理任务
    tasks: List[Tuple[int, Dict[str, Any]]] = []
    for i, item in enumerate(data):
        iid = get_instance_id(item)
        if SKIP_ALREADY_DONE and iid in done_ids:
            continue
        tasks.append((i, item))

    logger.info(f"Need to process {len(tasks)} records (skipped {len(data) - len(tasks)} already done).")

    write_lock = threading.Lock()
    err_lock = threading.Lock()

    # 统计
    context_failed = 0     # 该 context 所有 runs 都无效 or worker异常
    total_bad_runs = 0     # 所有样本累计被过滤的 run 数
    total_valid_runs = 0   # 所有样本累计有效 run 数

    with open(OUTPUT_JSONL_PATH, "a", encoding="utf-8") as wf:

        def worker(i: int, item: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]], int, int, int]:
            """
            返回：
              - i
              - updated_item（成功则可写入，否则 None）
              - ctx_fail_flag（1=失败，不写入；0=成功写入）
              - bad_runs（该条样本过滤的 run 数）
              - valid_runs（该条样本有效 run 数）
            """
            context = item.get("context")
            if context is None:
                raise KeyError(f"Record {i} missing key: context")

            # 每条样本可覆盖次数
            num_scores = item.get("num_scores", NUM_SCORES_DEFAULT)
            try:
                num_scores = int(num_scores)
            except Exception:
                num_scores = NUM_SCORES_DEFAULT
            if num_scores <= 0:
                num_scores = NUM_SCORES_DEFAULT

            agent = get_thread_agent()

            valid_runs: List[Dict[str, Any]] = []
            bad_runs = 0

            for _ in range(num_scores):
                try:
                    sim_test_report, sim_reward, sim_response_content = agent.get_simulated_test_report_and_reward(context)
                except Exception as e:
                    sim_test_report = ""
                    sim_reward = 0
                    sim_response_content = f"ERROR11111111: {type(e).__name__}: {e}"

                if is_bad_simulation(sim_response_content):
                    bad_runs += 1
                    continue

                # reward 强制转 float（你后面要 avg）
                try:
                    sim_reward_f = float(sim_reward)
                except Exception:
                    bad_runs += 1
                    continue

                valid_runs.append({
                    "simulated_reward": sim_reward_f,
                    "sim_test_report": sim_test_report,
                    "sim_response_content": sim_response_content,
                })

                if SLEEP_BETWEEN_CALLS_SEC > 0:
                    time.sleep(SLEEP_BETWEEN_CALLS_SEC)

            if len(valid_runs) == 0:
                # 全部无效：不写入
                return i, None, 1, bad_runs, 0

            rewards = [r["simulated_reward"] for r in valid_runs]
            avg_score = sum(rewards) / float(len(rewards))

            # 写回 item
            item["num_scores"] = num_scores
            item["simulated_reward_runs"] = valid_runs
            item["simulated_reward_avg"] = avg_score
            item["sim_valid_runs"] = len(valid_runs)
            item["sim_bad_runs"] = bad_runs
            item["sim_total_runs"] = num_scores

            ）
            item["simulated_reward_last"] = valid_runs[-1]["simulated_reward"]
            item["sim_test_report_last"] = valid_runs[-1]["sim_test_report"]
            item["sim_response_content_last"] = valid_runs[-1]["sim_response_content"]

            return i, item, 0, bad_runs, len(valid_runs)

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = [ex.submit(worker, i, item) for (i, item) in tasks]

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Simulating reward (multi-score)"):
                try:
                    i, updated_item, ctx_fail_flag, bad_runs, valid_runs = fut.result()
                except Exception as e:
                    with err_lock:
                        context_failed += 1
                    print(f"[worker exception] {type(e).__name__}: {e}")
                    continue

                with err_lock:
                    total_bad_runs += int(bad_runs)
                    total_valid_runs += int(valid_runs)

                if ctx_fail_flag == 1 or updated_item is None:
                    with err_lock:
                        context_failed += 1
                    continue

                with write_lock:
                    wf.write(json.dumps(updated_item, ensure_ascii=False) + "\n")
                    wf.flush()

    logger.info("Done.")
    logger.info(f"context_failed (no valid run / exception): {context_failed}")
    logger.info(f"total_valid_runs: {total_valid_runs}")
    logger.info(f"total_bad_runs(filtered): {total_bad_runs}")
    logger.info(f"Saved JSONL to: {OUTPUT_JSONL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulator Reward Runner (multi-score, no binarize, no metrics)")

    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON/JSONL file")
    parser.add_argument("--config_path", type=str, required=True, help="Path to simulator config YAML")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer directory")

    parser.add_argument("--output_path", type=str, default=None, help="Path to output JSONL file")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of concurrent workers")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep time between calls in seconds")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite/reprocess all (disable resume)")

    # ✅ 多次打分次数（默认 3），也可被 item['num_scores'] 覆盖
    parser.add_argument("--num_scores", type=int, default=3, help="Number of simulated reward scorings per context (default: 3)")

    args = parser.parse_args()

    INPUT_JSON_PATH = args.input_path
    SIMULATOR_CONFIG_PATH = args.config_path
    TOKENIZER_PATH = args.tokenizer_path

    NUM_WORKERS = args.num_workers
    SLEEP_BETWEEN_CALLS_SEC = args.sleep
    SKIP_ALREADY_DONE = not args.overwrite
    NUM_SCORES_DEFAULT = args.num_scores

    if args.output_path:
        OUTPUT_JSONL_PATH = args.output_path
    else:
        base_name, _ = os.path.splitext(INPUT_JSON_PATH)
        OUTPUT_JSONL_PATH = f"{base_name}_sim_reward_multi_score.jsonl"

    main()