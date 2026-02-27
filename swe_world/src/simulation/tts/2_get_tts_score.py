#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

# =========================
# ✅ 你只需要改这里：输入/输出配置
# =========================
INPUT_FILES = [
    # 把你多个输出文件路径写在这里（jsonl）
    "/path/to/run1_sim_reward_multi_score.jsonl",
    "/path/to/run2_sim_reward_multi_score.jsonl",
    "/path/to/run3_sim_reward_multi_score.jsonl",
]

# 选中的最终 submit（每个 instance 一行）
OUTPUT_SELECTED_JSONL = "./tts_selected.jsonl"

# 可选：保存 summary
OUTPUT_SUMMARY_JSON = "./tts_summary.json"

# 并列随机的 seed（保证可复现）
RANDOM_SEED = 2026


# =========================
# 工具函数
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
    raise KeyError("Cannot find instance_id in item")


def get_real_reward(item: Dict[str, Any]) -> Optional[int]:
    # 你说最终 reward 用 real_reward；兼容 eval_reward
    if item.get("real_reward") is not None:
        try:
            return int(item["real_reward"])
        except Exception:
            pass
    if item.get("eval_reward") is not None:
        try:
            return int(item["eval_reward"])
        except Exception:
            pass
    return None


def get_avg_score(item: Dict[str, Any]) -> Optional[float]:
    v = item.get("simulated_reward_avg")
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def get_valid_runs(item: Dict[str, Any]) -> int:
    # 每条候选里有效 run 数（多次打分过滤后保留下来的次数）
    if item.get("sim_valid_runs") is not None:
        try:
            return int(item["sim_valid_runs"])
        except Exception:
            pass
    runs = item.get("simulated_reward_runs")
    if isinstance(runs, list):
        return len(runs)
    return 0


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# =========================
# 主流程
# =========================
def main():
    rng = random.Random(RANDOM_SEED)

    # instance_id -> list of (file_idx, file_path, item)
    bucket: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = defaultdict(list)

    # 读取并聚合
    for fi, path in enumerate(INPUT_FILES):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        data = read_jsonl(path)
        for item in data:
            iid = get_instance_id(item)
            bucket[iid].append((fi, path, item))

    total_instances = 0
    reward_ones = 0

    # 真正 tts@k 的 k：每个 instance 有多少“候选文件”产出了有效模拟（valid_runs>0）
    k_effective_list: List[int] = []

    # 额外：选中候选内部的有效 runs（你可能也想看模型单次打分过滤后的可用数）
    selected_valid_runs_list: List[int] = []

    tie_count = 0
    missing_avg_count = 0
    missing_reward_count = 0

    selected_rows: List[Dict[str, Any]] = []

    for iid, candidates in bucket.items():
        total_instances += 1

        # k_effective：来自不同 output 文件的候选中，有多少个提供了有效轨迹
        k_eff = 0
        scored_candidates = []

        for (fi, path, item) in candidates:
            if get_valid_runs(item) > 0:
                k_eff += 1

            avg = get_avg_score(item)
            if avg is None:
                continue
            scored_candidates.append((avg, fi, path, item))

        k_effective_list.append(k_eff)

        # 选择 best
        if len(scored_candidates) == 0:
            missing_avg_count += 1
            _, path, item = rng.choice(candidates)
            chosen_avg = None
            chosen_reason = "fallback_no_avg"
            chosen_path = path
            chosen_item = item
        else:
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            best_avg = scored_candidates[0][0]
            best_group = [x for x in scored_candidates if x[0] == best_avg]

            if len(best_group) > 1:
                tie_count += 1
                chosen_avg, _, chosen_path, chosen_item = rng.choice(best_group)
                chosen_reason = "tie_random"
            else:
                chosen_avg, _, chosen_path, chosen_item = scored_candidates[0]
                chosen_reason = "max_avg"

        # 最终 reward：取被选中候选的 real_reward/eval_reward
        rr = get_real_reward(chosen_item)
        if rr is None:
            missing_reward_count += 1
            rr = 0

        if rr == 1:
            reward_ones += 1

        selected_valid_runs_list.append(get_valid_runs(chosen_item))

        # 输出：保存选中项 + meta
        out_obj = dict(chosen_item)
        out_obj["_tts_selected_from"] = chosen_path
        out_obj["_tts_selected_reason"] = chosen_reason
        out_obj["_tts_selected_avg"] = chosen_avg
        out_obj["_tts_k_effective"] = k_eff
        out_obj["_tts_final_reward"] = rr
        selected_rows.append(out_obj)

    # 统计
    reward_ratio = reward_ones / total_instances if total_instances else 0.0
    avg_k_effective = sum(k_effective_list) / len(k_effective_list) if k_effective_list else 0.0
    avg_selected_valid_runs = (
        sum(selected_valid_runs_list) / len(selected_valid_runs_list)
        if selected_valid_runs_list else 0.0
    )

    summary = {
        "num_instances": total_instances,
        "final_reward_1_count": reward_ones,
        "final_reward_1_ratio": reward_ratio,
        # 真正的 tts@k：每个 instance 有多少个候选（来自不同文件）有有效模拟
        "avg_tts_k_effective": avg_k_effective,
        "min_tts_k_effective": min(k_effective_list) if k_effective_list else None,
        "max_tts_k_effective": max(k_effective_list) if k_effective_list else None,
        # 选中候选内部有效 runs（一个候选内部多次打分过滤后剩余多少）
        "avg_valid_runs_in_selected_candidate": avg_selected_valid_runs,
        "tie_count": tie_count,
        "missing_avg_count": missing_avg_count,
        "missing_reward_count": missing_reward_count,
        "inputs": INPUT_FILES,
        "seed": RANDOM_SEED,
        "output_selected": OUTPUT_SELECTED_JSONL,
        "output_summary": OUTPUT_SUMMARY_JSON,
    }

    # 写 selected jsonl
    os.makedirs(os.path.dirname(OUTPUT_SELECTED_JSONL) or ".", exist_ok=True)
    with open(OUTPUT_SELECTED_JSONL, "w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 写 summary json
    os.makedirs(os.path.dirname(OUTPUT_SUMMARY_JSON) or ".", exist_ok=True)
    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()