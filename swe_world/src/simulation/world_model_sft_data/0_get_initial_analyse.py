import json
from pathlib import Path
from tqdm import tqdm  # 如果不用可以删掉
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import time
import fire  # 如果不用可以删掉
import random
import threading  # ✅ [NEW] 断点/写文件加锁用

from typing import Dict, Any  # ✅ 补上类型导入

# ✅ [NEW] 仅新增：支持 load_from_disk
from datasets import load_from_disk

sys.path.append(str(Path(__file__).resolve().parent.parent))

from r2egym.agenthub.agent.simulator import SimulatorAgent
from r2egym.agenthub.utils.log import get_logger

# ✅ 补上 logger 实例
logger = get_logger("InitialAnalysisBatch")

# ✅ [NEW] 写 jsonl 时用锁，避免多线程同时写
file_lock = threading.Lock()


def _build_meta_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    从一条 json 记录中抽取生成 initial_analysis 所需的字段。
    这里假设 json 中的字段名为：
        - problem_statement
        - hints_text   (如果没有，可以尝试 hints)
        - repo
        - patch        (gold patch)
        - test_patch
        - FAIL_TO_PASS
        - PASS_TO_PASS
    你可以根据真实字段名微调。
    """
    meta = {
        "problem_statement": item.get("problem_statement", ""),
        "hints_text": item.get("hints_text", item.get("hints", "")),
        "repo": item.get("repo", ""),
        "patch": item.get("patch", ""),
        "test_patch": item.get("test_patch", ""),
        "FAIL_TO_PASS": item.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": item.get("PASS_TO_PASS", []),
    }
    return meta


# ✅ [NEW] 仅新增：把 HF datasets Dataset/DatasetDict 转成 list[dict]
def _dataset_to_list(ds) -> list:
    # datasets 新版本可能有 to_list()
    if hasattr(ds, "to_list"):
        return ds.to_list()

    # 兼容：用 to_dict()（列式）还原成行式 list[dict]
    d = ds.to_dict()
    if not d:
        return []
    n = len(next(iter(d.values())))
    return [{k: v[i] for k, v in d.items()} for i in range(n)]


# ✅ [NEW] 断点机制：读取已存在输出文件(jsonl/json)中的 instance_id，作为已完成集合
def _load_done_instance_ids(output_path: Path) -> set:
    done = set()
    if not output_path.exists():
        return done

    try:
        # 优先按 jsonl 逐行读
        with open(output_path, "r", encoding="utf-8") as f:
            first = f.read(1)
            f.seek(0)

            # 兼容老输出是 JSON 数组
            if first == "[":
                arr = json.load(f)
                if isinstance(arr, list):
                    for obj in arr:
                        if isinstance(obj, dict):
                            iid = obj.get("instance_id")
                            if iid is not None and "- Initial analysis unavailable due to LLM error." not in obj["initial_analysis"]:
                                done.add(iid)
                return done

            # jsonl
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    iid = obj.get("instance_id")
                    if iid is not None and "- Initial analysis unavailable due to LLM error." not in obj["initial_analysis"]:
                        done.add(iid)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint from {output_path}: {e}", exc_info=True)

    return done

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def batch_generate_initial_analysis(
    input_file: str,
    output_file: str,
    llm_name: str,
    llm_base_url: str | None = None,
    max_workers: int = 8,
) -> None:
    """
    从 input_file 读入 json 数据，使用 SimulatorAgent.get_initial_analysis
    并行生成 initial_analysis 字段，并把结果写入 output_file。

    当前实现假设 input_file 是一个 JSON 数组：list[dict]。
    每条记录会增加：
        - "initial_analysis": str

    ✅ 输出改为 JSONL：每得到一个结果就写一行。
    ✅ 支持断点：根据 instance_id 跳过 output_file 里已经存在的记录。
    """

    logger.info(f"Loading input from: {input_file}")

    # 1. 读取 input_file
    input_path = Path(input_file)

    # ✅ [NEW] 如果是目录：用 load_from_disk 读取 HF datasets 的磁盘格式
    if input_path.is_dir():
        logger.info(f"Detected directory input. Loading dataset from disk: {input_file}")
        ds = load_from_disk(input_file)

        # 兼容 DatasetDict：优先 train，其次取第一个 split
        if hasattr(ds, "keys") and callable(getattr(ds, "keys")):
            keys = list(ds.keys())
            split = "train" if "train" in keys else (keys[0] if keys else None)
            if split is None:
                raise ValueError(f"DatasetDict at {input_file} has no splits.")
            logger.info(f"Loaded DatasetDict. Using split: {split}")
            ds = ds[split]

        data = _dataset_to_list(ds)
        # data = random.sample(data, 20)
    else:
        # 原逻辑：json 文件
        if input_path.suffix == ".jsonl":
            data = read_jsonl(input_file)
        else:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # data = random.sample(data, 20)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    logger.info(f"Loaded {len(data)} records from input.")

    # ✅ [NEW] 断点：读取输出文件中已完成的 instance_id
    output_path = Path(output_file)
    done_ids = _load_done_instance_ids(output_path)
    if done_ids:
        logger.info(f"Checkpoint loaded from {output_file}: {len(done_ids)} instance_ids found.")

    # ✅ [NEW] 只处理未完成的 item（按 instance_id）
    pending = []
    skipped = 0
    for idx, item in enumerate(data):
        iid = item.get("instance_id", None)
        if iid is not None and iid in done_ids:
            skipped += 1
            continue
        pending.append((idx, item))

    logger.info(f"Skip {skipped} records (already in output). Pending: {len(pending)}.")

    if len(pending) == 0:
        logger.info("Nothing to do. All records are already processed.")
        return

    # 2. 初始化 SimulatorAgent（共享一个实例，多线程用）
    sim_agent = SimulatorAgent(llm_name=llm_name, llm_base_url=llm_base_url, logger=logger)

    # 3. 定义一个 worker 函数
    def worker(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        对单条数据调用 get_initial_analysis，并返回带 initial_analysis 的新 dict。
        """
        meta = _build_meta_from_item(item)
        analysis = sim_agent.get_problem_summary(meta)
        # 拷贝一份，避免在原 data 上直接改（也可以直接改，看你习惯）
        new_item = dict(item)
        new_item["problem_summary"] = analysis
        logger.info(f"[{idx}] initial_analysis generated.")
        return {"index": idx, "item": new_item}

    # 4. 多线程并发处理 + ✅ [NEW] 结果实时写 jsonl
    logger.info(f"Start generating initial_analysis with max_workers={max_workers}...")
    start_time = time.time()

    # ✅ [NEW] 追加写：断点续跑时保留已有输出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as fw:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(worker, idx, item): idx
                for idx, item in pending
            }

            for future in tqdm(as_completed(future_to_idx), desc="Analyzing", total=len(pending)):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    out_item = res["item"]

                    # ✅ [NEW] 每得到一个结果就写一行 JSONL
                    with file_lock:
                        fw.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                        fw.flush()

                except Exception as e:
                    logger.error(f"Error in worker for index {idx}: {e}", exc_info=True)
                    # ✅ [NEW] 失败也写一行，方便断点继续（同一个 instance_id 不会重复跑）
                    # 注意：如果没有 instance_id，就无法可靠断点；这里仍然写出错误记录。
                    # err_item = dict(data[idx])
                    # err_item["initial_analysis"] = f"ERROR: failed to generate initial analysis: {e}"
                    # with file_lock:
                    #     fw.write(json.dumps(err_item, ensure_ascii=False) + "\n")
                    #     fw.flush()

    elapsed = time.time() - start_time
    logger.info(f"Finished generating initial_analysis for {len(pending)} records in {elapsed:.2f}s.")
    logger.info(f"Output JSONL appended to: {output_file}")
    logger.info("Done.")


# 命令行入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch generate initial_analysis using SimulatorAgent.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file or a dataset directory.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file (append mode).")
    parser.add_argument("--llm_name", type=str, required=True, help="LLM model name for SimulatorAgent.")
    parser.add_argument("--llm_base_url", type=str, default=None, help="Optional base URL for LLM API.")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of threads for concurrent processing.")
    args = parser.parse_args()

    batch_generate_initial_analysis(
        input_file=args.input_file,
        output_file=args.output_file,
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        max_workers=args.max_workers,
    )
