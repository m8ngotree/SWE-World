import os
import json
import time
import argparse  # ✅ [NEW] 引入 argparse
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm

# ✅ 直接从你工程里导入（按你之前用法）
from r2egym.agenthub.agent.simulator import SimulatorAgent
from r2egym.agenthub.utils.log import get_logger

# =========================
# ✅ [NEW] 多线程相关
# =========================
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# =========================
# 配置区：全局变量声明（具体值由 argparse 在 main 中注入）
# =========================
NUM_WORKERS = 16  # 默认值
SKIP_ALREADY_DONE = True  # 默认值

INPUT_JSON_PATH = None
OUTPUT_JSONL_PATH = None
SUMMARY_JSON_PATH = None

# SimulatorAgent 初始化参数
SIMULATOR_CONFIG_PATH = None
SIMULATOR_CONFIG: List[Dict[str, Any]] = []  # 保持为空列表，如果有 yaml 会优先读取 yaml
TOKENIZER_PATH = None

# 每次调用间隔
SLEEP_BETWEEN_CALLS_SEC = 0.0


# =========================
# IO：兼容 JSON array 或 JSONL
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
    
    for item in data:
        if "real_reward" in item:
            item["eval_reward"] = item["real_reward"]
    
    return data


# =========================
# Metrics：混淆矩阵 + 常用指标
# =========================
def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    assert len(y_true) == len(y_pred)
    n = len(y_true)

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    acc = safe_div(tp + tn, n)
    prec = safe_div(tp, tp + fp)          # PPV
    rec = safe_div(tp, tp + fn)           # TPR
    f1 = safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
    spec = safe_div(tn, tn + fp)          # TNR
    bal_acc = 0.5 * (rec + spec)

    # MCC
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = 0.0
    if denom > 0:
        mcc = (tp * tn - fp * fn) / (denom ** 0.5)

    return {
        "n": n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "tp_rate": safe_div(tp, n),
        "tn_rate": safe_div(tn, n),
        "fp_rate": safe_div(fp, n),
        "fn_rate": safe_div(fn, n),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "pos_true_rate": safe_div(sum(y_true), n),
        "pos_pred_rate": safe_div(sum(y_pred), n),
    }


# =========================
# ✅ [NEW] 基于 instance_id 的断点续跑工具
# =========================
def get_instance_id(item: Dict[str, Any]) -> str:
    """
    尽量兼容不同数据结构：优先 item['instance_id']，其次 item['ds']['instance_id'] 等。
    你明确说要用 instance_id 做断点，因此找不到就直接报错。
    """
    if "instance_id" in item and item["instance_id"] is not None:
        return str(item["instance_id"])
    ds = item.get("ds")
    if isinstance(ds, dict) and ds.get("instance_id") is not None:
        return str(ds["instance_id"])
    meta = item.get("metadata")
    if isinstance(meta, dict) and meta.get("instance_id") is not None:
        return str(meta["instance_id"])
    raise KeyError("Cannot find instance_id in item (tried item['instance_id'], item['ds']['instance_id'], item['metadata']['instance_id'])")

def load_done_instance_ids_and_metrics(output_jsonl_path: str) -> Tuple[set, List[int], List[int]]:
    """
    已经成功写入 OUTPUT_JSONL 的都认为 done（因为你的原逻辑只有成功才写）。
    同时把这些已完成的样本的 y_true/y_pred 读出来，保证续跑时 metrics 是全量的。
    """
    done_ids = set()
    y_true_exist: List[int] = []
    y_pred_exist: List[int] = []

    if not os.path.exists(output_jsonl_path):
        return done_ids, y_true_exist, y_pred_exist

    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                iid = get_instance_id(obj)
                done_ids.add(iid)
            except Exception:
                # 如果旧 output 行里拿不到 instance_id，就忽略（不影响主流程）
                continue

            # 兼容你现在的字段：eval_reward / simulated_reward
            if obj.get("eval_reward") is not None and obj.get("simulated_reward") is not None:
                try:
                    y_true_exist.append(int(obj["eval_reward"]))
                    y_pred_exist.append(int(obj["simulated_reward"]))
                except Exception:
                    pass

    return done_ids, y_true_exist, y_pred_exist


# =========================
# ✅ [NEW] 每线程一个 agent（避免共享对象线程不安全）
# =========================
_thread_local = threading.local()

def get_thread_agent() -> SimulatorAgent:
    agent = getattr(_thread_local, "agent", None)
    if agent is None:
        # 注意：这里的全局变量已经在 main 中被更新为 args 传入的值
        agent = SimulatorAgent(
            simulator_config=SIMULATOR_CONFIG,              # 你也可以留空，靠 yaml 覆盖
            simulator_config_path=SIMULATOR_CONFIG_PATH,    # 有 path 优先读 path（按你类逻辑）
            tokenizer_path=TOKENIZER_PATH,
            logger=get_logger("SimulatorAgent"),
        )
        _thread_local.agent = agent
    return agent


# =========================
# 主流程
# =========================
def main():
    logger = get_logger("sim_reward_runner")

    logger.info(f"Input Path: {INPUT_JSON_PATH}")
    logger.info(f"Output Path: {OUTPUT_JSONL_PATH}")
    logger.info(f"Config Path: {SIMULATOR_CONFIG_PATH}")
    logger.info(f"Tokenizer Path: {TOKENIZER_PATH}")

    data = load_json_or_jsonl(INPUT_JSON_PATH)
    logger.info(f"Loaded {len(data)} records from: {INPUT_JSON_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_JSONL_PATH), exist_ok=True)

    # ✅ [NEW] 断点续跑：读取已完成的 instance_id + 既有 metrics 部分
    done_ids = set()
    y_true: List[int] = []
    y_pred: List[int] = []
    if SKIP_ALREADY_DONE:
        done_ids, y_true_exist, y_pred_exist = load_done_instance_ids_and_metrics(OUTPUT_JSONL_PATH)
        y_true.extend(y_true_exist)
        y_pred.extend(y_pred_exist)
        logger.info(f"[checkpoint] Found {len(done_ids)} done instance_ids in: {OUTPUT_JSONL_PATH}")

    sim_error = 0

    # ✅ [NEW] 用锁保护文件写与 metrics append（线程安全）
    write_lock = threading.Lock()
    metric_lock = threading.Lock()

    # ✅ [NEW] 过滤待处理任务（按 instance_id）
    tasks: List[Tuple[int, Dict[str, Any]]] = []
    for i, item in enumerate(data):
        iid = get_instance_id(item)  # 找不到就直接报错（按你要求基于 instance_id）
        if SKIP_ALREADY_DONE and iid in done_ids:
            continue
        tasks.append((i, item))

    logger.info(f"Need to process {len(tasks)} records (skipped {len(data) - len(tasks)} already done).")

    # ✅ [NEW] 以追加方式打开，保证续跑继续写
    with open(OUTPUT_JSONL_PATH, "a", encoding="utf-8") as wf:

        def worker(i: int, item: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]], int, Optional[Tuple[int, int]]]:
            """
            返回：
              - i
              - updated_item(成功则为写入项，失败则 None)
              - err_flag (1表示错误/不写入，0表示成功写入)
              - metrics_pair: (eval_reward, sim_reward) 成功则返回，否则 None
            """
            # 你说文件里有 context / eval_reward
            context = item.get("context")
            if context is None:
                raise KeyError(f"Record {i} missing key: context")

            eval_reward = item.get("eval_reward")
            if eval_reward is None:
                raise KeyError(f"Record {i} missing key: eval_reward")

            agent = get_thread_agent()

            # 调用你已有的函数（不做 retry）
            try:
                sim_test_report, sim_reward, sim_response_content = agent.get_simulated_test_report_and_reward(context)
                sim_reward = int(sim_reward)
            except Exception as e:
                # 不中断：标记错误，不写入（保持你原先“错误不写入output”的语义）
                sim_test_report = ""
                sim_reward = 0
                sim_response_content = f"ERROR11111111: {type(e).__name__}: {e}"
                print(f"{sim_response_content}")

            # 你原逻辑：出现 ERROR 字样则记错误并跳过写入
            if ("ERROR11111111" in sim_response_content) or ("Combined test-report & reward LLM failed" in sim_response_content):
                return i, None, 1, None

            # 保存每条数据的模拟结果（保持你原逻辑字段名）
            item["simulated_reward"] = sim_reward
            item["sim_test_report"] = sim_test_report
            item["sim_response_content"] = sim_response_content

            return i, item, 0, (int(eval_reward), int(sim_reward))

        # ✅ [NEW] 多线程执行（用 as_completed + tqdm）
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = [ex.submit(worker, i, item) for (i, item) in tasks]

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Calling get_reward_new (multithread)"):
                try:
                    i, updated_item, err_flag, metrics_pair = fut.result()
                except Exception as e:
                    # worker 自己抛出的异常（比如缺字段），这里也记为 error
                    sim_error += 1
                    print(f"Record worker exception: {type(e).__name__}: {e}")
                    continue

                if err_flag == 1 or updated_item is None or metrics_pair is None:
                    sim_error += 1
                    continue

                # ✅ 写文件（追加 + 锁）
                with write_lock:
                    wf.write(json.dumps(updated_item, ensure_ascii=False) + "\n")
                    wf.flush()

                # ✅ 指标统计（锁）
                with metric_lock:
                    y_true.append(metrics_pair[0])
                    y_pred.append(metrics_pair[1])

                # 可选 sleep（每个成功样本后）
                if SLEEP_BETWEEN_CALLS_SEC > 0:
                    time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    metrics = compute_metrics(y_true, y_pred)

    # 打印混淆矩阵和指标
    n = metrics["n"]
    tp, tn, fp, fn = metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]

    print("\n================ Confusion Matrix ================")
    print(f"Total N = {n}")
    print(f"TP (pred=1,true=1): {tp} ({metrics['tp_rate']:.4f})")
    print(f"TN (pred=0,true=0): {tn} ({metrics['tn_rate']:.4f})")
    print(f"FP (pred=1,true=0): {fp} ({metrics['fp_rate']:.4f})")
    print(f"FN (pred=0,true=1): {fn} ({metrics['fn_rate']:.4f})")
    print(f"Sim Error Count    : {sim_error}")

    print("\n================ Metrics ================")
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Precision (PPV)   : {metrics['precision']:.4f}")
    print(f"Recall (TPR)      : {metrics['recall']:.4f}")
    print(f"F1                : {metrics['f1']:.4f}")
    print(f"Specificity (TNR) : {metrics['specificity']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"MCC               : {metrics['mcc']:.4f}")
    print(f"True Positive Rate (label=1 proportion): {metrics['pos_true_rate']:.4f}")
    print(f"Pred Positive Rate (pred=1 proportion) : {metrics['pos_pred_rate']:.4f}")

    # 保存 summary
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved per-item output JSONL to: {OUTPUT_JSONL_PATH}")
    logger.info(f"Saved summary metrics to: {SUMMARY_JSON_PATH}")
    logger.info("Done.")


if __name__ == "__main__":
    # ✅ [NEW] 解析命令行参数
    parser = argparse.ArgumentParser(description="Simulator Reward Runner")
    
    # 必填参数
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON/JSONL file")
    parser.add_argument("--config_path", type=str, required=True, help="Path to simulator config YAML")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer directory")
    
    # 选填参数
    parser.add_argument("--output_path", type=str, default=None, help="Path to output JSONL file (default: derived from input + suffix)")
    parser.add_argument("--summary_path", type=str, default=None, help="Path to summary metrics JSON file (default: derived from output)")
    
    # 调整参数
    parser.add_argument("--num_workers", type=int, default=16, help="Number of concurrent workers")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep time between calls in seconds")
    parser.add_argument("--overwrite", action="store_true", help="If set, skip breakpoint resume and overwrite/reprocess all")

    args = parser.parse_args()

    # 将参数注入全局变量
    INPUT_JSON_PATH = args.input_path
    SIMULATOR_CONFIG_PATH = args.config_path
    TOKENIZER_PATH = args.tokenizer_path
    
    NUM_WORKERS = args.num_workers
    SLEEP_BETWEEN_CALLS_SEC = args.sleep
    SKIP_ALREADY_DONE = not args.overwrite  # 默认 True (skip), 如果传入 --overwrite 则为 False

    # 路径推导逻辑
    if args.output_path:
        OUTPUT_JSONL_PATH = args.output_path
    else:
        # 如果未指定输出路径，使用通用后缀 (替代原先硬编码的 _qwen3_reward_...)
        base_name, _ = os.path.splitext(INPUT_JSON_PATH)
        OUTPUT_JSONL_PATH = f"{base_name}_sim_reward_out.jsonl"
    
    if args.summary_path:
        SUMMARY_JSON_PATH = args.summary_path
    else:
        # 基于 Output Path 推导 Summary Path
        base_name, _ = os.path.splitext(OUTPUT_JSONL_PATH)
        # 如果 output 包含 _out，去掉它再加 _metrics；否则直接加
        clean_base = base_name.replace("_out", "") if "_out" in base_name else base_name
        SUMMARY_JSON_PATH = f"{clean_base}_metrics.json"

    # 执行主函数
    main()