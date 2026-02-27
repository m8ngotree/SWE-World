import glob
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

from rllm.data.dataset import DatasetRegistry

# 你把这里改成：本地 json/jsonl 的文件路径或目录路径
# - 可以是单个文件："/path/to/train.jsonl"
# - 也可以是目录："/path/to/my_data_dir"（会递归搜 json/jsonl）
INPUT_SOURCES = [
    
]

# 一个“数据集名字”，用于 DatasetRegistry 下面建目录
# 最终会写到 rllm/data/datasets/<DATASET_NAME>/
DATASET_NAME = "swe_rl_data"

# shard 设置
TRAIN_NUM_SHARDS = 1
TEST_NUM_SHARDS = 1

# 如果文件名里能区分 train/test，可以用这个规则
# - 例如文件名含 "train" 就算 train
# - 含 "test" / "val" / "valid" 就算 test
# 不满足的会归到 train（你也可以改）
SPLIT_RULES = {
    "train": ["train"],
    "test": ["test", "val", "valid", "dev"],
}


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in jsonl file {path} at line {line_no}: {e}") from e


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # 支持：json 顶层是 list 或 dict
        if isinstance(obj, list):
            return [dict(x) for x in obj]
        if isinstance(obj, dict):
            # 单条样本 dict -> 变成长度为 1 的 list
            return [dict(obj)]
        raise ValueError(f"{path}: JSON root must be list or dict, got {type(obj)}")
    elif ext == ".jsonl":
        return [dict(x) for x in iter_jsonl(path)]
    else:
        raise ValueError(f"Unsupported file type: {path} (only .json/.jsonl)")


def discover_input_files(sources: List[str]) -> List[str]:
    """把 INPUT_SOURCES 里的文件/目录展开成所有 json/jsonl 文件列表。"""
    files: List[str] = []
    for src in sources:
        if os.path.isdir(src):
            # 递归找
            files.extend(glob.glob(os.path.join(src, "**", "*.json"), recursive=True))
            files.extend(glob.glob(os.path.join(src, "**", "*.jsonl"), recursive=True))
        elif os.path.isfile(src):
            files.append(src)
        else:
            raise FileNotFoundError(f"Input source not found: {src}")
    # 去重 + 排序（保证可复现）
    return sorted(list(dict.fromkeys(files)))


def infer_split_from_filename(path: str) -> str:
    """根据文件名猜 split；猜不到就默认 train。"""
    name = os.path.basename(path).lower()
    for split, keywords in SPLIT_RULES.items():
        for kw in keywords:
            if kw in name:
                return split
    return "train"


def make_process_fn():
    def process_fn(row: Dict[str, Any]) -> Dict[str, Any]:
        # 这里写你的预处理逻辑
        row_dict = dict(row)
        # problem_statement = row_dict.get("problem_statement", "")
        return row_dict

    return process_fn


def shard_list(data: List[Dict[str, Any]], num_shards: int) -> List[List[Dict[str, Any]]]:
    """把 data 均匀切成 num_shards 份。"""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if len(data) == 0:
        return [[] for _ in range(num_shards)]

    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
    for i, item in enumerate(data):
        shards[i % num_shards].append(item)
    return shards


def register_shards(dataset_name: str, split: str, data: List[Dict[str, Any]], num_shards: int):
    """
    把一个 split 的 data 切 shard 并逐个 register：
    - split=train -> train_00000, train_00001, ...
    - split=test  -> test_00000, ...
    """
    shards = shard_list(data, num_shards=num_shards)
    registered = []
    for shard_id, shard_data in enumerate(shards):
        shard_split_name = f"{split}_{shard_id:05d}"
        ds = DatasetRegistry.register_dataset(dataset_name, shard_data, shard_split_name)
        registered.append(ds)
        print(f"[OK] Registered {dataset_name}:{shard_split_name} with {len(shard_data)} examples")
    return registered


def prepare_swe_data_from_local_json(
    input_sources: List[str],
    dataset_name: str,
    train_num_shards: int = 10,
    test_num_shards: int = 1,
) -> Tuple[List[Any], List[Any]]:
    """
    从本地 json/jsonl 读取 -> 预处理 -> 分 shard -> 存 parquet 并 register
    """
    if not input_sources:
        raise ValueError("INPUT_SOURCES is empty. Please provide json/jsonl files or directories.")

    process_fn = make_process_fn()

    files = discover_input_files(input_sources)
    print(f"Discovered {len(files)} json/jsonl files.")
    for p in files:
        print(f" - {p}")

    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for path in files:
        split = infer_split_from_filename(path)
        print(f"\nLoading {path}  -> split={split}")
        rows = load_json_or_jsonl(path)
        print(f"Loaded {len(rows)} raw rows")

        processed = [process_fn(r) for r in rows]
        print(f"Processed {len(processed)} rows")

        if split == "test":
            test_rows.extend(processed)
        else:
            train_rows.extend(processed)

    print("\n=== Summary before sharding ===")
    print(f"Train examples: {len(train_rows)}")
    print(f"Test  examples: {len(test_rows)}")

    train_datasets = register_shards(dataset_name, "train", train_rows, num_shards=train_num_shards)
    test_datasets = []
    if len(test_rows) > 0:
        test_datasets = register_shards(dataset_name, "test", test_rows, num_shards=test_num_shards)

    return train_datasets, test_datasets


if __name__ == "__main__":
    train_datasets, test_datasets = prepare_swe_data_from_local_json(
        input_sources=INPUT_SOURCES,
        dataset_name=DATASET_NAME,
        train_num_shards=TRAIN_NUM_SHARDS,
        test_num_shards=TEST_NUM_SHARDS,
    )

    print("\nSummary:")
    print(f"Total train shard datasets: {len(train_datasets)}")
    print(f"Total test shard datasets: {len(test_datasets)}")
