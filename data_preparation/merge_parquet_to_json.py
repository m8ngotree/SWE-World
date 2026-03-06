import os
import json
from pathlib import Path

import pandas as pd

def parquet_dir_to_json(input_dir: str, output_json: str):
    """
    读取目录下所有 .parquet 文件，按行合并后保存为一个 JSON 文件（列表形式）。
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"{input_dir} 不是有效的目录")

    # 找到目录下所有 .parquet 文件
    parquet_files = sorted(input_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"在目录 {input_dir} 下没有找到 .parquet 文件")

    print(f"找到 {len(parquet_files)} 个 parquet 文件，将开始合并：")
    for p in parquet_files:
        print(" -", p.name)

    # 逐个读取并合并
    dfs = []
    for p in parquet_files:
        df = pd.read_parquet(p)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    # 用 pandas 自带的 to_json 避免 ndarray 等类型的问题
    json_str = merged_df.to_json(orient="records", force_ascii=False)

    # 如果你想要带缩进的好看格式，就再 parse 一次再 dump
    data = json.loads(json_str)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已保存到: {output_json}, 共 {len(merged_df)} 条记录")


if __name__ == "__main__":
    input_file = ""  
    output_file = ""  
    parquet_dir_to_json(input_file, output_file)
