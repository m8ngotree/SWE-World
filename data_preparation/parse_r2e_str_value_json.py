import json
from pathlib import Path
from typing import Any

TARGET_FIELDS = ["parsed_commit_content", "execution_result_content", "expected_output_json"]


def try_parse_json_string(s: str) -> Any:
    """
    尝试把字符串解析成 JSON（dict/list/str/number/bool/null）
    失败则抛异常给上层处理。
    """
    s2 = s.strip()

    # 常见场景：字符串本身被包了一层引号，里面才是 JSON
    # 例如: "\"{\\\"a\\\": 1}\""
    # 先尝试一次正常 json.loads，不行再尝试“二次解析”
    try:
        return json.loads(s2)
    except json.JSONDecodeError:
        pass

    # 二次解析：先把外层当 JSON 字符串解析成普通字符串，再解析一次
    try:
        inner = json.loads(s2)  # 如果 s2 是一个 JSON string，这里能变成普通 str
        if isinstance(inner, str):
            return json.loads(inner.strip())
    except Exception:
        pass

    raise json.JSONDecodeError("Not a valid JSON string", s2, 0)


def convert_record_fields(rec: dict, target_fields=TARGET_FIELDS, strict: bool = False) -> tuple[int, int]:
    """
    把单条记录里的 target_fields 从 JSON 字符串 -> JSON 对象
    返回 (converted_count, skipped_count)
    - strict=True：解析失败就报错
    - strict=False：解析失败就跳过，不改原值
    """
    converted = 0
    skipped = 0

    for k in target_fields:
        if k not in rec:
            skipped += 1
            continue

        v = rec[k]

        # 已经是 dict/list 等 JSON 类型就不动
        if isinstance(v, (dict, list, int, float, bool)) or v is None:
            skipped += 1
            continue

        # 只处理字符串
        if isinstance(v, str):
            if v.strip() == "":
                # 空字符串：通常不是 JSON，按跳过处理
                skipped += 1
                continue
            try:
                rec[k] = try_parse_json_string(v)
                converted += 1
            except Exception:
                if strict:
                    raise
                skipped += 1
            continue

        # 其他类型：不处理
        skipped += 1

    return converted, skipped


def normalize_to_records(obj: Any, list_key: str | None = None) -> tuple[list[dict], str]:
    """
    返回 (records, mode)
    mode:
      - "list": 顶层本来就是 list
      - "dict_list": 顶层是 dict，records 来自 obj[list_key]
      - "dict": 顶层是 dict，但当作单条记录
    """
    if isinstance(obj, list):
        if not all(isinstance(x, dict) for x in obj):
            raise TypeError("顶层是 list，但其中存在非 dict 元素")
        return obj, "list"

    if isinstance(obj, dict):
        if list_key is not None:
            data = obj.get(list_key)
            if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
                raise TypeError(f'指定的 list_key="{list_key}" 不是 list[dict]')
            return data, "dict_list"
        return [obj], "dict"

    raise TypeError("输入 JSON 顶层必须是 list 或 dict")


def process_file(input_path: str, output_path: str, list_key: str | None = None, strict: bool = False) -> None:
    input_path = str(Path(input_path))
    output_path = str(Path(output_path))

    with open(input_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    records, mode = normalize_to_records(obj, list_key=list_key)

    total_converted = 0
    total_skipped = 0
    for rec in records:
        c, s = convert_record_fields(rec, strict=strict)
        total_converted += c
        total_skipped += s

    # 写回：保持原始顶层结构
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj if mode != "dict" else records[0], f, ensure_ascii=False, indent=2)

    print(f"完成：{output_path}")
    print(f"converted={total_converted}, skipped={total_skipped}, records={len(records)}")


if __name__ == "__main__":
    # 1) 顶层是 list[dict]
    # process_file("input.json", "output.json", list_key=None, strict=False)

    input_file = ""
    output_file = ""
    process_file(input_file, output_file, list_key=None, strict=False)
