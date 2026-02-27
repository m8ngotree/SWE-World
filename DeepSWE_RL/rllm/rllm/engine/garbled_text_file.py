#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import unicodedata
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

from tqdm import tqdm


# ============================================================
# ✅ 全部配置写死在这里：不需要 argparse
# ============================================================

# 1) 你要扫描的目录（改这里）
TARGET_DIR = ""   # TODO: 改成你的目录

# 2) 只处理文件名为纯数字的 jsonl：1.jsonl / 22.jsonl ...
NUMERIC_JSONL_PATTERN = re.compile(r"^\d+\.jsonl$")

# 3) 是否预先统计行数（用于 tqdm 显示 total；会读两遍文件）
SHOW_TOTAL = True

# 4) JSON 解析失败或缺 chat_completion 字段：是否算作“乱码轨迹”
COUNT_MALFORMED_AS_GARBLED = True

# 5) 乱码检测阈值（固定）
DETECTOR_CONFIG = dict(
    min_len=10,                     # 太短不判
    min_script_chars=6,             # 某脚本至少出现多少个字母类字符才算“有效”
    min_script_prop=0.01,           # 某脚本占所有字母类字符比例达到多少才算“有效”
    script_diversity_threshold=3,   # 有效脚本种类数 >= 3 认为混杂明显
    symbol_ratio_threshold=0.40,    # 标点+符号占比过高
    control_ratio_threshold=0.02,   # 控制字符占比过高
    replacement_char_threshold=1,   # 出现 '�' 的次数阈值
)


# ============================================================
# 1) Unicode 脚本粗分类（按字符范围）
# ============================================================
_SCRIPT_RANGES = [
    ("Latin",        [(0x0041, 0x007A), (0x00C0, 0x02AF), (0x1E00, 0x1EFF)]),
    ("CJK",          [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)]),
    ("Hiragana",     [(0x3040, 0x309F)]),
    ("Katakana",     [(0x30A0, 0x30FF), (0x31F0, 0x31FF)]),
    ("Hangul",       [(0xAC00, 0xD7AF), (0x1100, 0x11FF)]),
    ("Cyrillic",     [(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)]),
    ("Arabic",       [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)]),
    ("Devanagari",   [(0x0900, 0x097F)]),
    ("Thai",         [(0x0E00, 0x0E7F)]),
    ("Hebrew",       [(0x0590, 0x05FF)]),
    ("Greek",        [(0x0370, 0x03FF)]),
]

def _script_of_char(ch: str) -> str:
    cp = ord(ch)
    for name, ranges in _SCRIPT_RANGES:
        for a, b in ranges:
            if a <= cp <= b:
                return name
    return "Other"


# ============================================================
# 2) 乱码/语言混杂检测（参数来自 DETECTOR_CONFIG）
# ============================================================
def detect_garbled_text(text: str, *, return_debug: bool = False):
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    min_len = DETECTOR_CONFIG["min_len"]
    min_script_chars = DETECTOR_CONFIG["min_script_chars"]
    min_script_prop = DETECTOR_CONFIG["min_script_prop"]
    script_diversity_threshold = DETECTOR_CONFIG["script_diversity_threshold"]
    symbol_ratio_threshold = DETECTOR_CONFIG["symbol_ratio_threshold"]
    control_ratio_threshold = DETECTOR_CONFIG["control_ratio_threshold"]
    replacement_char_threshold = DETECTOR_CONFIG["replacement_char_threshold"]

    n = len(text)
    if n < min_len:
        return (False, {"reason": "too_short", "len": n}) if return_debug else False

    cat_counter = Counter()
    script_counter = Counter()
    effective_script_chars = 0
    replacement_count = text.count("\uFFFD")  # '�'

    for ch in text:
        if ch.isspace():
            cat_counter["space"] += 1
            continue

        cat = unicodedata.category(ch)  # e.g. 'Ll', 'Po', 'So', 'Cc'
        major = cat[0]                  # L, N, P, S, C, M, Z
        cat_counter[major] += 1

        if major == "L":
            sc = _script_of_char(ch)
            script_counter[sc] += 1
            effective_script_chars += 1

    # 有效脚本：满足出现次数 + 占比
    effective_scripts = []
    for sc, cnt in script_counter.items():
        if sc == "Other":
            continue
        prop = cnt / max(1, effective_script_chars)
        if cnt >= min_script_chars and prop >= min_script_prop:
            effective_scripts.append((sc, cnt, prop))
    effective_scripts.sort(key=lambda x: x[1], reverse=True)
    script_diversity = len(effective_scripts)

    # 符号/标点占比
    num_non_space = n - cat_counter["space"]
    symbol_and_punct = cat_counter["S"] + cat_counter["P"]
    symbol_ratio = symbol_and_punct / max(1, num_non_space)

    # 控制字符占比
    control_ratio = cat_counter["C"] / max(1, num_non_space)

    # 连续怪符号串占比
    weird_punct = re.findall(r"[{}\[\]()<>\|\\/@`~^$*=+_]{3,}", text)
    weird_punct_score = sum(len(x) for x in weird_punct) / max(1, num_non_space)

    strong_conditions = [
        replacement_count >= replacement_char_threshold,
        control_ratio >= control_ratio_threshold,
        script_diversity >= script_diversity_threshold,
        symbol_ratio >= symbol_ratio_threshold,
    ]

    weak_combo = (
        (script_diversity >= 2 and symbol_ratio >= 0.22) or
        (script_diversity >= 2 and weird_punct_score >= 0.08) or
        (symbol_ratio >= 0.35 and weird_punct_score >= 0.10)
    )

    is_garbled = any(strong_conditions) and (weak_combo or script_diversity >= script_diversity_threshold)

    if not return_debug:
        return is_garbled

    debug = {
        "len": n,
        "non_space": num_non_space,
        "replacement_count": replacement_count,
        "category_major_counts": dict(cat_counter),
        "effective_script_chars": effective_script_chars,
        "script_counts": dict(script_counter),
        "effective_scripts_top": effective_scripts[:10],
        "script_diversity": script_diversity,
        "symbol_ratio": round(symbol_ratio, 4),
        "control_ratio": round(control_ratio, 6),
        "weird_punct_score": round(weird_punct_score, 4),
        "thresholds": dict(DETECTOR_CONFIG),
    }
    return is_garbled, debug


# ============================================================
# 3) 读取 jsonl 并统计每个文件乱码比例
# ============================================================
def list_numeric_jsonl_files(dir_path: str) -> List[str]:
    files = []
    for name in os.listdir(dir_path):
        if NUMERIC_JSONL_PATTERN.match(name):
            files.append(os.path.join(dir_path, name))
    files.sort(key=lambda p: int(os.path.basename(p).split(".")[0]))
    return files

def safe_get_chat_completion(obj: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    兼容结构：
      - obj["chat_completion"]
      - obj["data"]["chat_completion"]
    """
    if not isinstance(obj, dict):
        return None

    cc = obj.get("chat_completion")
    if isinstance(cc, list):
        return cc

    data = obj.get("data")
    if isinstance(data, dict):
        cc2 = data.get("chat_completion")
        if isinstance(cc2, list):
            return cc2

    return None

def trajectory_has_garbled_assistant(chat_completion: List[Dict[str, Any]]) -> bool:
    """
    任意 assistant message 的 content 检测为乱码 => 该行算“乱码轨迹”
    """
    for msg in chat_completion:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        if detect_garbled_text(content, return_debug=False):
            print(f"[Garbled] {content}\n====================")
            return True
    return False

def count_lines_fast(path: str) -> int:
    cnt = 0
    with open(path, "rb") as f:
        for _ in f:
            cnt += 1
    return cnt

def process_one_file(path: str) -> Tuple[int, int, float]:
    """
    返回：(total_lines, garbled_lines, ratio)
    """
    total = count_lines_fast(path) if SHOW_TOTAL else None

    garbled = 0
    seen = 0
    name = os.path.basename(path)

    with open(path, "r", encoding="utf-8") as f:
        it = tqdm(f, total=total, desc=f"[scan] {name}", unit="lines", dynamic_ncols=True)
        for line in it:
            line = line.strip()
            if not line:
                continue
            seen += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if COUNT_MALFORMED_AS_GARBLED:
                    garbled += 1
                # 更新进度显示
                it.set_postfix({"garbled": garbled, "total": seen, "ratio": f"{garbled/max(1,seen):.3%}"})
                continue

            cc = safe_get_chat_completion(obj)
            if cc is None:
                if COUNT_MALFORMED_AS_GARBLED:
                    garbled += 1
                it.set_postfix({"garbled": garbled, "total": seen, "ratio": f"{garbled/max(1,seen):.3%}"})
                continue

            if trajectory_has_garbled_assistant(cc):
                garbled += 1

            it.set_postfix({"garbled": garbled, "total": seen, "ratio": f"{garbled/max(1,seen):.3%}"})

    ratio = garbled / max(1, seen)
    return seen, garbled, ratio


# ============================================================
# 4) 入口：直接运行
# ============================================================
def main():
    if not os.path.isdir(TARGET_DIR):
        raise FileNotFoundError(f"TARGET_DIR not found or not a directory: {TARGET_DIR}")

    files = list_numeric_jsonl_files(TARGET_DIR)[:9]
    if not files:
        print(f"No numeric jsonl files found in: {TARGET_DIR} (pattern: ^\\d+\\.jsonl$)")
        return

    print(f"TARGET_DIR: {TARGET_DIR}")
    print(f"Found {len(files)} files.")
    print(f"DETECTOR_CONFIG: {DETECTOR_CONFIG}")
    print(f"SHOW_TOTAL={SHOW_TOTAL}, COUNT_MALFORMED_AS_GARBLED={COUNT_MALFORMED_AS_GARBLED}")
    print("")

    results = []
    overall_total = 0
    overall_garbled = 0

    for path in files:
        total, garbled, ratio = process_one_file(path)
        name = os.path.basename(path)
        results.append((name, total, garbled, ratio))
        overall_total += total
        overall_garbled += garbled
        print(f"[DONE] {name}  garbled={garbled}/{total}  ratio={ratio:.3%}")

    overall_ratio = overall_garbled / max(1, overall_total)

    print("\n=== Summary ===")
    for name, total, garbled, ratio in results:
        print(f"{name:>10s}  garbled={garbled:>6d}/{total:<6d}  ratio={ratio:.3%}")
    print(f"{'ALL':>10s}  garbled={overall_garbled:>6d}/{overall_total:<6d}  ratio={overall_ratio:.3%}")


if __name__ == "__main__":
    main()
