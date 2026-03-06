#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import difflib
import json
import sys
from typing import Any, Dict, List, Tuple, Optional


FULL_REPOS = [
    "pandas-dev/pandas",
    "numpy/numpy",
    "biolab/orange3",
    "python-pillow/Pillow",
    "aio-libs/aiohttp",
    "tornadoweb/tornado",
    "scrapy/scrapy",
    "Pylons/pyramid",
    "nedbat/coveragepy",
    "datalad/datalad",
]


# 常见短名 -> full repo 显式映射（最稳）
SHORT_TO_FULL = {
    "pandas": "pandas-dev/pandas",
    "numpy": "numpy/numpy",
    "orange3": "biolab/orange3",
    "pillow": "python-pillow/Pillow",
    "aiohttp": "aio-libs/aiohttp",
    "tornado": "tornadoweb/tornado",
    "scrapy": "scrapy/scrapy",
    "pyramid": "Pylons/pyramid",
    "coveragepy": "nedbat/coveragepy",
    "coverage": "nedbat/coveragepy",
    "datalad": "datalad/datalad",
}


def map_repo_name(short_name: str) -> str:
    """
    根据数据集里的 repo_name(短名) 映射到完整 repo。
    先用显式字典；不命中再在 FULL_REPOS 里做宽松匹配。
    """
    if not short_name:
        return short_name

    key = short_name.strip()
    low = key.lower()

    if low in SHORT_TO_FULL:
        return SHORT_TO_FULL[low]

    # fallback：尝试按 repo 路径最后一段匹配（如 .../Pillow -> pillow）
    for full in FULL_REPOS:
        tail = full.split("/")[-1].lower()
        if tail == low:
            return full

    # 仍失败就原样返回（但会让你后续可定位问题）
    return short_name


def is_test_file(path: str) -> bool:
    # 你给的规则原样实现
    return (
        path.endswith("_test.py")
        or path.startswith("test_")
        or path.split("/")[-1].startswith("test_")
        or "tests" in path.split("/")
        or "Tests" in path.split("/")
        or "test" in path.split("/")
        or "Test" in path.split("/")
    )


def parse_log_pytest(log: str | None) -> dict[str, str]:
    """
    Parser for test logs generated with Sympy framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    if log is None:
        return {}
    test_status_map = {}
    if "short test summary info" not in log:
        return test_status_map
    log = log.split("short test summary info")[1]
    log = log.strip()
    log = log.split("\n")
    for line in log:
        if "PASSED" in line:
            test_name = ".".join(line.split("::")[1:])
            test_status_map[test_name] = "PASSED"
        elif "FAILED" in line:
            test_name = ".".join(line.split("::")[1:]).split(" - ")[0]
            test_status_map[test_name] = "FAILED"
        elif "ERROR" in line:
            try:
                test_name = ".".join(line.split("::")[1:])
            except IndexError:
                test_name = line
            test_name = test_name.split(" - ")[0]
            test_status_map[test_name] = "ERROR"
    return test_status_map

def _short_hash(h: Optional[str]) -> str:
    if not h:
        return ""
    return h[:7]

def build_git_patch_for_file(file_diff: Dict[str, Any]) -> Tuple[str, str]:
    header_path = ((file_diff.get("header") or {}).get("file") or {}).get("path")
    minus_path = (file_diff.get("minus_file") or {}).get("path")
    plus_path = (file_diff.get("plus_file") or {}).get("path")

    # 选一个可用的 path（避免 /dev/null 抢占）
    path = (
        header_path
        or (plus_path if plus_path and plus_path != "/dev/null" else None)
        or (minus_path if minus_path and minus_path != "/dev/null" else None)
        or "UNKNOWN_FILE"
    )

    old_content = file_diff.get("old_file_content") or ""
    new_content = file_diff.get("new_file_content") or ""

    from_label = "/dev/null" if minus_path == "/dev/null" else f"a/{path}"
    to_label = "/dev/null" if plus_path == "/dev/null" else f"b/{path}"

    old_lines = old_content.splitlines(keepends=False)
    new_lines = new_content.splitlines(keepends=False)

    udiff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=from_label,
            tofile=to_label,
            lineterm="\n",
        )
    )

    patch_lines: List[str] = []
    patch_lines.append(f"diff --git a/{path} b/{path}\n")
    patch_lines.extend(udiff_lines)

    patch_text = "".join(patch_lines)
    if not patch_text.endswith("\n"):
        patch_text += "\n"
    return path, patch_text


def convert_one_item(item: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    repo_short = item.get("repo_name", "")
    repo_full = map_repo_name(repo_short)

    parsed_commit = item.get("parsed_commit_content") or {}
    base_commit = parsed_commit.get("old_commit_hash")
    new_commit_hash = parsed_commit.get("new_commit_hash")

    problem_statement = item.get("problem_statement", "")
    commit_date = parsed_commit.get("commit_date", item.get("commit_date"))
    docker_image = item.get("docker_image", item.get("docker_images"))

    file_diffs = parsed_commit.get("file_diffs") or []

    patch_parts: List[str] = []
    test_patch_parts: List[str] = []
    non_test_file_content: Dict[str, Dict[str, str]] = {}
    test_file_content: Dict[str, Dict[str, str]] = {}
    modified_not_test_files: List[str] = []
    modified_test_files: List[str] = []

    for fd in file_diffs:
        path = ((fd.get("header") or {}).get("file") or {}).get("path") or "UNKNOWN_FILE"
        test_flag = is_test_file(path)

        old_c = fd.get("old_file_content")
        new_c = fd.get("new_file_content")
        old_c = "" if old_c is None else old_c
        new_c = "" if new_c is None else new_c

        real_path, file_patch = build_git_patch_for_file(fd)

        if test_flag:
            test_patch_parts.append(file_patch)
            test_file_content[real_path] = {"old_file_content": old_c, "new_file_content": new_c}
            modified_test_files.append(real_path)
        else:
            patch_parts.append(file_patch)
            non_test_file_content[real_path] = {"old_file_content": old_c, "new_file_content": new_c}
            modified_not_test_files.append(real_path)

    # 合并成完整 patch
    patch = "".join(patch_parts).rstrip() + ("\n" if patch_parts else "")
    test_patch = "".join(test_patch_parts).rstrip() + ("\n" if test_patch_parts else "")

    # 去重但保序
    def dedup_keep_order(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    modified_not_test_files = dedup_keep_order(modified_not_test_files)
    modified_test_files = dedup_keep_order(modified_test_files)

    # 校验 modified_files
    original_modified_files = item.get("modified_files") or []
    union_set = set(modified_not_test_files) | set(modified_test_files)
    original_set = set(original_modified_files)

    modified_files_match = (union_set == original_set)
    modified_files_diff = None
    if not modified_files_match:
        missing = sorted(list(original_set - union_set))
        extra = sorted(list(union_set - original_set))
        modified_files_diff = {"missing_in_parsed": missing, "extra_in_parsed": extra}
        msg = (
            f"[WARN] modified_files mismatch for repo={repo_short} new_commit={new_commit_hash}\n"
            f"  missing_in_parsed={missing}\n"
            f"  extra_in_parsed={extra}\n"
        )
        print(msg, file=sys.stderr)
        if strict:
            raise ValueError(msg)
    
    if len(modified_not_test_files) != item["num_non_test_files"]:
        msg = (
            f"[WARN] num_non_test_files mismatch for repo={repo_short} new_commit={new_commit_hash}\n"
            f"  expected={item['num_non_test_files']}\n"
            f"  actual={len(modified_not_test_files)}\n"
        )
        print(msg, file=sys.stderr)
        if strict:
            raise ValueError(msg)

    # 处理 execution_result_content
    exec_res = item.get("execution_result_content") or {}
    test_file_names = exec_res.get("test_file_names") or []
    test_file_codes = exec_res.get("test_file_codes") or []

    test_exec_content: Dict[str, str] = {}
    n = min(len(test_file_names), len(test_file_codes))
    for i in range(n):
        test_exec_content[test_file_names[i]] = test_file_codes[i]
    if len(test_file_names) != len(test_file_codes):
        print(
            f"[WARN] test_file_names/codes length mismatch for repo={repo_short} new_commit={new_commit_hash}: "
            f"{len(test_file_names)} vs {len(test_file_codes)}",
            file=sys.stderr,
        )

    old_stdout = exec_res.get("old_commit_res_stdout")
    new_stdout = exec_res.get("new_commit_res_stdout")
    old_map = parse_log_pytest(old_stdout)
    new_map = parse_log_pytest(new_stdout)

    fail_to_pass: List[str] = []
    pass_to_pass: List[str] = []
    old_error: List[str] = []
    new_error: List[str] = []

    all_tests = set(old_map.keys()) | set(new_map.keys())
    for t in all_tests:
        os_ = old_map.get(t)
        ns_ = new_map.get(t)

        if os_ == "ERROR":
            old_error.append(t)
        if ns_ == "ERROR":
            new_error.append(t)

        if os_ in {"FAILED", "ERROR"} and ns_ == "PASSED":
            fail_to_pass.append(t)
        if os_ == "PASSED" and ns_ == "PASSED":
            pass_to_pass.append(t)

    fail_to_pass = sorted(fail_to_pass)
    pass_to_pass = sorted(pass_to_pass)
    old_error = sorted(old_error)
    new_error = sorted(new_error)

    if len(fail_to_pass) == 0:
        msg = (
            f"[WARN] no FAIL_TO_PASS tests for repo={repo_short} new_commit={new_commit_hash}\n"
            f"  fail_to_pass={fail_to_pass}\n"
            f"  pass_to_pass={pass_to_pass}\n"
        )
        print(msg, file=sys.stderr)
    
    if len(pass_to_pass) == 0:
        print(
            f"[WARN] no PASS_TO_PASS tests for repo={repo_short} new_commit={new_commit_hash}\n"
            f"  fail_to_pass={fail_to_pass}\n"
            f"  pass_to_pass={pass_to_pass}\n",
            file=sys.stderr,
        )

    if old_error:
        print(
            f"[WARN] old_commit_res_stdout has ERROR tests for repo={repo_short} new_commit={new_commit_hash}: "
            f"{old_error}",
            file=sys.stderr,
        )
    if new_error:
        print(
            f"[WARN] new_commit_res_stdout has ERROR tests for repo={repo_short} new_commit={new_commit_hash}: "
            f"{new_error}",
            file=sys.stderr,
        )

    # 输出结构：FAIL_TO_PASS / PASS_TO_PASS 按你要求存 str(JSON list)
    out: Dict[str, Any] = {
        "repo": repo_full,
        "repo_name": repo_short,
        "base_commit": base_commit,
        "new_commit_hash": new_commit_hash,
        "commit_hash": item.get("commit_hash", new_commit_hash),
        "problem_statement": problem_statement,
        "commit_date": commit_date,
        "docker_image": docker_image,
        "patch": patch,
        "test_patch": test_patch,
        "non_test_file_content": non_test_file_content,
        "test_file_content": test_file_content,
        "modified_not_test_files": modified_not_test_files,
        "modified_test_files": modified_test_files,
        "modified_files_match": modified_files_match,
        "modified_files_diff": modified_files_diff,
        "test_exec_content": test_exec_content,
        "FAIL_TO_PASS": json.dumps(fail_to_pass, ensure_ascii=False),
        "PASS_TO_PASS": json.dumps(pass_to_pass, ensure_ascii=False),
        "old_commit_res_stdout_error": old_error,
        "new_commit_res_stdout_error": new_error,
    }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 r2e-gym-parsed*.json（list of dict）")
    ap.add_argument("--output", required=True, help="输出转换后的 json")
    ap.add_argument("--strict", action="store_true", help="遇到 modified_files 不一致就直接报错退出")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"Input JSON must be a list, got {type(data)}")

    out_list: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        try:
            out_list.append(convert_one_item(item, strict=args.strict))
        except Exception as e:
            print(f"[ERROR] failed at index={i}: {e}", file=sys.stderr)
            raise

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=2, ensure_ascii=False)

    print(f"Done. Converted {len(out_list)} items -> {args.output}")


if __name__ == "__main__":
    main()
