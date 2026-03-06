import os
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from typing import Tuple, List, Dict
from tqdm import tqdm  

def clone_one_repo(repo: str, target_dir: str, github_prefix: str) -> Tuple[str, str, str]:
    """
    克隆单个仓库。

    返回:
        (repo, status, message)
        status: "success" | "failed" | "skipped"
    """
    # 构造 SSH / HTTPS 地址
    repo = repo.strip()
    if not repo:
        return repo, "skipped", "Empty repo name"

    ssh_or_https_url = f"{github_prefix}{repo}.git"

    # 本地目录名：用 repo 的最后一段，比如 astropy/astropy -> astropy__astropy
    repo_name = repo.replace("/", "__")
    repo_path = os.path.join(target_dir, repo_name)

    # 目录已存在：跳过
    if os.path.exists(repo_path):
        return repo, "skipped", f"Path already exists: {repo_path}"

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    try:
        result = subprocess.run(
            ["git", "clone", ssh_or_https_url, repo_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,  # 我们自己检查 returncode
        )
        if result.returncode == 0:
            return repo, "success", f"Cloned to {repo_path}"
        else:
            msg = f"git clone failed, code={result.returncode}, stderr={result.stderr.strip()}"
            return repo, "failed", msg
    except Exception as e:
        return repo, "failed", f"Exception: {e}"


def load_repos_from_json(json_path: str) -> List[str]:
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x) for x in data]
    elif isinstance(data, dict):
        # 如果你的 JSON 是 { "name": "owner/repo", ... } 这种形式
        # 默认取字典的 value
        return [str(v) for v in data.values()]
    else:
        raise ValueError("Unsupported JSON format, expected list or dict")


def main():
    parser = argparse.ArgumentParser(description="Multi-threaded GitHub repo cloner")
    parser.add_argument(
        "--json",
        required=True,
        help="JSON 文件路径，内容为仓库名列表，如 ['astropy/astropy', 'numpy/numpy']",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="所有仓库 clone 到的本地目标目录",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并发线程数（默认 8）",
    )
    parser.add_argument(
        "--https",
        action="store_true",
        help="使用 HTTPS 而不是 SSH（默认用 SSH: git@github.com:owner/repo.git）",
    )

    args = parser.parse_args()

    if args.https:
        github_prefix = "https://github.com/"
    else:
        github_prefix = "git@github.com:"

    json_path = args.json
    target_dir = args.target
    max_workers = args.workers

    print(f"📂 JSON 文件: {json_path}")
    print(f"📁 目标目录: {target_dir}")
    print(f"🔢 并发线程数: {max_workers}")
    print(f"🔗 前缀: {github_prefix}")

    repos = load_repos_from_json(json_path)
    print(f"📜 共 {len(repos)} 个仓库待处理")

    results: List[Tuple[str, str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_repo: Dict = {
            executor.submit(clone_one_repo, repo, target_dir, github_prefix): repo
            for repo in repos
        }

        for future in tqdm(as_completed(future_to_repo), total=len(repos), desc="Cloning repos", ncols=90):
            repo = future_to_repo[future]
            try:
                r_repo, status, message = future.result()
            except Exception as e:
                r_repo, status, message = repo, "failed", f"Future exception: {e}"
            results.append((r_repo, status, message))
            prefix = {
                "success": "✅",
                "failed": "❌",
                "skipped": "⚠️",
            }.get(status, "•")
            print(f"{prefix} {r_repo}: {message}")

    # 统计汇总
    success_count = sum(1 for _, s, _ in results if s == "success")
    failed_count = sum(1 for _, s, _ in results if s == "failed")
    skipped_count = sum(1 for _, s, _ in results if s == "skipped")

    print("\n===== Summary =====")
    print(f"✅ 成功: {success_count}")
    print(f"⚠️ 跳过(已存在/无效): {skipped_count}")
    print(f"❌ 失败: {failed_count}")

    if failed_count > 0:
        print("\n❌ 失败的仓库列表:")
        for repo, status, msg in results:
            if status == "failed":
                print(f" - {repo}: {msg}")


if __name__ == "__main__":
    main()
