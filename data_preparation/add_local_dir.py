import json
import os
from tqdm import tqdm


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"read json file: {file_path}, size: {len(data)}")
    return data


def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"write json file: {file_path}, size: {len(data)}")


input_file = "YOUR_FILE_PATH"
output_file = "YOUR_FILE_ADD_LOCAL_REPO_PATH"

# 存储相应数据集的仓库的路径
target_dir = ""

data = read_json(input_file)

failed_num = 0
repo_name_set = set()
data_new = []

for item in tqdm(data):

    # 读取 repo_name 或 repo
    repo_raw = item.get("repo") or item.get("repo_name")

    if repo_raw is None:
        print("repo field missing:", item)
        failed_num += 1
        continue

    repo_name_set.add(repo_raw)

    # 替换 / -> __
    repo_name = repo_raw.replace("/", "__")

    # 构造 local_repo_path
    repo_path = f"{target_dir}/{repo_name}"
    item["local_repo_path"] = repo_path

    # 检查 repo 是否存在
    if not os.path.exists(repo_path):
        failed_num += 1
        print(f"repo: {repo_raw} does not exist.")
    else:
        data_new.append(item)

print(f"failed_num: {failed_num}")
print(f"repo_name_set size: {len(repo_name_set)}")

write_json(output_file, data_new)