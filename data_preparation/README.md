# SWE-World Data Preparation

This document describes the data preprocessing pipeline used in **SWE-World**.
The goal is to download raw SWE datasets, convert them into JSON format, clone the required repositories with full commit history, and prepare cached specifications for efficient world-model simulation.

---

# 1. Download Raw Datasets

First, download open-source SWE datasets.

```bash
bash data_preparation/download_swe_datasets.sh
```

This script downloads the raw open-source SWE datasets (mainly in **Parquet** format).

---

# 2. Convert Parquet to JSON

The downloaded datasets are stored in **Parquet** format and need to be converted into **JSON**.

```bash
python data_preparation/merge_parquet_to_json.py
```

This script merges and converts the raw dataset files into JSON format.

---

# 3. (Optional) Convert R2E Dataset to SWE-Bench Style

For inference compatibility, the **R2E-Gym dataset** can optionally be converted into a **SWE-Bench-style format**.

Scripts:

```bash
python data_preparation/parse_r2e_str_value_json.py
python data_preparation/parse_r2e_gym_to_swebench_style.py
```

This step ensures the dataset follows a consistent schema compatible with SWE-Bench style tasks.

---

# 4. Repository Lists

The directory `repos_list/` contains the repository lists for different datasets:

```
data_preparation/repos_list/

r2e_gym_repos.json (R2E-Gym-Subset: https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset)
swe_gym_repos.json (SWE-Gym: https://huggingface.co/datasets/SWE-Gym/SWE-Gym)
swe_rebench_repos.json (SWE-rebench: https://huggingface.co/datasets/nebius/SWE-rebench)
swebench_verified_repos.json (SWE-Bench-Verified: https://huggingface.co/datasets/R2E-Gym/SWE-Bench-Verified)
```

Each file records the GitHub repositories associated with the dataset tasks.

---

# 5. Download Repositories with Full Commit History

After downloading the datasets, clone all required repositories locally.
These repositories must include **full commit histories**, which are required for **world-model simulation**.

Scripts:

```bash
bash data_preparation/crawl_repos.sh
```

or

```bash
python data_preparation/crawl_repos.py
```

This step downloads the repositories listed in `repos_list/`.

---

# 6. Add Local Repository Paths to Dataset

Once repositories are downloaded, we add their **local directory paths** to each dataset instance.

```bash
python data_preparation/add_local_dir.py
```

This step links dataset entries with their corresponding local repositories.

---

# 7. (Optional) Offline Unit Test Generation (Optional)
Generate unit tests for `swe-gym`, `swebench-verified`, and `swe-rebench` to facilitate direct loading during evaluation. 
This process generates two types of caches:
1. JSON files for unit tests mapped to each `instance_id` within the specified `base_dir`.
2. A dataset containing the `make_test_spec` field. 

During inference, the system first checks if the dataset contains the `make_test_spec` field, then checks for the cached directory in the workspace. If neither is found, it reverts to the default loading method.

**Note:** If `base_dir` is specified, you must update the corresponding `base_dir` in `./swe_world/src/r2egym/agenthub/runtime/docker.py`.

```bash
python data_preparation/make_test_spec.py \
    --base_dir /your/custom/cache/path \
    --data_file_path /your/path/to/data.json
```

For a completed conversion demo, refer to: `./data_examples/inference_data/ood_data`
The corresponding unit tests cache is located at: `./data_examples/inference_data/ood_data_unit_tests_cache

---

# Pipeline Overview

The full preprocessing pipeline is:

```
Download datasets
        ↓
Convert Parquet → JSON
        ↓
(Optional) Convert R2E → SWE-Bench style
        ↓
Load repository lists
        ↓
Clone repositories with full commit history
        ↓
Add local repository paths to dataset
        ↓
(Optional) Generate cached test specifications
```

After these steps, the dataset is fully prepared for **SWE-World simulation, training, and evaluation pipelines**.
