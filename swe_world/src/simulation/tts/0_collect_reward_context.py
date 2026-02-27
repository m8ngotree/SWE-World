#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect the last item from all *_simulation_data_collection.json files
under a given directory, and keep those whose
data['context']['type'] == 'reward_calculation'.

Usage:
    python collect_reward_last_items.py \
        --input_dir /path/to/search \
        --output_file output.json
"""

import os
import json
import argparse


def find_target_files(root_dir):
    """
    Recursively find all files ending with
    '_simulation_data_collection.json'
    """
    target_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith("_simulation_data_collection.json"):
                target_files.append(os.path.join(root, file))
    return target_files


def read_last_item(file_path):
    """
    Read the last item from a JSON file.
    Supports:
        1. JSON array format
        2. JSONL format
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return None

        # Try JSON array
        try:
            data = json.loads(content)
            if isinstance(data, list) and len(data) > 0:
                return data[-1]
        except json.JSONDecodeError:
            pass

        # Try JSONL (last line)
        lines = content.splitlines()
        if len(lines) > 0:
            try:
                return json.loads(lines[-1])
            except json.JSONDecodeError:
                return None

    except Exception as e:
        print(f"[Error reading {file_path}] {e}")

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Root directory to search")
    parser.add_argument("--output_file", required=True, help="Output JSON file")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file

    files = find_target_files(input_dir)
    print(f"Found {len(files)} candidate files.")

    collected = []

    for file_path in files:
        last_item = read_last_item(file_path)

        if not last_item:
            continue

        try:
            if (
                "context" in last_item
                and isinstance(last_item["context"], dict)
                and last_item["context"].get("type") == "reward_calculation"
            ):
                collected.append(last_item)
        except Exception as e:
            print(f"[Error processing {file_path}] {e}")

    print(f"Collected {len(collected)} matching items.")

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(collected, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()