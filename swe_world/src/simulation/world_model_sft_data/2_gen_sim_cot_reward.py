import argparse
import json
import os
import sys
import time
import yaml
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import litellm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("YOUR_TOKENIZER", use_fast=True)

def read_jsonl(path: str):
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(obj)
    print(f"[read_jsonl] {path} -> {len(samples)} samples")
    return samples

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    print(f"[read_json] {path} -> {len(obj)} samples")
    return obj



COT_SYSTEM_PROMPT = """You are an expert software engineering test runner simulator and a world-class SWE bug-fix evaluator.

In this task, the TRUE test execution outcome (a JSON containing the final `test_report` and `reward`) is already provided in the input.
You MUST copy that JSON exactly at the end of your answer.

However, your reasoning MUST be written as if you do NOT know the true outcome.
You must only use the provided context (analysis, problem statement, command, code, patches, test lists, etc.) to do a forward simulation
that *could have produced* the true outcome.

The input context you receive is organized into multiple sections. Each section has a specific meaning and purpose:

1. Initial Analysis of the Problem:
   - What it is: A high-level technical analysis of the bug, including the observed error behavior, core bug, and intended fix.
   - How to use it: Use this to quickly understand what is broken, what the correct behavior should be, and what the fix is aiming to change.

2. Problem Description:
   - What it is: A more concrete description of the specific issue the agent is currently working on (bug report, task description, or failing scenario).
   - How to use it: Use this to understand the exact wrong behavior being observed and the expected correct behavior.

3. Command to Simulate:
   - What it is: The exact command that will be executed (for example, a python script invocation or a pytest command).
   - How to use it: Use this to determine the entry point, what tests or scripts are run, and how execution will proceed (including arguments, options, and test discovery).

4. Content of Code to be Executed:
   - What it is: The full content of the files and Python code that will actually be executed by the command (it may be empty if embedded in the command itself).
   - How to use it: Use this as the ground truth for runtime semantics, carefully simulating how this code will behave when run.

5. Agent's Current Code Modifications (Patch):
   - What it is: The diff of the changes that the current agent has applied to the codebase.
   - How to use it: Use this to understand exactly what logic changed, which modules/functions are affected, and how control flow or data handling is now different.

6. Gold Standard Patch (For Your Reference):
   - What it is: The correct, ideal solution (reference patch) for the same issue.
   - How to use it: Use this to compare with the agent's patch, see whether they are functionally equivalent or where they differ, and reason about possible behavioral differences.

7. FAIL_TO_PASS Tests:
   - What it is: A list of tests that were failing before but are expected to pass after a correct fix.
   - How to use it: Use these to judge whether the patch fixes the targeted bug: if any FAIL_TO_PASS test fails, the reward must be 0.

8. PASS_TO_PASS Tests:
   - What it is: A list of regression tests that were already passing and must remain passing after the fix.
   - How to use it: Use these to check for regressions: if any PASS_TO_PASS test fails, the reward must be 0.

9. True Execution Result (JSON):
   - What it is: The real outcome of running the command in a real docker environment, containing `test_report` and `reward`.
   - How to use it: You must copy this JSON exactly at the end of your answer, but you must NOT use it to guide, justify, or leak into your reasoning. Your reasoning must be written as if you have not seen this result.

---

### Your Task

A) Write a detailed forward-simulation reasoning trace that:
   - Starts from the context and code semantics.
   - Compares agent patch vs gold patch to judge whether they are functionally equivalent.
   - Predicts (as expectations) which FAIL_TO_PASS and PASS_TO_PASS tests should pass/fail and why.
      - For the PASS_TO_PASS test cases, you should carefully analyze and simulate whether the current agent patch would break any existing functionality in the repository, and use that to determine whether the tests would pass.
      - For the FAIL_TO_PASS test cases, you should carefully analyze and simulate whether the current agent patch can fix the repository’s existing issue, and use that to determine whether the tests would pass.
      - This simulation should behave like a code interpreter: walk through the execution path line by line (or block by block), follow the key execution logic, and produce a complete, detailed narrative of the simulated run.
   - Based on the analysis, produce the test report for the PASS_TO_PASS and FAIL_TO_PASS tests. If all of them pass, the reward is 1; otherwise, it is 0.

B) CRITICAL CAUSALITY REQUIREMENT (NO LEAKING / NO BACKWARD EXPLANATION)
   - Your reasoning MUST NOT mention that you have been given the true result.
   - Your reasoning MUST NOT quote, paraphrase, or reference any concrete lines from the true `test_report`.
   - Do NOT write post-hoc explanations like “we see X in the output, therefore …”.
   - If you mention outcomes, phrase them as expectations derived from the code, e.g.:
     “This test is expected to pass because …”, “It likely fails due to …”.
   - Do NOT copy any distinctive strings from the True Execution Result into your reasoning.

C) After you finish the reasoning, append the provided True Execution Result JSON EXACTLY as-is.
   - Do not reformat it, do not add keys, do not change whitespace, do not add extra fields.
   - Do not add any extra commentary after the JSON.

---

### Output Format

You must output exactly:

<sim_reasoning>
...plain-text forward simulation reasoning...
</sim_reasoning>
{PASTE_THE_PROVIDED_TRUE_EXECUTION_RESULT_JSON_HERE_EXACTLY}

Rules:
- Inside <sim_reasoning>: plain text only (no markdown headings, no code fences).
- After </sim_reasoning>: immediately paste the JSON exactly.
- Do not output anything else.
"""



def encode_truncate_decode(s: str, max_tokens: int = 32768, add_special_tokens: bool = False) -> str:
        """
        先 encode 得到 token ids -> 截断到 max_tokens -> decode 回字符串
        max_tokens 计数是否包含 special tokens 由 add_special_tokens 决定
        """

        ids = tokenizer.encode(s, add_special_tokens=add_special_tokens)
        len1 = len(ids)
        ids = ids[:max_tokens]
        if len1 > len(ids):
            print(f"ha·_trunc: {len1} -> {len(ids)}")
        else:
            print(f"not_trunc")
        out = tokenizer.decode(
            ids,
            skip_special_tokens=True,              # 去掉 [CLS]/[SEP]/<s></s> 等
            clean_up_tokenization_spaces=False     # 尽量不“自动美化空格”，更忠实
        )
        return out


def build_user_prompt(example: Dict[str, Any]) -> str:
    """
    将单条样本打包成 user prompt 文本。
    """
    ctx = example.get("context", {}) or {}

    initial_analysis = ctx.get("initial_analysis", "No initial analysis provided.")
    problem_statement = ctx.get("problem_statement", "No problem description provided.")
   #  human_hints = ctx.get("human_hints", "No human hints provided.")

    agent_patch = ctx.get("agent_patch", "No agent patch provided.")
    gold_patch = ctx.get("gold_patch", "No gold patch provided.")
   #  test_patch = ctx.get("test_patch", "No test patch provided.")

    repo = ctx.get("repo", "Unknown repository")

    f2p = ctx.get("FAIL_TO_PASS", [])
    p2p = ctx.get("PASS_TO_PASS", [])

    # 统一转成字符串展示
    if not isinstance(f2p, str):
        try:
            f2p_str = json.dumps(f2p, indent=2, ensure_ascii=False)
        except Exception:
            f2p_str = str(f2p)
    else:
        f2p_str = f2p

    if not isinstance(p2p, str):
        try:
            p2p_str = json.dumps(p2p, indent=2, ensure_ascii=False)
        except Exception:
            p2p_str = str(p2p)
    else:
        p2p_str = p2p

    execution_code_content = ctx.get("execution_code_content", {}) or {}
    command_to_simulate = ctx.get("command_to_simulate", "No command provided.")

    # 格式化被执行的代码内容
    exec_code_blocks: List[str] = []
    for path, content in execution_code_content.items():
        exec_code_blocks.append(
            f"--- Content of `{path}` (the file being executed) ---\n"
            f"\n{content}\n"
        )
    if not exec_code_blocks:
        exec_code_block_str = "No explicit execution file content provided."
    else:
        exec_code_block_str = "\n\n".join(exec_code_blocks)

    true_execution_result_json = example["real_output_dict_str"]


    exec_code_block_str = encode_truncate_decode(exec_code_block_str, max_tokens=65536)
    gold_patch = encode_truncate_decode(gold_patch, max_tokens=32768)
    agent_patch = encode_truncate_decode(agent_patch, max_tokens=65536)
    p2p_str = encode_truncate_decode(p2p_str, max_tokens=10240)

    # true_execution_result_json_1 = encode_truncate_decode(true_execution_result_json, max_tokens=32768)

    return f"""### 1. Initial Analysis of the Problem
{initial_analysis}

### 2. Problem Description
{problem_statement}

### 3. Command to Simulate
```bash
{command_to_simulate}
```

### 4. Content of Code to be Executed

{exec_code_block_str}

### 5. Agent's Current Code Modifications (Patch)

```diff
{agent_patch}
```

### 6. Gold Standard Patch (For Your Reference)

```diff
{gold_patch}
```

### 7. FAIL_TO_PASS Tests (Must All Pass for reward=1)

{f2p_str}

### 8. PASS_TO_PASS Tests (Must All Pass for reward=1)

{p2p_str}

### 9. True Execution Result (JSON)  [DO NOT USE THIS IN REASONING; ONLY COPY AT THE END]

{true_execution_result_json}

### YOUR TASK

Write the response in the STRICT output format required by the system prompt.
Remember:

* Reasoning must be a forward simulation based only on context.
* Do NOT reference/quote the true result in reasoning.
* Append the true JSON exactly after </sim_reasoning>.
"""



def strip_internal_think_tags(content: str) -> str:
    """
    如果底层模型是一个 "thinking" 模型，返回内容里可能会包含它自己的 <think>...</think>。
    为了保证对外只保留我们想要的内容，这里把前面的 internal <think> 部分去掉。
    """
    return content.strip()


def extract_sim_reasoning_content(text: str) -> str:
    """
    提取 <sim_reasoning>...</sim_reasoning> 中间的内容。
    """
    start_tag = "<sim_reasoning>"
    end_tag = "</sim_reasoning>"

    text = text.strip()
    text = text.split("</think>")[-1].strip()

    start_idx = text.find(start_tag)
    end_idx = text.rfind(end_tag)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        # 没有找到标签，就退化为整体 strip
        return text.strip()

    inner = text[start_idx + len(start_tag): end_idx]
    return inner.strip()



def generate_cot_for_example(
    example: Dict[str, Any],
    model_config_list: List[Dict[str, str]], # [MODIFIED] 接收配置列表
    temperature: float,
    max_tokens: int,
    max_retries: int,
    request_timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
    """
    调用 LLM，为一条样本生成 <sim_reasoning>... 思维链。
    支持在多个模型节点间随机分发。
    """
    user_prompt = build_user_prompt(example)
    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_err: Optional[str] = None
    completion_text: Optional[str] = None
    raw_content: Optional[str] = None
    
    # 确保有可用的模型配置
    if not model_config_list:
        return {**example, "cot_completion": "", "cot_error": "No model config provided", "cot_raw_content": ""}

    for attempt in range(1, max_retries + 1):
        # [MODIFIED] 每次尝试前，随机选择一个模型配置，实现负载均衡
        current_config = random.choice(model_config_list)
        current_model = current_config["model_name"]
        current_base_url = current_config.get("base_url")
        current_api_key = current_config.get("api_key")

        try:
            resp = litellm.completion(
                model=current_model,
                messages=messages,
                api_base=current_base_url,
                api_key=current_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=request_timeout,
            )
            raw_content = resp.choices[0].message.content or ""
            # 去掉内部 <think>，只保留我们需要的 <sim_reasoning> 包裹内容
            content = extract_sim_reasoning_content(raw_content)
            completion_text = content
            last_err = None
            break
        except Exception as e:  # noqa: BLE001
            # 记录下当前使用的是哪个模型失败了
            print(f"[generate_cot_for_example] {current_model} failed: {repr(e)}")
            last_err = f"gen cot attempt {attempt} failed on {current_model}: {repr(e)}"
            time.sleep(1 + attempt * 0.5)

    out = dict(example)
    if completion_text is not None:
        out["cot_completion"] = completion_text
        out["cot_error"] = None
        out["cot_raw_content"] = raw_content
    else:
        out["cot_completion"] = ""
        out["cot_error"] = last_err or "unknown error"
        out["cot_raw_content"] = raw_content

    return out


# ===================== 4. 多进程并行处理 =====================

def process_wrapper(
        example: Dict[str, Any],
        model_config_list: List[Dict[str, str]], # [MODIFIED] 接收列表
        temperature: float,
        max_tokens: int,
        max_retries: int,
        request_timeout: Optional[int],
    ) -> Dict[str, Any]:
    """
    单独拿出来作为多进程入口函数（必须是 top-level 才能被 pickle）。
    只负责处理数据，不写入文件。
    """
    return generate_cot_for_example(
        example=example,
        model_config_list=model_config_list, # [MODIFIED] 传递列表
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        request_timeout=request_timeout,
    )

# ===================== 5. 主函数 =====================

def main() -> None:
    parser = argparse.ArgumentParser(
    description="Generate simulator chain-of-thought (<sim_reasoning>) for data using litellm."
    )
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径，内容为 list[object]")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径，每行一条样本")
    
    # [MODIFIED] 替换原有的单模型参数为配置文件路径
    parser.add_argument("--model-config", required=True, help="模型配置文件路径 (YAML格式)，包含 models 列表")
    
    parser.add_argument("--workers", type=int, default=4, help="并行进程数（默认 4）")
    parser.add_argument("--temperature", type=float, default=0.3, help="采样 temperature（默认 0.3）")
    parser.add_argument("--max-tokens", type=int, default=8192, help="单条回复最大 token 数（默认 4096）")
    parser.add_argument("--max-retries", type=int, default=3, help="请求失败重试次数（默认 3）")
    parser.add_argument("--request-timeout", type=int, default=1800, help="请求超时时间秒数，可选")
    parser.add_argument("--start-idx", type=int, default=0, help="从第几条数据开始处理")
    parser.add_argument("--end-idx", type=int, default=-1, help="处理到第几条数据结束")

    args = parser.parse_args()

    print(f"111111111111")

    # [MODIFIED] 加载模型配置
    with open(args.model_config, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
        model_config_list = config_data.get('models', [])
        if not model_config_list:
            print(f"ERROR: No models found in {args.model_config}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(model_config_list)} models from {args.model_config}")

    # 读取输入数据
    if args.input.endswith(".json"):
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif args.input.endswith(".jsonl"):
        data = read_jsonl(args.input)
    else:
        print("ERROR: input file must be JSON or JSONL.", file=sys.stderr)
        sys.exit(1)

    end = args.end_idx if args.end_idx != -1 else len(data)
    end = min(end, len(data))
    data = data[args.start_idx:end]
    print(f"start index: {args.start_idx}, end index: {args.end_idx}")



    num_examples = len(data)
    print(f"Loaded {num_examples} examples from {args.input}")
    

    # 断点继续
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            existing_data = [json.loads(line) for line in f]
        # 注意：这里假设数据里有 instance_id 字段
        # existing_idxs = [item.get("instance_id") for item in existing_data if "instance_id" in item]
        existing_idxs = []
        for item in existing_data:
            if "instance_id" in item:
                # existing_idxs.append(item.get("instance_id"))
                if item["cot_error"] is None: # 不能返回时错误的
                    #print(f'existing output item {item["instance_id"]} has cot_error: {item["cot_error"]}')
                    existing_idxs.append(item.get("instance_id"))

        print(f"Found {len(existing_idxs)} existing examples in {args.output}")
        data = [item for item in data if item.get("instance_id") not in existing_idxs]
        print(f"after flitering existing examples, {len(data)} examples left to process")
        num_examples = len(data)
    else:
        print(f"{args.output} not exists, start processing from scratch")

    # 并行处理
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx: Dict[Any, int] = {}
        for idx, example in tqdm(enumerate(data), total=num_examples, desc="Generating COTs"):
            example["real_output_dict"] = json.loads(example["real_output_dict_str"])
            future = executor.submit(
                process_wrapper,
                example,
                model_config_list, # [MODIFIED] 传入模型列表
                args.temperature,
                args.max_tokens,
                args.max_retries,
                args.request_timeout,
            )
            future_to_idx[future] = idx

        processed = 0
        for future in tqdm(as_completed(future_to_idx), total=num_examples, desc="Waiting for results"):
            idx = future_to_idx[future]
            try:
                result = future.result()
                # 拿到结果后立即写入文件
                with open(args.output, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False))
                    f.write("\n")
            except Exception as e:  # noqa: BLE001
                # 兜底：如果子进程挂了，就把错误记录在该样本上
                example = data[idx]
                result = dict(example)
                result["cot_completion"] = ""
                result["cot_error"] = f"worker crashed: {repr(e)}"
                instance_id = example.get('instance_id', 'unknown_id')
                print(f"[worker crash] instance_id: {instance_id} {result['cot_error']}")
                
                # 异常情况下也要写入文件
                # with open(args.output, "a", encoding="utf-8") as f:
                #     f.write(json.dumps(result, ensure_ascii=False))
                #     f.write("\n")
            
            processed += 1
            if processed % 10 == 0 or processed == num_examples:
                print(f"Processed {processed}/{num_examples} examples")

    print(f"Done. All {num_examples} examples with simulator COT written to {args.output}")

if __name__ == "__main__":
   main()