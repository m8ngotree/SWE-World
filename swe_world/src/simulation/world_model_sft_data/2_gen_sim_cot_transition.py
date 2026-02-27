
"""
使用一个“长思考”模型，为已有的 (context + real_stdout/real_stderr/real_error_code) 数据
补全一段思维链（chain-of-thought），只输出思维链本身，不包含真实输出 JSON。

模型输出目标格式：

    <sim_reasoning>这里是一整段详细的思维链...</sim_reasoning>

注意：
- 不能用 <think>...</think>，这是推理模型自己的系统级思考格式。
- 教师模型在输入中可以看到真实结果，但在输出中只写思维链。

用法示例：

python generate_cot_traces_only_reasoning.py \
  --input data_with_real_outputs.json \
  --output data_with_cot.jsonl \
  --model-config models.yaml \
  --workers 4 \
  --temperature 0.3

依赖：
- pip install litellm pyyaml
"""

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


# ===================== 1. 思维链补全的 system prompt =====================

COT_SYSTEM_PROMPT = """You are an expert Python code execution simulator and a world-class software engineer.

In this task, you are NOT asked to guess the result of running code from scratch — the true execution result
(stdout, stderr, exit_code) is already provided in the input and will be copied at the end of your answer.

However, your internal reasoning MUST be written as if you do NOT know the true execution result.
You should only use the provided context (code, patches, tests, descriptions, etc.) to simulate
what will happen when the command is executed.

The input context you receive is organized into multiple sections. Each section has a specific meaning and purpose:

1. Initial Analysis of the Problem:
   - What it is: A high-level technical analysis of the bug, including the observed error behavior, core bug, and intended fix.
   - How to use it: Use this to quickly understand what is broken, what the correct behavior should be, and what the fix is aiming to change.

2. Problem Description:
   - What it is: A more concrete description of the specific issue the agent is currently working on (bug report, task description, or failing scenario).
   - How to use it: Use this to understand the exact wrong behavior being observed and the expected correct behavior.

3. Human Discussion (Hints):
   - What it is: Human notes, comments, or discussions about the bug and its solution.
   - How to use it: Use this to pick up extra insights, corner cases, and design constraints that influence how the fix should behave.

4. Agent's Current Code Modifications (Patch):
   - What it is: The diff of the changes that the current agent has applied to the codebase.
   - How to use it: Use this to understand exactly what logic changed, which modules/functions are affected, and how control flow or data handling is now different.

5. Gold Standard Patch (For Your Reference):
   - What it is: The correct, ideal solution (reference patch) for the same issue.
   - How to use it: Use this to compare with the agent's patch, see whether they are functionally equivalent or where they differ, and reason about possible behavioral differences.

6. Original Content of Modified Files:
   - What it is: The full content of the relevant files before any changes were applied.
   - How to use it: Use this to understand the original behavior and to see, by comparison with the patches, how the behavior has been changed.

7. Content of Code to be Executed:
   - What it is: The full content of the files and Python code that will actually be executed by the command.
   - How to use it: Use this as the ground truth for the runtime semantics, carefully simulating how this code will behave when run.

8. Command to Simulate:
   - What it is: The exact command that is executed (for example, a python script invocation or a pytest command).
   - How to use it: Use this to determine the entry point, what tests or scripts are run, and how execution will proceed (including arguments, options, and test discovery).

In addition, you are given the True Execution Result:
- A JSON object containing `stdout`, `stderr`, and `exit_code`.
- This is the real outcome of running the command in that context.
- You must copy these values exactly at the end of your answer, but you must NOT use them to guide or justify your reasoning.
  Your reasoning should be written as if you are predicting the result solely from the context, without seeing the actual outcome.

---

### Your Task

Using the input context and your understanding of Python execution:

- Walk through a detailed, step-by-step reasoning process that predicts what will happen when the command is run.
  - Understand the high-level bug and intended fix from the initial analysis, problem description, and human discussion.
  - Analyze the agent's patch and how it differs from both the original code and the gold patch.
  - Carefully analyze the code that will actually be executed, as well as the command that triggers it, and fully simulate the code's runtime behavior step by step:
    - If the command runs tests, reason about how tests are collected, how the modified code affects each test, and which tests are expected to pass or fail.
    - If the command runs a script, reason through the control flow, data transformations, and error handling, and identify where prints or tracebacks will occur.
    - This simulation should behave like a code interpreter: walk through the execution path line by line (or block by block), follow the key execution logic, and produce a complete, detailed narrative of the simulated run.
  - From this simulation, arrive at what you expect the stdout, stderr, and exit_code to be.

Important causality requirement:

- Your reasoning must be written as a forward simulation: starting from the given code, patches, and tests, and reasoning towards the true execution result.
- Do NOT write post-hoc explanations like "all 8 tests passed, so …" or "we see this warning in the output, therefore …".
- Do NOT reference the fact that you have been given the true execution result, and do NOT describe your reasoning as "explaining" an already-known output.
- If you mention test outcomes or outputs, phrase them as expectations derived from the code (e.g., "this test is expected to pass because …"), not as observations of an already-run command.

---

### Output Format

Output the response in the following strict format:

<sim_reasoning>
[Step-by-step execution tracing based solely on the input context to produce the true execution result.]
</sim_reasoning>[Insert the provided True Execution Result JSON exactly as is]

**Requirements:**
- The content inside `<sim_reasoning>` must be plain text analysis.
- The JSON must be appended immediately after `</sim_reasoning>`.
- Do not add any other markdown, headings, or explanatory text outside these blocks."""


# ===================== 2. 构造 user prompt（拆分 context 各字段） =====================

def build_user_prompt(example: Dict[str, Any]) -> str:
    """
    将单条样本打包成 user prompt 文本。
    """
    ctx = example.get("context", {}) or {}

    initial_analysis = ctx.get("initial_analysis", "No initial analysis provided.")
    problem_statement = ctx.get("problem_statement", "No problem description provided.")
    human_hints = ctx.get("human_hints", "No human hints provided.")

    agent_patch = ctx.get("agent_patch", "No agent patch provided.")
    gold_patch = ctx.get("gold_patch", "No gold patch provided.")
    test_patch = ctx.get("test_patch", "No test patch provided.")

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

    original_files_content = ctx.get("original_files_content", {}) or {}
    execution_code_content = ctx.get("execution_code_content", {}) or {}
    command_to_simulate = ctx.get("command_to_simulate", "No command provided.")

    # 格式化原始文件内容
    original_files_blocks: List[str] = []
    for path, content in original_files_content.items():
        original_files_blocks.append(
            f"--- Original content of `{path}` (before any changes) ---\n"
            f"\n{content}\n"
        )
    if not original_files_blocks:
        original_files_block_str = "No files were modified."
    else:
        original_files_block_str = "\n\n".join(original_files_blocks)

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

    # 真实执行结果
    real_stdout = example.get("real_stdout", "")
    real_stderr = example.get("real_stderr", "")
    real_error_code = example.get("real_error_code", 1)

    return f"""
You are given one training example for a Python code execution simulator.

The information is structured into the same sections described in your system instructions.

### 1. Initial Analysis of the Problem
{initial_analysis}

### 2. Problem Description
{problem_statement}

### 3. Human Discussion (Hints)
{human_hints}

### 4. Agent's Current Code Modifications (Patch)
```diff
{agent_patch}
````

### 5. Gold Standard Patch (For Your Reference)

```diff
{gold_patch}
```

### 6. Original Content of Modified Files

{original_files_block_str}

### 7. Content of Code to be Executed

{exec_code_block_str}

### 8. Command to Simulate

```bash
{command_to_simulate}
```

### 9. Additional Repository & Test Context

#### Repository

{repo}

#### Test Patch (changes to tests)

```diff
{test_patch}
```

#### FAIL_TO_PASS (should be failing before, passing after the fix)

{f2p_str}

#### PASS_TO_PASS (should keep passing after the fix)

{p2p_str}

### 10. True Execution Result

This is the actual result of running the command in that context:

{{
  "stdout": {real_stdout},
  "stderr": {real_stderr},
  "exit_code": {real_error_code}
}}

### 11. Your Task

Your task is to complete the internal chain-of-thought as specified in the system instructions.
"""

# ===================== 3. 调用 LLM 生成一条样本的 COT =====================

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

    # 如果输出文件已存在，先清空 (注释掉以保留原有逻辑)
    # if os.path.exists(args.output):
    #     print(f"Warning: Output file {args.output} already exists. Overwriting.")
    #     open(args.output, "w").close()

    # 读取输入数据
    if args.input.endswith(".json"):
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif args.input.endswith(".jsonl"):
        data = read_jsonl(args.input)
    else:
        print("ERROR: input file must be JSON or JSONL.", file=sys.stderr)
        sys.exit(1)

    data = data[args.start_idx:min(args.end_idx, len(data))]
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