import json
import time
import re
from typing import Dict, Any, Tuple, List
import os
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("", use_fast=True)

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


def parse_command_output(output_string):
    """
    解析包装脚本的输出字符串，提取 stdout 和 stderr。

    参数:
        output_string (str): 包含 [STDOUT] 和 [STDERR] 标记的完整输出字符串。

    返回:
        tuple: (stdout_content, stderr_content)，均为去除首尾空白的字符串。
    """
    stdout_marker = "[STDOUT]"
    stderr_marker = "[STDERR]"

    # 初始化返回结果
    stdout_content = ""
    stderr_content = ""

    # 1. 提取 STDERR
    # 逻辑：STDERR 在输出的最后部分，位于 [STDERR] 标记之后
    if stderr_marker in output_string:
        # 使用 rsplit 从右侧分割，确保处理的是脚本打印的最后一个标记
        # 防止用户命令本身的输出里恰好包含了 "[STDERR]" 字符串
        parts = output_string.rsplit(stderr_marker, 1)
        
        # parts[0] 是 [STDERR] 之前的内容
        # parts[1] 是 [STDERR] 之后的内容 (即实际的 stderr)
        if len(parts) > 1:
            stderr_content = parts[1].strip()
            remaining_part = parts[0]
        else:
            remaining_part = output_string
    else:
        remaining_part = output_string

    # 2. 提取 STDOUT
    # 逻辑：STDOUT 位于剩余部分的 [STDOUT] 标记之后
    if stdout_marker in remaining_part:
        # 使用 rsplit 从右侧分割，理由同上，取最后一个标记之后的内容
        parts = remaining_part.rsplit(stdout_marker, 1)
        
        if len(parts) > 1:
            stdout_content = parts[1].strip()

    return stdout_content, stderr_content


system_prompt = """You are an expert software engineering test runner and evaluator.
Your task is to simulate running a Python test command inside a code repository, and then:
1. A reasoning section enclosed within `<think>` and `</think>` tags that contains your full internal reasoning process.
2. Produce a realistic test report summarizing which tests pass or fail.
3. Decide a final reward value based on the status of specific tests.

### Key Information You Must Use:
1. **Initial Analysis of the Problem**: This section contains a core analysis of the issue, including the description of the error behavior, the core bug, how the issue manifests, and the intended fix. It is crucial to understanding the problem and will guide the simulation process. Use this to help you quickly identify the issue and how it should be addressed.

2. **Problem Description**: This section describes the specific issue the agent is currently working on and trying to fix. Use this to understand the exact problem the agent is attempting to resolve.

3. **Command to Simulate**: This is the command that will be executed. It contains all the information about what needs to be run, including the files to be executed and any other relevant details. Use this to simulate the execution and generate the correct output. If the "Content of Code to be Executed" is empty, the code is embedded within the command itself.

4. **Content of Code to be Executed**: This is the actual code that will be executed. Pay close attention to this content as the simulated output must strictly correspond to the code being executed. If this section is empty, the specific code to execute is provided in the "Command to Simulate" section.

5. **Agent's Current Code Modifications (Patch)**: This section highlights the changes that the agent has made to the codebase. These changes are the ones you need to analyze carefully to simulate the feedback. Focus on these modifications when generating your simulated output.

6. **Gold Standard Patch (For Your Reference)**: This is the correct solution to the issue. You should compare the agent's current changes with the gold standard solution to ensure that the simulated result is as accurate as possible. If the agent's patch is **functionally equivalent** to the gold patch (i.e., it resolves the issue in the same way), then the simulation feedback should match the expected output as defined by the gold standard.

7. **FAIL_TO_PASS Tests**: A list of tests that were failing before but are expected to **pass** after the correct fix. For the reward to be 1, **every** test in this list must pass.

8. **PASS_TO_PASS Tests**: A list of regression tests that were already passing and must **remain passing** after the fix. For the reward to be 1, **every** test in this list must pass.

### Your Task
Given all of the above context:
* Simulate the execution of the command under the current agent patch.
* Focus especially on the tests in FAIL_TO_PASS and PASS_TO_PASS:
  - If **all** FAIL_TO_PASS tests pass and **all** PASS_TO_PASS tests pass, then the reward **must be 1**.
  - If **any** FAIL_TO_PASS test fails or errors, the reward **must be 0**.
  - If **any** PASS_TO_PASS test fails or errors, the reward **must be 0**.

### Output Format
- **Reasoning (`<think>` block)**: First, provide your detailed internal reasoning, analysis, and step-by-step thought process enclosed within `<think>` and `</think>` tags. This section explains how you arrived at the simulated result.
- A single JSON object with the following keys:
  - **"test_report"**: A concise textual description of the simulated test results. Mention which FAIL_TO_PASS and PASS_TO_PASS tests pass or fail, and any important errors.
  - **"reward"**: An integer, either 0 or 1, following the rules above.

The final structure MUST look conceptually like this:
`<think> ...your full reasoning process here... </think>
{"test_report": "...", "reward": 0/1}`

Be sure to use all the context provided, including any discrepancies between the agent's modifications and the gold standard patch, to generate the most accurate simulated output.
"""


def read_jsonl(path: str):
    samples = []

    r0 = 0
    r1 = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj["real_reward"] == 1:
                r1 += 1
            else:
                r0 += 1
            samples.append(obj)
    print(f"[read_jsonl] {path} -> {len(samples)} samples")
    print(f"[read_jsonl] {path} -> r1: {r1}, percentage: {r1 / (r0 + r1)}, r0: {r0}, percentage: {r0 / (r0 + r1)}")
    return samples

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    print(f"[read_json] {path} -> {len(obj)} samples")
    return obj



def get_simulated_feedback(sample):
    use_flag = True
    context = sample["context"]
    # --- [NEW] Build a much richer user prompt from the detailed context ---
        
    # 1. Format the content of the files being executed
    exec_code_blocks = []
    for path, content in context.get('execution_code_content', {}).items():
        exec_code_blocks.append(f"--- Content of `{path}` (the file being executed) ---\n\n{content}\n")
    
    if not exec_code_blocks:
        exec_code_block_str = "No explicit execution code content provided."
    else:
        exec_code_block_str = "\n\n".join(exec_code_blocks)
        
    instance_id = sample["instance_id"].lower()

    init_analysis = context["initial_analysis"]
    
    problem_statement = context.get("problem_statement", "No problem description provided.")
    agent_patch = context.get("agent_patch", "No changes from the agent yet.")
    gold_patch = context.get("gold_patch", "No gold patch provided.")
    command_to_simulate = context.get("command_to_simulate", "No command provided.")

    if init_analysis == "":
        print(f"[get_simulated_feedback] {instance_id} has no init_analysi")
        
    f2p = context.get("FAIL_TO_PASS", [])
    p2p = context.get("PASS_TO_PASS", [])

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

    # exec_code_block_str = encode_truncate_decode(exec_code_block_str, max_tokens=65536)
    # gold_patch = encode_truncate_decode(gold_patch, max_tokens=32768)
    # agent_patch = encode_truncate_decode(agent_patch, max_tokens=65536)
    # p2p_str = encode_truncate_decode(p2p_str, max_tokens=10240)
        
    # 3. Assemble the full prompt
    user_prompt = f"""
### 1. Initial Analysis of the Problem
{init_analysis}

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

### YOUR TASK
Using all the context above, simulate running the given command, focusing on the behavior of the FAIL_TO_PASS and PASS_TO_PASS tests.

Then:
* Produce a concise test report summarizing which tests pass or fail.
* Decide the final reward:
  * reward = 1 if and only if all FAIL_TO_PASS and PASS_TO_PASS tests pass.
  * reward = 0 otherwise.

Your answer must include a `<think>...</think>` block containing your full reasoning process, followed immediately by a single valid JSON object with keys `"test_report"` and `"reward"`.
"""

    sim_cot = sample["cot_completion"]
    sim_error = sample["cot_error"]

    if sim_error: # 确保 sim_error 为 None
        print(f"sim_error:{sim_error}")
        return [], False, sample["real_reward"]
    
    if "Initial analysis unavailable due to LLM err" in init_analysis:
        print("这个数据没有init_analysis，跳过！！")
        return [], False, sample["real_reward"]

 
    real_output_dict_str = sample["real_output_dict_str"]

    try:
        real_obj = json.loads(real_output_dict_str)
        real_output_dict_str = json.dumps(real_obj, ensure_ascii=False)
    except Exception:
        # 这里最好直接跳过或记录 bad case
        print(f"11111 error")
        use_flag = False


    assistant_content = f"<think>{sim_cot}</think>\n{real_output_dict_str}"

    messages = {
        "input":[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content},
    ]}
    # print(user_prompt)
    return messages, use_flag, sample["real_reward"]

# ### 6. Original Content of Modified Filesw
# {chr(10).join(original_files_blocks) if original_files_blocks else 'No files were modified.'}


def process_data(file_list, output_jsonl):
    """
    folders: 顶层 folder 路径列表
    output_jsonl: 最终输出的 .jsonl 文件路径
    """
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)
    
    
    prompt_cnt = 0
    r0 = 0
    r1 = 1
    
    with open(output_jsonl, "w", encoding="utf-8") as outf:
        for file in tqdm(file_list, desc="Processing files"):
            if file.endswith(".jsonl"):
                data = read_jsonl(file)
            else:
                data = read_json(file)

            for item in tqdm(data, desc="Processing items"):
                messages, use_flag,real_reward = get_simulated_feedback(item)
                if use_flag:
                    prompt_cnt+=1
                    if real_reward == 1:
                        r1 += 1
                    else:
                        r0 += 1
                    outf.write(json.dumps(messages, ensure_ascii=False) + "\n")
    
    print(f"prompt_cnt: {prompt_cnt}")
    print(f"r0: {r0}, percent: {r0/prompt_cnt:.2f}, r1: {r1}, percent: {r1/prompt_cnt:.2f}")


if __name__ == "__main__":
    # 1. 把你的 folder 路径写在这里

    file_list= [
        
    ]

    # 2. 指定最终输出的 jsonl
    out_file = ""
    
    process_data(file_list, out_file)
    print("Done! Check", out_file)
        