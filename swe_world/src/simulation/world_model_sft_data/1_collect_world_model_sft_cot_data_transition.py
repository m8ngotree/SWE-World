import json
import time
import re
from typing import Dict, Any, Tuple, List
import os
from tqdm import tqdm



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


system_prompt = """You are an expert Python code execution simulator and a world-class software engineer. 
Your task is to predict the output of a given Python command within a specific code repository context.
Analyze all the provided information: the initial analysis, the problem description, human hints, the agent's current changes, the ideal "gold" solution, and the original content of the modified files.

Before giving the final answer, you MUST first think through the reasoning process in your mind and then provide the user with the simulated output. Your response MUST consist of two parts in the following order:
1. A reasoning section enclosed within `<think>` and `</think>` tags that contains your full internal reasoning process, e.g. `<think> reasoning process here </think>`.
2. Immediately after the closing `</think>` tag, a single JSON object containing only 'stdout', 'stderr', and 'exit_code'.

Do not add any explanations or text outside of the `<think>...</think>` block and the JSON block.

### Key Information You Must Use:
1. **Initial Analysis of the Problem**: This section contains a core analysis of the issue, including the description of the error behavior, the core bug, how the issue manifests, and the intended fix. It is crucial to understanding the problem and will guide the simulation process. Use this to help you quickly identify the issue and how it should be addressed.

2. **Problem Description**: This section describes the specific issue the agent is currently working on and trying to fix. Use this to understand the exact problem the agent is attempting to resolve.

3. **Human Discussion (Hints)**: This section includes the human discussion or thoughts that went into solving the problem. It can help you understand the process of identifying and fixing the issue, and provide you with insights into why the fix works or any special considerations to keep in mind.

4. **Agent's Current Code Modifications (Patch)**: This section highlights the changes that the agent has made to the codebase. These changes are the ones you need to analyze carefully to simulate the feedback. Focus on these modifications when generating your simulated output.

5. **Gold Standard Patch (For Your Reference)**: This is the correct solution to the issue. You should compare the agent's current changes with the gold standard solution to ensure that the simulated result is as accurate as possible. If the agent's patch is **functionally equivalent** to the gold patch (i.e., it resolves the issue in the same way), then the simulation feedback should match the expected output as defined by the gold standard.

6. **Original Content of Modified Files**: These are the contents of the files before the modifications were made. This helps you understand what the agent has changed and how the current code compares to the original codebase.

7. **Content of Code to be Executed**: This is the actual code that will be executed. Pay close attention to this content as the simulated output must strictly correspond to the code being executed. If this section is empty, the specific code to execute is provided in the "Command to Simulate" section.

8. **Command to Simulate**: This is the command that will be executed. It contains all the information about what needs to be run, including the files to be executed and any other relevant details. Use this to simulate the execution and generate the correct output. If the "Content of Code to be Executed" is empty, the code is embedded within the command itself.

### Your Task:
- Use all the above information to generate the most realistic and accurate simulated output.
- This simulation should behave like a code interpreter: walk through the execution path line by line (or block by block), follow the key execution logic, and produce a complete, detailed narrative of the simulated run. This detailed narrative MUST appear inside the `<think>` and `</think>` block.
- For commands that reproduce errors (e.g., `python reproduce_issue.py`), refer to the **Initial Analysis of the Problem** and the **Problem Description** to understand the nature of the error. Then, simulate the result based on the actual code content and the problem analysis.
- For test commands (e.g., commands that include `pytest`), carefully compare the agent's modifications with the gold standard patch. If the tests are "FAIL_TO_PASS" tests, analyze whether the agent's changes fix the issue as described in the tests. For "PASS_TO_PASS" tests, ensure that the agent's changes do not break existing functionality.
- It is important to note that the execution result must strictly follow the current code content being executed, and no fabricated test output should be added. The same test case may have multiple versions, and the test content may change across versions. Therefore, each case should be analyzed specifically based on its content.
- Your simulated output should reflect the most likely and realistic results of the command execution based on the context provided. Be precise and clear in your simulated outputs, focusing on realistic error messages, test outputs, or successful execution results.

### Format of the Output:
- **Reasoning (`<think>` block)**: First, provide your detailed internal reasoning, analysis, and step-by-step thought process enclosed within `<think>` and `</think>` tags. This section explains how you arrived at the simulated result.
- **stdout**: In the JSON object that follows the `</think>` tag, this field is the standard output of the command, if applicable. For example, in the case of a test run, this should reflect the results of the tests, such as which tests passed or failed.
- **stderr**: In the same JSON object, this field contains any error messages that might be produced by the command. For example, if the command produces a syntax error or another exception, this should contain the appropriate traceback or error message.
- **exit_code**: In the same JSON object, this field is the exit code of the command. `0` indicates success, while `1` (or any other non-zero value) indicates failure.

The final structure MUST look conceptually like this:
`<think> ...your full reasoning process here... </think>
{"stdout": "...", "stderr": "...", "exit_code": 0}`

### Example Scenarios:
- **Successful Command Execution**: If the command executes successfully, the `<think>` block should contain the reasoning that leads you to this conclusion, walking through the execution path in a code-interpreter-like manner. `stdout` should contain the output generated by executing the code, and `exit_code` should be `0`. Ensure that the output corresponds directly to the expected result of the executed code.
- **Runtime Error (e.g., SyntaxError)**: If there is a runtime error, such as a syntax error, the `<think>` block should explain why this error occurs based on the code analysis, again behaving like a line-by-line (or block-by-block) interpreter of the execution path. The `stderr` field in the JSON should contain the error traceback, and `exit_code` should be `1`.
- **Test Failures (pytest)**: If a test fails, the `<think>` block should explain which part of the code or patch causes the failure, how the execution flows to that point, and how it manifests. The simulated output in `stdout` should include the failure details (e.g., pytest output) and an appropriate `exit_code`.

Be sure to use all the context provided, including any discrepancies between the agent's modifications and the gold standard patch, to generate the most accurate simulated output. Remember: apart from the `<think>...</think>` reasoning block and the final JSON object, you MUST NOT output any other text.
"""


def read_jsonl(path: str):
    samples = []
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



def get_simulated_feedback(sample):
    use_flag = True
    context = sample["context"]
    # --- [NEW] Build a much richer user prompt from the detailed context ---
        
    # 1. Format the content of the files being executed
    exec_code_blocks = []
    for path, content in context.get('execution_code_content', {}).items():
        exec_code_blocks.append(f"--- Content of `{path}` (the file being executed) ---\n\n{content}\n")
        
    # 2. Format the original content of all modified files
    original_files_blocks = []
    for path, content in context.get('original_files_content', {}).items():
        original_files_blocks.append(f"--- Original content of `{path}` (before any changes) ---\n\n{content}\n")
        
    instance_id = sample["instance_id"].lower()
    init_analysis = context["initial_analysis"]

    if init_analysis == "":
        print(f"[get_simulated_feedback] {instance_id} has no init_analysi")
        
    
    # 3. Assemble the full prompt
    user_prompt = f"""### 1. Initial Analysis of the Problem
{init_analysis}

### 2. Problem Description
{context.get('problem_statement', 'No problem description provided.')}

### 3. Human Discussion (Hints)
{context.get('human_hints', 'No hints provided.')}

### 4. Agent's Current Code Modifications (Patch)
```diff
{context.get('agent_patch', 'No changes from the agent yet.')}
```

### 5. Gold Standard Patch (For Your Reference)
```diff
{context.get('gold_patch', 'No gold patch provided.')}
```

### 6. Original Content of Modified Files
{chr(10).join(original_files_blocks) if original_files_blocks else 'No original files provided.'}

### 7. Content of Code to be Executed
{chr(10).join(exec_code_blocks)}

### 8. Command to Simulate
```bash
{context.get('command_to_simulate', 'No command provided.')}
```

### YOUR TASK
Based on all the context above, provide the simulated output for the command in section #8.
"""
    real_stdout = sample.get("real_stdout", "")
    real_stderr = sample.get("real_stderr", "")
    real_error_code = sample.get("real_error_code", 1)

    sim_cot = sample["cot_completion"]
    sim_error = sample["cot_error"]

    assert sim_error is None # 确保 sim_error 为 None

    real_output_dict = {
        "stdout": real_stdout,
        "stderr": real_stderr,
        "exit_code": real_error_code
    }
    real_output_dict_str = json.dumps(real_output_dict)

    assistant_content = f"<think>{sim_cot}</think>\n{real_output_dict_str}"

    messages = {
        "input":[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content},
    ]}
    # print(user_prompt)
    return messages, use_flag

def process_data(file_list, output_jsonl):
    """
    folders: 顶层 folder 路径列表
    output_jsonl: 最终输出的 .jsonl 文件路径
    """
    
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)
    
    
    prompt_cnt = 0
    with open(output_jsonl, "w", encoding="utf-8") as outf:
        for file in tqdm(file_list, desc="Processing files"):
            if file.endswith(".jsonl"):
                data = read_jsonl(file)
            else:
                data = read_json(file)

            for item in tqdm(data, desc="Processing items"):
                messages, use_flag = get_simulated_feedback(item)
                if use_flag:
                    prompt_cnt+=1
                    outf.write(json.dumps(messages, ensure_ascii=False) + "\n")
    
    print(f"prompt_cnt: {prompt_cnt}")


if __name__ == "__main__":
    # 1. 把你的 folder 路径写在这里
    file_list= [
       
    ]

    # 2. 指定最终输出的 jsonl
    out_file = ""
    

    process_data(file_list, out_file,init_json_list)
    print("Done! Check", out_file)
        