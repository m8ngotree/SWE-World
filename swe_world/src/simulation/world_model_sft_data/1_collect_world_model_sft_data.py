import json
import time
import re
from typing import Dict, Any, Tuple, List
import os

import shlex

# ---------------------------------------------------------------------------
# Language-aware utilities — single source of truth in lang_utils.
# PYTHON_CMD_PATTERN is kept as a local alias for backward compatibility.
# ---------------------------------------------------------------------------
from r2egym.agenthub.environment.lang_utils import (
    CMD_PATTERNS,
    Language,
    detect_language,
    is_execution_command,
    is_python_execution_command,
    extract_test_report as _lang_extract_test_report,
)

PYTHON_CMD_PATTERN = CMD_PATTERNS[Language.PYTHON]


from typing import Tuple


def trim_test_report_trailing_noise(
    test_report: str,
    equal_marker: str = "=",
) -> Tuple[str, bool]:
    """
    尝试裁剪 test_report 末尾的“噪声”。

    规则：
    1. 从下往上找到最后一行包含 equal_marker（默认 '===='）的行。
    2. 看这行之后的 tail 是否包含 'Traceback'：
       - 若 tail 中包含 'Traceback'，且不包含
         'AssertionError: plugin is not registered'：认为是重要错误，不裁剪。
       - 若 tail 中不包含 'Traceback'，或者虽有 Traceback 但包含
         'AssertionError: plugin is not registered'：视为可忽略，裁剪掉 tail。
    
    返回:
        (new_report, changed)
        - new_report: 裁剪后的报告（或原始报告）
        - changed: 是否执行了裁剪
    """
    original_report = test_report

    lines = test_report.splitlines(keepends=True)
    if not lines:
        return original_report, False

    # 找最后一个包含等号的行
    last_equal_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if equal_marker in lines[i]:
            last_equal_idx = i
            break

    if last_equal_idx is None:
        # 没有等号行，按照你的规则就不裁剪
        return original_report, False

    tail_text = "".join(lines[last_equal_idx + 1:])

    # 如果尾部包含 Traceback，但错误是“plugin is not registered”，也可以忽略
    has_traceback = "Traceback" in tail_text
    has_plugin_not_registered = "AssertionError: plugin is not registered" in tail_text

    if has_traceback and not has_plugin_not_registered:
        # 有 Traceback 且不是我们豁免的那个错误，不能删
        return original_report, False

    # 否则可以裁剪尾部，保留到等号行
    trimmed_report = "".join(lines[:last_equal_idx + 1]).rstrip()
    return trimmed_report, True


def extract_test_report(log: str, lang: Language = Language.PYTHON) -> Tuple[bool, str, str]:
    """
    Language-aware test report extractor — thin wrapper around lang_utils.

    Returns:
        (success, reason, test_report)
    """
    return _lang_extract_test_report(log, lang=lang)


def has_inline_code(code: str) -> bool:
    """
    根据 shell 命令字符串，粗略判断是否包含“内联代码执行”：
      - python -c '...'
      - heredoc: << EOF / <<'EOF' / <<EOF 等

    Args:
        code: 一整条 shell 命令字符串

    Returns:
        bool: 如果检测到内联代码，返回 True；否则 False。
    """
    try:
        cmd_parts = shlex.split(code)
    except ValueError:
        # shlex 失败时退化为简单的空格切分
        cmd_parts = code.split()

    for part in cmd_parts:
        try:
            inner_cmd = shlex.split(part)
        except ValueError:
            inner_cmd = part.split()

        i = 0
        while i < len(inner_cmd):
            inner_part = inner_cmd[i]

            # 1. 检测 heredoc：<<、<<EOF、<<'EOF' 等
            #   原代码只判断 == "<<", 这里稍微放宽一点：只要以 "<<" 开头就认为是 heredoc
            if inner_part.startswith("<<"):
                return True

            # 2. 检测 python -c 形式的内联代码
            if (
                inner_part in ("python", "python3")
                and i + 1 < len(inner_cmd)
                and inner_cmd[i + 1] == "-c"
            ):
                return True

            i += 1

    return False
import random

# is_python_execution_command is imported from lang_utils at the top of this
# file.  The local definition is removed to avoid divergence.

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

Your output MUST be a single JSON object containing 'stdout', 'stderr', and 'exit_code'. Do not add any explanations or text outside of this JSON block.

### Key Information You Must Use:
1. **Initial Analysis of the Problem**: This section contains a core analysis of the issue, including the description of the error behavior, the core bug, how the issue manifests, and the intended fix. It is crucial to understanding the problem and will guide the simulation process. Use this to help you quickly identify the issue and how it should be addressed.

2. **Problem Description**: This section describes the specific issue the agent is currently working on and trying to fix. Use this to understand the exact problem the agent is attempting to resolve.

3. **Command to Simulate**: This is the command that will be executed. It contains all the information about what needs to be run, including the files to be executed and any other relevant details. Use this to simulate the execution and generate the correct output. If the "Content of Code to be Executed" is empty, the code is embedded within the command itself.

4. **Content of Code to be Executed**: This is the actual code that will be executed. Pay close attention to this content as the simulated output must strictly correspond to the code being executed. If this section is empty, the specific code to execute is provided in the "Command to Simulate" section.

5. **Agent's Current Code Modifications (Patch)**: This section highlights the changes that the agent has made to the codebase. These changes are the ones you need to analyze carefully to simulate the feedback. Focus on these modifications when generating your simulated output.

6. **Gold Standard Patch (For Your Reference)**: This is the correct solution to the issue. You should compare the agent's current changes with the gold standard solution to ensure that the simulated result is as accurate as possible. If the agent's patch is **functionally equivalent** to the gold patch (i.e., it resolves the issue in the same way), then the simulation feedback should match the expected output as defined by the gold standard.

### Your Task:
- Use all the above information to generate the most realistic and accurate simulated output.
- For commands that reproduce errors (e.g., `python reproduce_issue.py`), refer to the **Initial Analysis of the Problem** and the **Problem Description** to understand the nature of the error. Then, simulate the result based on the actual code content and the problem analysis.
- For test commands (e.g., commands that include `pytest`), carefully compare the agent's modifications with the gold standard patch. If the tests are "FAIL_TO_PASS" tests, analyze whether the agent's changes fix the issue as described in the tests. For "PASS_TO_PASS" tests, ensure that the agent's changes do not break existing functionality.
- It is important to note that the execution result must strictly follow the current code content being executed, and no fabricated test output should be added. The same test case may have multiple versions, and the test content may change across versions. Therefore, each case should be analyzed specifically based on its content.
- Your simulated output should reflect the most likely and realistic results of the command execution based on the context provided. Be precise and clear in your simulated outputs, focusing on realistic error messages, test outputs, or successful execution results.

### Format of the Output:
- **stdout**: The standard output of the command, if applicable. For example, in the case of a test run, this should reflect the results of the tests, such as which tests passed or failed.
- **stderr**: Any error messages that might be produced by the command. For example, if the command produces a syntax error or another exception, this should contain the appropriate traceback or error message.
- **exit_code**: The exit code of the command. `0` indicates success, while `1` (or any other non-zero value) indicates failure.

### Example Scenarios:
- **Successful Command Execution**: If the command executes successfully, `stdout` should contain the output generated by executing the code, and `exit_code` should be `0`. Ensure that the output corresponds directly to the expected result of the executed code.
- **Runtime Error (e.g., SyntaxError)**: If there is a runtime error, such as a syntax error, the `stderr` should contain the error traceback, and `exit_code` should be `1`.
- **Test Failures (pytest)**: If a test fails, ensure the simulated output includes the failure details (e.g., pytest output) and an appropriate `exit_code`.

Be sure to use all the context provided, including any discrepancies between the agent's modifications and the gold standard patch, to generate the most accurate simulated output.
"""


def get_simulated_feedback(sample):
        use_flag = True
        context = sample["context"]
        # --- [NEW] Build a much richer user prompt from the detailed context ---
        
        # Detect language from context (instance_id contains repo info when available)
        instance_id = sample.get("instance_id", "")
        repo_name = context.get("repo", instance_id.split("__")[0] if "__" in instance_id else "")
        sample_lang = detect_language(repo_name=repo_name)

        # Filter: only simulate execution commands for the detected language
        command = context.get('command_to_simulate', 'No command provided.')
        if not is_execution_command(command, lang=sample_lang):
            print(f"Not an execution command [{sample_lang.name}]: [{command}], skip this sample")
            use_flag = False
            return None, use_flag
        
        if context["execution_code_content"] == {}:
            if has_inline_code(command):
                # print(f"this command has inline code: {command}")
                pass
            else:
                print(f"this command has no exec code: {command}, exec content: {context['execution_code_content']},skip this sample")
                use_flag = False
                return None, use_flag

        # 1. Format the content of the files being executed
        exec_code_blocks = []
        for path, content in context.get('execution_code_content', {}).items():
            exec_code_blocks.append(f"--- Content of `{path}` (the file being executed) ---\n\n{content}\n")
        
        # 2. Format the original content of all modified files
        original_files_blocks = []
        for path, content in context.get('original_files_content', {}).items():
            original_files_blocks.append(f"--- Original content of `{path}` (before any changes) ---\n\n{content}\n")
        
        init_analysis = context.get('initial_analysis', 'No initial analysis provided.') if include_detailed_context else "(Omitted to reduce context length)"
        # 3. Assemble the full prompt
        user_prompt = f"""
### 1. Initial Analysis of the Problem
{init_analysis}

### 2. Problem Description
{context.get('problem_statement', 'No problem description provided.')}

### 3. Command to Simulate
```bash
{context.get('command_to_simulate', 'No command provided.')}
```

### 4. Content of Code to be Executed
{chr(10).join(exec_code_blocks)}

### 5. Agent's Current Code Modifications (Patch)
```diff
{context.get('agent_patch', 'No changes from the agent yet.')}
```

### 6. Gold Standard Patch (For Your Reference)
```diff
{context.get('gold_patch', 'No gold patch provided.')}
```"""
        real_output =sample["real_output"]
        
        end_text = "\n### YOUR TASK\nBased on all the context above, provide the simulated output for the given command. Your response must be only the JSON object, with no other text."
        # print(sample.keys())

        real_stdout, real_stderr = parse_command_output(real_output)
        real_output_dict = {
            "stdout": real_stdout,
            "stderr": real_stderr,
            "exit_code": sample["real_error_code"]
        }
        real_output_dict_str = json.dumps(real_output_dict)

        sample["real_stdout"] = real_stdout
        sample["real_stderr"] = real_stderr

        messages = {"input":[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": real_output_dict_str},
        ]}

     
        messages = {"input": messages}
        # print(user_prompt)
        return messages, use_flag

# ### 6. Original Content of Modified Filesw
# {chr(10).join(original_files_blocks) if original_files_blocks else 'No files were modified.'}


def get_simulated_test_report_and_reward(
        sample
    ):
        use_flag = True
        context = sample["context"]

        real_output = sample["real_output"]
        real_reward = sample["real_reward"]

        # Detect language for this sample (same logic as get_simulated_feedback)
        instance_id = sample.get("instance_id", "")
        repo_name = context.get("repo", instance_id.split("__")[0] if "__" in instance_id else "")
        sample_lang = detect_language(repo_name=repo_name)

        is_success, fail_reason, test_report = extract_test_report(real_output, lang=sample_lang)
        test_report = test_report.rstrip()
        if is_success:
            if not test_report.endswith("="):
                test_report, trimmed_status = trim_test_report_trailing_noise(test_report)
                if not trimmed_status:
                    # print(f"Test report does not end with '==', test_report:\n{test_report}\neeeeeeeeeed")
                    use_flag = False
                    return None, False
        else:
            # print(f"extract_test_report failed: {fail_reason}, real_output:\n{real_output}\nttttttttttttttd")
            use_flag = False
            return None, False
        
        instance_id = sample["instance_id"].lower()
        
        init_analysis = context.get('initial_analysis', 'No initial analysis provided.') if include_detailed_context else "(Omitted to reduce context length)"

        system_prompt = """You are an expert software engineering test runner and evaluator.
Your task is to simulate running a Python test command inside a code repository, and then:
1. Produce a realistic test report summarizing which tests pass or fail.
2. Decide a final reward value based on the status of specific tests.

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
You must output a single JSON object with the following keys:
- **"test_report"**: A concise textual description of the simulated test results. Mention which FAIL_TO_PASS and PASS_TO_PASS tests pass or fail, and any important errors.
- **"reward"**: An integer, either 0 or 1, following the rules above.

Be sure to use all the context provided, including any discrepancies between the agent's modifications and the gold standard patch, to generate the most accurate simulated output.
"""
        # Format execution code content
        exec_code_blocks = []
        for path, content in context.get("execution_code_content", {}).items():
            exec_code_blocks.append(
                f"--- Content of `{path}` (the file being executed) ---\n\n{content}\n"
            )

        # Format original files content
        original_files_blocks = []
        for path, content in context.get("original_files_content", {}).items():
            original_files_blocks.append(
                f"--- Original content of `{path}` (before any changes) ---\n\n{content}\n"
            )
        original_content_str = (
            "\n".join(original_files_blocks) if original_files_blocks else "No original files provided."
        )

        # Format FAIL_TO_PASS / PASS_TO_PASS
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

        user_prompt = f"""
### 1. Initial Analysis of the Problem
{init_analysis}

### 2. Problem Description
{context.get("problem_statement", "No problem description provided.")}

### 3. Command to Simulate
```bash
{context.get("command_to_simulate", "No command provided.")}
```

### 4. Content of Code to be Executed
{chr(10).join(exec_code_blocks) if exec_code_blocks else "No explicit execution code content provided."}

### 5. Agent's Current Code Modifications (Patch)
```diff
{context.get("agent_patch", "No changes from the agent yet.")}
```

### 6. Gold Standard Patch (For Your Reference)
```diff
{context.get("gold_patch", "No gold patch provided.")}
```

### 7. FAIL_TO_PASS Tests (Must All Pass for reward=1)
{f2p_str}

### 8. PASS_TO_PASS Tests (Must All Pass for reward=1)
{p2p_str}"""     

        end_text_reward = """
### YOUR TASK
Using all the context above, simulate running the given command, focusing on the behavior of the FAIL_TO_PASS and PASS_TO_PASS tests.

Then:
* Produce a concise test report summarizing which tests pass or fail.
* Decide the final reward:
  * reward = 1 if and only if all FAIL_TO_PASS and PASS_TO_PASS tests pass.
  * reward = 0 otherwise.

Your answer must be a single JSON object with keys "test_report", "reward". Do not include any text outside of that JSON object.
"""

        real_output_dict = {
            "test_report": test_report,
            "reward": real_reward
        }
        real_output_dict_str = json.dumps(real_output_dict)

        messages = {"input":[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": real_output_dict_str}
        ]}

      
        
        return messages, use_flag

def process_folders(folders, output_jsonl):
    """
    folders: 顶层 folder 路径列表
    output_jsonl: 最终输出的 .jsonl 文件路径
    """
    result_dict ={}
    
    
    # 如果输出文件已存在，先删掉，避免重复追加
    # exit()
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)
    
    reward_cnt = 0
    # not_step = 0 
    step_cnt = 0   
    collect_cnt = 0
    with open(output_jsonl, "w", encoding="utf-8") as outf:
        for folder in folders:
            # 找到 folder 下所有 .json 文件（递归子目录也可）
            json_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.json')]
            all_num = len(json_files)
            jsonl_num_now = 0
            for json_file in json_files:
                jsonl_num_now+=1
                print(f"正在处理的文件({jsonl_num_now}/{all_num})：{json_file}")
                with open(json_file, "r", encoding="utf-8") as f:
                    try:
                        samples = json.load(f)
                    except Exception as e:
                        print(f"解析 JSON 失败：{json_file}, error: {e}")
                        continue  # 直接跳过这个文件
                messages_list = []
                for sample in samples:
                    if sample["context"]["type"] == "reward_calculation":
                        continue # don't process reward_calculation
                        messages,use_flag = get_simulated_test_report_and_reward(sample)
                        if use_flag:
                            reward_cnt+=1

                    elif sample["context"]["type"] == "step_execution":
                        # not_step+=1            
                        # continue # don't process step_execution

                        messages,use_flag = get_simulated_feedback(sample)
                        if use_flag:
                            step_cnt+=1
                            messages_list.append(messages)
                        # continue
                        # pass
                    else:
                        print(f"sample type error: {sample['context']['type']}")
                        continue

                    # if use_flag:
                    #     collect_cnt+=1
                    #     outf.write(json.dumps(messages, ensure_ascii=False) + "\n")

                messages_list_sample = random.sample(messages_list, min(6, len(messages_list)))
                for message in messages_list_sample:
                    collect_cnt+=1
                    outf.write(json.dumps(messages, ensure_ascii=False) + "\n")

    print(f"reward_cnt: {reward_cnt}")
    # print(f"not_step: {not_step}")
    print(f"step_cnt: {step_cnt}")
    print(f"collect_cnt: {collect_cnt}")

if __name__ == "__main__":
    # 1. 把你的 folder 路径写在这里
    folder_list= [
        # dir of inference result
]

    # 2. 指定最终输出的 jsonl
    out_file = ""
    
    process_folders(folder_list, out_file)
    print("Done! Check", out_file)