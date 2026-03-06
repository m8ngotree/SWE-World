import litellm
import json
import time
import re
import random
from typing import Dict, Any, Tuple, List, Optional
from r2egym.agenthub.utils.log import get_logger
from transformers import AutoTokenizer


def safe_get_usage_prompt_tokens(resp: Any):
    """
    Try to fetch prompt tokens from litellm response, if available.
    Different providers may return different shapes.
    """
    try:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return None
        # litellm often returns usage as dict-like
        if isinstance(usage, dict):
            return usage.get("prompt_tokens") or usage.get("input_tokens")
        # or an object with attributes
        return getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
    except Exception:
        return None


def safe_get_usage(resp: Any):
    """
    Try to fetch prompt tokens from litellm response, if available.
    Different providers may return different shapes.
    """
    try:
        usage = getattr(resp, "usage", None)
        return usage
    except Exception:
        return None


def safe_load_yaml(yaml_path: str) -> Dict[str, Any]:
    import yaml
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


class SimulatorAgent:
    """
    An agent that uses an LLM to simulate the output of Python code execution,
    leveraging a rich context about the software engineering task.
    """

    def __init__(self, simulator_config: dict = None, logger=None, simulator_config_path: str = None, tokenizer_path: str = "/code/songhuatong/sft_models/qwen_25_32B_mixed_step_reward_30k_dockerllm_128k_json_format"):
        """
        Args:
            simulator_config: A dict containing two keys: 'swt' and 'swr'.
        """
        simulator_config = simulator_config or {}
        self.simulator_config = simulator_config.get("swt", [])
        self.simulator_reward_config = simulator_config.get("swr", [])

        self.simulator_config_path = simulator_config_path
        self.logger = logger if logger else get_logger("SimulatorAgent")

        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        else:
            self.tokenizer = None

        # ===== ONLY MODIFIED: swt/swr yaml keys =====
        if simulator_config_path:  # 有path优先读取path的
            yaml_data = safe_load_yaml(simulator_config_path)
            config_data = (yaml_data or {}).get('swt', [])
            reward_config_data = (yaml_data or {}).get('swr', [])
            if not config_data:
                raise ValueError(f"No models found in {simulator_config_path}")
            self.simulator_config = config_data
            self.simulator_reward_config = reward_config_data
        # ========================================================

        if not self.simulator_config:
            raise ValueError("simulator_config is empty! Cannot initialize SimulatorAgent.")

        # [MODIFIED] The system prompt is enhanced to guide the LLM on output formatting.
        self.system_prompt = """You are an expert Python code execution simulator and a world-class software engineer. 
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

    def update_simulator_config(self):
        if not self.simulator_config_path:
            print("No simulator_config_path provided")
            return
        yaml_data = safe_load_yaml(self.simulator_config_path)
        config_data = (yaml_data or {}).get('swt', [])
        reward_config_data = (yaml_data or {}).get('swr', [])
        if not config_data:
            print(f"No models found in {self.simulator_config_path}")
            return
        if not reward_config_data:
            print(f"No reward_config found in {self.simulator_config_path}")
            return
        self.simulator_config = config_data
        self.simulator_reward_config = reward_config_data

    def _select_model_config(self) -> Tuple[str, str, str]:
        """
        Randomly selects a model configuration from the list.
        Returns: (model_name, base_url, api_key)
        """
        self.update_simulator_config() # 动态更新
        choice = random.choice(self.simulator_config)
        return choice["model_name"], choice.get("base_url"), choice.get("api_key")

    def _select_reward_model_config(self) -> Tuple[str, str, str]:
        """
        Randomly selects a model configuration from the list.
        Returns: (model_name, base_url, api_key)
        """
        self.update_simulator_config() # 动态更新
        choice = random.choice(self.simulator_reward_config)
        return choice["model_name"], choice.get("base_url"), choice.get("api_key")

    def encode_truncate_decode(self, s: str, max_tokens: int = 32768, add_special_tokens: bool = False) -> str:
        """
        先 encode 得到 token ids -> 截断到 max_tokens -> decode 回字符串
        max_tokens 计数是否包含 special tokens 由 add_special_tokens 决定
        """
        if not self.tokenizer:
            return s

        ids = self.tokenizer.encode(s, add_special_tokens=add_special_tokens)
        len1 = len(ids)
        ids = ids[:max_tokens]
        if len1 > len(ids):
            print(f"ha_trunc: {len1} -> {len(ids)}")
        else:
            pass
        out = self.tokenizer.decode(
            ids,
            skip_special_tokens=True,              # 去掉 [CLS]/[SEP]/<s></s> 等
            clean_up_tokenization_spaces=False     # 尽量不“自动美化空格”，更忠实
        )
        return out

    def get_initial_analysis(self, meta: Dict[str, Any]) -> str:
        """
        基于当前任务的关键信息，生成一段简洁但抓住核心的技术分析，
        用作后续模拟执行（SimulatorAgent）的 initial_analysis。
        """

        problem_statement = meta.get("problem_statement", "No problem statement provided.")
        hints_text = meta.get("hints_text", "No human hints provided.")
        repo = meta.get("repo", "Unknown repository")
        patch = meta.get("patch", "No gold patch provided.")
        test_patch = meta.get("test_patch", "No test patch provided.")
        f2p = meta.get("FAIL_TO_PASS", [])
        p2p = meta.get("PASS_TO_PASS", [])

        # 统一转成字符串方便展示
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

        system_prompt = """You are a senior software engineer and bug-fixing expert.

Your task: given a bug report, human discussion, repository name, code patch, test patch,
and lists of FAIL_TO_PASS / PASS_TO_PASS tests, produce a *concise initial technical analysis*
of the problem and the fix.

**Goals of the analysis:**
- Identify the core problem / bug being fixed.
- Explain the key symptoms or incorrect behavior.
- Describe which part of the codebase (modules / functions) is conceptually responsible.
- Summarize the essence of the fix: what is changed and why it fixes the bug.
- Mention how the tests (FAIL_TO_PASS / PASS_TO_PASS) relate to the fix: what behavior they verify.
- Call out any subtle constraints / corner cases that are important.

**Style & format requirements:**
- Output MUST be plain text in English.
- Use 5-10 short bullet points (markdown "- " style).
- Each bullet should be one or two sentences, focused and technical.
- Do NOT repeat the raw diff or test lists; summarize their intent.
- Be specific but not verbose.
"""

        user_prompt = f"""
### Repository
{repo}

### Problem Statement
{problem_statement}

### Human Discussion / Hints
{hints_text}

### Gold Patch (code fix)
```diff
{patch}
```

### Test Patch (changes to tests)

```diff
{test_patch}
```

### FAIL_TO_PASS (should be failing before, passing after the fix)

{f2p_str}

### PASS_TO_PASS (should keep passing after the fix)

{p2p_str}

### Your Task

Produce a concise *initial technical analysis* that captures:

* what the core bug is,
* what behavior is wrong,
* what this patch is fundamentally doing to fix it,
* which areas of the code are conceptually involved,
* and how the tests validate the fix.

Remember: 5–10 bullet points, markdown "- " bullets, plain English, no extra commentary.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        max_retries = 1
        for attempt in range(max_retries):
            # [NEW] Select a model for this request
            current_model_name, current_base_url, current_api_key = self._select_model_config()

            try:
                response = litellm.completion(
                    model=current_model_name,
                    messages=messages,
                    api_base=current_base_url,
                    temperature=0.0,
                    api_key=current_api_key
                )
                raw_content = response.choices[0].message.content or ""

                # 兼容 thinking 模型：去掉 <think> ... </think> 部分
                analysis = raw_content.split("</think>")[-1].strip()
                if not analysis:
                    raise ValueError("Empty initial analysis from LLM")

                return analysis

            except Exception as e:
                self.logger.error(
                    f"Error getting initial analysis on attempt {attempt + 1}: {e}",
                    exc_info=True,
                )
                if attempt == max_retries - 1:
                    # 最后一次失败：返回一个兜底的简单分析，避免上层崩掉
                    fallback = (
                        "- Initial analysis unavailable due to LLM error. "
                        "Assume the patch fixes the described bug and the FAIL_TO_PASS/PASS_TO_PASS "
                        "tests reflect the intended behavior change."
                    )
                    return fallback
                time.sleep(1)


    def get_simulated_feedback(self, context: Dict[str, Any]) -> Tuple[str, str, int]:
        """
        Queries the LLM to get simulated feedback for a Python command.
        """
        start_time = time.time()
        max_retries = 1

        sim_type = context.get("type")

        # [NEW] Flags to control context size reduction
        include_original_content = True
        include_detailed_context = True # Controls initial_analysis and human_hints

        for attempt in range(max_retries):
            # --- [NEW] Build prompt dynamically inside the loop based on flags ---

            # 1. Format the content of the files being executed
            exec_code_blocks = []
            for path, content in context.get('execution_code_content', {}).items():
                exec_code_blocks.append(f"--- Content of `{path}` (the file being executed) ---\n\n{content}\n")

            # 2. Format the original content of all modified files
            # [MODIFIED] Condition based on include_original_content
            if include_original_content:
                original_files_blocks = []
                for path, content in context.get('original_files_content', {}).items():
                    original_files_blocks.append(f"--- Original content of `{path}` (before any changes) ---\n\n{content}\n")
                original_content_str = chr(10).join(original_files_blocks) if original_files_blocks else 'No original files provided.'
            else:
                original_content_str = "(Omitted to reduce context length)"

            if not exec_code_blocks:
                exec_code_block_str = "No explicit execution code content provided."
            else:
                exec_code_block_str = "\n\n".join(exec_code_blocks)

            # [MODIFIED] Determine content for analysis and hints based on include_detailed_context
            init_analysis = context.get('initial_analysis', 'No initial analysis provided.') if include_detailed_context else "(Omitted to reduce context length)"
            human_hints = "No hints provided."
            original_content_str = 'No original files provided.'
            problem_statement = context.get("problem_statement", "No problem description provided.")
            agent_patch = context.get("agent_patch", "No changes from the agent yet.")
            gold_patch = context.get("gold_patch", "No gold patch provided.")
            command_to_simulate = context.get("command_to_simulate", "No command provided.")
            # 3. Assemble the full prompt


            init_analysis = self.encode_truncate_decode(init_analysis, max_tokens=5120) # 5k
            problem_statement = self.encode_truncate_decode(problem_statement, max_tokens=10240) # 10k
            command_to_simulate = self.encode_truncate_decode(command_to_simulate, max_tokens=5120) # 5k
            exec_code_block_str = self.encode_truncate_decode(exec_code_block_str, max_tokens=15360) # 15k
            agent_patch = self.encode_truncate_decode(agent_patch, max_tokens=15360) # 15k
            gold_patch = self.encode_truncate_decode(gold_patch, max_tokens=10240) # 10k

            user_prompt = f"""
### 1. Initial Analysis of the Problem
{init_analysis}

### 2. Problem Description
{problem_statement}

### 3. Human Discussion (Hints)
{human_hints}

### 4. Command to Simulate
```bash
{command_to_simulate}
```

### 5. Content of Code to be Executed
{exec_code_block_str}

### 6. Original Content of Modified Files
{original_content_str}

### 7. Agent's Current Code Modifications (Patch)
```diff
{agent_patch}
```

### 8. Gold Standard Patch (For Your Reference)
```diff
{gold_patch}
```

### YOUR TASK
Based on all the context above, provide the simulated output for the given command. Your response must be only the JSON object, with no other text.
"""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Select a model for this request
            current_model_name, current_base_url, current_api_key = self._select_model_config()
            print(f"current model: {current_model_name}, base_url: {current_base_url}, api_key: {current_api_key}")

            try:
                response = litellm.completion(
                    model=current_model_name,
                    messages=messages,
                    api_base=current_base_url,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    api_key=current_api_key,
                    timeout=650,
                    num_retries=0
                )

                usage = safe_get_usage(response)

                response_content = response.choices[0].message.content # 原始的回复

                sim_result = json.loads(response_content.split("</think>")[-1]) # 对于thinking model原始的回复中可能包含think内容，这部分要去掉，才能json解析成功

                stdout = sim_result.get("stdout", "")
                stderr = sim_result.get("stderr", "")
                exit_code = sim_result.get("exit_code", 1) # Default to 1 (error) if not provided

                combined_output = f"[STDOUT]\n\n{stdout}\n\n[STDERR]\n\n{stderr}"

                self.logger.info(f"[{sim_type}]Simulator LLM processing time: {time.time() - start_time:.2f} seconds")
                return response_content, combined_output.strip(), exit_code

            except Exception as e:
                self.logger.error(f"[{sim_type}]Error processing simulator LLM response on attempt {attempt + 1}: {e}", exc_info=True)
                if attempt == max_retries - 1:
                    error_message = f"[STDOUT]\n\n\n\n[STDERR]\n\nFailed to get environment feedback!" 
                    self.logger.info(f"[{sim_type}]Simulator LLM processing time (failed): {time.time() - start_time:.2f} seconds")
                    return error_message, error_message, -1

                # [NEW] Check for Context Window Error (robust check using name or string)
                is_context_error = (
                    "ContextWindowExceededError" in str(type(e).__name__) or 
                    "ContextWindowExceededError" in str(e) or 
                    "context_length_exceeded" in str(e)
                )

                if is_context_error:
                    self.logger.warning(f"[{sim_type}]Context window exceeded on attempt {attempt + 1}. Adjusting context for next attempt.")

                    if include_original_content:
                        include_original_content = False
                        continue # Retry immediately
                    elif include_detailed_context:
                        self.logger.info(f"[{sim_type}]Reducing context: Removing 'Initial Analysis' and 'Human Hints'.")
                        include_detailed_context = False
                        continue # Retry immediately

                time.sleep(0.5)


    def get_reward_for_test_summary(self, summary: str, f2p_files: List[str], p2p_files: List[str]) -> Tuple[int, str, str]:
        """
        Determines the final reward (0 or 1) based on a summary of test results.
        Returns:
            A tuple of (reward, response_content, reason).
        """
        system_prompt = """You are an expert software engineering test analyst. Your task is to determine if a bug is fixed based on a summary of test results.
The tests are categorized into "FAIL_TO_PASS" (tests that should now pass if the fix is correct) and "PASS_TO_PASS" (regression tests that should continue to pass).

- To award a reward of 1, ALL FAIL_TO_PASS tests must pass, AND ALL PASS_TO_PASS tests must pass.
- If ANY FAIL_TO_PASS test fails, the reward is 0.
- If ANY PASS_TO_PASS test fails, the reward is 0.

Analyze the provided summary and output a single JSON object with two keys: "reward" (an integer, 0 or 1) and "reason" (a brief explanation for your decision). Do not add any text outside the JSON block.
"""
        user_prompt = f"""
### Test Summary
{summary}

### FAIL_TO_PASS Test Files (Must Pass)
{json.dumps(f2p_files, indent=2)}

### PASS_TO_PASS Test Files (Must Pass)
{json.dumps(p2p_files, indent=2)}

### Your Task
Based on the summary, determine the final reward. Provide your answer in the required JSON format.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        start_time = time.time()
        max_retries = 1
        for attempt in range(max_retries):
            # [NEW] Select a model for this request
            current_model_name, current_base_url, current_api_key = self._select_model_config()

            try:
                response = litellm.completion(
                    model=current_model_name,
                    messages=messages,
                    api_base=current_base_url,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    api_key=current_api_key,
                    timeout=650,
                    num_retries=0
                )

                usage = safe_get_usage(response)

                response_content = response.choices[0].message.content

                sim_result = json.loads(response_content.split("</think>")[-1])
                reward = int(sim_result.get("reward", 0))
                reason = str(sim_result.get("reason", "No reason provided."))

                self.logger.info(f"Reward-judge LLM processing time: {time.time() - start_time:.2f} seconds")
                return reward, response_content, reason
            except Exception as e:
                self.logger.error(f"Error processing reward-judge LLM response on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    self.logger.info(f"Reward-judge LLM processing time (failed): {time.time() - start_time:.2f} seconds")
                    return 0, f"Error: Reward-judge LLM failed. Details: {e}", f"Error: Reward-judge LLM failed. Details: {e}"
                time.sleep(1)


    def get_simulated_test_report_and_reward(self, context: Dict[str, Any]) -> Tuple[str, int, str]:
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

        exec_code_blocks = []
        for path, content in context.get('execution_code_content', {}).items():
            exec_code_blocks.append(f"--- Content of `{path}` (the file being executed) ---\n\n{content}\n")

        if not exec_code_blocks:
            exec_code_block_str = "No explicit execution code content provided."
        else:
            exec_code_block_str = "\n\n".join(exec_code_blocks)

        init_analysis = context.get("initial_analysis", "No initial analysis provided.")

        problem_statement = context.get("problem_statement", "No problem description provided.")
        agent_patch = context.get("agent_patch", "No changes from the agent yet.")
        gold_patch = context.get("gold_patch", "No gold patch provided.")
        command_to_simulate = context.get("command_to_simulate", "No command provided.")

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

        init_analysis = self.encode_truncate_decode(init_analysis, max_tokens=5120) # 5k
        problem_statement = self.encode_truncate_decode(problem_statement, max_tokens=10240) # 10k
        command_to_simulate = self.encode_truncate_decode(command_to_simulate, max_tokens=5120) # 5k
        exec_code_block_str = self.encode_truncate_decode(exec_code_block_str, max_tokens=15360) # 15k
        agent_patch = self.encode_truncate_decode(agent_patch, max_tokens=15360) # 15k
        gold_patch = self.encode_truncate_decode(gold_patch, max_tokens=10240) # 10k
        p2p_str = self.encode_truncate_decode(p2p_str, max_tokens=5120) # 5k

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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        start_time = time.time()
        max_retries = 1
        for attempt in range(max_retries):
            # ===== ONLY MODIFIED: reward selection uses swr config =====
            current_model_name, current_base_url, current_api_key = self._select_reward_model_config()
            # ========================================================
            print(f"current model: {current_model_name}, base_url: {current_base_url}, api_key: {current_api_key}")
            try:
                self.logger.info(
                    f"Querying combined test-report & reward LLM, attempt {attempt + 1} using {current_model_name}..."
                )
                response = litellm.completion(
                    model=current_model_name,
                    messages=messages,
                    api_base=current_base_url,
                    temperature=0.7,
                    # response_format={"type": "json_object"},
                    api_key=current_api_key,
                    timeout=1200, # yuanben是900
                    num_retries=0
                )

                usage = safe_get_usage(response)

                response_content = response.choices[0].message.content or ""
                self.logger.info(
                    f"Combined LLM raw response: {response_content}"
                )
                try:
                    sim_result = json.loads(response_content.split("</think>")[-1].strip())

                    test_report = str(sim_result.get("test_report", ""))
                    reward = int(sim_result["reward"])
                except Exception as e:
                    print(f"注意这里reward模拟失败了一次！原因是：{e}；reward模型生成的内容是：{response_content}")
                    reward = -10086
                self.logger.info(
                    f"Combined test-report & reward LLM processing time: {time.time() - start_time:.2f} seconds"
                )
                return test_report, reward, response_content

            except Exception as e:
                self.logger.error(
                    f"Error processing combined test-report & reward LLM response on attempt {attempt + 1}: {e}",
                    exc_info=True,
                )
                if attempt == max_retries - 1:
                    error_msg = f"Error: Combined test-report & reward LLM failed. Details: {e}"
                    self.logger.info(
                        f"Combined test-report & reward LLM processing time (failed): {time.time() - start_time:.2f} seconds"
                    )
                    return "", 0, error_msg
                time.sleep(1)


class EvaluateAgent:
    """
    An agent that uses an LLM as a judge to compare simulated results against
    real execution results and provide a structured evaluation.
    """
    def __init__(self, llm_name: str, llm_base_url: str = None, llm_api_key: str = None, logger=None):
        self.llm_name = llm_name
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.logger = logger if logger else get_logger("EvaluateAgent")

    def evaluate_step_execution(self, real_output: str, real_code: int, sim_output: str, sim_code: int) -> Tuple[str, Dict[str, Any]]:
        """
        [MODIFIED] Asks an LLM to judge if the simulated execution is functionally
        equivalent to the real one, focusing on the overall outcome.
        """
        # [MODIFIED] Simplified system prompt
        system_prompt = """
You are an expert software test analyst. Your task is to determine if a `SIMULATED` command output is functionally equivalent to the `REAL` output.

**Your Goal:** Determine if the simulation was successful. A successful simulation means an agent observing the simulated output would make the same decision as if it observed the real output.

**Evaluation Criteria:**
1.  **Overall Outcome:** Do both results indicate the same outcome (e.g., success, failure, specific error)? The exit codes are a strong hint (`0` for success, non-zero for failure).
2.  **Functional Equivalence:** Are the outputs functionally the same?
    -   **Ignore minor differences**: Timestamps, memory addresses, absolute file paths, and minor whitespace changes are irrelevant.
    -   **Focus on key information**: For test runs, care about which tests passed or failed and the error messages for failures. For tracebacks, care about the exception type and the location of the error. For program output, care about the final result.

Your output MUST be a single JSON object with the following keys:
-   `is_success` (boolean): `true` if the simulation is functionally equivalent to the real execution, `false` otherwise.
-   `reason` (string): A brief explanation for your decision. If not successful, explain the key functional difference. For example: "The simulation correctly identified a test failure, but reported the wrong exception type." or "The simulation succeeded while the real execution failed with a TimeoutError."
"""
        # [MODIFIED] Simplified user prompt
        user_prompt = f"""
### Real Execution Result
**Exit Code:** {real_code}
```
{real_output}
```

### Simulated Execution Result
**Exit Code:** {sim_code}
```
{sim_output}
```

### Your Task
Is the simulated result functionally equivalent to the real result? Provide your evaluation in the required JSON format, with no other text.
"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        try:
            response = litellm.completion(
                model=self.llm_name,
                messages=messages,
                api_base=self.llm_base_url,
                temperature=0.6,
                response_format={"type": "json_object"},
                api_key=self.llm_api_key
            )
            response_content = response.choices[0].message.content
            evaluation_result = json.loads(response_content.split("</think>")[-1])

            if 'is_success' not in evaluation_result or 'reason' not in evaluation_result:
                raise ValueError("LLM response is missing required keys 'is_success' or 'reason'.")

            return response_content, evaluation_result

        except Exception as e:
            self.logger.error(f"Error querying evaluation LLM: {e}")
            return f"Error querying evaluation LLM: {e}", {"is_success": False, "reason": f"Error during LLM evaluation: {e}"}

    def evaluate_reward_calculation(self, real_reward: int, sim_reward: int) -> Dict[str, Any]:
        """
        Judges if the simulated final reward matches the real reward.
        """
        reward_match = (real_reward == sim_reward)
        return {
            "is_success": reward_match,
            "reward_match": reward_match,
            "reason": f"Simulated reward ({sim_reward}) {'matches' if reward_match else 'does not match'} real reward ({real_reward})."
        }


if __name__ == "__main__":
    # Example usage
    yaml_file = "src/simulation/models_yaml/example_world_models.yaml"
    config = safe_load_yaml(yaml_file)

    print(config)
