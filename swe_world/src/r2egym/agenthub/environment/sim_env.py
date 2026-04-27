import os
import time
import gym
import logging
import shlex
from typing import Dict, Any, Tuple, List
from pathlib import Path
import re  # [FIX] Added missing import
import json  # [NEW] For parsing FAIL_TO_PASS/PASS_TO_PASS if stored as JSON string
import random  # [NEW] For random sampling PASS_TO_PASS tests
import concurrent.futures
from tqdm import tqdm
from r2egym.agenthub.action import Action
from r2egym.agenthub.observation import Observation
from r2egym.agenthub.runtime.local import LocalRuntime
from r2egym.agenthub.agent.simulator import SimulatorAgent
from r2egym.agenthub.agent.commands import Command, ParseCommand, ParseCommandBash
from r2egym.agenthub import SUPPORTED_REPOS, SKIP_FILES, SKIP_FILES_NEW, CMD_TIMEOUT
from r2egym.agenthub.environment.lang_utils import Language, detect_language
from r2egym.agenthub.utils.log import get_logger

from r2egym.agenthub.runtime.docker import (
    SKIP_FILES_COLLECT_CONTEXT,
    SKIP_FILES_ORIGINAL_CONTEXT
)

# 为了从patch中解析到修改的文件
from swebench.harness.utils import get_modified_files


from r2egym.agenthub.environment.simulation_utils import (
    PYTHON_CMD_PATTERN,
    is_python_execution_command,
    _get_files_from_patch,
    _load_test_list,
    _rewrite_testbed_paths,
    _parse_test_script,
    extract_init_and_test_swebv,
    get_test_files_from_spec,
    extract_exec_targets_from_code,
)


FOBIDEN_CMD = [
    "pkill",
    "pgrep",
    "kill",
    "killall",
    # "xargs",
    "rm",
    "rmdir",
    # "find",
    # "chmod",
    # "chown",
    "chattr",
    "mv",
    "cp",
    "ln",
    "dd",
    "mkfs",
    "wipefs",
    "parted",
    "fdisk",
    "sgdisk",
    "blkdiscard",
    "shred",
    "mount",
    "umount",
    "losetup",
    "ip",
    "ifconfig",
    "route",
    "iptables",
    "nft",
    "ufw",
    "sysctl",
    "ulimit",
    "reboot",
    "shutdown",
    "poweroff",
    "systemctl",
    "service",
    "cron",
    "crontab",
    "docker",
    "podman",
    "nerdctl",
    "kubectl",
    "helm",
    "ray",
    # "conda",
    # "pip",
    "apt",
    "apt-get",
    "yum",
    "dnf",
]


FOBIDEN_COMPLEX_CMD = [
    # 1) Infinite / unbounded output
    # "cat /dev",
    "dd if=/dev/zero of=/dev/stdout",
    "openssl rand",
    "seq 1 999999999999",
    # "find / -type f -print",
    # "grep -R \"\" /",
    "tar cf - /",
    "zip -r - /",
    "base64 /dev/urandom",
    # "curl <URL>  # printing huge response to stdout",
    # "wget -O - <URL>  # printing huge response to stdout",

    # 2) Hang / never return (blocking / interactive / follow)
    "tail -f /var/log/syslog",
    # "tail -f <FILE>",
    "sleep 999999",
    # "while true; do :; done",
    # "cat  # waits for stdin",
    # "read x  # waits for stdin",
    "top",
    "htop",
    # "watch <CMD>",
    # "ssh <HOST>",
    # "ping <HOST>",
    # "nc -l <PORT>",

    # 3) Resource bombs (process/CPU/memory)
    # ":(){ :|:& };:  # fork bomb",
    # "while true; do <CMD> & done  # process storm",
    # "xargs -P 1000 <CMD>  # excessive parallelism",
    "make -j 999",
    "python -c \"a=' '*(10**10)\"",
    "python -c \"a=bytearray(10**10)\"",

    # 4) Disk bombs (write until disk full / huge sparse files)
    "dd if=/dev/zero of=bigfile",
    "fallocate -l 1T bigfile",
    "truncate -s 1T bigfile",
    "cat /dev/urandom > bigfile",

    # 5) Subtle “looks harmless but explodes output”
    "head -n 5 /dev/zero",
    "hexdump /dev/urandom",
    "strings /dev/urandom",
    "cat huge.bin",
    "/dev/zero"
]



class SimulatedEnv(gym.Env):
    """
    A simulated environment that intelligently handles simple and compound commands,
    dispatching them to either a local shell or an LLM-based Python simulator.
    """

    def __init__(self, args, simulator_agent: SimulatorAgent, logger=None, step_timeout: int = 300):
        self.args = args
        self.logger = logger if logger else get_logger("SimulatedEnv")
        self.simulator = simulator_agent

        # The runtime is now initialized in reset(), just like in RepoEnv
        self.runtime: LocalRuntime = None
        self.commands: List[Command] = []
        self.cmd_parser = ParseCommandBash()
        self.step_timeout = step_timeout
        self.done = False
        self.observation = None
        self.base_commit: str = None
        self.collected_contexts = []
        self.context_count = 0

        self.agent_patch = None

        self.reset()  # Initial setup


    def get_collected_contexts(self) -> List[Dict]:
        return self.collected_contexts

    def step_raw(self, action: Action, timeout: int = None) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Executes an action using the 'whole command dispatch' strategy. If a python
        command is detected anywhere, the entire command is simulated. Otherwise,
        the entire command is executed locally.
        """
        if not timeout:
            timeout = self.step_timeout

        start_time = time.time()

        if action.function_name in ['finish', 'submit']:
            self.logger.info("Agent issued 'finish' command.")
            self.done = True
            observation = Observation("Episode finished.", 0, action)
            info = {"total_time": time.time() - start_time}
            return observation, 0.0, self.done, info

        full_command_str = action.to_bashcmd()
        # self.logger.info(f"full_command_str: {full_command_str}")
        # self.logger.info(f"self.runtime.repo_path: {self.runtime.repo_path}")
        # 将其中的 /testbed 路径替换为实际的 repo_path
        full_command_str = _rewrite_testbed_paths(full_command_str, self.runtime.repo_path)
        # self.logger.info(f"full_command_str after rewrite: {full_command_str}")
        bash_output, error_code = "", 0
        raw_simulation = None  # 新增：默认没有 raw_simulation

        # self.logger.info(f"EXECUTING local-only command: [{full_command_str}]")
        output, err_code_str = self.runtime.run(full_command_str, timeout=timeout)
        bash_output = output
        try:
            error_code = int(float(err_code_str))
        except (ValueError, TypeError):
            error_code = -1

        total_time = time.time() - start_time
        # 这里把 raw_simulation 传入 Observation（可能为 None）
        observation = Observation(bash_output, error_code, action, raw_simulation=raw_simulation)
        info = {"total_time": total_time}

        # In your original code, you set self.observation here. Let's keep that pattern.
        self.observation = observation
        return self.observation, 0.0, self.done, info



    # def step(self, action: Action, timeout: int = None) -> Tuple[Observation, float, bool, Dict[str, Any]]:
    #     """
    #     Executes an action using the 'whole command dispatch' strategy. If a python
    #     command is detected anywhere, the entire command is simulated. Otherwise,
    #     the entire command is executed locally.
    #     """
    #     if not timeout:
    #         timeout = self.step_timeout

    #     start_time = time.time()

    #     if action.function_name in ['finish', 'submit']:
    #         self.logger.info("Agent issued 'finish' command.")
    #         self.done = True
    #         observation = Observation("Episode finished.", 0, action)
    #         info = {"total_time": time.time() - start_time}
    #         return observation, 0.0, self.done, info

    #     full_command_str = action.to_bashcmd()
    #     self.logger.info(f"full_command_str: {full_command_str}")
    #     self.logger.info(f"self.runtime.repo_path: {self.runtime.repo_path}")
    #     # 将其中的 /testbed 路径替换为实际的 repo_path
    #     full_command_str = _rewrite_testbed_paths(full_command_str, self.runtime.repo_path)
    #     self.logger.info(f"full_command_str after rewrite: {full_command_str}")
    #     bash_output, error_code = "", 0
    #     raw_simulation = None  # 新增：默认没有 raw_simulation

    #     # [MODIFIED] Use the "whole command dispatch" strategy.
    #     # 1. Check the entire command string for any python execution.
    #     # contains_python = bool(PYTHON_CMD_PATTERN.search(full_command_str))

    #     # if contains_python:
    #     if is_python_execution_command(full_command_str):
    #         # 2. If it contains python, simulate the WHOLE command.
    #         self.logger.info(f"SIMULATING command with Python execution: [{full_command_str}]")

    #         agent_patch = self.runtime.get_patch()
    #         gold_patch = self.args.ds.get("golden_patch", self.args.ds.get("patch", ""))


    #         exec_code_content = {}
    #         has_inline_code = False
    #         try:
    #             exec_files, exec_dirs, has_inline_code = extract_exec_targets_from_code(full_command_str, self.runtime.repo_path, "simulated")
    #             self.logger.info(f"exec_files: {exec_files}")
    #             self.logger.info(f"exec_dirs: {exec_dirs}")
    #             self.logger.info(f"has_inline_code: {has_inline_code}")

    #             all_exec_files: set[str] = set(exec_files)

    #             if exec_dirs:
    #                 # 按 pytest 的规则在目录下递归查找测试文件：
    #                 #   test_*.py 和 *_test.py
    #                 for exec_dir in exec_dirs:
    #                     # 这里假设当前工作目录已经是仓库根目录，如果需要可以加上 cd /testbed &&
    #                     # find_cmd = (
    #                     #     f"find {exec_dir} -type f "
    #                     #     r"\( -name 'test_*.py' -o -name '*_test.py' \)"
    #                     # )
    #                     # find_cmd = f"find {exec_dir} -type f"
    #                     find_cmd = f"find {exec_dir} -type f"
    #                     find_output, error_code = self.runtime.run(find_cmd, timeout=timeout)
    #                     if error_code != "0":
    #                         self.logger.error(
    #                             f"find tests under {exec_dir} failed with code {error_code}, output: {find_output}"
    #                         )
    #                         continue
    #                     self.logger.info(f"find_output:\n{find_output}\n ----file end----")
    #                     for path in find_output.splitlines():
    #                         path = path.strip()
    #                         if not path:
    #                             continue
    #                         # 去掉前导 ./，保持和其它路径风格一致
    #                         normalized = path.lstrip("./")
    #                         all_exec_files.add(normalized)
                
    #             # 过滤一下不必要的文件
    #             INTERESTING_EXTS = [".py", ".rst", ".txt", ".dat", ".html"]
                
    #             self.logger.info(f"all_exec_files before filter len: {len(all_exec_files)}, all_exec_files: {all_exec_files}")
    #             all_exec_files = [path for path in all_exec_files if os.path.splitext(path)[1] in INTERESTING_EXTS]
    #             self.logger.info(f"all_exec_files after filter len: {len(all_exec_files)}, all_exec_files: {all_exec_files}")

    #             MAX_TEST_FILE_NUM = 20 # 最多只能测试20个测试文件，超过了就随机sample一下
    #             if len(all_exec_files) > MAX_TEST_FILE_NUM:
    #                 all_exec_files = sorted(random.sample(all_exec_files, MAX_TEST_FILE_NUM))
    #                 self.logger.info(f"all_exec_files exceeds MAX_TEST_FILE_NUM, sample {MAX_TEST_FILE_NUM} files, all_exec_files: {all_exec_files}")

    #             # self.logger.info(f"all_exec_files: {all_exec_files}")
    #             # 逐个读取所有参与执行的文件内容
    #             for path in sorted(all_exec_files):
    #                 self.logger.info(f"Reading exec file content from: {path}")
    #                 file_content = self.runtime.read_file(path)
    #                 if not file_content.startswith("Error:"):
    #                     exec_code_content[path] = file_content
    #                     self.logger.info(f"Successfully read file: {path}")
    #                 else:
    #                     self.logger.error(f"Error reading file [{path}]: {file_content}")

                
    #         except ValueError:
    #              self.logger.warning(f"Could not parse command for execution content: {full_command_str}")


    #         # 如果没有找到可以执行的代码文件，以及代码内容不是在命令中体现，直接返回观察
    #         if not exec_code_content and not has_inline_code:
    #             self.logger.warning(f"No executable code found in command: {full_command_str}")
    #             observation = Observation("No executable files, paths, or code were found in the command. Please modify the command!.", 0, action, raw_simulation="No executable files, paths, or code were found in the command. Please modify the command!")
    #             info = {"total_time": time.time() - start_time}
    #             return observation, 0.0, self.done, info

    #         # agent_files = _get_files_from_patch(agent_patch)
    #         agent_files = get_modified_files(agent_patch)
    #         agent_files = set(file for file in agent_files if not any(skip_file in file for skip_file in SKIP_FILES_COLLECT_CONTEXT))
    #         # gold_files = _get_files_from_patch(gold_patch)
    #         gold_files = get_modified_files(gold_patch)
    #         all_modified_files = agent_files.union(gold_files)

    #         original_files_content: Dict[str, str] = {}
    #         for file_path in all_modified_files:
    #             if file_path in SKIP_FILES_ORIGINAL_CONTEXT: # 跳过在原始仓库中不存在的文件，这些文件是system prompt中要求model创建的，在原始的仓库中无法找到的文件
    #                 continue

    #             try:
    #                 backup_file = Path(self.runtime.repo_backup_path) / file_path
    #                 if backup_file.is_file():
    #                     original_files_content[file_path] = backup_file.read_text(errors='replace')
    #                 else:
    #                     original_files_content[file_path] = f"Info: File '{file_path}' does not exist in the baseline commit."
    #             except Exception as e:
    #                 original_files_content[file_path] = f"Error: Could not read original file. Details: {e}"


    #         context = {
    #             "type": "step_execution",
    #             "initial_analysis": self.args.ds.get("initial_analysis", "N/A"),
    #             "problem_statement": self.runtime.get_task_instruction(),
    #             "human_hints": self.args.ds.get("hints_text", ""),
    #             "agent_patch": agent_patch,
    #             "gold_patch": gold_patch,
    #             "original_files_content": original_files_content,
    #             # ✅ 这里 execution_code_content 可能是：
    #             # - 普通 python 命令：某个被执行的脚本
    #             # - pytest 命令：被选中的测试文件集合（无论是否显式目录）
    #             "execution_code_content": exec_code_content,
    #             "command_to_simulate": full_command_str,  # Pass the ENTIRE command
    #             "has_inline_code": has_inline_code,
    #         }

    #         # UPDATED: get_simulated_feedback now returns (raw_response, combined_output, exit_code)
    #         raw_simulation, bash_output, error_code = self.simulator.get_simulated_feedback(context)

    #         self.collected_contexts.append({
    #             "instance_id": self.args.ds.get("instance_id"),
    #             "context_id": self.context_count,
    #             "context": context,
    #             "simulated_output": bash_output,
    #             "simulation_error_code": error_code,
    #             "raw_simulation": raw_simulation
    #         })
    #         self.context_count += 1
    #     else:
    #         # 3. If no python is found, execute the WHOLE command locally.
    #         self.logger.info(f"EXECUTING local-only command: [{full_command_str}]")
    #         output, err_code_str = self.runtime.run(full_command_str, timeout=timeout)
    #         bash_output = output
    #         try:
    #             error_code = int(float(err_code_str))
    #         except (ValueError, TypeError):
    #             error_code = -1

    #     total_time = time.time() - start_time
    #     # 这里把 raw_simulation 传入 Observation（可能为 None）

    #     # 替换一下本地的信息
    #     bash_output = bash_output.replace(self.runtime.repo_path, "/testbed")
    #     observation = Observation(bash_output, error_code, action, raw_simulation=raw_simulation)
    #     info = {"total_time": total_time}

    #     # In your original code, you set self.observation here. Let's keep that pattern.
    #     self.observation = observation
    #     return self.observation, 0.0, self.done, info

    def step(self, action: Action, timeout: int = None) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Executes an action using the 'whole command dispatch' strategy. If a python
        command is detected anywhere, the entire command is simulated. Otherwise,
        the entire command is executed locally.
        """
        if not timeout:
            timeout = self.step_timeout

        start_time = time.time()

        if action.function_name in ['finish', 'submit']:
            # self.logger.info("Agent issued 'finish' command.")
            self.done = True
            observation = Observation("Episode finished.", 0, action)
            info = {"total_time": time.time() - start_time}
            return observation, 0.0, self.done, info

        full_command_str = action.to_bashcmd()
        # print(f"full_command_str: {full_command_str}")
        # self.logger.info(f"self.runtime.repo_path: {self.runtime.repo_path}")
        # 将其中的 /testbed 路径替换为实际的 repo_path
        full_command_str = _rewrite_testbed_paths(full_command_str, self.runtime.repo_path)
        # self.logger.info(f"full_command_str after rewrite: {full_command_str}")
        bash_output, error_code = "", 0
        raw_simulation = None  # 新增：默认没有 raw_simulation

        # 拦截所有包含 'pip install' 的命令（包括 -e ., -r requirements.txt, 或普通包安装）
        # 直接返回成功，不进行真实执行或模拟，节省时间
        if "pip install" in full_command_str:
            # self.logger.info(f"full_command_str: {full_command_str}, Intercepted 'pip install', skipping execution and returning success.")
            bash_output = "Successfully installed packages."
            # 返回 0 表示成功
            # 我添加了 raw_simulation 参数以保持与 Observation 类的统一性（如果 Observation 类支持）
            observation = Observation(bash_output, 0, action)
            info = {"total_time": time.time() - start_time}
            return observation, 0.0, self.done, info
        
        for cmd in FOBIDEN_CMD:
            if f"{cmd} " in full_command_str:
                bash_output = f"This command {cmd} is not allowed. Please try other approaches."
                print(f"action: {full_command_str} find {cmd} which is not allowed.")
                observation = Observation(bash_output, 0, action)
                info = {"total_time": time.time() - start_time}
                return observation, 0.0, self.done, info
        
        for cmd in FOBIDEN_COMPLEX_CMD:
            if f"{cmd}" in full_command_str:
                bash_output = f"This command {cmd} is not allowed. Please try other approaches."
                print(f"action: {full_command_str} find {cmd} which is not allowed.")
                observation = Observation(bash_output, 0, action)
                info = {"total_time": time.time() - start_time}
                return observation, 0.0, self.done, info
        
        # [MODIFIED] Use the "whole command dispatch" strategy.
        # 1. Check the entire command string for any python execution.
        # contains_python = bool(PYTHON_CMD_PATTERN.search(full_command_str))

        # if contains_python:
        if is_python_execution_command(full_command_str):
            # 2. If it contains python, simulate the WHOLE command.
            # self.logger.info(f"SIMULATING command with Python execution: [{full_command_str}]")

            agent_patch = self.runtime.get_patch()
            gold_patch = self.args.ds.get("golden_patch", self.args.ds.get("patch", ""))


            exec_code_content = {}
            has_inline_code = False
            try:
                exec_files, exec_dirs, has_inline_code = extract_exec_targets_from_code(full_command_str, self.runtime.repo_path, "simulated")
                # self.logger.info(f"exec_files: {exec_files}")
                # self.logger.info(f"exec_dirs: {exec_dirs}")
                # self.logger.info(f"has_inline_code: {has_inline_code}")

                all_exec_files: set[str] = set(exec_files)

                if exec_dirs:
                    # 按 pytest 的规则在目录下递归查找测试文件：
                    #   test_*.py 和 *_test.py
                    for exec_dir in exec_dirs:
                        # 这里假设当前工作目录已经是仓库根目录，如果需要可以加上 cd /testbed &&
                        # find_cmd = (
                        #     f"find {exec_dir} -type f "
                        #     r"\( -name 'test_*.py' -o -name '*_test.py' \)"
                        # )
                        # find_cmd = f"find {exec_dir} -type f"
                        find_cmd = f"find {exec_dir} -type f"
                        find_output, error_code = self.runtime.run(find_cmd, timeout=timeout)
                        if error_code != "0":
                            self.logger.error(
                                f"find tests under {exec_dir} failed with code {error_code}, output: {find_output}"
                            )
                            continue
                        # self.logger.info(f"find_output:\n{find_output}\n ----file end----")
                        for path in find_output.splitlines():
                            path = path.strip()
                            if not path:
                                continue
                            # 去掉前导 ./，保持和其它路径风格一致
                            normalized = path.lstrip("./")
                            all_exec_files.add(normalized)
                
                # 过滤一下不必要的文件
                INTERESTING_EXTS = [".py", ".rst", ".txt", ".dat", ".html"]
                
                self.logger.info(f"all_exec_files before filter len: {len(all_exec_files)}, all_exec_files: {all_exec_files}")
                all_exec_files = [path for path in all_exec_files if os.path.splitext(path)[1] in INTERESTING_EXTS]
                self.logger.info(f"all_exec_files after filter len: {len(all_exec_files)}, all_exec_files: {all_exec_files}")

                MAX_TEST_FILE_NUM = 20 # 最多只能测试20个测试文件，超过了就随机sample一下
                if len(all_exec_files) > MAX_TEST_FILE_NUM:
                    all_exec_files = sorted(random.sample(all_exec_files, MAX_TEST_FILE_NUM))
                    self.logger.info(f"all_exec_files exceeds MAX_TEST_FILE_NUM, sample {MAX_TEST_FILE_NUM} files, all_exec_files: {all_exec_files}")

                # self.logger.info(f"all_exec_files: {all_exec_files}")
                # 逐个读取所有参与执行的文件内容
                for path in sorted(all_exec_files):
                    self.logger.info(f"Reading exec file content from: {path}")
                    file_content = self.runtime.read_file(path)
                    if not file_content.startswith("Error:"):
                        exec_code_content[path] = file_content
                        self.logger.info(f"Successfully read file: {path}")
                    else:
                        self.logger.error(f"Error reading file [{path}]: {file_content}")

                
            except ValueError:
                 self.logger.warning(f"Could not parse command for execution content: {full_command_str}")


            # 如果没有找到可以执行的代码文件，以及代码内容不是在命令中体现，直接返回观察
            if not exec_code_content and not has_inline_code:
                self.logger.warning(f"No executable code found in command: {full_command_str}")
                observation = Observation("No executable files, paths, or code were found in the command. Please modify the command!.", 0, action, raw_simulation="No executable files, paths, or code were found in the command. Please modify the command!")
                info = {"total_time": time.time() - start_time}
                return observation, 0.0, self.done, info

            # agent_files = _get_files_from_patch(agent_patch)
            agent_files = get_modified_files(agent_patch)
            agent_files = set(file for file in agent_files if not any(skip_file in file for skip_file in SKIP_FILES_COLLECT_CONTEXT))
            # gold_files = _get_files_from_patch(gold_patch)
            if not self.runtime.r2egym:
                gold_files = get_modified_files(gold_patch)
            else:
                gold_files = self.args.ds["modified_not_test_files"]
            all_modified_files = agent_files.union(gold_files)

            original_files_content: Dict[str, str] = {}
            for file_path in all_modified_files:
                if file_path in SKIP_FILES_ORIGINAL_CONTEXT: # 跳过在原始仓库中不存在的文件，这些文件是system prompt中要求model创建的，在原始的仓库中无法找到的文件
                    continue

                try:
                    backup_file = Path(self.runtime.repo_backup_path) / file_path
                    if backup_file.is_file():
                        original_files_content[file_path] = backup_file.read_text(errors='replace')
                    else:
                        original_files_content[file_path] = f"Info: File '{file_path}' does not exist in the baseline commit."
                except Exception as e:
                    original_files_content[file_path] = f"Error: Could not read original file. Details: {e}"


            context = {
                "type": "step_execution",
                "initial_analysis": self.args.ds.get("initial_analysis", "N/A"),
                "problem_statement": self.runtime.get_task_instruction(),
                "human_hints": self.args.ds.get("hints_text", ""),
                "agent_patch": agent_patch,
                "gold_patch": gold_patch,
                "original_files_content": original_files_content,
                # ✅ 这里 execution_code_content 可能是：
                # - 普通 python 命令：某个被执行的脚本
                # - pytest 命令：被选中的测试文件集合（无论是否显式目录）
                "execution_code_content": exec_code_content,
                "command_to_simulate": full_command_str,  # Pass the ENTIRE command
                "has_inline_code": has_inline_code,
            }

            # UPDATED: get_simulated_feedback now returns (raw_response, combined_output, exit_code)
            raw_simulation, bash_output, error_code = self.simulator.get_simulated_feedback(context)

            self.collected_contexts.append({
                "instance_id": self.args.ds.get("instance_id"),
                "context_id": self.context_count,
                "context": context,
                "simulated_output": bash_output,
                "simulation_error_code": error_code,
                "raw_simulation": raw_simulation
            })
            self.context_count += 1
        else:
            # 3. If no python is found, execute the WHOLE command locally.
            # self.logger.info(f"EXECUTING local-only command: [{full_command_str}]")
            output, err_code_str = self.runtime.run(full_command_str, timeout=timeout)
            bash_output = output
            try:
                error_code = int(float(err_code_str))
            except (ValueError, TypeError):
                error_code = -1

        total_time = time.time() - start_time
        # 这里把 raw_simulation 传入 Observation（可能为 None）

        # 替换一下本地的信息
        bash_output = bash_output.replace(self.runtime.repo_path, "/testbed")
        observation = Observation(bash_output, error_code, action, raw_simulation=raw_simulation)
        info = {"total_time": total_time}
        print(f"len bash_output: {len(bash_output)}, bytesoutput: {len(bash_output.encode('utf-8'))}")

        # In your original code, you set self.observation here. Let's keep that pattern.
        self.observation = observation
        return self.observation, 0.0, self.done, info


    def _calculate_simulated_reward_swebv(self, max_workers: int = 4) -> Tuple[float, str]:
        """
        [NEW] Simulates the final test run to calculate a reward.
        It parses the test script, runs necessary git commands, simulates all
        required tests in parallel, and asks a judge LLM for the final reward.
        """
        self.logger.info("Starting simulated reward calculation...")
        
        # 1. Prepare environment by running pre-test git commands from test script
        # This is a simplified parser for the SWE-bench run_tests.sh format
        # test_script_content = self.args.ds.get("run_tests") # In SWE-bench, this contains the test commands
        # if not test_script_content:
        #     return 0.0, "Error: Could not find 'run_tests' in dataset to determine test setup."

        pre_test_commands = []
        pre_test_commands.append(f"cd {self.runtime.repo_path}") # 先进入仓库目录
        
        pytest_command = None
        # lines = test_script_content.split('\n')
        # for line in lines:
        #     if PYTHON_CMD_PATTERN.search(line):
        #         pytest_command = line
        #         # Stop processing after the first python command is found
        #         # break
        #         continue 
        #     if line.strip().startswith(('git')):
        #         pre_test_commands.append(line)
        
        # self.logger.info(f"Found {len(pre_test_commands)} pre-test setup commands to execute.")
        # # for cmd in pre_test_commands:
        # #     self.logger.info(f"Executing setup command: {cmd}")
        # #     self.runtime.run(cmd)

        # pre_test_commands_str = "\n".join(pre_test_commands)
        # self.logger.info(f"pre_test_commands_str:\n{pre_test_commands_str}")

        # pre_test_commands, pytest_command = _parse_test_script(test_script_content, self.runtime.repo_path)
        if self.runtime.swegym:
            pre_test_commands, test_command = extract_init_and_test_swebv(self.runtime.test_spec.eval_script_list, test_command_index = -2, init_command_end_index = -2, repo_path=self.runtime.repo_path, is_sim=True)
        elif self.runtime.r2egym:
            pre_test_commands = []
            r2e_gym_test_files = sorted(self.args.ds["test_exec_content"].keys())
            # print(f"r2e_gym_test_files: {r2e_gym_test_files}")
            test_command = "pytest -rA " + " ".join(r2e_gym_test_files) # 先暂时直接这样生成测试命令
        else:
            pre_test_commands, test_command = extract_init_and_test_swebv(self.runtime.test_spec.eval_script_list, repo_path=self.runtime.repo_path, is_sim=True)
        
        if not self.runtime.r2egym:
            pre_test_commands_str = "\n".join(pre_test_commands)
            
        # self.logger.info(f"pre_test_commands: {pre_test_commands}")
        # self.logger.info(f"test_command: {test_command}")

        # 获取当前修改了哪些文件
        # self.logger.info(f"Getting modified files... 111111")
        agent_patch = self.runtime.get_patch()
        # self.logger.info(f"Getting modified files... :\n{agent_patch}")
        # self.logger.info(f"Getting modified files... 222222")
        gold_patch = self.args.ds.get("golden_patch", self.args.ds.get("patch", ""))

        # 写到临时文件中
        if not self.runtime.r2egym:
            pre_test_command_file_path = os.path.join(self.runtime.temp_dir, "pre_test_commands.sh")
            self.runtime.write_file(pre_test_command_file_path, pre_test_commands_str)
            self.logger.info(f"Wrote pre-test commands to file at: {pre_test_command_file_path}")
        
        

        all_modified_files = self.runtime.get_modified_files() # 获取当前修改了哪些文件
        self.logger.info(f"Getting modified files... 333333")
        if not self.runtime.r2egym:
            gold_files = get_modified_files(gold_patch)
        else:
            gold_files = self.args.ds["modified_not_test_files"]
        self.logger.info(f"Getting modified files... 4444")
        all_modified_files = set(all_modified_files + gold_files)
        self.logger.info(f"all_modified_files: {len(all_modified_files)} Modified files:\n{all_modified_files}")
        all_modified_files_original_content: Dict[str, str] = {} # 记录修改的文件的原始内容
        for modified_file_path in all_modified_files:
            if modified_file_path in SKIP_FILES_ORIGINAL_CONTEXT: # 跳过在原始仓库中不存在的文件，这些文件是system prompt中要求model创建的，在原始的仓库中无法找到的文件
                continue

            try:
                backup_file = Path(self.runtime.repo_backup_path) / modified_file_path
                if backup_file.is_file():
                    all_modified_files_original_content[modified_file_path] = backup_file.read_text(errors='replace')
                else:
                    all_modified_files_original_content[modified_file_path] = f"Info: File '{modified_file_path}' does not exist in the baseline commit."
            except Exception as e:
                all_modified_files_original_content[modified_file_path] = f"Error: Could not read original file. Details: {e}"


        # 2. Identify all test files to be simulated (all F2P and P2P)
        f2p_list = _load_test_list(self.args.ds.get("FAIL_TO_PASS", "[]"))
        p2p_list = _load_test_list(self.args.ds.get("PASS_TO_PASS", "[]"))

        # self.logger.info(f"FAIL_TO_PASS: {len(f2p_list)}, PASS_TO_PASS: {len(p2p_list)}")

        # f2p_files = sorted(list({t.split('::')[0] for t in f2p_list}))
        # p2p_files = sorted(list({t.split('::')[0] for t in p2p_list}))
        # all_test_files = sorted(list(set(f2p_files + p2p_files)))
        if not self.runtime.r2egym:
            all_test_files = get_test_files_from_spec(
                self.args.ds.get("FAIL_TO_PASS", "[]"),
                self.args.ds.get("PASS_TO_PASS", "[]"),
                test_command,
                self.args.ds.get("test_patch", "")
            )
        else:
            all_test_files = r2e_gym_test_files

        self.logger.info(f"Total tests to simulate: {len(all_test_files)} files")
        # self.logger.info(f"Total tests to simulate: {len(all_test_files)} files ({len(f2p_files)} F2P, {len(p2p_files)} P2P)")

        if not all_test_files:
            self.logger.info("No FAIL_TO_PASS or PASS_TO_PASS tests were specified11.")
            return 1.0, "Success: No FAIL_TO_PASS or PASS_TO_PASS tests were specified."

        # 3. Simulate each test file in parallel
        simulation_results = {}
        problem_statement = self.runtime.get_task_instruction()
        
        execution_code_content_dict = {}
        def _simulate_single_file(file_path: str, execution_code_content_dict: dict):
            if not self.runtime.r2egym and not (Path(self.runtime.repo_path) / file_path).is_file():
                msg = f"--- SIMULATED STDERR ---\nERROR: Test file '{file_path}' not found in the repository."
                self.logger.error(msg)
                context_to_collect = {
                    "instance_id": self.args.ds.get("instance_id"),
                    "context": None,
                    "simulated_output": msg,
                    "simulated_error_code": -1,
                    "raw_simulation": None,
                }
                return context_to_collect, None, msg, -1, execution_code_content_dict

            self.logger.info(f"Simulating test file: {file_path}")
            
            if not self.runtime.r2egym:
                exec_file_content = self.runtime.read_file(file_path)
            else:
                exec_file_content = self.args.ds["test_exec_content"][file_path]

            # Build context for this single file
            context = {
                "type": "reward_calculation",
                "initial_analysis": self.args.ds.get("initial_analysis", ""),
                "problem_statement": problem_statement,
                "human_hints": self.args.ds.get("hints_text", ""),
                "agent_patch": agent_patch,
                "gold_patch": gold_patch,
                "execution_code_content": {file_path:exec_file_content},
                "original_files_content": all_modified_files_original_content,
                "command_to_simulate": test_command,
                "FAIL_TO_PASS": self.args.ds.get("FAIL_TO_PASS", []),
                "PASS_TO_PASS": self.args.ds.get("PASS_TO_PASS", []),
            }

            execution_code_content_dict[file_path] = exec_file_content

            # UPDATED: get_simulated_feedback now returns (raw_response, combined_output, exit_code)
            raw_simulation, output, exit_code = self.simulator.get_simulated_feedback(context)
            

            context_to_collect = {
                "instance_id": self.args.ds.get("instance_id"),
                # "context_id": self.context_count,
                "context": context,
                "simulated_output": output,
                "simulated_error_code": exit_code,
                "raw_simulation": raw_simulation
            }
            # # Determine pass/fail based on exit code and output
            # # A simple heuristic: fail if exit code is not 0 or "failed" is in output
            # if exit_code != 0 or " failed" in output.lower():
            #     return file_path, f"FAILED:\n{output}"
            # else:
            #     return file_path, "PASSED"
            return context_to_collect, raw_simulation, output, exit_code, execution_code_content_dict

        # 4. 执行 pre_test_commands.sh
        if not self.runtime.r2egym:
            pre_test_command_output, pre_test_command_exit_code = self.runtime.run(f"bash {pre_test_command_file_path}")
            # self.logger.info(f"Pre-test commands output:\n{pre_test_command_output}")
            # self.logger.info(f"Pre-test commands exit code:\n{pre_test_command_exit_code}")
            if pre_test_command_exit_code != "0":
                return 0.0, f"FAILED: Pre-test commands failed with exit code: {pre_test_command_exit_code}\n{pre_test_command_output}"

        # 5. 模拟每个测试文件
        start_all_test_files_simulation = time.time()
        # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     self.logger.info(f"Simulating {len(all_test_files)} test files in parallel...")
        #     print(f"Simulating {len(all_test_files)} test files in parallel..., test_files: {all_test_files}")
        #     future_to_file = {executor.submit(_simulate_single_file, fp): fp for fp in all_test_files}
        #     # 增加tqdm
        #     for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(all_test_files), desc="Simulating test files"):
        #         file_path = future_to_file[future]
        #         try:
        #             context_to_collect, raw_simulation, output, exit_code = future.result()
        #             simulation_results[file_path] = {
        #                 "output": output,
        #                 "exit_code": exit_code,
        #                 "raw_simulation": raw_simulation
        #             }
        #             context_to_collect["context_id"] = self.context_count
        #             self.collected_contexts.append(context_to_collect)
        #             self.context_count += 1
        #         except Exception as exc:
        #             simulation_results[file_path] = {
        #                 "output": f"FAILED: Exception during simulation: {exc}",
        #                 "exit_code": -1,
        #                 "raw_simulation": None
        #             }
        #     self.logger.info(f"Simulation results: {simulation_results}")

        self.logger.info(f"Simulating {len(all_test_files)} test files in series...")
        print(f"Simulating {len(all_test_files)} test files in series..., test_files: {all_test_files}")
        
        for fp in tqdm(all_test_files[:2], total=len(all_test_files), desc="Simulating test files"):
            file_path = fp
            try:
                context_to_collect, raw_simulation, output, exit_code, execution_code_content_dict = _simulate_single_file(fp, execution_code_content_dict)
                simulation_results[file_path] = {
                    "output": output,
                    "exit_code": exit_code,
                    "raw_simulation": raw_simulation
                }
                context_to_collect["context_id"] = self.context_count
                self.collected_contexts.append(context_to_collect)
                self.context_count += 1
            except Exception as exc:
                simulation_results[file_path] = {
                    "output": f"FAILED: Exception during simulation: {exc}",
                    "exit_code": -1,
                    "raw_simulation": None
                }
        # self.logger.info(f"Simulation results: {simulation_results}")

        self.logger.info(f"Simulation of all {len(all_test_files)} test files completed in {time.time() - start_all_test_files_simulation:.2f} seconds.")
        # 4. Assemble a summary and ask the judge LLM for the reward
        summary = "Summary of Simulated Test Results:\n\n"
        for file_path, result in simulation_results.items():
            summary += f"File: {file_path}\nOutput: {result['output']}\nExit Code: {result['exit_code']}\n----------------------------\n"
        
        self.logger.info("All simulations complete. Asking judge LLM for final reward.")
        # UPDATED: get_reward_for_test_summary now returns (reward, raw_response, reason)

        start_final_reward_calculation = time.time()
        final_reward, raw_simulation, reason = self.simulator.get_reward_for_test_summary(summary, f2p_list, p2p_list)
        self.logger.info(f"Final reward calculation completed 111in {time.time() - start_final_reward_calculation:.2f} seconds.")

        full_output = f"{summary}\n--- Reward Judge ---\nReward: {final_reward}\nReason: {reason}"
        
        self.logger.info(f"Simulated reward calculation finished. Reward: {final_reward}")

        context_sim_reward = {
                "type": "merged_reward_calculation",
                "initial_analysis": self.args.ds.get("initial_analysis", ""),
                "problem_statement": problem_statement,
                "human_hints": self.args.ds.get("hints_text", ""),
                "agent_patch": agent_patch,
                "gold_patch": gold_patch,
                "execution_code_content": execution_code_content_dict,
                "original_files_content": all_modified_files_original_content,
                "command_to_simulate": test_command,
                "FAIL_TO_PASS": self.args.ds.get("FAIL_TO_PASS", []),
                "PASS_TO_PASS": self.args.ds.get("PASS_TO_PASS", []),
            }

        self.collected_contexts.append({
            "instance_id": self.args.ds.get("instance_id"),
            "data_source": self.runtime.data_source,
            "context_id": self.context_count,
            "context": context_sim_reward,
            "agent_patch": agent_patch,
            "gold_patch": gold_patch,
            "simulated_summary": summary,
            "simulated_reward": final_reward,
            "simulated_reason": reason,
            "raw_simulation": raw_simulation,
            "finish_flag": True
        })

        # 最终增加这个agent_patch属性
        self.agent_patch = agent_patch
        self.context_count += 1
        return float(final_reward), full_output


    def get_reward_context(self):
        if not self.collected_contexts:
            return None
        if "finish_flag" in self.collected_contexts[-1]:
            return self.collected_contexts[-1]
        return None

    def _calculate_simulated_reward_swebv_hls_fix_train_and_inference(self, max_workers: int = 4) -> Tuple[float, str]:
        """
        [NEW] Simulates the final test run to calculate a reward.
        It parses the test script, runs necessary git commands, simulates all
        required tests in parallel, and asks a judge LLM for the final reward.
        """
        self.logger.info("Starting simulated reward calculation...")
        
        # 1. Prepare environment by running pre-test git commands from test script
        # This is a simplified parser for the SWE-bench run_tests.sh format
        # test_script_content = self.args.ds.get("run_tests") # In SWE-bench, this contains the test commands
        # if not test_script_content:
        #     return 0.0, "Error: Could not find 'run_tests' in dataset to determine test setup."

        pre_test_commands = []
        pre_test_commands.append(f"cd {self.runtime.repo_path}") # 先进入仓库目录
        
        pytest_command = None
        # lines = test_script_content.split('\n')
        # for line in lines:
        #     if PYTHON_CMD_PATTERN.search(line):
        #         pytest_command = line
        #         # Stop processing after the first python command is found
        #         # break
        #         continue 
        #     if line.strip().startswith(('git')):
        #         pre_test_commands.append(line)
        
        # self.logger.info(f"Found {len(pre_test_commands)} pre-test setup commands to execute.")
        # # for cmd in pre_test_commands:
        # #     self.logger.info(f"Executing setup command: {cmd}")
        # #     self.runtime.run(cmd)

        # pre_test_commands_str = "\n".join(pre_test_commands)
        # self.logger.info(f"pre_test_commands_str:\n{pre_test_commands_str}")

        # pre_test_commands, pytest_command = _parse_test_script(test_script_content, self.runtime.repo_path)
        if self.runtime.swegym:
            pre_test_commands, test_command = extract_init_and_test_swebv(self.runtime.test_spec.eval_script_list, test_command_index = -2, init_command_end_index = -2, repo_path=self.runtime.repo_path, is_sim=True)
        elif self.runtime.r2egym:
            pre_test_commands = []
            r2e_gym_test_files = sorted(self.args.ds["test_exec_content"].keys())
            test_command = "pytest -rA " + " ".join(r2e_gym_test_files) # 先暂时直接这样生成测试命令
        else:
            pre_test_commands, test_command = extract_init_and_test_swebv(self.runtime.test_spec.eval_script_list, repo_path=self.runtime.repo_path, is_sim=True)
        
        if not self.runtime.r2egym:
            pre_test_commands_str = "\n".join(pre_test_commands)
            
        # self.logger.info(f"pre_test_commands: {pre_test_commands}")
        # self.logger.info(f"test_command: {test_command}")

        # 获取当前修改了哪些文件
        # self.logger.info(f"Getting modified files... 111111")
        agent_patch = self.runtime.get_patch()
        # self.logger.info(f"Getting modified files... :\n{agent_patch}")
        # self.logger.info(f"Getting modified files... 222222")
        gold_patch = self.args.ds.get("golden_patch", self.args.ds.get("patch", ""))

        # 写到临时文件中
        if not self.runtime.r2egym:
            pre_test_command_file_path = os.path.join(self.runtime.temp_dir, "pre_test_commands.sh")
            self.runtime.write_file(pre_test_command_file_path, pre_test_commands_str)
            self.logger.info(f"Wrote pre-test commands to file at: {pre_test_command_file_path}")
        
        
        all_modified_files = self.runtime.get_modified_files() # 获取当前修改了哪些文件
        self.logger.info(f"Getting modified files... 333333")
        if not self.runtime.r2egym:
            gold_files = get_modified_files(gold_patch)
        else:
            gold_files = self.args.ds["modified_not_test_files"]
        # self.logger.info(f"Getting modified files... 4444")
        all_modified_files = set(all_modified_files + gold_files)
        # self.logger.info(f"all_modified_files: {len(all_modified_files)} Modified files:\n{all_modified_files}")
        all_modified_files_original_content: Dict[str, str] = {} # 记录修改的文件的原始内容
        for modified_file_path in all_modified_files:
            if modified_file_path in SKIP_FILES_ORIGINAL_CONTEXT: # 跳过在原始仓库中不存在的文件，这些文件是system prompt中要求model创建的，在原始的仓库中无法找到的文件
                continue

            try:
                backup_file = Path(self.runtime.repo_backup_path) / modified_file_path
                if backup_file.is_file():
                    all_modified_files_original_content[modified_file_path] = backup_file.read_text(errors='replace')
                else:
                    all_modified_files_original_content[modified_file_path] = f"Info: File '{modified_file_path}' does not exist in the baseline commit."
            except Exception as e:
                all_modified_files_original_content[modified_file_path] = f"Error: Could not read original file. Details: {e}"


        # 2. Identify all test files to be simulated (all F2P and P2P)
        f2p_list = _load_test_list(self.args.ds.get("FAIL_TO_PASS", "[]"))
        p2p_list = _load_test_list(self.args.ds.get("PASS_TO_PASS", "[]"))

        self.logger.info(f"FAIL_TO_PASS: {len(f2p_list)}, PASS_TO_PASS: {len(p2p_list)}")

        # f2p_files = sorted(list({t.split('::')[0] for t in f2p_list}))
        # p2p_files = sorted(list({t.split('::')[0] for t in p2p_list}))
        # all_test_files = sorted(list(set(f2p_files + p2p_files)))
        if not self.runtime.r2egym:
            all_test_files = get_test_files_from_spec(
                self.args.ds.get("FAIL_TO_PASS", "[]"),
                self.args.ds.get("PASS_TO_PASS", "[]"),
                test_command,
                self.args.ds.get("test_patch", "")
            )
        else:
            all_test_files = r2e_gym_test_files

        # self.logger.info(f"Total tests to simulate: {len(all_test_files)} files")
        # self.logger.info(f"Total tests to simulate: {len(all_test_files)} files ({len(f2p_files)} F2P, {len(p2p_files)} P2P)")

        if not all_test_files:
            return 1.0, "Success: No FAIL_TO_PASS or PASS_TO_PASS tests were specified."

        # 3. Simulate each test file in parallel
        # simulation_results = {}
        problem_statement = self.runtime.get_task_instruction()
        


        # 4. 执行 pre_test_commands.sh
        if not self.runtime.r2egym:
            pre_test_command_output, pre_test_command_exit_code = self.runtime.run(f"bash {pre_test_command_file_path}")
            # self.logger.info(f"Pre-test commands output:\n{pre_test_command_output}")
            # self.logger.info(f"Pre-test commands exit code:\n{pre_test_command_exit_code}")
            if pre_test_command_exit_code != "0":
                return 0.0, f"FAILED: Pre-test commands failed with exit code: {pre_test_command_exit_code}\n{pre_test_command_output}"

        # 5. 收集上下文
        execution_code_content = {}
        for file_path in all_test_files:

            if not self.runtime.r2egym:
                exec_file_content = self.runtime.read_file(file_path)
            else:
                exec_file_content = self.args.ds["test_exec_content"][file_path]
            execution_code_content[file_path] = exec_file_content

        # UPDATED: get_reward_for_test_summary now returns (reward, raw_response, reason)

        start_final_reward_calculation = time.time()

        context = {
            "type": "reward_calculation",
            "initial_analysis": self.args.ds.get("initial_analysis", ""),
            "problem_statement": problem_statement,
            "human_hints": self.args.ds.get("hints_text", ""),
            "agent_patch": agent_patch,
            "gold_patch": gold_patch,
            "execution_code_content": execution_code_content, # 。。。
            "original_files_content": all_modified_files_original_content, # 。。。
            "command_to_simulate": test_command,
            "FAIL_TO_PASS": self.args.ds.get("FAIL_TO_PASS", []),
            "PASS_TO_PASS": self.args.ds.get("PASS_TO_PASS", []),
        }

        # final_reward, raw_simulation, reason = self.simulator.get_reward_for_test_summary(summary, f2p_list, p2p_list)
        # time_reward = time.time()
        if not agent_patch:
            test_report, final_reward, response_content = "" ,0 ,""
        else:
            test_report, final_reward, response_content = self.simulator.get_simulated_test_report_and_reward(context)

        self.logger.info(f"Final reward calculation completed in {time.time() - start_final_reward_calculation:.2f} seconds.")


        full_output = f"{test_report}\n--- Reward Judge ---\nReward: {final_reward}"
        
        self.logger.info(f"Simulated reward calculation finished. Reward: {final_reward}")
        self.collected_contexts.append({
            "instance_id": self.args.ds.get("instance_id"),
            "data_source": self.runtime.data_source,
            "context_id": self.context_count,
            "context": context,
            "agent_patch": agent_patch,
            "gold_patch": gold_patch,
            "test_report":test_report,
            "simulated_reward": final_reward,
            "raw_simulation": response_content,
            "finish_flag": True
        })

        self.agent_patch = agent_patch
        self.context_count += 1

        return float(final_reward), full_output


    def reset(self) -> Observation:
        """
        Resets the underlying local runtime and returns an initial observation.
        Also detects the repository language and updates the simulator so that
        the correct SWT/SWR system prompts are used for this episode.
        """
        self.logger.info("Resetting SimulatedEnv...")
        if self.runtime:
            self.runtime.close()

        self.runtime = LocalRuntime(ds=self.args.ds, logger=self.logger)
        self.done = False

        # Detect language from the repo field in the dataset entry and inform
        # the simulator so it uses the right system prompts.
        repo_name = self.args.ds.get("repo", "")
        detected_lang = detect_language(repo_name=repo_name)
        self.simulator.lang = detected_lang
        self.logger.info(
            f"Reset done. Local repo at: {self.runtime.repo_path}  "
            f"[language={detected_lang.name}]"
        )

        initial_obs_text = "Environment reset. Workspace is ready."
        self.observation = Observation(initial_obs_text, 0, Action("reset", {}))

        self.collected_contexts = []
        self.context_count = 0
        return self.observation

    def add_commands(self, cmd_files: List[Path]):
        """
        Parses command files and makes them available in the local runtime's PATH.
        """
        cmds = []

        # Create a bin directory in the local runtime if it doesn't exist
        bin_path = os.path.join(self.runtime.temp_dir, 'bin')
        os.makedirs(bin_path, exist_ok=True)

        # Add this bin directory to the PATH for the runtime's subprocess calls
        # os.environ['PATH'] = f"{bin_path}:{os.environ['PATH']}"
        self.logger.info(f"Added {bin_path} to PATH for local command execution.")

        for cmd_file in cmd_files:
            parsed_commands = self.cmd_parser.parse_command_file(str(cmd_file))
            cmds.extend(parsed_commands)

            # [FIX] Correctly handle file extensions
            base_name = os.path.basename(cmd_file)
            cmd_name, ext = os.path.splitext(base_name)

            dest_name = cmd_name
            dest_path = os.path.join(bin_path, dest_name)

            # Copy the script to the bin directory
            from shutil import copy, copymode
            copy(cmd_file, dest_path)
            copymode(cmd_file, dest_path)  # Copy permissions (e.g., executable)
            os.chmod(dest_path, 0o755)  # Ensure it's executable

            self.logger.info(f"Copied command '{cmd_name}' to '{dest_path}'.")

        self.commands = cmds
        # self.logger.info(f"Added {len(cmds)} commands to the environment.")
        # self.logger.info(f"Available commands: {[cmd.name for cmd in cmds]}")

    def close(self):
        """Cleans up the underlying local runtime."""
        if self.runtime:
            self.runtime.close()
        # pass
