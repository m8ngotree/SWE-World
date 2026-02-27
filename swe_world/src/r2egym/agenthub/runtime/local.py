import os
import subprocess
import tempfile
import shutil
import uuid
import re
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

from r2egym.agenthub.runtime.base import ExecutionEnvironment
from r2egym.agenthub.utils.log import get_logger

from r2egym.agenthub.runtime.docker import (
    SKIP_FILES_COLLECT_CONTEXT,
    SKIP_FILES_ORIGINAL_CONTEXT
)

from swebench.harness.test_spec.test_spec import TestSpec
from r2egym.agenthub.trajectory.swebench_utils import make_test_spec
from swebench_fork_swegym.harness.test_spec import make_test_spec as make_test_spec_swegym
from swebench_fork_swerebench.harness.test_spec.test_spec import make_test_spec as make_test_spec_swerebench

base_dir = Path("your_path") # # you can change this to your testspec cache directory

# 判断是否存了test_spec
def check_cache_exists(file_path) -> bool:
    """
    检查缓存文件是否存在
    """
    print(f"check whether in: {file_path.exists()}")
    return file_path.exists()

# 读取存取的缓存
def load_from_json(filename):
    if "swe_rebench" in str(filename):
        from swebench_fork_swerebench.harness.test_spec.test_spec import TestSpec
    elif "swe_gym" in str(filename):
        from swebench_fork_swegym.harness.test_spec import TestSpec
    elif "swe_bench" in str(filename):
        from swebench.harness.test_spec.test_spec import TestSpec
    else:
        from r2egym.agenthub.trajectory.swebench_utils import TestSpec 
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 如果需要转回对象，可以解包
    return TestSpec(**data)

CMD_TIMEOUT = 300  # Default command timeout

class LocalRuntime(ExecutionEnvironment):
    """
    LocalRuntime interacts with a local shell, managing a temporary git repository.
    It mirrors the file system and git operations of DockerRuntime but executes
    them on the local machine for use in a simulated environment.
    """

    def __init__(self, ds: Dict[str, Any], logger=None, base_workdir: str = "/tmp/swe_sim_workdir", **kwargs):
        self.ds = ds
        self.instance_id = ds['instance_id']
        self.logger = logger if logger else get_logger("LocalRuntime")
        self.base_workdir = base_workdir

        
        # 基于字段判断数据源，可以基于docker_image字段或者自带的数据源字段
        dataset_type_hint = ds.get("docker_image", "") or ds.get("image_name", "") or ds.get("dataset_type", "")
        self.swebench_verified = "swebench" in dataset_type_hint
        self.swesmith = "swesmith" in dataset_type_hint
        self.swegym = "xingyaoww" in dataset_type_hint
        self.swerebench = "swerebench" in dataset_type_hint
        self.r2egym = "namanjain12" in dataset_type_hint
        self.sweworld = "sweworld" in dataset_type_hint
        self.data_source = None

        if self.swesmith or self.swegym or self.swerebench or self.r2egym or self.sweworld:
            self.swebench_verified = False

        if self.swesmith:
            self.data_source = "swesmith"
            # image_name = self.ds['image_name'].replace('__', '_1776_')
            image_name = self.ds['image_name']
            self.swebench_verified = False
            # self.docker_image = f'jyangballin/{image_name}:latest'
            self.docker_image = f'docker.io/{image_name}:latest'

        if self.swegym:
            self.data_source = "swegym"
            self.logger.info(f"swegym now in local.py, starting to make the test spec")
            self.cache_file = base_dir / "swe_gym" /f"{self.instance_id}.json"
            # print(f"this is dockerruntime cache_file: \n{self.cache_file}")
            self.logger.info(f"this is dockerruntime cache_file: \n{self.cache_file}")
            if check_cache_exists(self.cache_file):
                self.test_spec = load_from_json(self.cache_file)
                # self.logger.info(f"has read cached swegym test_spec:\n{self.cache_file}")
            else:
                self.logger.info(f"load from github")
                self.test_spec = make_test_spec_swegym(self.ds)

            # self.logger.info(f"swegym test_spec:\n{self.test_spec}")

        if self.swerebench:
            self.data_source = "swerebench"
            self.logger.info(f"swerebench now in local.py, starting to make the test spec")
            self.cache_file = base_dir / "swe_rebench" /f"{self.instance_id}.json"
            # self.logger.info(f"this is dockerruntime cache_file: \n{self.cache_file}")
            if check_cache_exists(self.cache_file):
                self.test_spec = load_from_json(self.cache_file)
                # self.logger.info(f"has read cached swe-rebench test_spec:\n{self.cache_file}")
            else:
                self.test_spec = make_test_spec_swerebench(self.ds)
                self.logger.info(f"load from github")

            # self.logger.info(f"swerebench test_spec:\n{self.test_spec}")
        
        if self.r2egym:
            self.data_source = "r2egym"
            self.logger.info(f"r2egym now in docker.py")
            # self.test_spec = make_test_spec(self.ds)
            # self.logger.info(f"r2egym test_spec:\n{self.test_spec}")
        
        if self.sweworld:
            self.data_source = "sweworld"
            self.logger.info(f"sweworld now in local.py")
            
        if self.swebench_verified:
            self.data_source = "swebench_verified"
            # also create a test spec for swebench verified dockers (useful for grading)
            self.logger.info(f"now in docker.py, starting to make the test spec")
            self.cache_file = base_dir / "swe_bench_verified" /f"{self.instance_id}.json"
            # self.logger.info(f"this is dockerruntime cache_file: \n{self.cache_file}")
            if check_cache_exists(self.cache_file):
                self.test_spec = load_from_json(self.cache_file)
                # self.logger.info(f"has read cached swe-bench-verified test_spec:\n{self.cache_file}")
            else:
                self.logger.info(f"load from github")
                self.test_spec = make_test_spec(self.ds)

            self.logger.info("has finished make test spec")
        
        # --- Modified repo setup lifecycle ---
        # Initialize paths, will be set in setup_env
        self.temp_dir = None
        self.repo_path = None
        self.temp_backup_dir = None
        self.repo_backup_path = None

        self.setup_env()

    # --- [NEW] Dispatcher method inspired by docker.py ---
    def setup_env(self):
        """Dispatches to the correct repository setup method based on dataset type."""
        # 确保 base_workdir 存在
        os.makedirs(self.base_workdir, exist_ok=True)

        # 统一把临时目录创建到 base_workdir 下，前缀 sim_repo_
        self.temp_dir = tempfile.mkdtemp(prefix="sim_repo_", dir=self.base_workdir)
        self.repo_path = os.path.join(self.temp_dir, "testbed")  # working copy

        print(f"Created temporary repo directory111: {self.repo_path}")

        # 备份目录也放同一个 base_workdir 下（或者你也可以用另一个 base_backup_workdir）
        self.temp_backup_dir = tempfile.mkdtemp(prefix="sim_repo_backup_", dir=self.base_workdir)
        self.repo_backup_path = os.path.join(
            self.temp_backup_dir,
            f"{self.ds.get('instance_id')}_backup"
        )

        self.logger.info(f"Created temporary runtime directory at: {self.temp_dir}")

        commit_hash = self.ds.get('base_commit') or self.ds.get('instance_id')
        # repo_url = f"https://github.com/{self.ds['repo']}.git"
        

        source_repo_path = self.ds.get('local_repo_path')

        if not source_repo_path:
            self.logger.error("Dataset is missing the required 'local_repo_path' field.")
            raise ValueError("Dataset must specify a 'local_repo_path' to a pre-cloned repository.")
        
        if not os.path.isdir(source_repo_path):
            self.logger.error(f"The provided 'local_repo_path' is not a valid directory: {source_repo_path}")
            raise ValueError(f"The path '{source_repo_path}' is not a valid directory.")


        # Clone the repository first
        try:
            self.logger.info(f"Copying repository from local source '{source_repo_path}' to '{self.repo_path}'...")
            # shutil.copytree is used to recursively copy the entire directory
            
            if os.path.exists(self.repo_path): # 先删除存在的仓库
                shutil.rmtree(self.repo_path)

            # shutil.copytree(source_repo_path, self.repo_path)
            # self.logger.info("Repository copy successful.")

            # 使用copytree存在一定的问题，复制得到仓库不一定和原始的仓库是一致的，比如原始link的文件，会被真实复制，所以改为直接git clone
            output_clone, exit_code_clone = self.run(f"git clone {source_repo_path} {self.repo_path}", workdir=self.temp_dir)

            if exit_code_clone != "0":
                self.logger.error(f"Failed to clone {source_repo_path} to {self.repo_path}, output: {output_clone}, exit code: {exit_code_clone}")
                raise RuntimeError(f"Failed to clone {source_repo_path} to {self.repo_path}, output: {output_clone}, exit code: {exit_code_clone}")
            else:
                # self.logger.info(f"Successfully clone {source_repo_path} to {self.repo_path}, output: {output_clone}, exit code: {exit_code_clone}")
                self.logger.info(f"Successfully clone {source_repo_path} to {self.repo_path}")
                
            # Dispatch to specific setup routine
            if self.swesmith:
                self._setup_repo_swesmith(commit_hash)
            elif self.swegym or self.swerebench:
                self._setup_repo_swegym(commit_hash) # They share the same simple logic
            elif self.swebench_verified:
                self._setup_repo_swebench(commit_hash)
            else: # Default for R2E-Gym
                self._setup_repo_default(commit_hash)

            # Backup the repository
            self.logger.info(f"Backing up repository from '{self.repo_path}' to '{self.repo_backup_path}'...")
            if os.path.exists(self.repo_backup_path):
                shutil.rmtree(self.repo_backup_path)
            # shutil.copytree(self.repo_path, self.repo_backup_path)
            # self.logger.info("Repository backup successful.")

            output_bak, exit_code_bak = self.run(
                f"git clone {self.repo_path} {self.repo_backup_path}",
                workdir=self.temp_backup_dir  # 确保这个目录存在
            )
            if exit_code_bak != "0":
                self.logger.error(f"Failed to backup repo from {self.repo_path} to {self.repo_backup_path}, output: {output_bak}, exit_code: {exit_code_bak}")
                raise RuntimeError(f"Failed to backup repo from {self.repo_path} to {self.repo_backup_path}, output: {output_bak}, exit_code: {exit_code_bak}")
            else:
                # self.logger.info(f"Successfully backup repo from {self.repo_path} to {self.repo_backup_path}, output: {output_bak}, exit_code: {exit_code_bak}")
                self.logger.info(f"Successfully backup repo from {self.repo_path} to {self.repo_backup_path}")


        except Exception as e:
            self.logger.error(f"Failed to copy repository from '{source_repo_path}': {e}")
            raise

        # commit_hash = self.ds.get('base_commit') or self.ds.get('instance_id')
        self.logger.info(f"Setting git state for the new copy to commit: {commit_hash}")

        # # Dispatch to specific setup routine
        # if self.swesmith:
        #     self._setup_repo_swesmith(commit_hash)
        # elif self.swegym or self.swerebench:
        #     self._setup_repo_swegym(commit_hash) # They share the same simple logic
        # elif self.swebench_verified:
        #     self._setup_repo_swebench(commit_hash)
        # else: # Default for R2E-Gym
        #     self._setup_repo_default(commit_hash)
            
        self.logger.info("Local repository setup complete.")

    # --- [NEW] Specific setup methods for each dataset type ---
    def _setup_repo_default(self, commit_hash: str):
        """Default setup: just checkout the commit."""
        self.logger.info(f"Checking out default commit: {commit_hash}...")
        output, exit_code = self.run(f"git checkout {commit_hash}", workdir=self.repo_path)
        if exit_code != "0":
            self.logger.error(f"Failed to checkout commit: {commit_hash}, output: {output}, exit code: {exit_code}")
            raise RuntimeError(f"Failed to checkout commit: {commit_hash}, output: {output}, exit code: {exit_code}")
        else:
            # self.logger.info(f"Setup repo to commit: {commit_hash} successfully, output: {output}, exit code: {exit_code}")
            self.logger.info(f"Setup repo to commit: {commit_hash} successfully")
    def _setup_repo_swebench(self, commit_hash: str):
        """Setup for SWE-bench: just checkout the commit."""
        self.logger.info(f"Checking out SWE-bench commit: {commit_hash}...")
        output, exit_code = self.run(f"git checkout {commit_hash}", workdir=self.repo_path)
        if exit_code != "0":
            self.logger.error(f"Failed to checkout commit: {commit_hash}, output: {output}, exit code: {exit_code}")
            raise RuntimeError(f"Failed to checkout commit: {commit_hash}, output: {output}, exit code: {exit_code}")
        else:
            # self.logger.info(f"Setup repo to commit: {commit_hash} successfully, output: {output}, exit code: {exit_code}")
            self.logger.info(f"Setup repo to commit: {commit_hash} successfully")        


    def _setup_repo_swegym(self, commit_hash: str):
        """Setup for SWE-Gym/SWE-rebench: checkout and hard reset."""
        self.logger.info(f"Checking out and resetting for SWE-Gym/rebench commit: {commit_hash}...")
        output, exit_code = self.run(f"git checkout {commit_hash}", workdir=self.repo_path)
        if exit_code != "0":
            self.logger.error(f"Failed to checkout commit: {commit_hash}, output: {output}, exit code: {exit_code}")
            raise RuntimeError(f"Failed to checkout commit: {commit_hash}, output: {output}, exit code: {exit_code}")
        else:
            # self.logger.info(f"Setup repo to commit: {commit_hash} successfully, output: {output}, exit code: {exit_code}")
            self.logger.info(f"Setup repo to commit: {commit_hash} successfully")
        self.run("git reset --hard", workdir=self.repo_path)

    def _setup_repo_swesmith(self, commit_hash: str):
        """Setup for SWE-smith: fetch, checkout, and clean."""
        self.logger.info(f"Performing SWE-smith setup for commit: {commit_hash}...")
        self.run("git fetch --all", workdir=self.repo_path)
        output, exit_code = self.run(f"git checkout {commit_hash}", workdir=self.repo_path)
        if exit_code != "0":
            self.logger.error(f"Failed to checkout commit: {commit_hash}, output: {output}, exit code: {exit_code}")
            raise RuntimeError(f"Failed to checkout commit: {commit_hash}, output: {output}, exit code: {exit_code}")
        else:
            # self.logger.info(f"Setup repo to commit: {commit_hash} successfully, output: {output}, exit code: {exit_code}")
            self.logger.info(f"Setup repo to commit: {commit_hash} successfully")
        self.run("git clean -fdq", workdir=self.repo_path)

    def run(self, code: str, timeout: int = CMD_TIMEOUT, workdir: str = None, **kwargs) -> Tuple[str, str]:
        """
        Executes a command in the local shell within the repository's directory.
        """
        exec_workdir = workdir if workdir else self.repo_path
        
        try:
            # Add the local bin to the PATH for this specific command
            local_bin = os.path.join(self.temp_dir, 'bin')
            env = os.environ.copy()
            env['PATH'] = f"{local_bin}:{env['PATH']}"

            result = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                cwd=exec_workdir,
                timeout=timeout,
                errors='replace',
                env=env
            )
            
            output = result.stdout + result.stderr
            # output = result.stdout
            
            exit_code = result.returncode
            output = re.sub(r'\x1b\[[0-9;]*m|\r', '', output)

            if exit_code != 0:
                # self.logger.warning(f"Command failed with exit code {exit_code}:\n>>> {code}\n--- Output ---\n{output} ---- stderr: \n{result.stderr}")
            
                return output, str(exit_code)
                
            return output, str(exit_code)

        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out after {timeout}s: {code}")
            return f"The command took too long to execute (>{timeout}s)", "-1"
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return f"Error: {repr(e)}", "-1"

    # def get_patch(self) -> str:
    #     """Get the git diff of the current state of the repository."""
    #     output, _ = self.run("git add -A && git diff --cached")
    #     return output

    def get_patch(self) -> str:
        """
        Get the diff of the current state of the repository, excluding skip files.
        """
        # stage everything first
        self.run("git add -A")

        # 生成排除 pathspec
        exclude_specs = " ".join(
            f"':(exclude){fname}'" for fname in SKIP_FILES_COLLECT_CONTEXT
        )

        # 组合 diff 命令
        # 例如: git diff --cached -- . ':(exclude)pyproject.toml'
        cmd = f"git diff --cached -- . {exclude_specs}".strip()

        # self.logger.info(f"[get_patch] Running: {cmd}")

        output, _ = self.run(cmd)
        return output

    # 获取当前修改了哪些文件
    def get_modified_files(self) -> List[str]:
        """Get the list of modified files."""
        output, _ = self.run("git add -A && git diff --cached --name-only")
        return output.splitlines() if output else []

    def apply_patch(self, patch: str) -> Tuple[str, str]:
        """Applies a patch file using 'git apply'."""
        patch_path = os.path.join(self.temp_dir, f"patch_{uuid.uuid4()}.patch")
        with open(patch_path, "w", errors='replace') as f:
            f.write(patch)
        
        output, error_code = self.run(f"git apply --whitespace=fix {patch_path}", workdir=self.repo_path)
        os.remove(patch_path)
        return output, error_code

    # --- [NEW] Added for feature parity with DockerRuntime ---
    def create_file(self, file_path: str, content: str):
        """Creates a file (including parent directories) with the given content."""
        full_path = Path(self.repo_path) / file_path
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, errors='replace')
            # self.logger.info(f"Created/updated file at: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to create file {file_path}: {e}")

    def read_file(self, file_path: str) -> str:
        """Reads the content of a file within the repo."""
        # full_path = Path(self.repo_path) / file_path

        # 先判断一下是否是repo_path开头
        if file_path.startswith(self.repo_path):
            full_path = Path(file_path)
        else:
            full_path = Path(self.repo_path) / file_path
        if not full_path.exists():
            return f"Error: File not found at {file_path}, full_path: {full_path}"
        try:
            return full_path.read_text(errors='replace')
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, file_path: str, content: str):
        """Writes content to a file."""

        if file_path.startswith(self.repo_path):
            full_path = Path(file_path)
        else:
            full_path = Path(self.repo_path) / file_path
        try:
            full_path.write_text(content, errors='replace')
            # self.logger.info(f"Wrote content to file at: {full_path}")
        except Exception as e:
            self.logger.error(f"Failed to write content to file {file_path}: {e}")

    def close(self):
        """Cleans up all temporary directories created by this runtime."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            # self.logger.info(f"Cleaning up runtime directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        if self.temp_backup_dir and os.path.exists(self.temp_backup_dir):
            self.logger.info(f"Cleaning up backup directory: {self.temp_backup_dir}")
            shutil.rmtree(self.temp_backup_dir, ignore_errors=True)

        # 将它们设置为 None 防止意外的二次删除
        self.temp_dir = None
        self.temp_backup_dir = None

    def reset(self):
        """Resets the runtime by creating a fresh repository clone."""
        self.close()
        self.setup_env()

    def get_task_instruction(self) -> str:
        try:
            content = self.ds["problem_statement"]
            match = re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL)
            return match.group(1).strip() if match else content
        except Exception:
            return self.ds["problem_statement"]
    
    def checkout(self, commit_hash: str) -> Tuple[str, str]:
        """Checks out a specific git commit."""
        output, error_code = self.run(f"git checkout {commit_hash}", workdir=self.repo_path)
        return output, error_code