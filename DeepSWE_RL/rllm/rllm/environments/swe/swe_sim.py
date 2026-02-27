import json
import os
import yaml
import logging
import numpy as np
from typing import Any, Dict, List, Union

from datasets import Dataset, load_dataset
import time
# -----------------------------------------------------------------------------
# R2E-Gym 导入部分
# -----------------------------------------------------------------------------
try:
    import r2egym
    from r2egym.agenthub.action import Action
    from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
    from r2egym.logging import setup_logging, INFO
    
    # [NEW] 导入模拟环境和模拟Agent
    from r2egym.agenthub.environment.sim_env import SimulatedEnv
    from r2egym.agenthub.agent.simulator_muti_wo_original_new_user_new_reward import SimulatorAgent
except ImportError:
    r2egym = None
    EnvArgs = None
    RepoEnv = None
    Action = None
    SimulatedEnv = None
    SimulatorAgent = None

from rllm.environments.base.base_env import BaseEnv

# logger = logging.getLogger(__name__)

try:
    R2EGYM_PATH = os.path.dirname(r2egym.__file__)
except Exception:
    R2EGYM_PATH = ""

# -----------------------------------------------------------------------------
# 工具脚本路径配置
# -----------------------------------------------------------------------------
# 模拟环境通常需要特定的工具脚本，这里我们复用原有的工具路径
# 但注意：SimulatedEnv 使用 LocalRuntime，需要确保这些路径在训练节点上是可访问的

R2EGYM_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/file_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/search.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/finish.py"),
]

# OPENHANDS_COMMAND_FILES = [
#     os.path.join(R2EGYM_PATH, "agenthub/tools/str_replace_editor.py"),
#     os.path.join(R2EGYM_PATH, "agenthub/tools/execute_bash.py"),
#     os.path.join(R2EGYM_PATH, "agenthub/tools/submit.py"),
# ]

OPENHANDS_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools_for_rl/str_replace_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools_for_rl/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools_for_rl/submit.py"),
]

SWEAGENT_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/str_replace_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/submit.py"),
]

R2E_ENV_IDS = [
    "R2E-Gym/R2E-Gym-Subset",
    "R2E-Gym/R2E-Gym-V1",
    "R2E-Gym/R2E-Gym-Lite",
    "R2E-Gym/SWE-Bench-Verified",
    "R2E-Gym/SWE-Bench-Lite",
]
DEFAULT_R2E_ENV_ID = "R2E-Gym/R2E-Gym-Lite"




import json
from typing import Any, Dict, List, Union

def restore_top_level(
    item: Dict[str, Any],
    changed_field_name: str = "_changed_keys",
    drop_changed_field: bool = False,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    将 convert_top_level 处理后的单条记录恢复原样：
    - 只恢复顶层被转换过的 key：对 item[changed_keys] 中列出的字段做 json.loads
    - 其它字段保持不变
    - 默认删除 changed_field_name 字段

    参数：
    - drop_changed_field: 是否删除 changed_field_name
    - strict: 若 json.loads 失败是否抛异常；False 则保留原字符串
    """
    if not isinstance(item, dict):
        raise TypeError(f"Each record must be a dict, got: {type(item)}")

    changed_keys = item.get(changed_field_name, [])
    if changed_keys is None:
        changed_keys = []
    if not isinstance(changed_keys, list) or not all(isinstance(k, str) for k in changed_keys):
        raise TypeError(f"{changed_field_name} must be a list[str], got: {type(changed_keys)}")

    out = dict(item)  # shallow copy

    for k in changed_keys:
        if k not in out:
            # 有些 key 可能不存在了：跳过即可
            continue
        v = out[k]
        if not isinstance(v, str):
            # 按你的转换逻辑，这里理论上应当是 str；不是的话直接跳过或报错
            if strict:
                raise TypeError(f"Expected a JSON string for key '{k}', got: {type(v)}")
            print(f"转化数据的时候出现与预计的不匹配的： key: {key}, value: {v}")
            continue
        try:
            out[k] = json.loads(v)
        except Exception:
            if strict:
                raise ValueError(f"Failed to json.loads for key '{k}': {v[:200]}")  # 截断展示
            # 非严格：保持原字符串不动
            out[k] = v

    if drop_changed_field:
        out.pop(changed_field_name, None)

    return out


def restore_json_data(
    data: Any,
    changed_field_name: str = "_changed_keys",
    drop_changed_field: bool = True,
    strict: bool = False,
) -> Any:
    """
    支持两种输入：
    1) 顶层是 list[dict]：逐条恢复
    2) 顶层是 dict：恢复单条
    """
    if isinstance(data, list):
        return [
            restore_top_level(
                x,
                changed_field_name=changed_field_name,
                drop_changed_field=drop_changed_field,
                strict=strict,
            )
            for x in data
        ]
    elif isinstance(data, dict):
        return restore_top_level(
            data,
            changed_field_name=changed_field_name,
            drop_changed_field=drop_changed_field,
            strict=strict,
        )
    else:
        raise TypeError(f"Top-level JSON must be a list or dict, got: {type(data)}")


# [NEW] 简单的包装类，用于适配 SimulatedEnv 的参数接口
class SimEnvArgs:
    def __init__(self, ds: Dict[str, Any]):
        self.ds = ds



class SWEEnv(BaseEnv):
    """Software Engineering Environment for code-related tasks.
    Supports both Docker-based execution (RepoEnv) and LLM-based Simulation (SimulatedEnv).
    """

    def __init__(
        self,
        entry: dict | None = None,
        dataset: Dataset | None = None,
        idx: int | None = None,
        step_timeout: int = 90,
        reward_timeout: int = 300,
        backend: str = "kubernetes",  # 支持 'simulated', 'docker', 'kubernetes'
        delete_image: bool = False,
        verbose: bool = False,
        scaffold: str = "r2egym",
        # [NEW] 模拟器特有参数
        simulator_yaml: str | None = None,
        sim_reward_max_workers: int = 4,
    ):
        """Initialize the SWE environment.
        """
        # 数据集加载逻辑
        if entry is not None:
            self.entry = entry
            self.dataset = None
            self.idx = None
        else:
            if dataset is None:
                dataset = load_dataset(DEFAULT_R2E_ENV_ID, split="test")
            self.dataset = dataset

            if idx is None:
                idx = np.random.randint(0, len(self.dataset))
            assert 0 <= idx < len(self.dataset), "Selected index out of range"
            self.idx = idx
            self.entry = self.dataset[idx]


        
        # 恢复 entry

        if "_changed_keys" in entry:
            print(f"注意这里的存在嵌套的value，现在重新转为json")
            self.entry = restore_json_data(entry)

        self.step_timeout = step_timeout
        self.reward_timeout = reward_timeout
        self.total_steps = 0
        self.delete_image = delete_image
        self.backend = backend
        self.env = None
        self.verbose = verbose
        self.scaffold = scaffold
        
        # [NEW] 保存模拟器配置
        self.simulator_yaml = simulator_yaml
        self.sim_reward_max_workers = sim_reward_max_workers
        self.simulator_agent = None 


        # 输出这些配置信息
        print(f"!!!!!!!!!!!!!!!!!SWEEnv: Using backend: {self.backend}")
        print(f"!!!!!!!!!!!!!!!!!SWEEnv: Using scaffold: {self.scaffold}")
        print(f"!!!!!!!!!!!!!!!!!SWEEnv: Using simulator_yaml: {self.simulator_yaml}")
        print(f"!!!!!!!!!!!!!!!!!SWEEnv: Using sim_reward_max_workers: {self.sim_reward_max_workers}")
        print(f"!!!!!!!!!!!!!!!!!SWEEnv: Using simulator_agent: {self.simulator_agent}")

        assert scaffold in ["r2egym", "sweagent", "openhands"], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"

    def _init_simulator_agent(self, logger):
        """
        [NEW] Helper to initialize the SimulatorAgent based on YAML config.
        Only called when backend == 'simulated'.
        """
        if self.simulator_agent is not None:
            return self.simulator_agent

        if not self.simulator_yaml:
            raise ValueError("Backend is 'simulated' but 'simulator_yaml' is not provided.")

        try:
            with open(self.simulator_yaml, 'r') as f:
                config_data = yaml.safe_load(f)
                simulator_config_list = config_data.get('models', [])
                if not simulator_config_list:
                     raise ValueError(f"No models found in {self.simulator_yaml}")
        except Exception as e:
            print(f"Failed to load simulator YAML: {e}")
            raise e

        self.simulator_agent = SimulatorAgent(
            simulator_config=simulator_config_list,
            logger=logger,
            simulator_config_path=self.simulator_yaml
        )
        return self.simulator_agent

    def reset(self) -> tuple[str, dict]:
        """Reset the environment to initial state.
        Switch logic based on backend type.
        """
        # 1. 模拟环境 (Simulated Backend)
        if self.backend == "simulated":
            if "local_repo_path" not in self.entry:
                raise KeyError("Simulated backend requires entry['local_repo_path'].")

            if not self.env:
                # 初始化 Simulator Agent
                logger = setup_logging(
                    name=self.entry["docker_image"].replace("/", "_"),
                    log_file=None,
                    console=True,
                    level=INFO,
                )

                sim_agent = self._init_simulator_agent(logger)
                
                # 构造参数包装器
                sim_env_args = SimEnvArgs(ds=self.entry)
                
                # 初始化 SimulatedEnv (使用 LocalRuntime)
                # 注意：LocalRuntime 依赖 self.entry 中的 'local_repo_path'
                # 确保传入的 dataset entry 包含此字段
                self.env = SimulatedEnv(
                    args=sim_env_args,
                    simulator_agent=sim_agent,
                    logger=logger,
                    step_timeout=self.step_timeout
                )
            else:
                self.env.reset()
        
        # 2. 真实环境 (Docker/Kubernetes Backend)
        else:
            if not self.env:
                env_args = EnvArgs(ds=self.entry)
                self.env = RepoEnv(
                    env_args, 
                    backend=self.backend, 
                    step_timeout=self.step_timeout, 
                    reward_timeout=self.reward_timeout, 
                    verbose=self.verbose
                )
            else:
                self.env.reset()

        # 3. 添加工具命令 (通用逻辑)
        print(f"!!!!!!!!!!!!!!!!!!![NEW] Using {self.scaffold} scaffold")
        if self.scaffold == "r2egym":
            print(f"!!!!!!!!!!!!!!!!!!![NEW] Using R2EGym scaffold")
            self.env.add_commands(R2EGYM_COMMAND_FILES)
        elif self.scaffold == "openhands":
            print(f"!!!!!!!!!!!!!!!!!!![NEW] Using OpenHands scaffold")
            self.env.add_commands(OPENHANDS_COMMAND_FILES)
        else:
            print(f"!!!!!!!!!!!!!!!!!!![NEW] Using SWEAgent scaffold")
            self.env.add_commands(SWEAGENT_COMMAND_FILES)
        
        
        self.total_steps = 0

        # 获取任务描述
        # LocalRuntime 和 DockerRuntime 都实现了 get_task_instruction
        if self.backend == "simulated":
             # SimulatedEnv 的 runtime 是 LocalRuntime
             task_instruction = self.env.runtime.get_task_instruction()
        else:
             task_instruction = self.env.get_task_instruction()

        return (
            task_instruction,
            {},
        )

    def compute_final_reward(self):
        """
        计算最终奖励。
        SimulatedEnv 使用基于 LLM 的模拟奖励计算。
        RepoEnv 使用真实的测试执行。
        """
        if self.backend == "simulated":
            # time.sleep(30)
            # return 1.0
            # # [NEW] 调用 SimulatedEnv 特有的模拟奖励计算方法
            # 返回值通常是 (reward, output_str)
            # reward, _ = self.env._calculate_simulated_reward_swebv(
            #     max_workers=self.sim_reward_max_workers
            # )
            #这里走和推理一致的逻辑
            reward, _ = self.env._calculate_simulated_reward_swebv_hls_fix_train_and_inference(
                max_workers=self.sim_reward_max_workers
            )
            return float(reward)
        else:
            # 原有的 Docker 环境奖励计算
            return self.env.compute_reward()

    def step(self, action: str | Action) -> tuple[str, float, bool, bool, dict]:
        """Take a step in the environment.
        """
        if isinstance(action, str):
            action_obj: Action = Action.from_string(action)
        else:
            action_obj = action

        if not action_obj.function_name:
            return "", 0, False, False, {}

        # SimulatedEnv.step 和 RepoEnv.step 的签名基本一致
        # SimulatedEnv 返回: (observation, reward, done, info)
        # 注意：SimulatedEnv 的 observation 包含 raw_simulation 字段，这里我们转 string
        obs, reward, done, info = self.env.step(action_obj)
        
        self.total_steps += 1
        
        # rllm 要求返回 (obs, reward, done, truncated, info)
        # 这里的 truncated 我们暂时设为 False，由外部 TimeLimit wrapper 控制或由 info 控制
        return str(obs), reward, done, False, info

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if self.env is not None:
            self.env.close()

        # 仅在 Docker 模式下才需要删除镜像
        if self.delete_image and self.backend != "simulated":
            docker_image = self.env.runtime.docker_image
            os.system(f"docker rmi {docker_image}")

    @staticmethod
    def from_dict(extra_info: dict | str) -> "SWEEnv":
        """Create an environment instance from JSON configuration.
        """
        import inspect

        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        sig = inspect.signature(SWEEnv.__init__)
        init_params = {}
        for param_name, param in sig.parameters.items():
            
            if param_name == "self":
                continue
            if param_name in extra_info:
                init_params[param_name] = extra_info[param_name]
        
        # 将整个 extra_info 作为 entry 传入
        init_params["entry"] = extra_info
        return SWEEnv(**init_params)