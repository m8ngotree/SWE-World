import re
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import concurrent.futures
import threading
import docker
from tqdm import tqdm

from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent

from r2egym.agenthub.agent.simulator import SimulatorAgent
from r2egym.agenthub.environment.sim_env import SimulatedEnv

from r2egym.docker_bash_utils.docker_list_tags import fetch_docker_tags
from r2egym.agenthub.utils.log import get_logger
from r2egym.logging import setup_logging, INFO
from r2egym.agenthub.utils.utils import get_parsed_commit

from fire import Fire
from r2egym.agenthub.utils.utils import match_dockerimage_to_repo
from r2egym.agenthub import SUPPORTED_REPOS
from datasets import load_dataset, load_from_disk
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
import time

##############################################################################
# Initialize Logger
##############################################################################
logger = get_logger(__name__)  # Initialize the logger

##############################################################################
# Initialize File Lock for Thread-Safe Writing
##############################################################################
file_lock = threading.Lock()


##############################################################################
# Utility Function
##############################################################################
def get_docker_images(repo_name) -> List[str]:
    """
    Fetches the list of Docker images available for the base image.

    Returns:
        A list of Docker image tags.
    """
    base_image = f"namanjain12/{repo_name}new"
    tags = fetch_docker_tags(base_image)
    docker_image_list = [f"{base_image}:{x['name']}" for x in tags]
    return docker_image_list


def prepull_docker_image(docker_image: str) -> bool:
    """Prepulls a single Docker image if it doesn't exist locally."""
    try:
        client = docker.from_env()
        
        # [FIX 3] A more robust way to check if the image exists
        try:
            client.images.get(docker_image)
            logger.info(f"Docker image already exists locally: {docker_image}")
            return True
        except docker.errors.ImageNotFound:
            # Image doesn't exist locally, so pull it
            logger.info(f"Pulling Docker image: {docker_image}")
            client.images.pull(docker_image)
            logger.info(f"Successfully pulled Docker image: {docker_image}")
            return True
    except Exception as e:
        logger.error(f"Failed to pull Docker image {docker_image}: {e}")
        return False

def prepull_docker_images(ds_selected: List[Dict], max_workers: Optional[int] = None) -> None:
    """
    Prepulls all Docker images in parallel before starting the main execution.
    
    Args:
        ds_selected: List of dataset entries containing docker_image keys
        max_workers: Maximum number of threads for parallel pulling
    """
    # Extract unique Docker images
    docker_images = list(set([ds_entry["docker_image"] for ds_entry in ds_selected]))
    logger.info(f"Starting parallel prepull of {len(docker_images)} unique Docker images...")
    
    # Use ThreadPoolExecutor for I/O bound operations like Docker pulls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pull tasks
        future_to_image = {
            executor.submit(prepull_docker_image, docker_image): docker_image
            for docker_image in docker_images
        }
        
        # Track results
        successful_pulls = []
        failed_pulls = []
        
        for future in concurrent.futures.as_completed(future_to_image):
            docker_image = future_to_image[future]
            try:
                success = future.result()
                if success:
                    successful_pulls.append(docker_image)
                else:
                    failed_pulls.append(docker_image)
            except Exception as e:
                logger.error(f"Exception during prepull of {docker_image}: {e}")
                failed_pulls.append(docker_image)
    
    logger.info(f"Prepull completed. Success: {len(successful_pulls)}, Failed: {len(failed_pulls)}")
    if failed_pulls:
        logger.warning(f"Failed to pull images: {failed_pulls}")


##############################################################################
# editagent Functions
##############################################################################
def run_agent_with_restarts(
    agent,
    env,
    max_steps=40,
    num_restarts=1,
    temperature=0.0,
    max_steps_absolute=50,
    use_fn_calling: bool = True,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 131072,
):
    """
    Iterative eval protocol:
    - normally run the agent
    - run for maximum num_iterations = 3 times
    - stop if trajectory.exit_reason == "agent"
    - otherwise continue iteratively till maximum iterations
    - finally choose the trajectory with the lowest number of steps
    - note restarts and iterative_evals are different (so just use one of them | add an assert flag)
    - also if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    """
    steps_per_agent = max_steps // num_restarts
    logger.warning(f"running {steps_per_agent} steps per agent")

    # only one of restarts > 1 and iterative_eval can be True
    iterative_eval = max_iterations > 1
    assert not (num_restarts > 1 and iterative_eval), "only one of restarts > 1 and iterative_eval can be True"
    logger.warning(f"Using iterations: {max_iterations}, using iterative protocol: {iterative_eval}")

    # if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    # if temperature is 0, create list of increasing temperatures up to 0.2
    if temperature == 0:
        temperatures = [0.0 + 0.1 * i for i in range(max_iterations)]
        temperatures = [min(t, 0.2) for t in temperatures]  # cap at 0.2
    else:
        temperatures = [temperature] * max_iterations
    logger.warning(f"Using temperatures: {temperatures}")

    # run the agent in iterative protocol
    trajectories = []
    for iteration in range(max_iterations):
        for idx in range(num_restarts):
            logger.warning(f"running agent at idx: {idx+1}")
            trajectory = agent.run(
                env,
                max_steps=steps_per_agent,
                temperature=temperatures[iteration],
                max_steps_absolute=max_steps_absolute,
                use_fn_calling=use_fn_calling,
                scaffold=scaffold,
                max_token_limit=max_tokens,
            )
            # remove reproduce.py
            # env.runtime.run('rm reproduce_issue.py')
        if trajectory.exit_reason == "agent":
            logger.warning(f"agent self-finished at iteration: {iteration}")
            return trajectory
        # otherwise continue iteratively
        trajectories.append(trajectory)
        # reset the env
        # env.reset()

    # choose the trajectory with the lowest number of steps
    trajectory = min(trajectories, key=lambda x: x.num_steps)
    return trajectory

def runagent(
    ds,
    traj_dir,
    exp_name: Optional[str] = None,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    llm_name="gpt-4o",
    llm_api_key: str = None, # 这个参数现在其实没有使用，如果要用，后面还是要修改一下
    temperature=0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 131072,
    ip: str = "",
    used_yaml: str= "",
    execution_mode: str = "docker",  # New argument: "docker" or "simulated"
    simulator_yaml: str = None, # [NEW] ONLY pass the yaml file path for simulator config
    sim_reward_max_workers: int = 4,
    cal_reward: bool = True,
) -> Optional[str]:
    """
    Runs the editagent agent on a specified Docker image.
    """

    if "instance_id" not in ds:
        insid = f"{ds['repo_name']}_{ds['docker_image']}"
        ds["instance_id"] = insid.replace("/", "_")
 
    logger = setup_logging(
        name=ds["docker_image"].replace("/", "_"),
        log_file=f"run_logs/{exp_name}/{ds['docker_image'].replace('/', '_')}.log",
        console=True,
        level=INFO,
    )
    logger.info(f"Starting editagent on Docker image: {ds['docker_image']}")
    logger.info(f"Using LLM: {llm_name}")
    logger.info(f"Max Steps: {max_steps}")

    assert scaffold in ["r2egym", "sweagent", "openhands"], f"Scaffold is {scaffold}, must be one of [r2egym, sweagent, openhands]"
    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # # Initialize environment arguments
    # env_args = EnvArgs(ds=ds)

    # # Initialize the RepoEnv
    # env = RepoEnv(env_args, logger=logger, backend=backend)
    logger.info(f"runagent with execution_mode: {execution_mode}")

    if execution_mode == "docker":
        logger.info(f"Using DOCKER execution mode on image: {ds['docker_image']}")
        env_args = EnvArgs(ds=ds)
        env = RepoEnv(env_args,ip=ip, logger=logger, backend=backend)
    elif execution_mode == "simulated":
        logger.info(f"Using SIMULATED execution mode for instance: {ds['docker_image']}")
        
        # [NEW] Load simulator config from YAML
        if not simulator_yaml:
            raise ValueError("Execution mode is 'simulated' but 'simulator_yaml' is not provided.")
            
        logger.info(f"Loading simulator configuration from {simulator_yaml}")
        try:
            with open(simulator_yaml, 'r') as f:
                config_data = yaml.safe_load(f)
                simulator_config_list = config_data.get('models', [])
                if not simulator_config_list:
                     raise ValueError(f"No models found in {simulator_yaml}")
                logger.info(f"Loaded {len(simulator_config_list)} models for simulation.")
        except Exception as e:
            logger.error(f"Failed to load simulator YAML: {e}")
            raise e

        # 1. Initialize the Simulator Agent with the loaded config list
        simulator_agent = SimulatorAgent(
            simulator_config=simulator_config_list, # [NEW] Pass the list of models
            logger=logger,
            simulator_config_path=simulator_yaml
        )
        
        # 2. Create EnvArgs-like object for SimulatedEnv
        class SimEnvArgs:
            def __init__(self, ds):
                self.ds = ds
        
        sim_env_args = SimEnvArgs(ds)
        
        # 3. Initialize the Simulated Environment
        env = SimulatedEnv(args=sim_env_args, simulator_agent=simulator_agent, logger=logger)
    elif execution_mode == "collect":
        logger.info(f"Using COLLECT execution mode for instance: {ds['docker_image']}")
        env_args = EnvArgs(ds=ds)
        env = RepoEnv(env_args, logger=logger,ip=ip, backend=backend, collect_contexts=True)
    else:
        raise ValueError(f"Invalid execution_mode: {execution_mode}. Must be 'docker' or 'simulated' or 'collect'.")

    print("has set up env")
    
    # set agent args
    if used_yaml:
        agent_args = AgentArgs.from_yaml(
            Path(used_yaml)
        )
    else:
        if use_fn_calling:
            assert scaffold != "sweagent", "SWEagent scaffold does not support fn calling"

            if execution_mode == "docker":
                agent_args = AgentArgs.from_yaml(
                    Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_fn_calling.yaml")
                )
            else:
                agent_args = AgentArgs.from_yaml(
                    Path(f"./src/r2egym/agenthub/config/{scaffold}_sim/edit_fn_calling.yaml")
                )
        else:
            if execution_mode == "docker":
                agent_args = AgentArgs.from_yaml(
                        Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_non_fn_calling.yaml")
                    )
            else:
                agent_args = AgentArgs.from_yaml(
                    Path(f"./src/r2egym/agenthub/config/{scaffold}_sim/edit_non_fn_calling.yaml")
                )
    agent_args.llm_name = llm_name

    # Initialize the agent
    agent = Agent(name="EditAgent", args=agent_args, logger=logger)

    # run agent editagent
    try:
        trajectory = run_agent_with_restarts(
            agent,
            env,
            max_steps=max_steps,
            num_restarts=num_restarts,
            temperature=temperature,
            max_steps_absolute=max_steps_absolute,
            use_fn_calling=use_fn_calling,
            max_iterations=max_iterations,
            scaffold=scaffold,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(
            f"Error during agent run for Docker image {ds['docker_image']}: {e}"
        )

        # close env and docker runtime
        env.close()
        return None

    # 输出推理完成
    logger.info(f"Inference completed for Docker image: {ds['docker_image']}")

    # For simulated mode, the final reward calculation is complex.
    if execution_mode == "docker" or execution_mode == "collect":
        logger.info(f"Getting GT outputs for Docker image: {ds['docker_image']}")
        reward_calc_time = time.time()
        reward, test_output = env.runtime._calculate_reward(get_test_output=True, timeout=max_reward_calc_time)
        logger.info(f"after _calculate_reward GT outputs: \n{test_output}")
        logger.info(f"after _calculate_reward GT reward: {reward}")
        reward_calc_time = time.time() - reward_calc_time
    else: # execution_mode == "simulated"
        logger.info(f"Calculating reward in SIMULATED mode for instance: {ds['docker_image']}")
        reward_calc_time = time.time()
        # [NEW] Call the new method on the SimulatedEnv instance
        # start_time = time.time()
        if cal_reward:
            # reward, test_output = env._calculate_simulated_reward_swebv(max_workers=sim_reward_max_workers)
            reward, test_output = env._calculate_simulated_reward_swebv_hls_fix_train_and_inference(max_workers=sim_reward_max_workers)
        else:
            reward = 0
            # 标记一下不计算reward
            test_output = "Remind: reward not calculated"
            print("[NEW] Reward not calculated")
        reward_calc_time = time.time() - reward_calc_time
        logger.info(f"Simulated reward: {reward}")

        collected_contexts = env.get_collected_contexts()

        if collected_contexts:
            collection_file_path = Path(traj_dir) / f"{ds['instance_id']}_simulation_contexts_collection.json"
            logger.info(f"Saving {len(collected_contexts)} collected contexts to {collection_file_path}")
            with open(collection_file_path, 'w') as f:
                json.dump(collected_contexts, f, indent=2, ensure_ascii=False)

    # --- [NEW] Data Saving Logic ---
    if execution_mode == "collect":
        collected_contexts = env.get_collected_contexts()
        if collected_contexts:
            collection_file_path = Path(traj_dir) / f"{ds['instance_id']}_simulation_data_collection.json"
            logger.info(f"Saving {len(collected_contexts)} collected contexts to {collection_file_path}")
            with open(collection_file_path, 'w') as f:
                json.dump(collected_contexts, f, indent=2, ensure_ascii=False)
        else:
            logger.warning(f"No collected data found for instance: {ds['instance_id']}")
    # --- End of Data Saving Logic ---

    env.close()

    # update the trajectory object
    trajectory.reward = reward
    trajectory.test_output = test_output
    trajectory.ds = ds
    trajectory.exp_name = exp_name
    trajectory.reward_calc_time = reward_calc_time # time taken to calculate reward
    logger.warning(f"time taken to calculate reward in seconds: {reward_calc_time:.2f}")

    logger.info(f"editagent completed for Docker image: {ds['docker_image']}")
    # close env and docker runtime
    logger.info(f"Closing environment for Docker image: {ds['docker_image']}")
    return trajectory.model_dump_json()


def runagent_multiple(
    dataset: str,
    split: str,
    k: int = 1,
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    start_idx=0,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    max_workers: Optional[int] = None,
    llm_name="gpt-4o",
    llm_api_key: Optional[str] = None,
    use_existing: bool = True,
    skip_existing: bool = False,
    temperature: float = 0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    prepull_images: bool = False,
    max_tokens: int = 131072,
    ip: str = "",
    used_yaml:str = "",
    execution_mode: str = "docker",
    simulator_yaml: str = None, # [NEW] ONLY pass this
    sim_reward_max_workers: int = 4,
    cal_reward: bool = True,
):
    """
    Runs the editagent agent on the first k Docker images.
    """
    # Load the dataset
    if dataset.endswith(".json"):
        with open(dataset, "r") as f:
            ds = json.load(f)
            print(f"loaded dataset from {dataset}, length: {len(ds)}")
    else:
        logger.info(f"use load_dataset")
        # ds = load_dataset(dataset, split=split)
        ds = load_from_disk(dataset)
        print(ds["instance_id"][:5])
        logger.info(f"{len(ds)}, {k}, {start_idx}")
        # shuffle the dataset
        
        selected_idx = range(start_idx, start_idx + k)
        ds_selected = [ds[i] for i in selected_idx]
        
        ds = ds.shuffle(seed=42)

    logger.info(f"{len(ds)}, {k}, {start_idx}")
    # shuffle the dataset
    # ds = ds.shuffle(seed=42)

    # get selected idxs
    selected_idx = range(start_idx, start_idx + k)
    ds_selected = [ds[i] for i in selected_idx]

    # print ds_selected stats
    logger.info(
        f"Dataset: {dataset}, Split: {split}, Num_total: {len(ds)}, Start Index: {start_idx}, k: {k}"
    )
    logger.info(f"Starting editagent on {len(ds_selected)} Docker images.")

    # 打印具体的docker_image列表
    logger.info("Selected docker_image list:")
    for entry in ds_selected:
        logger.info(f"- {entry['docker_image']}")

    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure traj_dir exists
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate a filename for the JSONL file
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"

    if use_existing:
        if jsonl_file.exists():
            with open(jsonl_file) as f:
                existing_dockers = []
                for line in f.readlines():
                    try:
                        existing_dockers.append(
                            Trajectory.load_from_model_dump_json(line).ds[
                                "docker_image"
                            ]
                        )
                    except:
                        print("error in jsonl file")

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["docker_image"] not in existing_dockers
            ]

            # Log the existing Docker images that were skipped
            logger.info(
                f"Skipping {len(existing_dockers)} Docker images already present in {jsonl_file}."
            )
            if existing_dockers:
                logger.info("Existing docker_image list:")
                for docker in existing_dockers:
                    logger.info(f"- {docker}")
        logger.info(f"经过筛选后，还需要处理的sample数量为：{len(ds_selected)}")
        import time
        time.sleep(10)

    if skip_existing:
        old_jsonl_files_glob = f"{exp_name[:-1]}*"
        for old_jsonl_file in traj_dir_path.glob(old_jsonl_files_glob):
            with open(old_jsonl_file) as f:
                existing_dockers = [
                    loadline["ds"]["docker_image"]
                    for line in f
                    for loadline in [json.loads(line)]
                    if loadline["reward"] == 1
                ]

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["docker_image"] not in existing_dockers
            ]

            # Log the existing Docker images that were skipped
            logger.info(
                f"Skipping {len(existing_dockers)} Docker images with reward=1 already present in {old_jsonl_file}."
            )
            if existing_dockers:
                logger.info("Existing docker_image list with reward=1:")
                for docker in existing_dockers:
                    logger.info(f"- {docker}")

    logger.info(
        f"Starting editagent on {len(ds_selected)} Docker images after filtering."
    )

    # Prepull all Docker images in parallel before starting main execution
    if ds_selected and prepull_images:
        logger.info("Prepulling Docker images before starting main execution...")
        prepull_docker_images(ds_selected, max_workers=max_workers)
        logger.info("Docker image prepull completed.")


    logger.info(f"running agent multiple with execution_mode: {execution_mode}")
    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {}
        for idx, ds_entry in enumerate(ds_selected, start=1):
            docker_image = ds_entry["docker_image"]
            docker_ip = ds_entry.get("ip","")
            if docker_ip:
                ip_used = docker_ip
                logger.info(f"{docker_image}自带ip, 直接使用对应的ip: {docker_ip}")
                logger.info(f"[{idx}/{len(ds_selected)}] 正在提交任务: {docker_image}")
            else:
                ip_used = ip
                logger.info(f"{docker_image}条目中不带ip, 使用提前设定好的ip: {ip}")
                logger.info(f"[{idx}/{len(ds_selected)}] 正在提交任务: {docker_image}")
            fut = executor.submit(
                runagent,
                ds=ds_entry,
                exp_name=exp_name,
                max_steps=max_steps,
                num_restarts=num_restarts,
                traj_dir=traj_dir,
                max_steps_absolute=max_steps_absolute,
                llm_name=llm_name,
                llm_api_key=llm_api_key,
                temperature=temperature,
                use_fn_calling=use_fn_calling,
                backend=backend,
                max_reward_calc_time=max_reward_calc_time,
                max_iterations=max_iterations,
                scaffold=scaffold,
                max_tokens=max_tokens,
                ip=ip_used,
                used_yaml=used_yaml,
                execution_mode=execution_mode,
                simulator_yaml=simulator_yaml, # [NEW] Pass only the yaml path
                sim_reward_max_workers=sim_reward_max_workers,
                cal_reward=cal_reward,
            )

            future_to_image[fut] = docker_image

        with open(jsonl_file, "a") as f:
            for future in concurrent.futures.as_completed(future_to_image):
                docker_image = future_to_image[
                    future
                ]  # <-- retrieve that stored docker_image
                try:
                    result = future.result()
                    if result is not None:
                        with file_lock:
                            f.write(result + "\n")
                except Exception as e:
                    # Use docker_image from above when logging
                    logger.error(f"Exception for Docker image {docker_image}: {e}")

    logger.info(f"editagent completed on {len(ds_selected)} Docker images.")


if __name__ == "__main__":
    # Expose functions via Fire
    Fire(
        {
            "runagent": runagent,
            "runagent_multiple": runagent_multiple,
        }
    )