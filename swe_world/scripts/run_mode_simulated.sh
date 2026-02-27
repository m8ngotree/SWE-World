# rollout based on world model

export OPENAI_API_KEY='YOUR_API_KEY'
export OPENAI_API_BASE='YOUR_API_BASE'


# use function calling
python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "results/mode_simulated" \
  --max_workers 10 \
  --start_idx 0 \
  --k 500 \
  --dataset "datasets/swe_bench_verified.json" \
  --split "test" \
  --llm_name 'openai/model' \
  --use_fn_calling False \
  --exp_name "rollout_swebench_verified_mode_simulated" \
  --temperature 1 \
  --max_steps 150 \
  --max_steps_absolute 150 \
  --execution_mode "simulated" \
  --scaffold "openhands" \
  --used_yaml "src/r2egym/agenthub/config/openhands/openhands_sp_fn_calling.yaml"


# don't use function calling
python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "results/mode_simulated" \
  --max_workers 10 \
  --start_idx 0 \
  --k 500 \
  --dataset "datasets/swe_bench_verified.json" \
  --split "test" \
  --llm_name 'openai/model' \
  --use_fn_calling False \
  --exp_name "rollout_swebench_verified_mode_simulated" \
  --temperature 1 \
  --max_steps 150 \
  --max_steps_absolute 150 \
  --execution_mode "simulated" \
  --scaffold "openhands" \
  --used_yaml "src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml"