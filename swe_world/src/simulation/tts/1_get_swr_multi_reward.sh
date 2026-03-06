
INPUT_PATH=reward_contexts.jsonl
CONFIG_PATH=src/simulation/models_yaml/example_world_models.yaml
TOKENIZER_PATH="YOUR_TOKENIZER_PATH"
OUTPUT_PATH=output_sim_reward.jsonl

NUM_WORKERS=16
NUM_SCORES=3

python sim_reward_multi_score.py \
    --input_path ${INPUT_PATH} \
    --config_path ${CONFIG_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --output_path ${OUTPUT_PATH} \
    --num_workers ${NUM_WORKERS} \
    --num_scores ${NUM_SCORES}