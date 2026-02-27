
# for reward model
python swe_world/swe_world/src/simulation/world_model_sft_data/get_sim_cot_reward.py \
  --input "" \
  --output "" \
  --model-config "" \
  --max-tokens 65526 \
  --workers 40 \
  --temperature 0.6 \
  --start-idx 0 \
  --end-idx 40000 \
  --request-timeout 1800

# for transition model
# python swe_world/swe_world/src/simulation/world_model_sft_data/get_sim_cot_transition.py \
#   --input "" \
#   --output "" \
#   --model-config "" \
#   --max-tokens 65526 \
#   --workers 40 \
#   --temperature 0.6 \
#   --start-idx 0 \
#   --end-idx 40000 \
#   --request-timeout 1800