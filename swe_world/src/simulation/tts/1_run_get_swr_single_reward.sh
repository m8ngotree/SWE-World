#!/bin/bash

# ================= 配置区 =================

# 1. 项目根目录 (用于设置 PYTHONPATH，确保能 import r2egym)
# 根据您的路径推测根目录如下，如有变动请修改

# 2. Python 脚本路径 (您保存的文件名)
PYTHON_SCRIPT="swe_world/swe_world/src/simulation/tts/get_swr_reward.py"
MODEL="swr"
# 3. 输入数据路径
INPUT_PATH=""

# 4. 输出数据路径 (根据您的习惯自定义后缀)
OUTPUT_PATH="_${MODEL}.jsonl"
OUTPUT_SUMMARY_PATH="_summary_reward_sim_${MODEL}.jsonl"

# 5. 模型配置与 Tokenizer
CONFIG_PATH=""
TOKENIZER_PATH=""

# 6. 运行参数
NUM_WORKERS=15  # 线程数
SLEEP_SEC=0.0   # 请求间隔

# ================= 运行区 =================

echo "Starting Simulation Evaluation..."
echo "Model:  $MODEL"
echo "Input:  $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Summary: $OUTPUT_SUMMARY_PATH"
echo "Config: $CONFIG_PATH"
echo "Tokenizer: $TOKENIZER_PATH" 
echo "Workers: $NUM_WORKERS"
echo "Sleep:  $SLEEP_SEC"


# 运行 Python 脚本
# --overwrite : 如果不想断点续跑（全部重跑），请在下面加上这个参数
python "$PYTHON_SCRIPT" \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --config_path "$CONFIG_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --summary_path "$OUTPUT_SUMMARY_PATH" \
    --num_workers $NUM_WORKERS \
    --sleep $SLEEP_SEC