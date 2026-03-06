# lanuch world models for simulation

# 1. launch swt
# Our SWT model: https://huggingface.co/RUC-AIBOX/SWE-World-SWT-32B-wo-cot  
SWT_DIR="YOUR_SWT_DIR"
vllm serve ${SWT_DIR} \
    --tensor-parallel-size=4 \
    --gpu-memory-utilization 0.8 \
    --served-model-name swt \
    --host 0.0.0.0 \
    --port 8012 \
    --dtype auto \
    --enable-prefix-caching


# 2. launch swr
# our SWR model: https://huggingface.co/RUC-AIBOX/SWE-World-SWR-32B-w-cot 
SWT_DIR="YOUR_SWR_DIR"
vllm serve ${SWT_DIR} \
    --tensor-parallel-size=4 \
    --gpu-memory-utilization 0.8 \
    --served-model-name swr \
    --host 0.0.0.0 \
    --port 8013 \
    --dtype auto \
    --enable-prefix-caching