#!/bin/bash

# check whether the model is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

# the first parameter from the command as the model name or path
MODEL_PATH=$1

# # 使用for循环启动8个服务实例，每个实例使用不同的GPU和端口
# for i in {4,5,6,7}
# we use for loop to create 8 vllm instances, with different GPUs
for i in 0 1 2 3 4 5 6 7
do
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.api_server \
        --model $MODEL_PATH \
        --gpu-memory-utilization=0.9 \
        --max-num-seqs=200 \
        --host 127.0.0.1 --tensor-parallel-size 1 \
        --port $((8000+i)) \
    &
done
