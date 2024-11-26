source ~/.bashrc
export CUDA_VISIBLE_DEVICES='4,5,6,7'
# Initialize Conda environment
eval "$(conda shell.bash hook)"


# Base paths and settings
initial_model="RLHFlow/LLaMA3-SFT-v2" 
#"meta-llama/Meta-Llama-3-8B-Instruct"
base_path="/mnt/raid5/xc/rlhf_online/dpo"
#"/home/wx/Iterative-RLHF-dev/test"
# mkdir $base_path
iteration_prefix="Test"




# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local model_path=$2
    local jsonl_input=$3
    local json_output=$4
    local model_output=$5
    local model_output_file=$6
    local i=$7

    
    # if [ $i -gt 0 ]; then
    #     conda activate vllm
    #     bash generation/run_8gpu.sh $model_path
    #     sleep 60
    #     python generation/gen_hf.py --ports 8004 8005 8006 8007 --eos_ids 128009 --tokenizer $initial_model --dataset_name_or_path $jsonl_input --output_dir $json_output --K 32 --temperature 1.0
    #     pkill -f "python -m vllm.entrypoints.api_server"
    #     accelerate launch annotate_data/get_multi_task_rewards.py --dataset_name_or_path $json_output --output_dir $model_output --K 32
    #     python ./generation/merge_data.py --base_path $model_output --output_dir $model_output_file --num_datasets 4
    # fi
    
    # conda activate test_online
    # sleep 5
    # accelerate launch --config_file ./configs/zero2_test.yaml dpo_iteration/run_dpo.py \
    #     --run_name $iteration --output_dir $iteration --model_name_or_path $model_path --ref_model $initial_model --learning_rate 5e-7 \
    #     --max_steps 1200 --choose_type max_min --train_dir $model_output_file --eval_dir $model_output_file --loss_type tdpo --lr_scheduler_type cosine \
    #     --len_penalty 0.0 --num_train_epochs 2 --gradient_accumulation_steps 16
    # conda activate vllm
    bash generation/register_server.sh $model_path
    sleep 140
    python generation/gen_hf.py --ports 8000 8001 8002 8003 8004 8005 8006 8007 --eos_ids 128009 --tokenizer $initial_model --dataset_name_or_path $jsonl_input --output_dir $json_output --K 8 --temperature 1.0
    pkill -f "python -m vllm.entrypoints.api_server"
    accelerate launch annotate_data/get_rewards.py --dataset_name_or_path $json_output --output_dir $model_output
    conda activate rlhflow
    cat <<EOT > dpo_config.yaml
run_name: $iteration
output_dir: $iteration
model_name_or_path: $model_path
ref_model: $model_path
learning_rate: 5.0e-7
num_train_epochs: 2
logging_steps: 2
gradient_checkpointing: true
do_train: true
do_eval: true
eval_steps: 10000
choose_type: max_min
train_dir: $model_output
eval_dir: $model_output
loss_type: sigmoid
lr_scheduler_type: cosine
max_length: 2048
max_prompt_length: 1000
eval_strategy: steps
bf16: true
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
report_to: wandb
EOT

    accelerate launch --config_file ./configs/zero2.yaml dpo_iteration/run_dpo.py dpo_config.yaml
}


# Main loop for iterations
for i in {1..9}
do
    iteration_name="LLaMA3_iter${i}"
    jsonl_input="RLHFlow/iterative-prompt-v1-iter${i}-20K"
    json_output="${base_path}/${iteration_prefix}${i}_${iteration_name}.json"
    model_output="${base_path}/${iteration_prefix}${i}_${iteration_name}_reward"
    model_output_file="${base_path}/${iteration_prefix}${i}_${iteration_name}_reward.json"
    
    # Determine the model path: first iteration uses the initial model, subsequent iterations use the previous iteration's model
    if [ $i -eq 1 ]; then
        model_path=$initial_model
    else
        previous_iteration=$((i-1))
        model_path="LLaMA3_iter${previous_iteration}"
    fi

    run_iteration $iteration_name $model_path $jsonl_input $json_output $model_output $model_output_file $i
done


