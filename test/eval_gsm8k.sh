# source ~/.bashrc
export CUDA_VISIBLE_DEVICES='4,5,6,7'
# eval "$(conda shell.bash hook)"

# model_path="LLaMA3_iter2"
# model_path="RLHFlow/LLaMA3-SFT"
# model_path="meta-llama/Meta-Llama-3-8B-Instruct"
# model_path="/mnt/raid5/xc/rlhf_online/exps/tdpo_armo_lc_online/LLaMA3_iter5"
model_path=$1
jsonl_input="openai/gsm8k"


conda activate vllm
bash generation/eval_vllm.sh $model_path
sleep 60
python eval/eval_gsm8k.py \
    --ports 8004 8005 8006 8007 \
    --eos_ids 128009 \
    --tokenizer $model_path \
    --dataset_name_or_path $jsonl_input \
    --K 1 \
    --temperature 0.0 \
    --ds_split 'test' \
    --dataset_key 'question'
pkill -f "python -m vllm.entrypoints.api_server";