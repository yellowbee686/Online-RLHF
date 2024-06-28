# First approach: initialize 4 VLLM processes and split the prompt set to the 4 agents
# The generated samples will be stored at output_dir + local_index + ".json

my_world_size=4 # how many gpu you use
# infer_model=meta-llama/Meta-Llama-3-8B-Instruct
infer_model=RLHFlow/LLaMA3-SFT
prompt_dir=RLHFlow/test_generation_2k
mkdir data
output_dir=./data/gen_data

CUDA_VISIBLE_DEVICES=1 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --seed 686 --K 4 --temperature 1.0 --local_index 0 --my_world_size ${my_world_size} --eos_ids 128009 &
CUDA_VISIBLE_DEVICES=2 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --seed 686 --K 4 --temperature 1.0 --local_index 1 --my_world_size ${my_world_size} --eos_ids 128009 &
CUDA_VISIBLE_DEVICES=6 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --seed 686 --K 4 --temperature 1.0 --local_index 2 --my_world_size ${my_world_size} --eos_ids 128009 &
CUDA_VISIBLE_DEVICES=7 python ./generation/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --seed 686 --K 4 --temperature 1.0 --local_index 3 --my_world_size ${my_world_size} --eos_ids 128009 &

wait
python ./generation/merge_data.py --base_path ${output_dir} --output_dir ./data/gen_data.json --num_datasets ${my_world_size}