model_path=RLHFlow/LLaMA3-SFT
initial_model=RLHFlow/LLaMA3-SFT
# mkdir models
accelerate launch --config_file ./configs/zero2_test.yaml ./dpo_iteration/run_dpo.py --run_name rlhflow_iter1 --output_dir ./models/rlhflow_iter1 --model_name_or_path $model_path --ref_model $initial_model --learning_rate 2e-7 --max_steps 1200 --choose_type max_min --train_dir ./data/data_with_rewards.json --eval_dir ./data/data_with_rewards.json --loss_type sigmoid --lr_scheduler_type cosine