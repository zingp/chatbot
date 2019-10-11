
python interact.py --model_checkpoint ./models --dataset_path ./dataset/personachat_self_original.json
# 直接加载
python interact.py --model_checkpoint ./models --dataset_file ./.dataset_cache/dataset_cache_OpenAIGPTTokenizer 
# 训练
python ./train.py --model_checkpoint ./gpt2-models --is_local_model True  --dataset_file ./.dataset_cache/dataset_cache_OpenAIGPTTokenizer


# 复现？
python -m torch.distributed.launch --nproc_per_node=8 ./train.py --model_checkpoint ./gpt2-models --is_local_model True  --dataset_file ./.dataset_cache/dataset_cache_OpenAIGPTTokenizer --gradient_accumulation_steps=4 --lm_coef=2.0 --max_history=2 --n_epochs=1 --num_candidates=4 --personality_permutations=2 --train_batch_size=2 --valid_batch_size=2 
