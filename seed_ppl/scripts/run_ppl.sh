# export MODEL_DIR="/ML-A100/team/mm/eamon/self_instruction/models/Qwen_72B"
# export TOKENIZERS_PARALLELISM=True
# model_path=${MODEL_DIR}
# outpath=outputs/seed_ppl_10k.jsonl

# export MODEL_DIR="/ML-A100/team/mm/eamon/self_instruction/models/Qwen_72B"
# export TOKENIZERS_PARALLELISM=True
# model_path=${MODEL_DIR}
# outpath=outputs/seed_ppl_20k.jsonl

export MODEL_DIR="/ML-A100/team/mm/eamon/self_instruction/models/Qwen_72B"
export TOKENIZERS_PARALLELISM=True
model_path=${MODEL_DIR}
outpath=outputs/seed_ppl_30k.jsonl

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
CUDA_VISIBLE_DEVICES=2,3 python3 ppl_generate.py \
    --model $model_path \
    --seed_task_path seed_tasks_hf65_with_cluster_id.jsonl \
    --output $outpath \
    --num_generation 30000
