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


# ip_address=$(hostname -i)
# echo "本机IP地址为：$ip_address"
# export NODE_ADDR=$ip_address

if [ ! -f "$outpath" ]; then
  # 文件不存在，创建文件
  touch "$outpath"
fi

python3 context_generate.py \
    --model $model_path \
    --seed_task_path seed_tasks_hf65_with_cluster_id.jsonl \
    --output $outpath \
    --num_generation 30000 \
    --gpus 8  \
    --batch_size 100 \
    --origin_samples 3 \
    --sample_methods 'inverse' \
    --form 'ppl'