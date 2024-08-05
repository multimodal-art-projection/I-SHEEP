set -x
set -e

lora_output_dir=/ML-A100/team/mm/eamon/self_instruction/seed_ppl/threshold_models
model_path=/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B
evaluate_model=/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B
threshold=8
datasize=50000
# iter_data_size=10k
# setting=on_base

# ================================================= iter0 =================================================
# -------------------------------------use base model to generatae d0 prompt-----------------------------------------------------
export MODEL_DIR="/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B"
export TOKENIZERS_PARALLELISM=True
model_path=${MODEL_DIR}
outpath=threshold_ouputs/threshold${threshold}_qwen_72B_base_d0_$datasize.jsonl

if [ ! -f "$outpath" ]; then
  # 文件不存在，创建文件
  touch "$outpath"
fi

# python3 context_generate_single.py \
#     --model $model_path \
#     --seed_task_path seed_tasks_hf65_with_cluster_id.jsonl \
#     --output $outpath \
#     --num_generation $datasize \
#     --gpus 8  \
#     --batch_size 100 \
#     --origin_samples 3 \
#     --sample_methods 'inverse' \
#     --form 'ppl'

# -------------------------------------use base model to generatae d0 output-----------------------------------------------------
cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl
SEED_TASK_PATH=threshold_ouputs/threshold${threshold}_qwen_72B_base_d0_$datasize.jsonl
OUTPUT_PATH=threshold_ouputs/threshold${threshold}_qwen_72B_base_d0_${datasize}_output.jsonl
# MODEL=/ML-A100/team/mm/eamon/self_instruction/models/Llama_3_8B

if [ ! -f "$OUTPUT_PATH" ]; then
  # 文件不存在，创建文件
  touch "$OUTPUT_PATH"
fi

cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl
# python3 generate_output.py --model ${model_path} --seed_task_path ${SEED_TASK_PATH} --gpus 8 --output ${OUTPUT_PATH} --batch_size 512

# -------------------------------------use base model to evaluate d0 output-----------------------------------------------------
# first filtered by rules
# output_filtered_by_rules=${OUTPUT_PATH/.jsonl/_filtered.json}
output_filtered_by_rules="${OUTPUT_PATH%.jsonl}_filtered.json"
# python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/utils/step_2_1_filter_output_and_instruction_to_sft.py --input_file $OUTPUT_PATH --output_file $output_filtered_by_rules 2>&1 | tee logs/step_2_1_filter_output_threshold${threshold}_qwen_72B_base_d0_${datasize}_output.log
# second self_evaluate 
# output_evaluated=${output_filtered_by_rules/.json/_evaluated.json}
output_evaluated="${output_filtered_by_rules%.json}_evaluated.json"
# python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/evaluate_prompt_parameters.py --model $evaluate_model --input_file $output_filtered_by_rules --output_file $output_evaluated
# third filterd by evaluate
# output_filtered_by_evaluate=${output_evaluated/.json/_filtered.json}
output_filtered_by_evaluate="${output_evaluated%.json}_filtered.json"
# python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/utils/step_3_post_filter_parameter_threshold.py --input_file $output_evaluated --output_file $output_filtered_by_evaluate --threshold $threshold


# ------------------------------------- use d0 to train base model, and get iter1 model --------------------------------------------
cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl/LLaMA-Factory/examples/lora_multi_gpu

d0=/ML-A100/team/mm/eamon/self_instruction/seed_ppl/$output_filtered_by_evaluate
dataset_name=threshold${threshold}_qwen_72B_d0_${datasize}_iter1_model
lora_output_path=$lora_output_dir/$dataset_name

if [ ! -d "$lora_output_path" ]; then
    mkdir -p "$lora_output_path"
else
    echo "Folder already exists: $lora_output_path"
fi

python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/LLaMA-Factory/examples/lora_multi_gpu/write_dataset_dict_to_info_json.py --dataset_name $dataset_name --file_name $d0

# sh deepspeed_training_lora_parameter_qwen_like_on_base_parameter.sh $model_path $dataset_name $lora_output_path | tee $lora_output_path/lora.log 2>&1

# CUDA_VISIBLE_DEVICES=0,1,2,3 python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/LLaMA-Factory/src/export_model.py \
#     --model_name_or_path $model_path \
#     --adapter_name_or_path $lora_output_dir/$dataset_name \
#     --template qwen_like \
#     --finetuning_type lora \
#     --export_dir $lora_output_dir/merged_$dataset_name \
#     --export_size 4 \
#     --export_legacy_format False | tee $lora_output_path/merge.log 2>&1

# # =================================================iter1=================================================

# -------------------------------------use iter1 model to generatae d1 prompt-----------------------------------------------------
MODEL=$lora_output_dir/merged_$dataset_name
export TOKENIZERS_PARALLELISM=True
cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl
outpath=threshold_ouputs/threshold${threshold}_qwen_72B_iter1_d1_$datasize.jsonl


if [ ! -f "$outpath" ]; then
  # 文件不存在，创建文件
  touch "$outpath"
fi

# python3 context_generate_iter2_single.py \
#     --model $MODEL \
#     --seed_task_path seed_tasks_hf65_with_cluster_id.jsonl \
#     --output $outpath \
#     --num_generation $datasize \
#     --gpus 8  \
#     --batch_size 100 \
#     --origin_samples 3 \
#     --sample_methods 'inverse' \
#     --form 'ppl'

# -------------------------------------use iter1 model to generatae d1 output-----------------------------------------------------
cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl
SEED_TASK_PATH=$outpath
OUTPUT_PATH="${outpath%.jsonl}_output.jsonl"

if [ ! -f "$OUTPUT_PATH" ]; then
  # 文件不存在，创建文件
  touch "$OUTPUT_PATH"
fi

cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl
# python3 generate_output_iter2.py --model ${MODEL} --seed_task_path ${SEED_TASK_PATH} --gpus 8 --output ${OUTPUT_PATH} --batch_size 512

# -------------------------------------use base model to evaluate d1 output-----------------------------------------------------
# first filtered by rules
output_filtered_by_rules="${OUTPUT_PATH%.jsonl}_filtered.json"
# python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/utils/step_2_1_filter_output_and_instruction_to_sft.py --input_file $OUTPUT_PATH --output_file $output_filtered_by_rules 2>&1 | tee logs/step_2_1_filter_output_threshold${threshold}_qwen_72B_base_d1_${datasize}_output.log
# second self_evaluate 
output_evaluated="${output_filtered_by_rules%.json}_evaluated.json"
# python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/evaluate_prompt_parameters.py --model $evaluate_model --input_file $output_filtered_by_rules --output_file $output_evaluated
# third filterd by evaluate
output_filtered_by_evaluate="${output_evaluated%.json}_filtered.json"
# python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/utils/step_3_post_filter_parameter_threshold.py --input_file $output_evaluated --output_file $output_filtered_by_evaluate --threshold $threshold


# ------------------------------------- use d1 to train base model, and get iter2 model --------------------------------------------

cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl/LLaMA-Factory/examples/lora_multi_gpu

d1=/ML-A100/team/mm/eamon/self_instruction/seed_ppl/$output_filtered_by_evaluate
dataset_name=threshold${threshold}_qwen_72B_d1_${datasize}_iter2_model
lora_output_path=$lora_output_dir/$dataset_name

if [ ! -d "$lora_output_path" ]; then
    mkdir -p "$lora_output_path"
else
    echo "Folder already exists: $lora_output_path"
fi

python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/LLaMA-Factory/examples/lora_multi_gpu/write_dataset_dict_to_info_json.py --dataset_name $dataset_name --file_name $d1

# sh deepspeed_training_lora_parameter_qwen_like_on_base_parameter.sh $model_path $dataset_name $lora_output_path | tee $lora_output_path/lora.log 2>&1

# CUDA_VISIBLE_DEVICES=0,1,2,3 python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/LLaMA-Factory/src/export_model.py \
#     --model_name_or_path $model_path \
#     --adapter_name_or_path $lora_output_dir/$dataset_name \
#     --template qwen_like \
#     --finetuning_type lora \
#     --export_dir $lora_output_dir/merged_$dataset_name \
#     --export_size 4 \
#     --export_legacy_format False | tee $lora_output_path/merge.log 2>&1

# # =================================================iter2=================================================

# -------------------------------------use iter2 model to generatae d2 prompt-----------------------------------------------------
MODEL=$lora_output_dir/merged_$dataset_name
export TOKENIZERS_PARALLELISM=True
cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl
outpath=threshold_ouputs/threshold${threshold}_qwen_72B_iter2_d2_$datasize.jsonl


if [ ! -f "$outpath" ]; then
  # 文件不存在，创建文件
  touch "$outpath"
fi

# python3 context_generate_iter2_single.py \
#     --model $MODEL \
#     --seed_task_path seed_tasks_hf65_with_cluster_id.jsonl \
#     --output $outpath \
#     --num_generation $datasize \
#     --gpus 8  \
#     --batch_size 100 \
#     --origin_samples 3 \
#     --sample_methods 'inverse' \
#     --form 'ppl'

# -------------------------------------use iter2 model to generatae d2 output-----------------------------------------------------
cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl
SEED_TASK_PATH=threshold_ouputs/threshold${threshold}_qwen_72B_iter2_d2_$datasize.jsonl
OUTPUT_PATH=threshold_ouputs/threshold${threshold}_qwen_72B_iter2_d2_${datasize}_output.jsonl

if [ ! -f "$OUTPUT_PATH" ]; then
  # 文件不存在，创建文件
  touch "$OUTPUT_PATH"
fi

cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl
# python3 generate_output_iter2.py --model ${MODEL} --seed_task_path ${SEED_TASK_PATH} --gpus 8 --output ${OUTPUT_PATH} --batch_size 512

# -------------------------------------use base model to evaluate d2 output-----------------------------------------------------
# first filtered by rules
output_filtered_by_rules="${OUTPUT_PATH%.jsonl}_filtered.json"
# python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/utils/step_2_1_filter_output_and_instruction_to_sft.py --input_file $OUTPUT_PATH --output_file $output_filtered_by_rules 2>&1 | tee logs/step_2_1_filter_output_threshold${threshold}_qwen_72B_base_d2_${datasize}_output.log
# second self_evaluate 
output_evaluated="${output_filtered_by_rules%.json}_evaluated.json"
python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/evaluate_prompt_parameters.py --model $evaluate_model --input_file $output_filtered_by_rules --output_file $output_evaluated
# third filterd by evaluate
output_filtered_by_evaluate="${output_evaluated%.json}_filtered.json"
python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/utils/step_3_post_filter_parameter_threshold.py --input_file $output_evaluated --output_file $output_filtered_by_evaluate --threshold $threshold

# ------------------------------------- use d2 to train base model, and get iter3 model --------------------------------------------

cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl/LLaMA-Factory/examples/lora_multi_gpu

d2=/ML-A100/team/mm/eamon/self_instruction/seed_ppl/$output_filtered_by_evaluate
dataset_name=threshold${threshold}_qwen_72B_d2_${datasize}_iter3_model
lora_output_path=$lora_output_dir/$dataset_name

if [ ! -d "$lora_output_path" ]; then
    mkdir -p "$lora_output_path"
else
    echo "Folder already exists: $lora_output_path"
fi

python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/LLaMA-Factory/examples/lora_multi_gpu/write_dataset_dict_to_info_json.py --dataset_name $dataset_name --file_name $d2

sh deepspeed_training_lora_parameter_qwen_like_on_base_parameter.sh $model_path $dataset_name $lora_output_path | tee $lora_output_path/lora.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/LLaMA-Factory/src/export_model.py \
    --model_name_or_path $model_path \
    --adapter_name_or_path $lora_output_dir/$dataset_name \
    --template qwen_like \
    --finetuning_type lora \
    --export_dir $lora_output_dir/merged_$dataset_name \
    --export_size 4 \
    --export_legacy_format False | tee $lora_output_path/merge.log 2>&1

