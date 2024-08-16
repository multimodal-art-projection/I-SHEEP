# I-SHEEP: Self-Alignment of LLM from Scratch through an Iterative Self-Enhancement Paradigm
This repository contains code and data for the [I-SHEEP paper](https://arxiv.org/abs/2408.08072), which introduces an iterative self-enhancement paradigm enabling LLMs to continuously self-align from scratch with nothing.

The checkpoints of I-SHEEP will be released in the HuggingFace soon (delayed due to server migration).

## Introduction
The I-SHEEP framework takes the base model and small seed dataset as input, aligns the base
model iteratively from scratch independently, and finally obtains the self-enhanced models and high-quality synthetic datasets.

As shown in the following figure, I-SHEEP begins with seed data and leverages LLMs powerful understanding and generation capabilities to create additional instruction-output pair data. 
We then perform self-assessment, allowing the model to monitor and assess its learning process. 
By filtering out incorrect cognitions and retaining accurate ones, LLMs can self-align by training themselves with these correct cognitions.
Through iterative repetition of this process, the model can continuously and autonomously align from scratch, relying solely on its internal knowledge.

![The Pipeline of I-SHEEP. ](https://github.com/multimodal-art-projection/I-SHEEP/blob/main/I-SHEEP.png)

## Background

Humans do not have an explicit SFT stage; instead, they learn the pattern of instruction following through repetition and low-resource, weakly supervised instruction data, actively self-align in various environments.

The Self-Instruct, Dromedary, and SALMON settings not only aim to address the scarcity of labeled resources but also hold significant importance in exploring how active, human-like alignment can emerge.

Self-Instruct, Dromedary, SALMON, and now I-SHEEP!

## Key Research Questions Explored
Can LLMs continuously self-enhance with nothing?

How far can this self-enhancement process go?

What factors influence continuous self-enhancement?

Detailed answers to these questions can be found in the results section of the paper.

## Usage
This work is still in progress. We may update the code and data as we make progress. Please be cautious about the version control. 
This is just an unoptimized version of the code, which is the same as what we used during the actual run.

###  Directly Run
Here are the scripts for I-SHEEP:
```
cd seed_ppl/scripts
sh qwen72B_all.sh
```

## Step-by-step Run
Each iteration executes instructions, including self-synthesize, self-assess, filter, and SFT. The specific code is as follows:
```
lora_output_dir=/ML-A100/team/mm/eamon/self_instruction/seed_ppl/qwen72B_models
model_path=/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B
evaluate_model=/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B
# threshold=-1
# iter_data_size=10k
# setting=on_base

# ================================================= iter0 =================================================
# -------------------------------------use base model to generatae d0 prompt-----------------------------------------------------
export MODEL_DIR="/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B"
export TOKENIZERS_PARALLELISM=True
model_path=${MODEL_DIR}
outpath=qwen72B_outputs/qwen_72B_base_d0_10k.jsonl

if [ ! -f "$outpath" ]; then
  # 文件不存在，创建文件
  touch "$outpath"
fi

python3 context_generate_single.py \
    --model $model_path \
    --seed_task_path seed_tasks_hf65_with_cluster_id.jsonl \
    --output $outpath \
    --num_generation 10000 \
    --gpus 8  \
    --batch_size 100 \
    --origin_samples 3 \
    --sample_methods 'inverse' \
    --form 'ppl'

# -------------------------------------use base model to generatae d0 output-----------------------------------------------------
cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl
SEED_TASK_PATH=qwen72B_outputs/qwen_72B_base_d0_10k.jsonl
OUTPUT_PATH=qwen72B_outputs/qwen_72B_base_d0_10k_output.jsonl
# MODEL=/ML-A100/team/mm/eamon/self_instruction/models/Llama_3_8B

if [ ! -f "$OUTPUT_PATH" ]; then
  # 文件不存在，创建文件
  touch "$OUTPUT_PATH"
fi

cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl
python3 generate_output.py --model ${model_path} --seed_task_path ${SEED_TASK_PATH} --gpus 8 --output ${OUTPUT_PATH} --batch_size 512

# -------------------------------------use base model to evaluate d0 output-----------------------------------------------------
# first filtered by rules
# output_filtered_by_rules=${OUTPUT_PATH/.jsonl/_filtered.json}
output_filtered_by_rules="${OUTPUT_PATH%.jsonl}_filtered.json"
python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/utils/step_2_1_filter_output_and_instruction_to_sft.py --input_file $OUTPUT_PATH --output_file $output_filtered_by_rules 2>&1 | tee logs/step_2_1_filter_output_qwen_72B_base_d0_10k_output.log
# second self_evaluate 
# output_evaluated=${output_filtered_by_rules/.json/_evaluated.json}
output_evaluated="${output_filtered_by_rules%.json}_evaluated.json"
python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/evaluate_prompt_parameters.py --model $evaluate_model --input_file $output_filtered_by_rules --output_file $output_evaluated
# third filterd by evaluate
# output_filtered_by_evaluate=${output_evaluated/.json/_filtered.json}
output_filtered_by_evaluate="${output_evaluated%.json}_filtered.json"
python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/utils/step_3_post_filter_parameter.py --input_file $output_evaluated --output_file $output_filtered_by_evaluate


# ------------------------------------- use d0 to train base model, and get iter1 model --------------------------------------------
cd /ML-A100/team/mm/eamon/self_instruction/seed_ppl/LLaMA-Factory/examples/lora_multi_gpu

d0=/ML-A100/team/mm/eamon/self_instruction/seed_ppl/$output_filtered_by_evaluate
dataset_name=qwen_72B_d0_iter1_model
lora_output_path=$lora_output_dir/$dataset_name

if [ ! -d "$lora_output_path" ]; then
    mkdir -p "$lora_output_path"
else
    echo "Folder already exists: $lora_output_path"
fi

python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/LLaMA-Factory/examples/lora_multi_gpu/write_dataset_dict_to_info_json.py --dataset_name $dataset_name --file_name $d0

sh deepspeed_training_lora_parameter_qwen_like_on_base_parameter.sh $model_path $dataset_name $lora_output_path | tee $lora_output_path/lora.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 python /ML-A100/team/mm/eamon/self_instruction/seed_ppl/LLaMA-Factory/src/export_model.py \
    --model_name_or_path $model_path \
    --adapter_name_or_path $lora_output_dir/$dataset_name \
    --template qwen_like \
    --finetuning_type lora \
    --export_dir $lora_output_dir/merged_$dataset_name \
    --export_size 4 \
    --export_legacy_format False | tee $lora_output_path/merge.log 2>&1

```

## citation
```
@inproceedings{Liang2024isheep,
  title  = {I-SHEEP: Self-Alignment of LLM from Scratch through an Iterative Self-Enhancement Paradigm},
  author = {Yiming Liang and Ge Zhang and Xingwei Qu and Tianyu Zheng and Jiawei Guo and Xeron Du and Zhenzhu Yang and Jiaheng Liu and Chenghua Lin and Lei Ma and Wenhao Huang and Jiajun Zhang},
  year   = {2024},
  url    = {https://openreview.net/forum?id=y8Ng9dkuK9&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DAAAI.org%2F2025%2FAI_Alignment_Track%2FAuthors%23your-submissions)}
}
```
