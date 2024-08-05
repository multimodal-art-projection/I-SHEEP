# We use 8 x 6 = 48 V100-32GB GPUs
# On AiMOS cluster [https://docs.cci.rpi.edu/clusters/DCS_Supercomputer/]
# salloc --nodes 8 --time 6:00:00 --gres=gpu:32g:6 srun bash scripts/finetune_dromedary_65b_non_verbose.sh

# Due to some unknown issues in HF datasets library, we recommend run `finetune.py`
# with --fake_run flag to prepare the dataset on your local machine,
# and then submit the slurm training job to the cluster.
set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3,
export MODEL_DIR="/cpfs/29cd2992fe666f2a/user/huangwenhao/xw/git/llama1-65B-hf"
export SAVE_DIR="/cpfs/29cd2992fe666f2a/user/huangwenhao/xw/models/selfins_15epoch/"
# export DATA_PATH="/ML-A100/home/gezhang/Dromedary/training/all_instances_82K.json"
# export DATA_PATH="/ML-A100/home/gezhang/Dromedary/training/all_data.json"
export DATA_PATH="/cpfs/29cd2992fe666f2a/user/huangwenhao/xw/code/ppl_test/nopploutput_10k.json"
export GPUS_PER_NODE=8
export NNODES=1
export MASTER_ADDR=localhost
export MASTER_PORT=9901
export TOTAL_NUM_GPUS=$(( $NNODES * $GPUS_PER_NODE ))
export SLURM_PROCID=0

verbose=1

if [ $verbose_value -eq 1 ]; then
    verbose_output=""
else
    verbose_output="--disable_verbose True"
fi

TOTAL_BATCH_SIZE=768
LEARNING_RATE=4e-4
NUM_EPOCHS=15
CKPT_STEPS=1000

MICRO_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=$(($TOTAL_BATCH_SIZE / $MICRO_BATCH_SIZE / $TOTAL_NUM_GPUS))

accelerate launch \
    --num_processes=$TOTAL_NUM_GPUS --num_machines=$NNODES --machine_rank=$SLURM_PROCID \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    --deepspeed_multinode_launcher "standard" \
    --deepspeed_config_file /cpfs/29cd2992fe666f2a/user/huangwenhao/xw/code/Dromedary/training/ds_config_zero3.json \
    finetune.py \
    --num_warmup_steps 0 \
    --batch_size $TOTAL_BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --ds_gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --base_model $MODEL_DIR \
    --output_dir "$SAVE_DIR/noppl_10k" \
    --run_tensorboard_dir True \
    --val_set_size 0 \
    --checkpointing_steps $CKPT_STEPS \
    --resume_from_checkpoint True \
    --data_path $DATA_PATH \
    --meta_prompt_pattern "../prompts/inference_prompts/dromedary_standard_prompt_distill.txt" \
    --add_eos_token False \
    --cutoff_len 1024 \
    --train_on_inputs False \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    $verbose_output
