export WANDB_PROJECT=llava-video-7B-Video-GRPO
export WANDB_NAME=dvd

# 禁用WANDB
export WANDB_MODE=disabled

mkdir -p ./ckpt/$WANDB_PROJECT/$WANDB_NAME

SCRIPT_DIR=$(cd "$(dirname $(dirname "${BASH_SOURCE[0]}"))" &>/dev/null && pwd)
export PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH

# --report_to wandb \
# CUDA_VISIBLE_DEVICES=0,1,2,3 
# torchrun --nproc_per_node="8" \
#     --nnodes="1" \
#     --node_rank="0" \
#     --master_addr="127.0.0.1" \
#     --master_port="12352" \
deepspeed src/open_r1_video/grpo.py \
    --deepspeed scripts/zero3.json \
    --output_dir ./ckpt/$WANDB_PROJECT/$WANDB_NAME \
    --model_name_or_path YOUR_PATH/llava_qwen_ckpt \
    --video_folder YOUR_PATH/dvd_dataset \
    --dataset_name xxx \
    --jsonl_path YOUR_PATH/dvd_dataset/train_dvd.jsonl \
    --max_prompt_length 8192 \
    --learning_rate 1e-6 \
    --num_generations 8 \
    --beta 0.04 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 20 \
    --save_total_limit 2 \
    --save_only_model true