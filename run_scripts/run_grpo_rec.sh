# ---------- user-specific bits ----------
RUN_NAME=$1
DATA_PATH=$2
MODEL_PATH=$3
MAX_STEPS=$4

if [ -z "$RUN_NAME" ] || [ -z "$DATA_PATH" ] || [ -z "$MODEL_PATH" ] || [ -z "$MAX_STEPS" ]; then
  echo "Usage: $0 <RUN_NAME> <DATA_PATH> <MODEL_PATH> <MAX_STEPS>"
  exit 1
fi

# ensure Qwen2.5-VL is in the name of the model path
if [[ "$MODEL_PATH" != *"Qwen2.5-VL"* ]]; then
  echo "Error: MODEL_PATH must contain 'Qwen2.5-VL' to specify the correct model."
  exit 1
fi

IMAGE_FOLDERS="dummy"  # not used right now, since we use global paths
SAVE_ROOT=/home/jovyan/workspace/saves/rl/$RUN_NAME
LOGFILE=/home/jovyan/shared/RodDeSc/experiments/logs/$RUN_NAME.out
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"

# ---------------------------------------

echo "==============================================="
echo "Run Name          : $RUN_NAME"
echo "Data Path         : $DATA_PATH"
echo "Model Path        : $MODEL_PATH"
echo "Max Steps         : $MAX_STEPS"
echo "Save Root         : $SAVE_ROOT"
echo "Log File          : $LOGFILE"
echo "Repo home         : $REPO_HOME"
echo "CUDA Devices      : $CUDA_VISIBLE_DEVICES"
echo "-----------------------------------------------"

source activate vlm-r1
conda info

# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${RUN_NAME}/log
TASK_TYPE="rec"
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="${REPO_HOME}/runs/${RUN_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# export WANDB_DISABLED=true

cd ${REPO_HOME}/src/open-r1-multimodal

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir $SAVE_ROOT \
    --resume_from_checkpoint True \
    --model_name_or_path $MODEL_PATH \
    --data_file_paths $DATA_PATH \
    --image_folders $IMAGE_FOLDERS \
    --is_reward_customized_from_vlm_module True \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 1000 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name $RUN_NAME \
    --data_seed 42 \
    --save_steps 100 \
    --num_generations 8 \
    --max_completion_length 2048 \
    --max_prompt_length 2048 \
    --max_pixels 262144 \
    --reward_funcs pdms format ade \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json \
    --max_steps $MAX_STEPS >> "$LOGFILE" 2>&1

echo "Training completed for ${RUN_NAME}"
