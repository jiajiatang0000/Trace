#!/bin/bash
#SBATCH --account=vjgo8416-ai-data-eng
#SBATCH --qos=turing
#SBATCH --job-name=CFMPSN_v1
#SBATCH --output=logs/train_6datasets_CFMPSN_v1_v1_%j.out
#SBATCH --error=logs/train_6datasets_CFMPSN_v1_v1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00

set -e

DATA_PATH="data/LLM-CL_Benchmark/LLM-CL-Benchmark_5000"
MODEL_PATH="models/llama-2-7b-chat"
OUTPUT_DIR_PREFIX="outputs/6datasets_CFMPSN_v1_"

LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.0
LR_CSTANCE_FOMC=1e-5         
LR_MEETINGBANK=1e-5
LR_ScienceQA=1e-5
LR_Py150=1e-5
LR_NumGLUE_cm=1e-5
EPOCHS_CSTANCE=5
EPOCHS_FOMC=3
EPOCHS_MEETINGBANK=7
EPOCHS_ScienceQA=3
EPOCHS_Py150=5
EPOCHS_NumGLUE_cm=5

PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
MAX_LENGTH=1024
WARMUP_STEPS=30
MAX_GRAD_NORM=0.3

eval "$(/bask/projects/v/vjgo8416-ai-data-eng/multi-policy/TRACE/miniconda3/bin/conda shell.bash hook)"
conda activate env_pipp
export TOKENIZERS_PARALLELISM=false

for task in "C-STANCE" "FOMC" "MeetingBank" "Py150" "ScienceQA" "NumGLUE-cm"; do
    if [ ! -f "$DATA_PATH/$task/train.json" ] || [ ! -f "$DATA_PATH/$task/eval.json" ]; then
        echo "ERROR: Missing train.json or eval.json for task '$task'!"
        exit 1
    fi
done

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M")
OUTPUT_DIR="${OUTPUT_DIR_PREFIX}${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR" "logs"
echo "Output dir created: $OUTPUT_DIR"

HF_CACHE_ROOT="/dev/shm/hf_${TIMESTAMP}_$$"
export HF_HOME="$HF_CACHE_ROOT"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_MODULES_CACHE="$HF_CACHE_ROOT/modules"
export TMPDIR="/dev/shm/tmp_${TIMESTAMP}_$$"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_MODULES_CACHE" "$TMPDIR"

trap 'rm -rf "$HF_CACHE_ROOT" "$TMPDIR"' EXIT

sync

TRAIN_ARGS=(
    --data_path "$DATA_PATH"
    --model_name_or_path "$MODEL_PATH"
    --output_dir "$OUTPUT_DIR"

    --epochs_CSTANCE "$EPOCHS_CSTANCE"
    --epochs_FOMC "$EPOCHS_FOMC"
    --epochs_MeetingBank "$EPOCHS_MEETINGBANK"

    --epochs_ScienceQA "$EPOCHS_ScienceQA"
    --epochs_Py150 "$EPOCHS_Py150"
    --epochs_NumGLUE_cm "$EPOCHS_NumGLUE_cm"

    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"

    --lr_CSTANCE_FOMC "$LR_CSTANCE_FOMC"
    --lr_MeetingBank "$LR_MEETINGBANK"
    --lr_ScienceQA "$LR_ScienceQA"
    --lr_Py150 "$LR_Py150"
    --lr_NumGLUE_cm "$LR_NumGLUE_cm"

    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --max_length "$MAX_LENGTH"
    --warmup_steps "$WARMUP_STEPS"
    --max_grad_norm "$MAX_GRAD_NORM"
    --bf16
)
echo
echo "========================================"
echo "Starting QLoRA Training on 6 Datasets"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "GPUs: 1 (single process, device_map='auto')"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "LoRA: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "Effective global batch: $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 1))"
echo "Learning Rates:"
echo "  - C-STANCE     : $LR_CSTANCE_FOMC"
echo "  - FOMC         : $LR_CSTANCE_FOMC"
echo "  - MeetingBank  : $LR_MEETINGBANK"
echo "  - ScienceQA    : $LR_ScienceQA"
echo "  - Py150        : $LR_Py150"
echo "  - NumGLUE-cm   : $LR_NumGLUE_cm"
echo "Epochs:"
echo "  - C-STANCE($EPOCHS_CSTANCE), FOMC($EPOCHS_FOMC), MeetingBank($EPOCHS_MEETINGBANK)"
echo "  - ScienceQA($EPOCHS_ScienceQA), Py150($EPOCHS_Py150), NumGLUE-cm($EPOCHS_NumGLUE_cm)"
echo "========================================"

python training/train_6datasets_CFMPSN.py "${TRAIN_ARGS[@]}"

echo
echo "QLoRA 6-Dataset Training Completed!"
echo "Results saved to: $OUTPUT_DIR"

