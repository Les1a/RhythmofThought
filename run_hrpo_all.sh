#!/bin/bash
###############################################################################
# run_hrpo_all.sh — Unified HRPO Training & Evaluation Script
#
# Runs HRPO (Hybrid Latent Reasoning via RL) training and evaluation for
# GSM8K, MATH, MMLU, and RAG tasks with paper-official hyperparameters.
#
# Smart skipping — by default won't re-train if checkpoints exist, won't re-eval if
# results exist. Pass --resume to continue training from the latest checkpoint
# instead of skipping (full state restore: optimizer, scheduler, RNG, global_step).
#
# Usage:
#   bash run_hrpo_all.sh [OPTIONS]
#
# Options:
#   --gpu ID              GPU device ID (default: 0)
#   --tasks TASKS         Comma-separated: gsm8k,math,mmlu,rag or "all" (default: all)
#   --model NAME          Model name (default: Qwen/Qwen2.5-3B-Instruct)
#   --paper-params        Use paper-original batch sizes instead of H200-optimized
#   --eval-only           Skip training, only evaluate existing checkpoints
#   --skip-eval           Skip evaluation after training
#   --resume              Resume training from the latest checkpoint in each task's
#                         experiment dir instead of skipping. Errors if a task's
#                         experiment dir is missing or contains no checkpoint-*.
#                         Hyperparameters must match the original run.
#   --mode MODE           Internal override: hrpo or thrpo (default: hrpo)
#   --thinking-time-loss-weight V
#                         Thinking-time auxiliary loss weight (default: 0.1)
#   --lr-time-cond V      Learning rate for time conditioning modules (default: 1e-4)
#   --exp-suffix NAME     Append a suffix to experiment dir names to avoid
#                         checkpoint collisions with previous runs
#   --max-steps N         Override trainer max_steps for smoke tests or short runs
#   --max-train-samples N Limit the loaded training dataset before tokenization
#   --no-wandb            Disable WandB logging
#   --prep-data           Run prepare_data.py for selected --tasks before training/eval
#   --dry-run             Print commands without executing
#   --help                Show this help message
###############################################################################
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR"
LOG_DIR="${WORK_DIR}/logs"
CONDA_ENV="rot"

# ========================= Paper-Official Hyperparameters ====================
MODEL="Qwen/Qwen2.5-3B-Instruct"
SEED=42
LR=5e-6
BETA=0.005
LORA_RANK=32
RESIDUAL_R_MIN=0.99
RESIDUAL_R_MAX=0.999
LR_RESIDUAL_GATE=1e-4
LR_RESIDUAL_LAMBDA=1e-3
WEIGHT_DECAY=0.1
WARMUP_RATIO=0.1
LR_SCHEDULER_TYPE="cosine"
OPTIMIZER="paged_adamw_8bit"
MAX_GRAD_NORM=0.1
TEMPERATURE=0.5

# ========================= H200-Optimized Batch Sizes ========================
# H200 141GB HBM3e: maximize BS, set GA=1 to eliminate accumulation overhead
# Effective batch = BS * GA (kept identical to paper for training dynamics)
GSM8K_BS=32;  GSM8K_GA=1    # paper: BS=8,  GA=4, eff=32 | group=4, seq=2048
MATH_BS=64;   MATH_GA=1     # paper: BS=16, GA=4, eff=64 | group=8, seq=4096
MMLU_BS=64;   MMLU_GA=1     # paper: BS=16, GA=4, eff=64 | group=8, seq=2048
RAG_BS=64;    RAG_GA=1      # paper: BS=16, GA=4, eff=64 | group=4, seq=3072
EVAL_BS=128                  # inference only, no gradients needed

# ========================= Defaults ==========================================
GPU_ID=0
TASKS="all"
EVAL_ONLY=false
SKIP_EVAL=false
NO_WANDB=false
PREP_DATA=false
DRY_RUN=false
RESUME=false
MODE="hrpo"
THINKING_TIME_LOSS_WEIGHT=0.1
LR_TIME_CONDITIONING=1e-4
EXP_SUFFIX=""
MAX_STEPS=-1
MAX_TRAIN_SAMPLES=""
MODE_LABEL="HRPO"
FAILED_TASKS=()

# ========================= Argument Parsing ==================================
show_help() {
    sed -n '/^# Usage:/,/^###/{ /^###/d; s/^# \?//; p }' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)        GPU_ID="$2"; shift 2 ;;
        --tasks)      TASKS="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --paper-params)
            GSM8K_BS=8;  GSM8K_GA=4
            MATH_BS=16;  MATH_GA=4
            MMLU_BS=16;  MMLU_GA=4
            RAG_BS=16;   RAG_GA=4
            EVAL_BS=32
            shift ;;
        --eval-only)  EVAL_ONLY=true; shift ;;
        --skip-eval)  SKIP_EVAL=true; shift ;;
        --resume)     RESUME=true; shift ;;
        --mode)       MODE="$2"; shift 2 ;;
        --no-wandb)   NO_WANDB=true; shift ;;
        --prep-data)  PREP_DATA=true; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --thinking-time-loss-weight) THINKING_TIME_LOSS_WEIGHT="$2"; shift 2 ;;
        --lr-time-cond)     LR_TIME_CONDITIONING="$2"; shift 2 ;;
        --exp-suffix)       EXP_SUFFIX="$2"; shift 2 ;;
        --max-steps)        MAX_STEPS="$2"; shift 2 ;;
        --max-train-samples) MAX_TRAIN_SAMPLES="$2"; shift 2 ;;
        --help|-h)    show_help ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$MODE" in
    hrpo) MODE_LABEL="HRPO" ;;
    thrpo) MODE_LABEL="THRPO" ;;
    *)
        echo "Unsupported mode for run_hrpo_all.sh: ${MODE} (expected hrpo or thrpo)"
        exit 1
        ;;
esac

if ! [[ "${MAX_STEPS}" =~ ^-?[0-9]+$ ]]; then
    echo "Invalid value for --max-steps: ${MAX_STEPS}"
    exit 1
fi

if [ -n "${MAX_TRAIN_SAMPLES}" ] && ! [[ "${MAX_TRAIN_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid value for --max-train-samples: ${MAX_TRAIN_SAMPLES}"
    exit 1
fi

# ========================= Utility Functions =================================
log() {
    local task="$1"; shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$task] $*"
}

normalize_exp_suffix() {
    local raw="$1"
    raw=$(printf '%s' "$raw" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    raw=$(printf '%s' "$raw" | tr -cs 'A-Za-z0-9._-' '-')
    raw=$(printf '%s' "$raw" | sed -e 's/^[.-]*//' -e 's/[.-]*$//')
    printf '%s' "$raw"
}

EXP_SUFFIX=$(normalize_exp_suffix "$EXP_SUFFIX")

get_exp_name() {
    local task="$1"
    local group_size="$2"
    local model_short="${MODEL##*/}"
    local suffix_segment=""
    [ -n "$EXP_SUFFIX" ] && suffix_segment="-${EXP_SUFFIX}"
    echo "./experiments/${model_short}-${task}-${MODE}-group${group_size}-lora${LORA_RANK}-rmin${RESIDUAL_R_MIN}-temp${TEMPERATURE}${suffix_segment}"
}

find_latest_checkpoint() {
    local exp_dir="$1"
    ls -d "${exp_dir}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1
}

dry_run_checkpoint_hint() {
    local exp_dir="$1"
    local ckpt
    ckpt=$(find_latest_checkpoint "$exp_dir")
    if [ -n "$ckpt" ]; then
        echo "$ckpt"
    else
        echo "${exp_dir}/checkpoint-<latest>"
    fi
}

check_dataset() {
    local task="$1"
    case "$task" in
        gsm8k) return 0 ;;
        math)
            if [ ! -d "../MATH/train" ]; then
                log "$task" "ERROR: Dataset not found at ../MATH/train/"
                return 1
            fi ;;
        mmlu)
            if [ ! -d "../MMLU_Train_Merged" ]; then
                log "$task" "ERROR: Dataset not found at ../MMLU_Train_Merged/"
                return 1
            fi ;;
        rag)
            if [ ! -d "../RAG_Train_Merged" ]; then
                log "$task" "ERROR: Dataset not found at ../RAG_Train_Merged/"
                return 1
            fi ;;
    esac
    return 0
}

common_train_args() {
    echo "--mode ${MODE} \
        --model_name ${MODEL} \
        ${EXP_SUFFIX:+--exp-suffix ${EXP_SUFFIX}} \
        --lora_rank ${LORA_RANK} \
        --lr ${LR} \
        --beta ${BETA} \
        --residual_r_min ${RESIDUAL_R_MIN} \
        --residual_r_max ${RESIDUAL_R_MAX} \
        --lr_residual_gate ${LR_RESIDUAL_GATE} \
        --lr_residual_Lambda ${LR_RESIDUAL_LAMBDA} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_ratio ${WARMUP_RATIO} \
        --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
        --optimizer ${OPTIMIZER} \
        --max_grad_norm ${MAX_GRAD_NORM} \
        --temperature ${TEMPERATURE} \
        --seed ${SEED} \
        $( [ "${MAX_STEPS}" -gt 0 ] && echo "--max_steps ${MAX_STEPS}" ) \
        $( [ -n "${MAX_TRAIN_SAMPLES}" ] && echo "--max_train_samples ${MAX_TRAIN_SAMPLES}" ) \
        $( [ "$MODE" = "thrpo" ] && echo "--thinking_time_loss_weight ${THINKING_TIME_LOSS_WEIGHT} --lr_time_conditioning ${LR_TIME_CONDITIONING}" )"
}

# ========================= Conda Activation ==================================
activate_env() {
    log "MAIN" "Activating conda environment: ${CONDA_ENV}"
    if [ "$DRY_RUN" = true ]; then
        log "MAIN" "[DRY-RUN] conda activate ${CONDA_ENV}"
        return 0
    fi
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        eval "$(conda shell.bash hook 2>/dev/null)"
    fi
    conda activate "${CONDA_ENV}"
    log "MAIN" "Python: $(which python)"
}

# ========================= Training Functions ================================
train_task() {
    local task="$1"
    local group_size="$2"
    local bs="$3"
    local ga="$4"
    local max_prompt="$5"
    local max_completion="$6"
    local dataset_root="${7:-}"

    local exp_name
    exp_name=$(get_exp_name "$task" "$group_size")

    log "$task" "Experiment: ${exp_name}"
    log "$task" "Effective batch size: $((bs * ga)) (BS=${bs} x GA=${ga})"
    log "$task" "Launch config: max_steps=${MAX_STEPS}, max_train_samples=${MAX_TRAIN_SAMPLES:-all}"

    # Resolve resume vs. skip vs. fresh-train. The python script does the actual
    # state restore; here we only decide whether to invoke it and with what flag.
    local resume_arg=""
    if [ -d "$exp_name" ] && ls "${exp_name}"/checkpoint-* &>/dev/null; then
        if [ "$RESUME" = true ]; then
            log "$task" "Resuming training from latest checkpoint in ${exp_name}"
            resume_arg="--resume"
        else
            log "$task" "Checkpoint already exists, skipping training (use --resume to continue)"
            return 0
        fi
    elif [ -d "$exp_name" ] && [ "$(ls -A "$exp_name" 2>/dev/null)" ]; then
        if [ "$RESUME" = true ]; then
            log "$task" "ERROR: --resume specified but no checkpoint-* found in ${exp_name}"
            return 1
        fi
        log "$task" "ERROR: Experiment dir exists but contains no checkpoint-* entries: ${exp_name}"
        log "$task" "ERROR: Clean it manually or use a different --exp-suffix"
        return 1
    elif [ "$RESUME" = true ]; then
        log "$task" "ERROR: --resume specified but ${exp_name} does not exist"
        return 1
    fi

    local logfile="${LOG_DIR}/${task}_train_$(date +%Y%m%d_%H%M%S).log"
    local script="hrpo_${task}.py"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python ${script} \\"
        [ -n "$dataset_root" ] && echo "    --dataset_root ${dataset_root} \\"
        echo "    --per_device_train_batch_size ${bs} --gradient_accumulation_steps ${ga} \\"
        echo "    --group_size ${group_size} --max_prompt_length ${max_prompt} --max_completion_length ${max_completion} \\"
        echo "    $(common_train_args)${resume_arg:+ ${resume_arg}}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python "$script" \
        ${dataset_root:+--dataset_root "$dataset_root"} \
        ${resume_arg} \
        --per_device_train_batch_size ${bs} \
        --gradient_accumulation_steps ${ga} \
        --group_size ${group_size} \
        --max_prompt_length ${max_prompt} \
        --max_completion_length ${max_completion} \
        $(common_train_args) \
        2>&1 | tee "${logfile}"
}

train_gsm8k() { train_task gsm8k 4 "$GSM8K_BS" "$GSM8K_GA" 1024 1024; }
train_math()  { train_task math  8 "$MATH_BS"  "$MATH_GA"  2048 2048 ../MATH; }
train_mmlu()  { train_task mmlu  8 "$MMLU_BS"  "$MMLU_GA"  1024 1024 ../MMLU_Train_Merged; }
train_rag()   { train_task rag   4 "$RAG_BS"   "$RAG_GA"   2048 1024 ../RAG_Train_Merged; }

# ========================= Evaluation Functions ==============================
eval_gsm8k() {
    local task="gsm8k"
    local exp_name
    exp_name=$(get_exp_name "$task" 4)
    local ckpt
    ckpt=$(find_latest_checkpoint "$exp_name")
    if [ "$DRY_RUN" = true ] && [ -z "$ckpt" ]; then
        ckpt=$(dry_run_checkpoint_hint "$exp_name")
    fi

    if [ -z "$ckpt" ]; then
        log "$task" "No checkpoint found in ${exp_name}, skipping eval"
        return 1
    fi

    if [ -f "${ckpt}/eval_results.json" ]; then
        log "$task" "Eval results already exist at ${ckpt}/eval_results.json, skipping"
        return 0
    fi

    log "$task" "Evaluating checkpoint: ${ckpt}"
    local logfile="${LOG_DIR}/gsm8k_eval_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_gsm8k.py --checkpoint_path ${ckpt} --batch_size ${EVAL_BS}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_gsm8k.py \
        --checkpoint_path "${ckpt}" \
        --batch_size ${EVAL_BS} \
        2>&1 | tee "${logfile}"
}

eval_math() {
    local task="math"
    local exp_name
    exp_name=$(get_exp_name "$task" 8)
    local ckpt
    ckpt=$(find_latest_checkpoint "$exp_name")
    if [ "$DRY_RUN" = true ] && [ -z "$ckpt" ]; then
        ckpt=$(dry_run_checkpoint_hint "$exp_name")
    fi

    if [ -z "$ckpt" ]; then
        log "$task" "No checkpoint found in ${exp_name}, skipping eval"
        return 1
    fi

    if [ -f "${ckpt}/eval_results.json" ]; then
        log "$task" "Eval results already exist at ${ckpt}/eval_results.json, skipping"
        return 0
    fi

    log "$task" "Evaluating checkpoint: ${ckpt}"
    local logfile="${LOG_DIR}/math_eval_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_math.py --checkpoint_path ${ckpt} --batch_size ${EVAL_BS}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_math.py \
        --checkpoint_path "${ckpt}" \
        --batch_size ${EVAL_BS} \
        2>&1 | tee "${logfile}"
}

eval_mmlu() {
    local task="mmlu"
    local exp_name
    exp_name=$(get_exp_name "$task" 8)
    local ckpt
    ckpt=$(find_latest_checkpoint "$exp_name")
    if [ "$DRY_RUN" = true ] && [ -z "$ckpt" ]; then
        ckpt=$(dry_run_checkpoint_hint "$exp_name")
    fi

    if [ -z "$ckpt" ]; then
        log "$task" "No checkpoint found in ${exp_name}, skipping eval"
        return 1
    fi

    if [ -f "${ckpt}/eval_results_mmlust.json" ]; then
        log "$task" "Eval results already exist at ${ckpt}/eval_results_mmlust.json, skipping"
        return 0
    fi

    log "$task" "Evaluating checkpoint: ${ckpt}"
    local logfile="${LOG_DIR}/mmlu_eval_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_mmlust.py --checkpoint_path ${ckpt} --batch_size ${EVAL_BS}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_mmlust.py \
        --checkpoint_path "${ckpt}" \
        --batch_size ${EVAL_BS} \
        2>&1 | tee "${logfile}"
}

eval_arcc() {
    local task="arcc"
    local exp_name
    exp_name=$(get_exp_name "mmlu" 8)  # ARC-C shares checkpoint with MMLU
    local ckpt
    ckpt=$(find_latest_checkpoint "$exp_name")
    if [ "$DRY_RUN" = true ] && [ -z "$ckpt" ]; then
        ckpt=$(dry_run_checkpoint_hint "$exp_name")
    fi

    if [ -z "$ckpt" ]; then
        log "$task" "No checkpoint found in ${exp_name}, skipping eval"
        return 1
    fi

    if [ -f "${ckpt}/eval_results_ai2_arc.json" ]; then
        log "$task" "Eval results already exist at ${ckpt}/eval_results_ai2_arc.json, skipping"
        return 0
    fi

    log "$task" "Evaluating checkpoint: ${ckpt}"
    local logfile="${LOG_DIR}/arcc_eval_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_arcc.py --checkpoint_path ${ckpt} --batch_size ${EVAL_BS}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_arcc.py \
        --checkpoint_path "${ckpt}" \
        --batch_size ${EVAL_BS} \
        2>&1 | tee "${logfile}"
}

eval_rag() {
    local task="rag"
    local exp_name
    exp_name=$(get_exp_name "$task" 4)
    local ckpt
    ckpt=$(find_latest_checkpoint "$exp_name")
    if [ "$DRY_RUN" = true ] && [ -z "$ckpt" ]; then
        ckpt=$(dry_run_checkpoint_hint "$exp_name")
    fi

    if [ -z "$ckpt" ]; then
        log "$task" "No checkpoint found in ${exp_name}, skipping eval"
        return 1
    fi

    # Check RAG eval datasets exist
    local rag_eval_dir="../RAG_Eval"
    local missing=false
    for sub in NQ_Eval TQ_Eval 2Wiki_Eval HotpotQA_Eval Bamboogle_Eval; do
        if [ ! -d "${rag_eval_dir}/${sub}" ]; then
            missing=true
            break
        fi
    done
    if [ "$missing" = true ]; then
        log "$task" "WARNING: RAG eval datasets not found at ${rag_eval_dir}/"
        log "$task" "Expected subdirs: NQ_Eval, TQ_Eval, 2Wiki_Eval, HotpotQA_Eval, Bamboogle_Eval"
        log "$task" "Skipping RAG evaluation"
        return 1
    fi

    local all_rag_done=true
    for code in nq tq 2wiki hotpotqa bamboogle; do
        if [ ! -f "${ckpt}/eval_results_${code}.json" ]; then
            all_rag_done=false
            break
        fi
    done
    if [ "$all_rag_done" = true ]; then
        log "$task" "All RAG eval results already exist at ${ckpt}/, skipping"
        return 0
    fi

    log "$task" "Evaluating checkpoint: ${ckpt}"
    local logfile="${LOG_DIR}/rag_eval_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_rag.py --checkpoint_path ${ckpt} --batch_size ${EVAL_BS}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_rag.py \
        --checkpoint_path "${ckpt}" \
        --batch_size ${EVAL_BS} \
        2>&1 | tee "${logfile}"
}

# ========================= Summary ==========================================
print_summary() {
    echo ""
    echo "============================================================"
    if [ "$MODE" = "thrpo" ]; then
        echo "                  THRPO Pipeline Summary"
    else
        echo "                   HRPO Pipeline Summary"
    fi
    echo "============================================================"
    echo "Model:  ${MODEL}"
    echo "GPU:    ${GPU_ID}"
    echo "Tasks:  ${TASKS}"
    if [ "$MODE" = "thrpo" ]; then
        echo "Time conditioning: thinking_time_loss_weight=${THINKING_TIME_LOSS_WEIGHT}, lr=${LR_TIME_CONDITIONING}"
    fi
    echo ""

    for task in "${TASK_LIST[@]}"; do
        local group_size
        case "$task" in
            gsm8k) group_size=4 ;;
            math)  group_size=8 ;;
            mmlu)  group_size=8 ;;
            rag)   group_size=4 ;;
        esac
        local exp_name
        exp_name=$(get_exp_name "$task" "$group_size")
        local ckpt
        ckpt=$(find_latest_checkpoint "$exp_name" 2>/dev/null || true)

        printf "  %-8s | " "$task"
        if [ -n "$ckpt" ]; then
            printf "checkpoint: %s" "$(basename "$ckpt")"
            if [ "$task" = "rag" ]; then
                # Show all 5 RAG benchmark accuracies
                local rag_results=""
                local rag_pending=false
                for code in nq tq 2wiki hotpotqa bamboogle; do
                    local rf="${ckpt}/eval_results_${code}.json"
                    if [ -f "$rf" ]; then
                        local racc
                        racc=$(python -c "
import json
d = json.load(open('${rf}'))
print(f\"{d['metrics']['accuracy']:.4f}\")
" 2>/dev/null || echo "N/A")
                        rag_results="${rag_results} ${code}=${racc}"
                    else
                        rag_pending=true
                    fi
                done
                if [ -n "$rag_results" ]; then
                    printf " |%s" "$rag_results"
                    if [ "$rag_pending" = true ]; then
                        printf " (partial)"
                    fi
                else
                    printf " | eval: pending"
                fi
            else
                local eval_file
                case "$task" in
                    gsm8k|math) eval_file="${ckpt}/eval_results.json" ;;
                    mmlu)       eval_file="${ckpt}/eval_results_mmlust.json" ;;
                esac
                if [ -f "$eval_file" ]; then
                    local acc
                    acc=$(python -c "
import json
d = json.load(open('${eval_file}'))
print(f\"{d['metrics']['accuracy']:.4f}\")
" 2>/dev/null || echo "N/A")
                    printf " | accuracy: %s" "$acc"
                else
                    printf " | eval: pending"
                fi
            fi
            # Show ARC-C results alongside MMLU (shared checkpoint)
            if [ "$task" = "mmlu" ]; then
                echo ""
                printf "  %-8s | " "arcc"
                printf "checkpoint: %s" "$(basename "$ckpt")"
                local arcc_file="${ckpt}/eval_results_ai2_arc.json"
                if [ -f "$arcc_file" ]; then
                    local arcc_acc
                    arcc_acc=$(python -c "
import json
d = json.load(open('${arcc_file}'))
print(f\"{d['metrics']['accuracy']:.4f}\")
" 2>/dev/null || echo "N/A")
                    printf " | accuracy: %s" "$arcc_acc"
                else
                    printf " | eval: pending"
                fi
            fi
        else
            printf "no checkpoint"
        fi
        echo ""
    done

    if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
        echo ""
        echo "  FAILED: ${FAILED_TASKS[*]}"
    fi
    echo "============================================================"
}

# ========================= Main ==============================================
main() {
    log "MAIN" "=========================================="
    log "MAIN" "${MODE_LABEL} Training & Evaluation Pipeline"
    log "MAIN" "=========================================="
    log "MAIN" "Model:       ${MODEL}"
    log "MAIN" "GPU:         ${GPU_ID}"
    log "MAIN" "Tasks:       ${TASKS}"
    if [ "$MODE" = "thrpo" ]; then
        log "MAIN" "Time cond:   enabled (thinking_time_loss_weight=${THINKING_TIME_LOSS_WEIGHT}, lr=${LR_TIME_CONDITIONING})"
    fi
    log "MAIN" "Eval only:   ${EVAL_ONLY}"
    log "MAIN" "Skip eval:   ${SKIP_EVAL}"
    log "MAIN" "Resume:      ${RESUME}"
    log "MAIN" "WandB:       $([ "$NO_WANDB" = true ] && echo 'disabled' || echo 'enabled')"

    # Parse task list
    if [ "$TASKS" = "all" ]; then
        TASK_LIST=(gsm8k math mmlu rag)
    else
        IFS=',' read -ra TASK_LIST <<< "$TASKS"
    fi

    # Setup
    mkdir -p "${LOG_DIR}"
    mkdir -p "${WORK_DIR}/experiments"
    cd "${WORK_DIR}"

    # Activate conda
    activate_env

    # Disable WandB if requested.
    # Prefer WANDB_MODE over WANDB_DISABLED to avoid upstream deprecation noise.
    if [ "$NO_WANDB" = true ]; then
        unset WANDB_DISABLED
        export WANDB_MODE=disabled
        log "MAIN" "WandB disabled"
    fi

    # Prepare datasets if requested or if RAG_Eval is missing
    if [ "$PREP_DATA" = true ]; then
        local task_arg
        task_arg=$(IFS=,; echo "${TASK_LIST[*]}")
        local stage_arg="all"
        [ "$EVAL_ONLY" = true ] && stage_arg="eval"
        [ "$SKIP_EVAL" = true ] && stage_arg="train"
        log "MAIN" "Preparing datasets for tasks=${task_arg} stage=${stage_arg}"
        if [ "$DRY_RUN" = true ]; then
            log "MAIN" "[DRY-RUN] python prepare_data.py --tasks ${task_arg} --stage ${stage_arg} --with-retrieval"
        else
            python prepare_data.py --tasks "$task_arg" --stage "$stage_arg" --with-retrieval
        fi
    elif [ "$SKIP_EVAL" != true ]; then
        # Auto-check: if rag is in task list and RAG_Eval doesn't exist, warn user + auto-prep
        for task in "${TASK_LIST[@]}"; do
            if [ "$task" = "rag" ] && [ ! -d "../RAG_Eval/NQ_Eval" ]; then
                log "MAIN" "WARNING: RAG eval datasets missing at ../RAG_Eval/"
                log "MAIN" "Auto-preparing via prepare_data.py --tasks rag --stage eval --with-retrieval"
                log "MAIN" "(NQ/TQ/Bamboogle will be closed-book if rank_bm25 is not installed)"
                if [ "$DRY_RUN" = true ]; then
                    log "MAIN" "[DRY-RUN] python prepare_data.py --tasks rag --stage eval --with-retrieval"
                else
                    python prepare_data.py --tasks rag --stage eval --with-retrieval
                fi
                break
            fi
        done
    fi

    # Validate datasets
    for task in "${TASK_LIST[@]}"; do
        if ! check_dataset "$task"; then
            log "MAIN" "Dataset check failed for ${task}, aborting"
            exit 1
        fi
    done
    log "MAIN" "All datasets validated"

    # Training phase
    if [ "$EVAL_ONLY" != true ]; then
        for task in "${TASK_LIST[@]}"; do
            log "MAIN" "==================== Training ${MODE_LABEL}: ${task} ===================="
            if ! "train_${task}"; then
                log "MAIN" "WARNING: Training ${task} failed"
                FAILED_TASKS+=("train_${task}")
            fi
        done
    fi

    # Evaluation phase
    if [ "$SKIP_EVAL" != true ]; then
        for task in "${TASK_LIST[@]}"; do
            log "MAIN" "==================== Evaluating ${MODE_LABEL}: ${task} ===================="
            if ! "eval_${task}"; then
                log "MAIN" "WARNING: Evaluation ${task} failed"
                FAILED_TASKS+=("eval_${task}")
            fi
            # ARC-C shares checkpoint with MMLU (paper trains on merged MMLU+ARC-C)
            if [ "$task" = "mmlu" ]; then
                log "MAIN" "==================== Evaluating ${MODE_LABEL}: arcc (from mmlu checkpoint) ===================="
                if ! eval_arcc; then
                    log "MAIN" "WARNING: Evaluation arcc failed"
                    FAILED_TASKS+=("eval_arcc")
                fi
            fi
        done
    fi

    # Print summary
    print_summary

    if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
        log "MAIN" "Completed with ${#FAILED_TASKS[@]} failure(s): ${FAILED_TASKS[*]}"
        exit 1
    else
        log "MAIN" "All tasks completed successfully!"
    fi
}

main
