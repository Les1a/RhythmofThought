#!/bin/bash
###############################################################################
# run_grpo_all.sh — GRPO Baseline Training & Evaluation Script
#
# Runs vanilla GRPO (without thinking residual) training and evaluation for
# GSM8K, MATH, MMLU, and RAG tasks, using the same hyperparameters as HRPO
# minus the thinking-residual-specific components.
#
# Smart skipping — by default won't re-train if checkpoints exist, won't re-eval if
# results exist. Pass --resume to continue training from the latest checkpoint
# instead of skipping (full state restore: optimizer, scheduler, RNG, global_step).
#
# Usage:
#   bash run_grpo_all.sh [OPTIONS]
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
#   --no-wandb            Disable WandB logging
#   --dry-run             Print commands without executing
#   --help                Show this help message
###############################################################################
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR"
LOG_DIR="${WORK_DIR}/logs"
CONDA_ENV="rot"

# ========================= Hyperparameters (same as HRPO, minus residual) ===
MODEL="Qwen/Qwen2.5-3B-Instruct"
SEED=42
LR=5e-6
BETA=0.005
LORA_RANK=32
WEIGHT_DECAY=0.1
WARMUP_RATIO=0.1
LR_SCHEDULER_TYPE="cosine"
OPTIMIZER="paged_adamw_8bit"
MAX_GRAD_NORM=0.1
TEMPERATURE=0.5

# ========================= H200-Optimized Batch Sizes ========================
GSM8K_BS=32;  GSM8K_GA=1
MATH_BS=64;   MATH_GA=1
MMLU_BS=64;   MMLU_GA=1
RAG_BS=64;    RAG_GA=1
EVAL_BS=128

# ========================= Defaults ==========================================
GPU_ID=0
TASKS="all"
EVAL_ONLY=false
SKIP_EVAL=false
NO_WANDB=false
DRY_RUN=false
RESUME=false
FAILED_TASKS=()

# ========================= Argument Parsing ==================================
show_help() {
    head -25 "$0" | tail -23 | sed 's/^# \?//'
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
        --no-wandb)   NO_WANDB=true; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --help|-h)    show_help ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ========================= Utility Functions =================================
log() {
    local task="$1"; shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$task] $*"
}

get_exp_name() {
    local task="$1"
    local group_size="$2"
    local model_short="${MODEL##*/}"
    echo "./experiments/${model_short}-${task}-grpo-group${group_size}-lora${LORA_RANK}-temp${TEMPERATURE}"
}

find_latest_checkpoint() {
    local exp_dir="$1"
    ls -d "${exp_dir}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1
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

# Common training args shared by all tasks (GRPO baseline: no residual args)
common_train_args() {
    echo "--model_name ${MODEL} \
        --lora_rank ${LORA_RANK} \
        --lr ${LR} \
        --beta ${BETA} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_ratio ${WARMUP_RATIO} \
        --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
        --optimizer ${OPTIMIZER} \
        --max_grad_norm ${MAX_GRAD_NORM} \
        --temperature ${TEMPERATURE} \
        --seed ${SEED}"
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
train_gsm8k() {
    local task="gsm8k"
    local group_size=4
    local exp_name
    exp_name=$(get_exp_name "$task" "$group_size")

    log "$task" "Experiment: ${exp_name}"
    log "$task" "Effective batch size: $((GSM8K_BS * GSM8K_GA)) (BS=${GSM8K_BS} x GA=${GSM8K_GA})"

    local resume_arg=""
    if [ -d "$exp_name" ] && ls "${exp_name}"/checkpoint-* &>/dev/null; then
        if [ "$RESUME" = true ]; then
            log "$task" "Resuming training from latest checkpoint in ${exp_name}"
            resume_arg="--resume"
        else
            log "$task" "Checkpoint already exists, skipping training (use --resume to continue)"
            return 0
        fi
    elif [ "$RESUME" = true ]; then
        log "$task" "ERROR: --resume specified but no checkpoint found in ${exp_name}"
        return 1
    fi

    local logfile="${LOG_DIR}/grpo_gsm8k_train_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python hrpo_gsm8k.py \\"
        echo "    --only_grpo${resume_arg:+ ${resume_arg}} \\"
        echo "    --per_device_train_batch_size ${GSM8K_BS} --gradient_accumulation_steps ${GSM8K_GA} \\"
        echo "    --group_size ${group_size} --max_prompt_length 1024 --max_completion_length 1024 \\"
        echo "    $(common_train_args)"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python hrpo_gsm8k.py \
        --only_grpo \
        ${resume_arg} \
        --per_device_train_batch_size ${GSM8K_BS} \
        --gradient_accumulation_steps ${GSM8K_GA} \
        --group_size ${group_size} \
        --max_prompt_length 1024 \
        --max_completion_length 1024 \
        --model_name "${MODEL}" \
        --lora_rank ${LORA_RANK} \
        --lr ${LR} \
        --beta ${BETA} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_ratio ${WARMUP_RATIO} \
        --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
        --optimizer ${OPTIMIZER} \
        --max_grad_norm ${MAX_GRAD_NORM} \
        --temperature ${TEMPERATURE} \
        --seed ${SEED} \
        2>&1 | tee "${logfile}"
}

train_math() {
    local task="math"
    local group_size=8
    local exp_name
    exp_name=$(get_exp_name "$task" "$group_size")

    log "$task" "Experiment: ${exp_name}"
    log "$task" "Effective batch size: $((MATH_BS * MATH_GA)) (BS=${MATH_BS} x GA=${MATH_GA})"

    local resume_arg=""
    if [ -d "$exp_name" ] && ls "${exp_name}"/checkpoint-* &>/dev/null; then
        if [ "$RESUME" = true ]; then
            log "$task" "Resuming training from latest checkpoint in ${exp_name}"
            resume_arg="--resume"
        else
            log "$task" "Checkpoint already exists, skipping training (use --resume to continue)"
            return 0
        fi
    elif [ "$RESUME" = true ]; then
        log "$task" "ERROR: --resume specified but no checkpoint found in ${exp_name}"
        return 1
    fi

    local logfile="${LOG_DIR}/grpo_math_train_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python hrpo_math.py \\"
        echo "    --only_grpo${resume_arg:+ ${resume_arg}} \\"
        echo "    --dataset_root ../MATH \\"
        echo "    --per_device_train_batch_size ${MATH_BS} --gradient_accumulation_steps ${MATH_GA} \\"
        echo "    --group_size ${group_size} --max_prompt_length 2048 --max_completion_length 2048 \\"
        echo "    $(common_train_args)"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python hrpo_math.py \
        --only_grpo \
        ${resume_arg} \
        --dataset_root ../MATH \
        --per_device_train_batch_size ${MATH_BS} \
        --gradient_accumulation_steps ${MATH_GA} \
        --group_size ${group_size} \
        --max_prompt_length 2048 \
        --max_completion_length 2048 \
        --model_name "${MODEL}" \
        --lora_rank ${LORA_RANK} \
        --lr ${LR} \
        --beta ${BETA} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_ratio ${WARMUP_RATIO} \
        --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
        --optimizer ${OPTIMIZER} \
        --max_grad_norm ${MAX_GRAD_NORM} \
        --temperature ${TEMPERATURE} \
        --seed ${SEED} \
        2>&1 | tee "${logfile}"
}

train_mmlu() {
    local task="mmlu"
    local group_size=8
    local exp_name
    exp_name=$(get_exp_name "$task" "$group_size")

    log "$task" "Experiment: ${exp_name}"
    log "$task" "Effective batch size: $((MMLU_BS * MMLU_GA)) (BS=${MMLU_BS} x GA=${MMLU_GA})"

    local resume_arg=""
    if [ -d "$exp_name" ] && ls "${exp_name}"/checkpoint-* &>/dev/null; then
        if [ "$RESUME" = true ]; then
            log "$task" "Resuming training from latest checkpoint in ${exp_name}"
            resume_arg="--resume"
        else
            log "$task" "Checkpoint already exists, skipping training (use --resume to continue)"
            return 0
        fi
    elif [ "$RESUME" = true ]; then
        log "$task" "ERROR: --resume specified but no checkpoint found in ${exp_name}"
        return 1
    fi

    local logfile="${LOG_DIR}/grpo_mmlu_train_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python hrpo_mmlu.py \\"
        echo "    --only_grpo${resume_arg:+ ${resume_arg}} \\"
        echo "    --dataset_root ../MMLU_Train_Merged \\"
        echo "    --per_device_train_batch_size ${MMLU_BS} --gradient_accumulation_steps ${MMLU_GA} \\"
        echo "    --group_size ${group_size} --max_prompt_length 1024 --max_completion_length 1024 \\"
        echo "    $(common_train_args)"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python hrpo_mmlu.py \
        --only_grpo \
        ${resume_arg} \
        --dataset_root ../MMLU_Train_Merged \
        --per_device_train_batch_size ${MMLU_BS} \
        --gradient_accumulation_steps ${MMLU_GA} \
        --group_size ${group_size} \
        --max_prompt_length 1024 \
        --max_completion_length 1024 \
        --model_name "${MODEL}" \
        --lora_rank ${LORA_RANK} \
        --lr ${LR} \
        --beta ${BETA} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_ratio ${WARMUP_RATIO} \
        --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
        --optimizer ${OPTIMIZER} \
        --max_grad_norm ${MAX_GRAD_NORM} \
        --temperature ${TEMPERATURE} \
        --seed ${SEED} \
        2>&1 | tee "${logfile}"
}

train_rag() {
    local task="rag"
    local group_size=4
    local exp_name
    exp_name=$(get_exp_name "$task" "$group_size")

    log "$task" "Experiment: ${exp_name}"
    log "$task" "Effective batch size: $((RAG_BS * RAG_GA)) (BS=${RAG_BS} x GA=${RAG_GA})"

    local resume_arg=""
    if [ -d "$exp_name" ] && ls "${exp_name}"/checkpoint-* &>/dev/null; then
        if [ "$RESUME" = true ]; then
            log "$task" "Resuming training from latest checkpoint in ${exp_name}"
            resume_arg="--resume"
        else
            log "$task" "Checkpoint already exists, skipping training (use --resume to continue)"
            return 0
        fi
    elif [ "$RESUME" = true ]; then
        log "$task" "ERROR: --resume specified but no checkpoint found in ${exp_name}"
        return 1
    fi

    local logfile="${LOG_DIR}/grpo_rag_train_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python hrpo_rag.py \\"
        echo "    --only_grpo${resume_arg:+ ${resume_arg}} \\"
        echo "    --dataset_root ../RAG_Train_Merged \\"
        echo "    --per_device_train_batch_size ${RAG_BS} --gradient_accumulation_steps ${RAG_GA} \\"
        echo "    --group_size ${group_size} --max_prompt_length 2048 --max_completion_length 1024 \\"
        echo "    $(common_train_args)"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python hrpo_rag.py \
        --only_grpo \
        ${resume_arg} \
        --dataset_root ../RAG_Train_Merged \
        --per_device_train_batch_size ${RAG_BS} \
        --gradient_accumulation_steps ${RAG_GA} \
        --group_size ${group_size} \
        --max_prompt_length 2048 \
        --max_completion_length 1024 \
        --model_name "${MODEL}" \
        --lora_rank ${LORA_RANK} \
        --lr ${LR} \
        --beta ${BETA} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_ratio ${WARMUP_RATIO} \
        --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
        --optimizer ${OPTIMIZER} \
        --max_grad_norm ${MAX_GRAD_NORM} \
        --temperature ${TEMPERATURE} \
        --seed ${SEED} \
        2>&1 | tee "${logfile}"
}

# ========================= Evaluation Functions ==============================
eval_gsm8k() {
    local task="gsm8k"
    local exp_name
    exp_name=$(get_exp_name "$task" 4)
    local ckpt
    ckpt=$(find_latest_checkpoint "$exp_name")

    if [ -z "$ckpt" ]; then
        log "$task" "No checkpoint found in ${exp_name}, skipping eval"
        return 1
    fi

    if [ -f "${ckpt}/eval_results.json" ]; then
        log "$task" "Eval results already exist at ${ckpt}/eval_results.json, skipping"
        return 0
    fi

    log "$task" "Evaluating checkpoint: ${ckpt}"
    local logfile="${LOG_DIR}/grpo_gsm8k_eval_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_gsm8k.py --only_grpo --checkpoint_path ${ckpt} --batch_size ${EVAL_BS}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_gsm8k.py \
        --only_grpo \
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

    if [ -z "$ckpt" ]; then
        log "$task" "No checkpoint found in ${exp_name}, skipping eval"
        return 1
    fi

    if [ -f "${ckpt}/eval_results.json" ]; then
        log "$task" "Eval results already exist at ${ckpt}/eval_results.json, skipping"
        return 0
    fi

    log "$task" "Evaluating checkpoint: ${ckpt}"
    local logfile="${LOG_DIR}/grpo_math_eval_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_math.py --only_grpo --checkpoint_path ${ckpt} --batch_size ${EVAL_BS}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_math.py \
        --only_grpo \
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

    if [ -z "$ckpt" ]; then
        log "$task" "No checkpoint found in ${exp_name}, skipping eval"
        return 1
    fi

    if [ -f "${ckpt}/eval_results_mmlust.json" ]; then
        log "$task" "Eval results already exist at ${ckpt}/eval_results_mmlust.json, skipping"
        return 0
    fi

    log "$task" "Evaluating checkpoint: ${ckpt}"
    local logfile="${LOG_DIR}/grpo_mmlu_eval_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_mmlust.py --only_grpo --checkpoint_path ${ckpt} --batch_size ${EVAL_BS}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_mmlust.py \
        --only_grpo \
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

    if [ -z "$ckpt" ]; then
        log "$task" "No checkpoint found in ${exp_name}, skipping eval"
        return 1
    fi

    if [ -f "${ckpt}/eval_results_ai2_arc.json" ]; then
        log "$task" "Eval results already exist at ${ckpt}/eval_results_ai2_arc.json, skipping"
        return 0
    fi

    log "$task" "Evaluating checkpoint: ${ckpt}"
    local logfile="${LOG_DIR}/grpo_arcc_eval_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_arcc.py --only_grpo --checkpoint_path ${ckpt} --batch_size ${EVAL_BS}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_arcc.py \
        --only_grpo \
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
    local logfile="${LOG_DIR}/grpo_rag_eval_$(date +%Y%m%d_%H%M%S).log"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_rag.py --only_grpo --checkpoint_path ${ckpt} --batch_size ${EVAL_BS}"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_rag.py \
        --only_grpo \
        --checkpoint_path "${ckpt}" \
        --batch_size ${EVAL_BS} \
        2>&1 | tee "${logfile}"
}

# ========================= Summary ==========================================
print_summary() {
    echo ""
    echo "============================================================"
    echo "              GRPO Baseline Pipeline Summary"
    echo "============================================================"
    echo "Model:  ${MODEL}"
    echo "GPU:    ${GPU_ID}"
    echo "Tasks:  ${TASKS}"
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
    log "MAIN" "GRPO Baseline Training & Evaluation"
    log "MAIN" "=========================================="
    log "MAIN" "Model:       ${MODEL}"
    log "MAIN" "GPU:         ${GPU_ID}"
    log "MAIN" "Tasks:       ${TASKS}"
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

    # Disable WandB if requested
    if [ "$NO_WANDB" = true ]; then
        export WANDB_DISABLED=true
        log "MAIN" "WandB disabled"
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
            log "MAIN" "==================== Training GRPO: ${task} ===================="
            if ! "train_${task}"; then
                log "MAIN" "WARNING: Training ${task} failed"
                FAILED_TASKS+=("train_${task}")
            fi
        done
    fi

    # Evaluation phase
    if [ "$SKIP_EVAL" != true ]; then
        for task in "${TASK_LIST[@]}"; do
            log "MAIN" "==================== Evaluating GRPO: ${task} ===================="
            if ! "eval_${task}"; then
                log "MAIN" "WARNING: Evaluation ${task} failed"
                FAILED_TASKS+=("eval_${task}")
            fi
            # ARC-C shares checkpoint with MMLU (paper trains on merged MMLU+ARC-C)
            if [ "$task" = "mmlu" ]; then
                log "MAIN" "==================== Evaluating GRPO: arcc (from mmlu checkpoint) ===================="
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
