#!/usr/bin/env bash
# srun.sh — Submit repository commands to Slurm with workspace defaults.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash srun.sh [OPTIONS] "COMMAND"
  bash srun.sh [OPTIONS] -- COMMAND [ARGS...]

Submit a non-interactive Slurm job with sane defaults for this repo:
  1x H200, 128G RAM, 24h, 1 task

Examples:
  bash srun.sh "bash run_thrpo_all.sh --tasks gsm8k"
  bash srun.sh --name thrpo-gsm8k -- bash run_thrpo_all.sh --tasks gsm8k
  bash srun.sh --dry-run "bash run_thrpo_all.sh --tasks gsm8k"

Options:
  --name NAME        Job name
  --gpus N           GPU count (default: 1)
  --gpu-type TYPE    GPU type (default: H200)
  --cpus N           CPUs per task (default: 8)
  --mem GB           Memory in GB (default: 128)
  --time HH:MM:SS    Time limit (default: 24:00:00)
  --env NAME         Conda env name (default: rot)
  --account NAME     Slurm account (default: gts-wl67-paid)
  --partition NAME   Optional Slurm partition
  --log-dir DIR      Log directory (default: ./logs/slurm)
  --dry-run          Print the sbatch command without submitting
  -h, --help         Show this help
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
job_name="${JOB_NAME:-rot-job}"
gpu_count="${GPU_COUNT:-1}"
gpu_type="${GPU_TYPE:-H200}"
cpus="${CPUS_PER_TASK:-8}"
mem_gb="${MEMORY_GB:-128}"
time_limit="${TIME_LIMIT:-24:00:00}"
conda_env="${CONDA_ENV_NAME:-rot}"
account="${SLURM_ACCOUNT_NAME:-gts-wl67-paid}"
partition="${SLURM_PARTITION_NAME:-}"
log_dir="${LOG_DIR:-${repo_root}/logs/slurm}"
dry_run=0
command_string=""

normalize_command_if_needed() {
  local cmd="$1"
  if [[ "${cmd}" =~ ^bash[[:space:]]+([A-Za-z0-9_./-]+)([[:space:]].*)?$ ]]; then
    local script="${BASH_REMATCH[1]}"
    local tail="${BASH_REMATCH[2]:-}"
    if [[ "${script}" != *.sh && -f "${repo_root}/${script}.sh" && ! -e "${repo_root}/${script}" ]]; then
      printf 'bash %s.sh%s' "${script}" "${tail}"
      return 0
    fi
  fi
  printf '%s' "${cmd}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      job_name="$2"
      shift 2
      ;;
    --gpus)
      gpu_count="$2"
      shift 2
      ;;
    --gpu-type)
      gpu_type="$2"
      shift 2
      ;;
    --cpus)
      cpus="$2"
      shift 2
      ;;
    --mem)
      mem_gb="$2"
      shift 2
      ;;
    --time)
      time_limit="$2"
      shift 2
      ;;
    --env)
      conda_env="$2"
      shift 2
      ;;
    --account)
      account="$2"
      shift 2
      ;;
    --partition)
      partition="$2"
      shift 2
      ;;
    --log-dir)
      log_dir="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Missing command after --" >&2
        exit 1
      fi
      printf -v command_string '%q ' "$@"
      command_string="${command_string% }"
      break
      ;;
    *)
      if [[ -n "${command_string}" ]]; then
        echo "Unexpected extra argument: $1" >&2
        exit 1
      fi
      command_string="$1"
      shift
      ;;
  esac
done

if [[ -z "${command_string}" ]]; then
  echo "Missing command to submit." >&2
  usage
  exit 1
fi

original_command_string="${command_string}"
command_string="$(normalize_command_if_needed "${command_string}")"

for value_name in gpu_count cpus mem_gb; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid numeric value for ${value_name}: ${value}" >&2
    exit 1
  fi
done

mkdir -p "${log_dir}"
job_script="$(mktemp "${log_dir}/sbatch_${job_name//[^A-Za-z0-9_.-]/_}_XXXXXX.sh")"
chmod 700 "${job_script}"
trap 'rm -f "${job_script}"' EXIT

{
  echo '#!/usr/bin/env bash'
  # Avoid nounset here: cluster/global shell init scripts often read unset vars.
  # The actual training scripts can enforce their own strict mode if needed.
  echo 'set -eo pipefail'
  echo
  echo 'if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then'
  echo '  source "$HOME/miniconda3/etc/profile.d/conda.sh"'
  echo 'elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then'
  echo '  source "$HOME/anaconda3/etc/profile.d/conda.sh"'
  echo 'else'
  echo '  source ~/.bashrc'
  echo 'fi'
  printf 'conda activate %q\n' "${conda_env}"
  printf 'cd %q\n' "${repo_root}"
  printf 'echo "[%s] Running on $(hostname) in %s"\n' '$(date "+%Y-%m-%d %H:%M:%S")' "${repo_root}"
  printf 'echo "Command: %s"\n' "${command_string}"
  printf 'eval %q\n' "${command_string}"
} > "${job_script}"

sbatch_cmd=(
  sbatch
  "--job-name=${job_name}"
  "--account=${account}"
  "--gres=gpu:${gpu_type}:${gpu_count}"
  "--cpus-per-task=${cpus}"
  "--mem=${mem_gb}G"
  "--ntasks=1"
  "--time=${time_limit}"
  "--output=${log_dir}/%x_%j.out"
  "--error=${log_dir}/%x_%j.err"
)

if [[ -n "${partition}" ]]; then
  sbatch_cmd+=("--partition=${partition}")
fi

sbatch_cmd+=("${job_script}")

echo "Submitting non-interactive Slurm job"
echo "  job_name=${job_name}"
echo "  account=${account}"
echo "  gpu=${gpu_type} x ${gpu_count}"
echo "  cpus_per_task=${cpus}"
echo "  memory=${mem_gb}G"
echo "  time_limit=${time_limit}"
echo "  conda_env=${conda_env}"
echo "  log_dir=${log_dir}"
echo "  command=${command_string}"
if [[ "${command_string}" != "${original_command_string}" ]]; then
  echo "  note=auto-normalized missing .sh suffix in bash command"
fi

if [[ "${dry_run}" -eq 1 ]]; then
  printf 'Dry run command:'
  printf ' %q' "${sbatch_cmd[@]}"
  printf '\n'
  exit 0
fi

"${sbatch_cmd[@]}"
