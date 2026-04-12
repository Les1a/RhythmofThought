#!/bin/bash
###############################################################################
# run_tgrpo_all.sh — TGRPO Training & Evaluation Wrapper
#
# Thin wrapper around run_grpo_all.sh that pins the mode to TGRPO
# (GRPO + time conditioning, no thinking residual).
###############################################################################
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_grpo_all.sh" --time-cond "$@"
