#!/bin/bash
###############################################################################
# run_thrpo_all.sh — THRPO Training & Evaluation Wrapper
#
# Thin wrapper around run_hrpo_all.sh that pins the mode to THRPO
# (HRPO + thinking-time conditioning). Time predictor history length and
# warmup fraction are forwarded to the underlying launcher.
###############################################################################
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_hrpo_all.sh" --mode thrpo "$@"
