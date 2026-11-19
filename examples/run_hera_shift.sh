#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DATA_PROTOCOL=fedsc
export DATASET="${DATASET:-Cifar10}" NUM_CLIENTS="${NUM_CLIENTS:-30}"
export ALGORITHMS="${ALGORITHMS:-HERAFL}"
export NUM_ROUNDS="${NUM_ROUNDS:-${HERA_ROUNDS:-200}}"
export LOCAL_EPOCHS="${LOCAL_EPOCHS:-${HERA_LOCAL_EPOCHS:-5}}"
export FEDSC_ALPHA="${FEDSC_ALPHA:-0.05}" FEDSC_IMB_FACTOR="${FEDSC_IMB_FACTOR:-0.1}"
export FEDSC_SEED="${FEDSC_SEED:-66}"
if [[ -n "${SHIFT_GOAL:-}" ]]; then
    export GOAL="$SHIFT_GOAL"
fi
exec bash "$SCRIPT_DIR/../start.sh" "$@"
