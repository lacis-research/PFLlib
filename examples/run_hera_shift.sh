#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_ROOT='/media/bolla/HD UTILITÁRIOS/testesherafl'
export TMPDIR="$EXPERIMENT_ROOT/tmp" MPLCONFIGDIR="$EXPERIMENT_ROOT/tmp/matplotlib" PYTHONPYCACHEPREFIX="$EXPERIMENT_ROOT/tmp/pycache-herafl-v3"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
SHIFT_GOAL="${SHIFT_GOAL:-hera-shift-$(TZ=America/Belem date +%Y%m%d-%H%M%S)-$$}"
cd "$EXPERIMENT_ROOT/PFLlib/system"
"$EXPERIMENT_ROOT/venv/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA indisponível"; print(torch.cuda.get_device_name(0))'
"$EXPERIMENT_ROOT/venv/bin/python" -u main.py \
 -algo HERAFL -data Cifar10DirichletFull30 -m VGG11 -dev cuda -did 0 \
 -nc 30 -ncl 10 -gr "${HERA_ROUNDS:-200}" -jr 0.2 -ls "${HERA_LOCAL_EPOCHS:-5}" -lbs 64 -lr 0.0001 \
 --policy-seed 42 --resource-preset paper --aligned-comparison --wait-for-all \
 --pruning-arms 0.5 0.5555555555555556 0.6111111111111112 0.6666666666666666 0.7222222222222222 0.7777777777777778 0.8333333333333333 0.8888888888888888 0.9444444444444444 1.0 \
 --bandit-alpha 0.5 --bandit-beta 0.75 --reward-weights 0.6 0.1 0.3 \
 --straggler-predictor true -go "$SHIFT_GOAL" \
 2>&1 | tee "$EXPERIMENT_ROOT/logs/$SHIFT_GOAL.log"
