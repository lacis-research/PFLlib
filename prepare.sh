#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"

# Reuse an existing installation.
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -x "$MINICONDA_DIR/bin/conda" ]]; then
    CONDA_BASE="$MINICONDA_DIR"
else
    INSTALLER="$SCRIPT_DIR/Miniconda3-latest-Linux-x86_64.sh"
    wget -O "$INSTALLER" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "$INSTALLER" -p "$MINICONDA_DIR"
    CONDA_BASE="$MINICONDA_DIR"
fi

# .bashrc can return early in noninteractive shells; load Conda directly.
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Keep the package cache local without editing shell configuration.
export PIP_CACHE_DIR="$SCRIPT_DIR/tmp"
mkdir -p "$PIP_CACHE_DIR"

conda env create -f "$SCRIPT_DIR/env_cuda_latest.yaml"
conda activate pfllib

echo 'Ambiente preparado. No seu terminal, execute: conda activate pfllib'
