#!/usr/bin/env bash
set -euo pipefail

COMPARE_ONLY=0
if [[ "${1:-}" == "--compare-only" ]]; then
    COMPARE_ONLY=1
    shift
fi
REQUESTED_GOAL="${GOAL:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Inicializa o Conda também em shells não interativos.
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -x "${MINICONDA_DIR:-$HOME/miniconda3}/bin/conda" ]]; then
    CONDA_BASE="${MINICONDA_DIR:-$HOME/miniconda3}"
else
    echo "Conda não encontrado. Instale-o ou informe MINICONDA_DIR." >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate pfllib

# Usa o Python do ambiente ativado, salvo sobrescrita explícita.
PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-cifar10}"
SCENARIO="${SCENARIO:-iid}"
NUM_ROUNDS="${NUM_ROUNDS:-300}"
NUM_CLIENTS="${NUM_CLIENTS:-30}" # vazio: usa config.json existente ou o padrão do gerador
SELECTION_FRACTION="${SELECTION_FRACTION:-0.2}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL="${MODEL:-}" # selecionado por dataset abaixo
DEVICE="${DEVICE:-cuda}" # main.py usa CPU se CUDA não estiver disponível
DEVICE_ID="${DEVICE_ID:-0}"
GOAL="${GOAL:-start-$(date +%Y%m%d-%H%M%S)-$$}"
ALGORITHMS="${ALGORITHMS:-HERAFL FedLoad FedAvg}"
STRAGGLER_PREDICTOR="${STRAGGLER_PREDICTOR:-true}"
BALANCE="${BALANCE:-balance}"
PARTITION="${PARTITION:-dir}"
COMPARE="${COMPARE:-1}"
# Tabela I e seção IV do artigo FedLoad.
BANDIT_ALPHA="${BANDIT_ALPHA:-0.5}"
BANDIT_BETA="${BANDIT_BETA:-0.75}"
PRUNING_ARMS="${PRUNING_ARMS:-0.5 0.5555555555555556 0.6111111111111112 0.6666666666666666 0.7222222222222222 0.7777777777777778 0.8333333333333333 0.8888888888888888 0.9444444444444444 1.0}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.6 0.1 0.3}"
RESOURCE_PRESET="${RESOURCE_PRESET:-paper}"
# Valores não informados no artigo: defaults da implementação local.
LEARNING_RATE="${LEARNING_RATE:-}" # padrão escolhido pelo modelo, compartilhado entre métodos
POLICY_SEED="${POLICY_SEED:-42}"
ROUND_TIMEOUT="${ROUND_TIMEOUT:-180}"
WAIT_FOR_ALL="${WAIT_FOR_ALL:-1}" # 1: agrega todas as atualizações selecionadas
read -r -a pruning_arms <<< "$PRUNING_ARMS"
read -r -a reward_weights <<< "$REWARD_WEIGHTS"

case "${DATASET,,}" in
    mnist) DATASET=MNIST ;;
    cifar10|cifar-10) DATASET=Cifar10 ;;
    cifar100|cifar-100) DATASET=Cifar100 ;;
    cinic10|cinic-10) DATASET=Cinic10 ;;
    fashionmnist|fashion-mnist) DATASET=FashionMNIST ;;
esac
if [[ -z "$MODEL" ]]; then
    case "$DATASET" in
        Cifar10) MODEL=VGG11 ;;
        Cinic10|Cifar100) MODEL=VGG16 ;;
        *) MODEL=CNN ;;
    esac
fi
if [[ -z "$LEARNING_RATE" ]]; then
    case "${MODEL,,}" in
        vgg11|vgg16) LEARNING_RATE=0.0001 ;;
        *) LEARNING_RATE=0.001 ;;
    esac
fi
case "$SCENARIO" in
    iid|idd) SCENARIO=iid ;;
    non-iid|non_iid|noniid) SCENARIO=non-iid ;;
    *) echo "SCENARIO deve ser iid ou non-iid." >&2; exit 1 ;;
esac

read -r -a algorithms <<< "$ALGORITHMS"
if (( ${#algorithms[@]} == 0 )); then
    echo "ALGORITHMS deve conter HERAFL, FedLoad e/ou FedAvg." >&2
    exit 1
fi
for i in "${!algorithms[@]}"; do
    case "${algorithms[$i],,}" in
        herafl|hera-fl|hera_fl) algorithms[$i]=HERAFL ;;
        fedload) algorithms[$i]=FedLoad ;;
        fedavg) algorithms[$i]=FedAvg ;;
        *) echo "Algoritmo não suportado: ${algorithms[$i]}" >&2; exit 1 ;;
    esac
done

# Recupera a comparação sem preparar datasets ou repetir treinamentos.
if [[ "$COMPARE_ONLY" == "1" ]]; then
    comparison_args=(--dataset "$DATASET" --algorithms "${algorithms[@]}")
    if [[ -n "$REQUESTED_GOAL" ]]; then
        comparison_args+=(--goal "$REQUESTED_GOAL")
    else
        comparison_args+=(--latest)
    fi
    exec "$PYTHON" tools/plot_comparison.py "${comparison_args[@]}" "$@"
fi

# Reutiliza partições compatíveis ou chama o gerador nativo automaticamente.
prepare_args=(--dataset "$DATASET" --scenario "$SCENARIO"
              --balance "$BALANCE" --partition "$PARTITION")
if [[ -n "$NUM_CLIENTS" ]]; then
    prepare_args+=(--num-clients "$NUM_CLIENTS")
fi
dataset_config="$("$PYTHON" tools/prepare_dataset.py "${prepare_args[@]}")"
read -r dataset_clients dataset_classes DATASET <<< "$dataset_config"

common_args=( 
    -data "$DATASET" -m "$MODEL" -gr "$NUM_ROUNDS"
    -nc "$dataset_clients" -ncl "$dataset_classes"
    -jr "$SELECTION_FRACTION" -ls "$LOCAL_EPOCHS" -lbs "$BATCH_SIZE"
    -lr "$LEARNING_RATE" --policy-seed "$POLICY_SEED"
    --resource-preset "$RESOURCE_PRESET"
    --aligned-comparison --pruning-round-timeout "$ROUND_TIMEOUT"
    --pruning-arms "${pruning_arms[@]}"
    --bandit-alpha "$BANDIT_ALPHA" --bandit-beta "$BANDIT_BETA"
    --reward-weights "${reward_weights[@]}"
    -dev "$DEVICE" -did "$DEVICE_ID" -go "$GOAL" --scenario "$SCENARIO"
)
case "$WAIT_FOR_ALL" in
    1) common_args+=(--wait-for-all) ;;
    0) ;;
    *) echo "WAIT_FOR_ALL deve ser 0 ou 1." >&2; exit 1 ;;
esac
cd "$SCRIPT_DIR/system"
for algorithm in "${algorithms[@]}"; do
    echo "Executando $algorithm: dataset=$DATASET, cenário=$SCENARIO, goal=$GOAL"
    algorithm_args=()
    if [[ "$algorithm" == "FedAvg" ]]; then
        algorithm_args+=(--record-resources)
    fi
    "$PYTHON" main.py "${common_args[@]}" "${algorithm_args[@]}" -algo "$algorithm" \
        --straggler-predictor "$STRAGGLER_PREDICTOR" "$@"
done

if [[ "$COMPARE" == "1" ]] && (( ${#algorithms[@]} > 1 )); then
    cd "$SCRIPT_DIR"
    if ! "$PYTHON" tools/plot_comparison.py --dataset "$DATASET" --goal "$GOAL" \
        --algorithms "${algorithms[@]}"; then
        echo "Comparação falhou; os resultados de treino foram preservados." >&2
        printf 'Para tentar novamente: DATASET=%q GOAL=%q ALGORITHMS=%q ./start.sh --compare-only\n' "$DATASET" "$GOAL" "${algorithms[*]}" >&2
        exit 1
    fi
fi
