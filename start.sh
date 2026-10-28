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
DATASET="${DATASET:-Cifar10}" # MNIST, Cifar10, Cifar100, Cinic10, FashionMNIST
SCENARIO="${SCENARIO:-noniid}"
NUM_ROUNDS="${NUM_ROUNDS:-200}"
NUM_CLIENTS="${NUM_CLIENTS:-30}" # vazio: usa config.json existente ou o padrão do gerador
SELECTION_FRACTION="${SELECTION_FRACTION:-.2}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL="${MODEL:-CNN}" # use MODEL=VGG16 ./start.sh para o experimento com VGG16
DEVICE="${DEVICE:-cuda}" # main.py usa CPU se CUDA não estiver disponível
DEVICE_ID="${DEVICE_ID:-0}"
GOAL="${GOAL:-start-$(date +%Y%m%d-%H%M%S)-$$}"
ALGORITHMS="${ALGORITHMS:-HERAFL FedLoad FedAvg}" #-HERAFL FedLoad FedAvg
HERA_SEMANTIC_POLICY="${HERA_SEMANTIC_POLICY:-linucb}" # linucb, heuristic ou legacy (controle preservado)
HERA_COMMUNICATION="${HERA_COMMUNICATION:-prototypes}" # ciclo global FedSC: transmite pesos + protótipos e agrega submodelos
HERA_GLOBAL_ALIGN_WEIGHT="${HERA_GLOBAL_ALIGN_WEIGHT:-1}" # 0: ablação sem regularização por protótipos
HERA_CLIENT_STORAGE="${HERA_CLIENT_STORAGE:-memory}" # estados locais na RAM da CPU; disk reduz o consumo de RAM
# Protótipos: cache e checkpoints são removidos após salvar resultados/gráficos.
# Preservar checkpoint global e estados dos clientes: HERA_COMMUNICATION=prototypes ALGORITHMS=HERAFL ./start.sh --hera-keep-checkpoints
HERA_PROTOTYPE_KIND="${HERA_PROTOTYPE_KIND:-relational}" # simple ou relational
HERA_PROTOTYPE_POLICY="${HERA_PROTOTYPE_POLICY:-resources}" # resources ou semantic
HERA_ALIGN_WEIGHT="${HERA_ALIGN_WEIGHT:-1}"
HERA_RPCL_WEIGHT="${HERA_RPCL_WEIGHT:-1}"
HERA_CPDR_WEIGHT="${HERA_CPDR_WEIGHT:-1}"
REWARD_VALIDATION_FRACTION="${REWARD_VALIDATION_FRACTION:-0.2}"
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
case "${MODEL,,}" in
    vgg11) MODEL=VGG11 ;;
    vgg16) MODEL=VGG16 ;;
esac
if [[ -z "$LEARNING_RATE" ]]; then
    case "${MODEL,,}" in
        vgg11|vgg16) LEARNING_RATE=0.01 ;;
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

# Falha antes de preparar dados se a configuração semântica for incompatível.
for algorithm in "${algorithms[@]}"; do
    if [[ "$algorithm" == "HERAFL" ]]; then
        case "$MODEL" in
            CNN|DNN|VGG11|VGG16) ;;
            *) echo "HERAFL semântico requer MODEL=CNN, DNN, VGG11 ou VGG16." >&2; exit 1 ;;
        esac
        case "$HERA_COMMUNICATION" in
            parameters|prototypes) ;;
            *) echo "HERA_COMMUNICATION deve ser parameters ou prototypes." >&2; exit 1 ;;
        esac
        case "$HERA_PROTOTYPE_KIND" in
            simple|relational) ;;
            *) echo "HERA_PROTOTYPE_KIND deve ser simple ou relational." >&2; exit 1 ;;
        esac
        case "$HERA_PROTOTYPE_POLICY" in
            resources|semantic) ;;
            *) echo "HERA_PROTOTYPE_POLICY deve ser resources ou semantic." >&2; exit 1 ;;
        esac
        case "$HERA_SEMANTIC_POLICY" in
            linucb|heuristic|legacy) ;;
            *) echo "HERA_SEMANTIC_POLICY deve ser linucb, heuristic ou legacy." >&2; exit 1 ;;
        esac
    fi
done

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
    --resource-preset "$RESOURCE_PRESET" --experiment-id "${GOAL}_${MODEL}"
    --aligned-comparison --detailed-evaluation --pruning-round-timeout "$ROUND_TIMEOUT"
    --reward-validation-fraction "$REWARD_VALIDATION_FRACTION"
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
    elif [[ "$algorithm" == "HERAFL" ]]; then
        algorithm_args+=(--hera-semantic --hera-semantic-policy "$HERA_SEMANTIC_POLICY"
                            --hera-communication "$HERA_COMMUNICATION")
        if [[ "$HERA_COMMUNICATION" == "prototypes" ]]; then
            algorithm_args+=(--hera-client-storage "$HERA_CLIENT_STORAGE"
                                --hera-prototype-kind "$HERA_PROTOTYPE_KIND"
                                --hera-prototype-policy "$HERA_PROTOTYPE_POLICY"
                                --hera-align-weight "$HERA_ALIGN_WEIGHT")
            if [[ "$HERA_PROTOTYPE_KIND" == "relational" ]]; then
                algorithm_args+=(--hera-rpcl-weight "$HERA_RPCL_WEIGHT" --hera-cpdr-weight "$HERA_CPDR_WEIGHT")
            fi
        else
            algorithm_args+=(--hera-global-align-weight "$HERA_GLOBAL_ALIGN_WEIGHT")
        fi
    fi
    "$PYTHON" main.py "${common_args[@]}" "${algorithm_args[@]}" -algo "$algorithm" \
        "$@"
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
