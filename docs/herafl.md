# HeraFL com ciclo global e perdas do código oficial do FedSC

## Configuração padrão

O `start.sh` usa `HERA_COMMUNICATION=prototypes`, `HERA_PROTOTYPE_KIND=relational`
e pesos RPCL/CPDR iguais a 1. O nome `prototypes` identifica a colaboração
semântica; **esse modo transmite pesos e protótipos**. Cada participante parte
do modelo global da rodada, e o servidor agrega os pesos recebidos.

Referência executável: `../FedSC`, commit
`a04dc8e630706a0c3f12f716c45857ead95a4940`. Por decisão explícita do usuário,
o código oficial prevalece nas divergências em relação ao
[PDF do FedSC](https://arxiv.org/abs/2506.21012).
A poda estruturada, a seleção de retenções por LinUCB, a validação para recompensa
e os perfis de recursos continuam sendo extensões do HeraFL. Não é uma reprodução
dos experimentos originais do FedSC (arquitetura, particionamento e orçamento).

## Ciclo de uma rodada

1. Selecionar participantes e retenções. Congelar os pesos globais e as referências
   semânticas da rodada anterior; nenhum cliente usa atualizações de outro cliente
   da mesma rodada.
2. Enviar o estado global completo e os protótipos ao participante. Reinicializar
   pesos e otimizador e podar a partir desse estado global, preservando a dimensão
   da representação. O cache local não é o ponto de partida da próxima participação.
3. Treinar com `CE + λ_RPCL RPCL + λ_CPDR CPDR`, usando somente a partição de treino.
   A primeira rodada usa CE porque ainda não existem protótipos.
4. Acumular características dos forwards da última época local, antes das
   atualizações correspondentes, e calcular suas médias por classe, como
   `Local.local_train`/`average_kns`. Retornar
   **pesos do submodelo, protótipos e telemetria**. A validação separada continua
   alimentando a recompensa; o teste não seleciona retenções.
5. Agregar apenas respostas aceitas, ponderadas pelo número de exemplos de treino.
   A agregação considera as coordenadas ativas da poda; coordenadas sem cobertura
   preservam o valor global anterior. Com retenção 1, isso é a média ponderada
   usual do FedSC/FedAvg. Se nenhuma resposta for aceita, o estado global é preservado.
6. Construir as referências conforme o comportamento executado de
   `Global.get_re_p` e `Global.get_co_p` do checkout oficial (detalhes abaixo).
7. Avaliar o mesmo modelo global agregado nas partições dos clientes.

## Perdas e distinção entre PDF e código dos autores

| Componente | Código oficial adotado | Diferença para o PDF |
|---|---|---|
| Similaridade RPCL | Normalização L2 e temperatura 0.05; **sem U explícito** | A equação 13 contém U |
| Positivo RPCL | Média dos protótipos relacionais da classe de cada amostra | A equação 14 soma exponenciais de vários positivos |
| Negativos RPCL | Lista compartilhada construída sobre todas as amostras do minibatch, com repetições | O texto descreve negativos por classe da amostra |
| CPDR | `mse_loss(features, targets)`, média sobre amostras e dimensões, sem normalizar características | A equação 15 impressa tem soma de normas por coordenada |
| Protótipos locais | Características dos forwards da última época | Coleta após treino no pseudocódigo |

**Não há varredura adicional do dataset por minibatch, nem backward através de U.**
O código anterior que introduzia esse trabalho foi removido. Os logs mostram
CE, CPDR, RPCL, perda total, tempo do forward das perdas semânticas e tempo de
backward + atualização. Essas informações são salvas em `clients[].loss_diagnostics`.

Há dois comportamentos relevantes do banco oficial que também são reproduzidos:

- `get_re_p` altera um tensor temporário, mas retorna as listas originais. Assim,
  não aplicamos a média angular da equação 9 nesse modo. `hera_semantic_neighbors`
  não controla esse banco oficial.
- `get_co_p` aplica pesos pela **posição em cada lista de classe**, e altera as
  próprias relações antes de retorná-las para RPCL. Os protótipos consistentes
  são a soma dessas relações ponderadas. Com classes ausentes, essa associação
  posicional pode diferir da associação por ID de cliente; ela é mantida para
  reproduzir o código, não apresentada como uma correção do algoritmo.

Alvos sem classe disponível usam a própria característica destacada, como no
código oficial. Situações em que os autores dividem por zero (todos os clientes
uniformes) ou empilham uma lista vazia de negativos geram erro explícito, sem
inventar outro esquema de pesos ou perda. A primeira rodada sem referências e
as ablações que desativam RPCL não exigem negativos.

O modo `simple` permanece uma ablação, e `parameters` mantém o alinhamento
normalizado anterior. A construção do banco e as perdas relacionais foram
comparadas numericamente, incluindo gradientes, com o código local dos autores
em CUDA (`tests/test_fedsc_official.py`). O checkout externo não é necessário
para treinar; é necessário somente para os testes de paridade.

Mantêm-se configuração experimental e mecanismos do HeraFL: CNN/VGG/DNN,
poda/LinUCB, partições existentes, validação para recompensa, política de timeout,
armazenamento, coleta adicional de H/U/D/Q e avaliação por cliente. A CNN atual
não passa a usar o ResNet10 nem as transformações de aumento de dados do script
original. Portanto, a paridade aqui é do caminho semântico descrito, não uma
reprodução integral dos experimentos publicados.

## Resultados, comunicação e retomada

O HDF5 registra `evaluation_type=global_model`, `rs_test_acc`, `rs_train_loss`,
`rs_test_auc` e os desvios entre clientes. A acurácia global soma acertos e amostras;
a AUC é a micro-AUC por cliente ponderada pelo tamanho do teste. A loss global
é CE avaliada no **treino**, não a perda composta de otimização.

Cada execução exporta `global_model.pt` e os gráficos:

- `global_accuracy_per_round_comparison.png`;
- `global_train_loss_per_round_comparison.png`;
- `global_test_auc_per_round_comparison.png`.

Os gráficos locais continuam usando os submodelos após treino e antes da agregação.
A comparação distingue essas métricas das globais. Resultados locais antigos não
são convertidos em resultados globais: é necessário treinar novamente.

A comunicação registra tamanhos serializados reais, incluindo JSON/base64:
parâmetros baixados/enviados, protótipos e controle. Respostas rejeitadas também
incorrem em custos. A avaliação contabiliza envio do modelo global aos clientes
e retorno de telemetria. Rede e energia continuam sendo estimativas dos perfis.

O protocolo é `herafl-global-prototypes-v3`. Checkpoints incluem pesos globais,
clientes, banco, política, métricas e RNG; retomadas mantêm a sequência de rodadas.
Checkpoints v1 e v2 são rejeitados, pois usam outros protocolos/perdas. Os
resultados v3 registram `loss_definition=fedsc-official` e o commit de referência.
`--hera-local-only` continua sendo a ablação de clientes independentes, sem modelo
global nem comunicação de aprendizado após a inicialização.

```bash
# HeraFL com ciclo global e perdas do código oficial do FedSC (padrão)
./start.sh

# Apenas HeraFL; gráficos globais também são gerados na própria execução
ALGORITHMS=HERAFL ./start.sh

# Guardar checkpoint para retomada
ALGORITHMS=HERAFL ./start.sh --hera-keep-checkpoints

# Ablação sem RPCL, mantendo CPDR e agregação global
HERA_RPCL_WEIGHT=0 ./start.sh

# Modo global anterior, com alinhamento normalizado
HERA_COMMUNICATION=parameters ./start.sh
```

Ao chamar `system/main.py` diretamente, use `--hera-communication prototypes
--hera-prototype-kind relational --reward-validation-fraction 0.2` além dos
argumentos do experimento. RPCL/CPDR têm padrão 1 nesse modo; pesos zero continuam
aceitos para ablações. O armazenamento em RAM é padrão; `--hera-client-storage disk`
descarrega clientes inativos para disco. Cache e checkpoints são removidos após
exportação bem-sucedida, exceto checkpoints solicitados explicitamente.

## Verificação de integração e tempo (28/10/2026)

Com o padrão do `start.sh` (CIFAR-10, CNN, 10 clientes, participação 0.6,
5 épocas locais, batch 64, LR 0.001, poda/LinUCB, RPCL=CPDR=1), a execução curta
`ALGORITHMS=HERAFL NUM_ROUNDS=2 GOAL=diagnostic-fedsc-official-v3 DEVICE=cuda ./start.sh`
terminou e exportou HDF5, política, modelo e três gráficos globais.

- Rodada 1: 9.81 s; rodada 2: **12.44 s**.
- Na rodada 2, soma do forward das perdas semânticas: 1.47 s; soma do backward
  com atualização: 1.89 s. Esses tempos são componentes, não o total da rodada.
- O log anterior fornecido pelo usuário registrava 462.99 s na rodada 2 com U.
  A comparação é entre duas execuções/versões, não um benchmark estatístico nem
  uma medição isolada de cada mudança. Nenhuma conclusão sobre convergência é feita.

Log: `tmp/hera-official-two-rounds.log`. Os três testes de paridade em CUDA
(`python -m unittest discover -s tests -p test_fedsc_official.py`) passaram.
Os resultados identificam protocolo v3, perdas `fedsc-official` e o commit local.
