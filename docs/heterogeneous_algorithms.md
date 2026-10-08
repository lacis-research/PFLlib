# FedLoad e HERAFL no PFLlib

A implementação adapta os métodos presentes em `mininetfed_iwcmc/mininetfed`
ao fluxo nativo do PFLlib. Os servidores herdam de `Server`; os clientes herdam
de `Client`. Treino, leitura das partições, seleção por `join_ratio`, avaliação,
checkpoints e arquivos HDF5 usam a infraestrutura do PFLlib. Não há dependência
de Mininet, MQTT, Docker, TensorFlow ou do diretório `mininetfed_iwcmc` em execução.

## Executar

Com o ambiente `pfllib` ativo e as partições do dataset já geradas, a partir de `PFLlib/system`:

```bash
python main.py -data MNIST -m CNN -algo FedLoad -gr 10 -did 0
python main.py -data MNIST -m CNN -algo HERAFL -gr 10 -did 0
```

Use `-dev cpu` para executar sem GPU. Os nomes `fedload`, `herafl`, `HeraFL`,
`hera-fl` e `hera_fl` também são aceitos e normalizados para os nomes acima.
A quantidade de clientes (`-nc`, padrão 20) precisa corresponder às partições disponíveis.

Exemplo com perfis de recursos diferentes para os clientes 0 e 1:

```bash
python main.py -data MNIST -m CNN -algo HERAFL -gr 10 -nc 2 \
  --client-profiles ../examples/heterogeneous_profiles.json
```

Também é possível usar esse arquivo com mais clientes: os IDs omitidos recebem
os valores padrão. IDs inexistentes e campos desconhecidos geram erro.

## Modelos e políticas

Há suporte à CNN `FedAvgCNN`, à DNN e à MLR originais do PFLlib. MNIST e Cifar10
usam as dimensões estabelecidas em `main.py`. Outros modelos, como ResNet e
MobileNet, geram uma mensagem explícita: conexões residuais e normalização
exigem um mapeamento de poda próprio. A MLR não tem camadas ocultas; portanto,
os braços de retenção produzem o mesmo modelo completo nesse caso.

A poda seleciona canais/neurônios ocultos pelo L1 dos pesos, mantém todas as
classes da saída e constrói um submodelo fisicamente menor. Os índices da
transição convolução → camada linear respeitam o flatten NCHW do PyTorch.
A agregação pondera cada posição pela quantidade de amostras dos clientes que
a receberam. Posições não cobertas conservam o peso global anterior.

As políticas LinUCB foram copiadas de `server/bandits/linucb_fedload.py` e
`server/bandits/linucb_hera.py` do projeto de origem; o preditor online veio de
`server/straggler.py`. A recompensa e o timeout adaptativo seguem os respectivos
controllers. O FedLoad usa o contexto de recursos; o HERAFL acrescenta risco
de atraso, previsão de tempo e orientação pela capacidade estática e dinâmica.

O treino usa SGD e a perda do `Client` original do PFLlib, com `-lr`, `-ls`,
`-lbs` e decay. O otimizador é reconstruído a cada submodelo, preservando o
cronograma de learning rate. O treino não reutiliza os modelos/otimizadores do
runtime TensorFlow. A avaliação global utiliza todos os clientes com cópias do
modelo completo. A acurácia local no teste alimenta a recompensa, portanto este
protocolo deve ser considerado ao comparar resultados com métodos que reservam
um conjunto de validação separado.

Assim como o FedAvg desta versão do PFLlib, o loop percorre `0..global_rounds`,
com avaliação antes do treino. Por exemplo, `-gr 1` faz duas iterações. O JSON
registra todas as iterações; o HDF5 registra as avaliações nos intervalos `-eg`.

## Recursos e parâmetros

Os clientes executam sequencialmente no processo do PFLlib. O tempo de treino
é medido (com sincronização CUDA quando aplicável). O tempo de comunicação é
estimado pelo tamanho dos parâmetros do submodelo e pela banda do perfil:
`tempo = bytes × 8 / (Mbps × 1e6)`. Download e upload têm o mesmo tamanho.
A energia é estimada por `potência × tempo`, separadamente para treino,
download e upload, seguindo o modelo de recursos usado na origem.
Esses bytes não incluem serialização, protocolo ou envio de índices de poda.
Não são medições de rede, RAPL ou energia da GPU.

Os perfis orientam a política e as estimativas; `cpu_capacity` não limita
fisicamente a CPU/GPU. `train_slow_rate` conserva a simulação de atraso e
redução de épocas locais do PFLlib. O tempo total do processo inclui avaliação
e operações do servidor; o `round_time` de cada cliente contém treino medido e
comunicação estimada, e não representa paralelismo real entre máquinas.

| Opção | Padrão | Significado |
| --- | --- | --- |
| `--pruning-arms` | `1.0 0.8 0.6 0.4` | Frações retidas em cada camada oculta; 1 conserva tudo |
| `--bandit-alpha` | `0.25` | Exploração LinUCB |
| `--bandit-beta` | `0.6` | Referência da exploração proporcional |
| `--policy-seed` | `42` | Seed do bandit, somada ao índice da repetição |
| `--reward-weights` | `0.6 0.2 0.2` | Pesos de acurácia, energia e tempo; soma igual a 1 |
| `--client-profiles` | ausente | Arquivo JSON com perfis por ID |
| `--pruning-round-timeout` | `300` | Prazo estimado em segundos; HERAFL pode aumentá-lo |

O timeout filtra atualizações depois do treino local; ele não interrompe o
processo de treino. `client_drop_rate` também filtra respostas. Se nenhum cliente
responder, o servidor conserva o modelo e não atualiza o bandit. `time_select`
aplica ainda `time_threthold` sobre o tempo da rodada do cliente.

Campos aceitos em cada perfil:

| Campo | Padrão | Unidade/uso |
| --- | --- | --- |
| `f_i` ou `cpu_capacity` | `0.5` | Capacidade normalizada em [0,1]; `f_i` tem precedência |
| `c_r_i` ou `compute_cost` | `0.5` | Custo inicial normalizado; `c_r_i` tem precedência |
| `throughput_mbps` | `10` | Banda positiva; contexto normaliza no intervalo 1–75 Mbps |
| `energy_efficiency` | `0.5` | Eficiência normalizada em [0,1] |
| `e_cmp`, `e_up`, `e_down` | `0.5` | Contexto energético normalizado em [0,1] |
| `e_cmp_w`, `e_up_w`, `e_down_w` | correspondente `e_*` | Potências não negativas em watts para as estimativas |
| `compute_time_scale_per_sample` | `0.002` | Segundos/amostra para normalizar custo de treino observado |
| `round_time_scale` | `60` | Segundos para normalização do tempo |
| `energy_scale` | `10` | Joules para normalização da energia |

Depois de cada resposta, `c_r_i` usa o tempo medido por amostra e época; o contexto
estático do HERAFL conserva os valores iniciais. `dataset_size` é a quantidade
de amostras do cliente dividida pela maior partição entre os clientes configurados.
Valores normalizados são limitados a [0,1]. Perfis omitidos começam homogêneos;
partições e tempos observados ainda podem diferir.

## Resultados e validação

Além do HDF5 e do checkpoint originais, os algoritmos salvam
`results/<dataset>_<algoritmo>_<goal>_<repetição>_policy.json`, com retenção,
contexto da escolha, informações LinUCB, status da resposta, recompensa e
custos por cliente e rodada. Energia/rede são identificadas como `profile_model`.
Novos clientes para fine tuning e avaliação DLG ainda não são suportados por
estes dois métodos e geram erro antes do treino.

Execute os testes a partir de `PFLlib`:

```bash
python -m unittest discover -s tests -v
```

Os testes verificam preservação do modelo com retenção 1, redução estrutural,
flatten de MNIST/Cifar10, agregação com cobertura parcial, política, preditor,
treino nativo dos dois servidores e preservação do modelo quando não há respostas.

## Gráficos comparativos

O comparador produz os mesmos seis tipos de gráficos de
`mininetfed_iwcmc/results/comparison`, com os mesmos nomes de arquivo,
cores dos métodos e unidades:

| Arquivo (PNG/PDF) | Métrica |
| --- | --- |
| `mean_accuracy_per_round_comparison` | Média aritmética da acurácia local dos clientes com respostas aceitas, após treino, em [0,1] |
| `global_accuracy_per_round_comparison` | Acurácia do modelo global nas partições de teste do PFLlib, ponderada pelo número de amostras, em [0,1] |
| `client_round_compute_capacity_pruning_bubble_comparison` | Cliente × rodada; cor indica retenção e formato indica capacidade baixa (<0.33), média (<0.66) ou alta |
| `total_energy_per_experiment_comparison` | Soma da energia de todos os registros de clientes e rodadas, em joules |
| `total_round_time_per_experiment_comparison` | Soma dos tempos de todos os clientes e rodadas, em segundos |
| `total_communication_per_experiment_comparison` | Soma dos bytes de download e upload |

O total de tempo soma os tempos dos clientes; não representa o tempo de parede
do experimento ou o máximo por rodada. Nos gráficos de totais, as anotações
percentuais usam HERAFL como referência, como no projeto de origem. O PFLlib
mantém seu protocolo de avaliação global antes do treino; a acurácia local é
avaliada depois do treino. Não é o conjunto de teste centralizado do MininetFed.

### Registrar os três métodos

HERAFL e FedLoad já registram os JSONs de recursos automaticamente.
FedAvg precisa da opção `--record-resources`. Ela mantém o treino, o otimizador
e a agregação do FedAvg original e acrescenta medições de tempo e avaliações
locais. As estimativas de energia e comunicação usam o mesmo modelo de perfis
aplicado aos outros dois métodos. A avaliação adicional não consome o estado
aleatório usado pelo treino seguinte.

Com `pfllib` ativo, a partir de `PFLlib/system`, execute os métodos com a mesma
configuração e um goal próprio para preservar os arquivos anteriores:

```bash
for algo in FedAvg FedLoad HERAFL; do
  python main.py -data MNIST -m CNN -algo "$algo" -gr 10 -did 0 \
    -go comparison --record-resources \
    --client-profiles ../examples/heterogeneous_profiles.json || break
done
```

Os perfis omitidos recebem os padrões. Os valores normalizados da capacidade
não restringem fisicamente a CPU/GPU. Energia e comunicação permanecem
estimativas de perfis, sem emulação Mininet. `--record-resources` registra
FedAvg; outros baselines ainda precisam de um adaptador de registro próprio.

### Gerar os gráficos

A partir de `PFLlib`:

```bash
python tools/plot_comparison.py --dataset MNIST --goal comparison \
  --algorithms HERAFL FedLoad FedAvg
```

Os gráficos PNG/PDF ficam em `results/comparison/MNIST_comparison/plots/`.
Para os resultados antigos, use `--goal test`: o script gera as mesmas vistas,
mas métodos sem JSON aparecem somente na acurácia global. Os recursos e a média
local ausentes não são reconstruídos ou preenchidos com zero. Para incluir
FedAvg antigo em todas as vistas, é necessário executá-lo novamente com registro.

Também é possível comparar qualquer método com resultados HDF5 do PFLlib:

```bash
python tools/plot_comparison.py --dataset Cifar10 --goal comparison \
  --algorithms HERAFL FedLoad FedAvg FedProx --runs 0 1 2
```

Por padrão, todas as repetições disponíveis para cada algoritmo são lidas;
`--runs` exige os mesmos índices em todos os algoritmos. As linhas mostram a
média, com faixa de ± um desvio padrão populacional quando há várias repetições.
Para rodadas ausentes, são usadas apenas as repetições disponíveis, sem interpolar.
Os totais são somados dentro de cada repetição e depois apresentados como média
entre repetições, com desvio padrão. A vista de capacidade mantém um painel por
método e repetição para não misturar escolhas de retenção entre execuções.
Se apenas parte das repetições possui JSON, as métricas de recursos usam essa
parte; o script informa a cobertura no terminal.

Se os experimentos usaram `-eg 5`, informe `--eval-gap 5`: o HDF5 original não
salva os índices das rodadas. Todos os experimentos comparados devem ter o mesmo
intervalo de avaliação. O JSON já contém os índices reais das rodadas.
Use a mesma configuração de dataset, partições, modelo, número de clientes,
rodadas, treino local, participação e perfis para uma comparação controlada.
O script não consegue verificar esses parâmetros nos arquivos HDF5 atuais.

Opções adicionais: `--results-dir`, `--output-dir`, `--formats png pdf svg` e
`--extra-metrics` (acrescenta os gráficos anteriores de perda/AUC e recursos por
rodada no diretório pai de `plots/`). Cada execução substitui os gráficos de
mesmo nome no diretório de saída; um `--output-dir` diferente preserva
comparações anteriores.
