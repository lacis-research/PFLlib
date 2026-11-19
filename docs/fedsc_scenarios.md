# Cenários de dados do FedSC na simulação

A implementação reproduz o cenário executável de `../FedSC`, commit
`a04dc8e630706a0c3f12f716c45857ead95a4940`: seleção de treino com cauda
longa exponencial, seguida por particionamento Dirichlet e balanceamento
final do FedSC. Os dados resultantes são carregados e usados no treino de
HERA-FL, FedLoad e FedAvg. Não requer importar ou executar o projeto vizinho.

O FedSC dessa versão prepara os dados uma vez antes das rodadas. Não possui
drift temporal, detector de drift ou cenário `step` funcional. A integração
mantém esse comportamento. São cenários de heterogeneidade de rótulos.

## Executar

Na raiz do PFLlib, com uma base existente em `dataset/Cifar10/`:

```bash
DATA_PROTOCOL=fedsc DATASET=Cifar10 NUM_CLIENTS=30 \
FEDSC_ALPHA=0.05 FEDSC_IMB_FACTOR=0.1 FEDSC_SEED=66 \
ALGORITHMS="HERAFL FedLoad FedAvg" ./start.sh
```

Para HERA-FL, o launcher portátil também está disponível:

```bash
HERA_ROUNDS=1 HERA_LOCAL_EPOCHS=1 bash examples/run_hera_shift.sh
```

Sem essas duas variáveis, o exemplo executa 200 rodadas e 5 épocas locais.
`DEVICE=cpu` permite execução sem GPU. `start.sh` mantém seus próprios
padrões de rodadas/épocas; `NUM_ROUNDS` e `LOCAL_EPOCHS` os substituem.

| Parâmetro | Padrão | Efeito |
| --- | --- | --- |
| `DATA_PROTOCOL` | `native` | `fedsc` ativa a preparação integrada |
| `FEDSC_ALPHA` | `0.05` | Concentração Dirichlet; menor tende a aumentar heterogeneidade |
| `FEDSC_IMB_FACTOR` | `0.1` | Fator exponencial entre classes, em `(0, 1]` |
| `FEDSC_SEED` | `66` | Seleção das amostras e particionamento |
| `POLICY_SEED` | `42` | Seed do treinamento; independente da preparação |

`NUM_CLIENTS` deve coincidir com a base de origem e ser pelo menos 10.
A entrada precisa ter `config.json`, `train/<cliente>.npz` e
`test/<cliente>.npz`; cada teste local deve conter todas as classes, como
no protocolo HERA anterior. Bases ausentes ou incompatíveis geram erro;
o modo FedSC não regenera nem sobrescreve a base original.

Cada combinação de parâmetros e conteúdo de origem gera um diretório
`dataset/<Base>FedSC_<assinatura>/`. Repetir a configuração reutiliza as
partições verificadas. Os três algoritmos recebem esse mesmo nome de dataset.
`DATA_PROTOCOL=fedsc ... ./start.sh --compare-only` resolve o mesmo dataset,
sem gerar partições nem treinar; use os parâmetros originais da preparação.

## Seleção das amostras

Para `N` exemplos de treino, `C` classes e classe `k` (de 0 a `C-1`), a
quantidade solicitada é:

```text
int((N / C) * imb_factor ** (k / (C - 1)))
```

As listas de índices de cada classe são embaralhadas em ordem de classe
com MT19937 e a seed informada. Selecionam-se os primeiros exemplos de cada
lista, limitados pela disponibilidade, exatamente como `train_long_tail`
do FedSC. Os índices selecionados passam pelo mesmo `clients_indices`.
Não há duplicação ou criação de imagens. O balanceamento final iguala
aproximadamente o total de exemplos por cliente, sem igualar as classes.

`FEDSC_IMB_FACTOR=1` remove a redução exponencial. Em uma base com classes
de tamanhos diferentes, ainda pode cortar as classes acima da média:
esse é o comportamento original do FedSC. Para preservar **todo** o treino
com apenas Dirichlet, continue usando `tools/prepare_hera_shift.py` sem
`--imb-factor`, como documentado em `herafl_class_shift.md`.

A seleção pode reduzir bastante o total de exemplos. Fatores que esvaziem
uma classe são rejeitados. O resorteio Dirichlet tem limite de 10.000
tentativas por bloco; em caso de não convergência, falha com uma mensagem
em vez de ficar indefinidamente no laço original. Os sorteios antes desse
limite são os mesmos do FedSC.

## Fidelidade e diferenças de protocolo

A seleção e os índices de particionamento foram comparados com o código
original do FedSC em fixtures de 10 e 100 classes, 10/17/30 clientes,
fatores 1, 0.1 e 0.01, incluindo classes de tamanhos diferentes.

A entrada é o pool de treino já preparado no PFLlib, na ordem dos clientes.
Pixels, normalização, rótulos e testes locais dessa base são preservados.
O FedSC original usa o treino oficial CIFAR e teste global; portanto,
reproduzir seu particionamento não torna as bases ou os resultados finais
numericamente idênticos aos experimentos do artigo. MNIST também pode usar
a preparação como extensão da simulação; a referência original usa CIFAR.

O modelo, poda, política, recompensa e agregação continuam sendo os de cada
algoritmo desta simulação. A integração é dos cenários de dados, não do
algoritmo de colaboração por protótipos do FedSC.

## Evidências gravadas e testes

O `config.json` do dataset inclui parâmetros, hashes dos arquivos de origem,
contagens por classe globais e locais, total selecionado/excluído e
`temporal_drift=false`. `indices.json` registra os índices usados por cliente,
os excluídos e os intervalos correspondentes aos arquivos originais.
A configuração do dataset é copiada para os resultados dos três algoritmos
no modo de comparação. HERA-FL mantém as métricas de classes vistas/ausentes.

```bash
conda activate pfllib
python -m unittest discover -s tests -p 'test_fedsc_scenarios.py' -v
python -m unittest discover -s tests -p 'test_hera_shift.py' -v
```

Os testes incluem equivalência dos índices ao FedSC original, conservação
dos exemplos selecionados, exclusões rastreáveis, testes locais inalterados,
reutilização de cenários e treino real dos três algoritmos em dados sintéticos
redistribuídos, com mudança dos pesos e gravação da configuração.
