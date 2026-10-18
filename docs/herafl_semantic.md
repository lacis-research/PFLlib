# Seleção semântica de retenção no HERAFL

## Escopo entregue

O experimento HERAFL agora é **FedLoad + abordagem semântica**. A solicitação
posterior substitui a exigência original do prompt de manter os mecanismos extras
do antigo HERAFL. A base usa `linucb_fedload`, seleção uniforme, os mesmos perfis
e timeout do FedLoad. Foram removidos do caminho experimental bônus de capacidade,
pisos de retenção, seleção estratificada e predição de stragglers.

`--hera-semantic` ativa a extensão sobre essa base, mantendo a poda física e a
agregação por coordenadas.
Acrescenta contexto semântico ao LinUCB e uma heurística comparável. O treino
continua CE. RPCL/CPDR **não foram implementadas**: pesos diferentes de zero
falham antecipadamente; temperatura e vizinhos são opções reservadas validadas.
Não há alegação de ganho de generalização, eficiência ou garantia de convergência.
Não houve alteração em `../FedSC/` nem execução da matriz científica.

Sem a opção, HERAFL é um controle equivalente ao FedLoad, inclusive sua recompensa baseada
no teste por padrão. `--reward-validation-fraction .2` é uma intervenção explícita,
aplicada também a FedLoad e FedAvg com `--aligned-comparison`. Ela separa uma
permutação determinística do treino nativo (seed de política + ID), sem consultar
o teste; validação não participa de treino, histogramas, protótipos nem peso de
agregação. Os clientes precisam ter ao menos dois exemplos. Não ajustar parâmetros
no teste; resultados com recompensa legada de teste não são avaliação independente.


O contexto de atualização segue FedLoad (recursos após resposta) quando a extensão
está desligada ou no controle preservado sem coleta. Quando a coleta semântica
está ativa, o contexto é congelado na decisão, para não usar indicadores
pós-treino na atualização da ação anterior. `--hera-semantic-policy legacy` significa
política FedLoad com representação preservada; não reativa o HERAFL antigo.
`--straggler-predictor`, `--hera-retention-floors` e `--hera-capacity-thresholds`
foram removidos do parser e do experimento. Os módulos históricos permanecem
somente como referência/testes e não são importados pelo novo HERAFL.

## Diagnóstico e referência

Referência: [Wang et al., FedSC, KDD 2025, §§3.3–4.2 e Algoritmo 1](https://arxiv.org/html/2506.21012v1),
[DOI](https://doi.org/10.1145/3711896.3736957).

| Mecanismo | Referência | Código local FedSC | Escolha desta entrega |
|---|---|---|---|
| Protótipo de classe | Eq. 3 | Features da última época durante atualização | Média em uma passagem final eval/no_grad, contagem única |
| Relações | Eqs. 7–9: diferenças de cossenos ao centro, M vizinhos e próprio cliente | `get_re_p` calcula distâncias à média replicada; modifica `temp` mas devolve listas originais | Não implementado; banco de médias ponderadas por suporte de classe |
| Consistência | Eqs. 10–12: discrepância à uniforme e sigmoid | `get_co_p` usa posição da lista como ID, modifica `re_p`, pode dividir por discrepância total zero | Não implementado; nenhuma cópia dessa ponderação na agregação de parâmetros |
| RPCL | Eqs. 13–14: normalização U por distância média e soma de positivos | Temperatura fixa, positivos médios, negativos misturados entre amostras | Não implementado; uma futura variante com temperatura fixa e InfoNCE multipositivo deverá ser identificada como variante |
| CPDR | Eq. 15: soma por dimensão de normas | MSE de features e alvos | Não implementado; MSE não é reprodução literal da Eq. 15 |
| Inicialização/faltantes | Classes efetivamente disponíveis | `torch.empty`, fallback com `except` genérico no treino | Banco vazio explícito, máscaras e suporte, nenhuma feature fictícia |
| Política de retenção | Não é contribuição do artigo | FedSC não seleciona submodelos | Extensão deste projeto; nenhuma transferência automática do teorema |

O ciclo executável FedLoad/HERAFL tem exatamente `global_rounds` rodadas, numeradas
1..R, e avaliação **após** agregação em múltiplos de `eval_gap`. A avaliação
compartilha o modelo global entre clientes; o banco mantém tensores independentes.

## Espaço de representação e redução física

Suporte: **FedAvgCNN, DNN, VGG11 e VGG16**, conferido antes de criar clientes.
MininetCNN genérico e MLR continuam no legado e são rejeitados no modo semântico. `forward_features`
e `forward_with_features` executam o backbone uma vez; `forward` conserva a saída
anterior (o DNN legado retorna log-softmax, preservado por compatibilidade).

`prune_model(..., preserve_representation=True)` conserva toda a saída da última
camada oculta e todas as classes, na ordem global. CNN poda convoluções e entradas
da camada oculta. **DNN tem somente uma camada oculta**: preservá-la sem outra
mudança impediria qualquer redução física. Portanto seleciona entradas originais
por soma L1 das colunas de `fc1`, desempata por índice e faz `index_select` antes
da camada. Isso reduz colunas físicas, preservando os IDs de todas as features.
É seleção L1 de entradas, não relevância semântica de pixels. Comparar sempre com
o controle de representação preservada para isolar esse efeito arquitetural.

Nas VGG11/VGG16 locais (imagens RGB 32×32), as cinco operações de pooling
produzem 512×1×1 antes das camadas densas. A poda mapeia cada canal global dessa
saída à coluna correspondente da primeira camada densa. A primeira camada densa
é reduzida; a segunda mantém todas as 4096 saídas na ordem global, podando apenas
suas entradas. A representação é a saída da segunda ReLU/dropout; em avaliação,
dropout e augmentation estão desativados. O classificador mantém todas as classes
e suas 4096 entradas. O caminho com features aplica augmentation uma única vez.
`fc` é uma propriedade de acesso à última camada, sem duplicar parâmetros nem
alterar chaves `layers.*` de checkpoints legados. Adam permanece o otimizador.
Isso não habilita por suposição MininetCNN com representação baseada em flatten.

Retenção 1 é numericamente equivalente ao modelo completo; a fração real de
parâmetros varia por arquitetura. Logs incluem fração real e tamanho do
classificador preservado. Coordenadas comuns significam identidade de neurônios,
não igualdade temporal das representações produzidas por modelos atualizados.

## Indicadores e contrato temporal

Schema `fedload-semantic-v2` (incompatível com checkpoints semânticos da base antiga). Payload inclui ID, rodada, retenção, dimensão, schema,
contagens, médias somente de classes presentes, indicadores/máscaras, cobertura e
rodada da referência. Não envia exemplos nem embeddings individuais.

Para C classes e n exemplos de treino efetivo:

* H = −Σ p_c log(p_c) / log(C), com denominador 1 para C=1.
* U = média da entropia softmax / log(C), mesma convenção; extraída em ordem
  determinística, eval/no_grad, depois do treino, sem augmentation.
* D = média ponderada por contagens de (1−cos(local_c, banco_c))/2 nas classes
  compartilhadas com ambas as normas >1e−12. Sem pares válidos, máscara falsa.
  Cobertura = número de exemplos dessas classes / n.
* Q = 1−Σ_c Σ_i ||z_i−m_c||² / Σ_i ||z_i||², limitado a [0,1]. Usa somas,
  contagens e soma de quadrados em O(Cd); energia ≤1e−12 torna Q indisponível.
  É compactação intraclasse relativa, não qualidade universal: classes com um
  exemplo parecem compactas, por isso o payload sempre inclui suporte.

Todos usam escalas fixas. H não define dificuldade. Q não repete D: é local e
não compara com banco. Ausência codifica valor 0 **e máscara 0**, distinguindo
zero observado. Contexto inclui idade normalizada, retenção de origem, cobertura,
rodadas da extração/referência; esses IDs de rodada são logs, não features
numéricas do bandit. Ablações zeram o indicador excluído e sua máscara, sem mudar
o schema. Indicadores expiram quando t−rodada > max_age; D também expira pela
idade da referência comparada. `max_age=0` força fallback.

A rodada t usa apenas histórico aceito até t−1. Banco começa vazio, aquecimento
padrão de uma rodada usa política original e CE. Todos recebem cópias do mesmo
snapshot anterior. Custos completos são calculados antes do timeout. Rejeitados
não atualizam parâmetros, banco, histórico ou bandit, mas seus custos ficam nos
logs. Não há preditor de stragglers nesta variante.

Depois da agregação por coordenadas, o próximo banco usa Σ n_kc m_kc / Σ n_kc,
com IDs de contribuidores por classe. Se houve respostas, classes não vistas
ficam ausentes; se não houve nenhuma, preservam-se modelo e banco e aumenta sua
idade. Histórico por cliente é separado. O bandit usa exatamente o contexto
congelado da decisão, nunca a extração recém-obtida para atualizar esse contexto.

## Políticas, restrições e recompensa

O LinUCB continua com matrizes por braço e a mesma exploração escalada pela média
de retenções. No modo semântico adiciona H/U/D/Q, quatro máscaras, idade, retenção
de origem, cobertura e disponibilidade. A contribuição semântica é o produto das
novas coordenadas com os coeficientes **específicos do braço**; coeficientes de
recursos, exploração e contribuição semântica são registrados separadamente. Inicialmente
os coeficientes são zero: o método precisa aprender com braços efetivamente
observados. O teste controlado verifica que coeficientes aprendidos distintos
permitem mudar a escolha quando apenas Z muda. Isso não demonstra ganho científico.

A heurística usa o mesmo conjunto de braços. Seja T a média das retenções disponíveis,
a a retenção, w=0.6, v=0.1, f_a a fração física e
`demand = U*Q + D*(1-H)`. Seu alvo é
`Tz = clip(T + w*(demand−0.5), min_arm, max_arm)` e a pontuação é
`FedLoad_exploit(a,R) + FedLoad_explore(a,R,Z) − w*abs(a−Tz) − v*f_a`.
Não há bônus de capacidade ou piso de retenção.
Pesos: `--hera-heuristic-weight`, `--hera-heuristic-cost-weight`. Sinais inválidos
são mascarados; não há regra universal “entropia maior exige modelo maior”.
Heurística e LinUCB resolvem empate por menor braço. O fallback preserva o sorteio
reproduzível do FedLoad quando seu braço é viável; há uma instância de
controle original atualizada com os mesmos braços/recompensas aceitos.

`--hera-semantic-memory-limit BYTES` filtra pela estimativa 3×bytes de parâmetros
para SGD (parâmetros, gradientes, cópia de trabalho) e 5× para Adam nas VGGs
(inclui os dois estados de momentos). **Não é memória de pico**: não inclui
ativações, allocator ou buffers do framework. `--hera-semantic-time-limit SECONDS`
filtra tempo previsto: último compute_time × razão das frações físicas, mais
comunicação prevista; sem histórico usa escala do perfil × amostras × (épocas+1), ou apenas épocas no controle sem coleta.
A previsão reserva payload de todas as classes mais metadados. Não é garantia de
prazo; o timeout real estimado/medido continua ativo. Os limites são opcionais e
não mudam o legado desativado. Comparar políticas semânticas com limites idênticos;
comparações com legado devem registrar que ele não aplica esses filtros extras.

Todos os braços configurados são candidatos, sem pisos por capacidade. Os limites
opcionais de recursos filtram somente a viabilidade física prevista do submodelo
semântico. Sem braço viável, status `infeasible`, nenhuma instalação ou treino;
participantes originalmente selecionados continuam registrados. Logs incluem
candidatos do FedLoad, viáveis, custos previstos, pontuações e fallback. Não se
treinam candidatos para obter recompensas contrafactuais.

A recompensa controle é a original: combinação configurável Uacc/UE/UT, com
referências históricas de energia e tempo da rodada. Recomenda-se validação
reservada para Uacc em **todas** as variantes. Ablação independente
`--hera-semantic-reward-weight w` usa
`(reward_control + w*(Q_valid − clip(round_time / time_scale,0,1)))/(1+w)`.
Q inválido contribui zero com máscara registrada; `time_scale` padrão 60 s é
configurável. Somente o braço observado recebe essa recompensa. Não é uma perda
diferenciável nem uma observação dos demais braços.

## Configuração, custos e resultados

| Opção | Padrão |
|---|---|
| `--hera-semantic` | false |
| `--hera-semantic-policy` | linucb; heuristic/legacy |
| `--hera-semantic-features` | H,U,D,Q |
| `--hera-semantic-max-age` | 5 |
| `--hera-semantic-warmup-rounds` | 1 |
| `--hera-semantic-reward-weight` | 0 |
| `--hera-rpcl-weight`, `--hera-cpdr-weight` | 0; valores positivos rejeitados |
| `--hera-semantic-temperature` | .05; reservado |
| `--hera-semantic-neighbors` | 1; reservado |
| `--hera-semantic-shuffle` | false |
| `--reward-validation-fraction` | 0 (controle legado com teste) |
| `--experiment-id` | default |
| `--detailed-evaluation` | false |

Pesos são finitos/não negativos; temperatura/escalas/limites positivos finitos;
idades/vizinhos/aquecimento inteiros não negativos. `legacy` + recompensa zero
preserva representação, mas evita extração e comunicação sem consumidor.
`--hera-semantic-shuffle` permuta conjuntamente valores, máscaras e metadados
entre participantes da mesma rodada usando seed+rodada, sem informação futura.

Contagem lógica determinística: 8 bytes de comprimento do cabeçalho, JSON UTF-8
canônico dos metadados (chaves ordenadas, sem espaços), depois, para cada classe,
ID int64 (8 bytes) e vetor float32 (4d bytes). Contagens e IDs de contribuidores
estão no JSON. É tamanho do protocolo lógico, não medição de sockets/pickle/TLS.
Download é contado por cliente, mesmo com snapshot compartilhado no servidor.
O DNN também contabiliza seu mapa de entradas int64 no download, separado como
`coordinate_download_bytes`; o servidor retém o mapa para interpretar o upload.
Esta entrega envia uma média por classe; se forem adicionados protótipos
relacionais, o protocolo deve contar cada vetor e seu ID, não apenas C×d.

`training_time`, `semantic_extraction_time`, bytes de parâmetros e semânticos
nas duas direções ficam separados. Tempo de comunicação = bytes×8/(Mbps×1e6),
energia = potência do perfil×tempo; extração soma ao compute_time/energia local.
`semantic_aggregation_time` é custo do servidor, sem atribuí-lo à energia local.
Tempo medido varia entre execuções; determinismo de decisões refere-se a contextos
controlados, não a idêntico wall-clock do SO.

Resultados têm ID/configuração, schema, ordem das features, recompensa decomposta,
histórico, suporte, cobertura, retenção e fração física, custos, idade do banco e
fallback. Diretórios distinguem política/features/experimento e timestamp. Leitura
de resultados antigos permanece suportada. Checkpoints de modelo legado continuam;
não há retomada de treinamento do servidor. `SemanticLinUCB.state_dict` valida
schema/features/braços/dimensões na carga; não representa um checkpoint completo
de execução (que precisaria também RNGs, banco, histórico e perfis).

`--detailed-evaluation` grava matriz resumida via suporte/recall, macro-F1 e
balanced accuracy nas classes **com suporte**, acurácia ponderada por exemplos,
média/std por cliente e agregados por capacidade. Ausência no teste é `null`.
Faixas analíticas são <.4, [.4,.7), ≥.7 de (CPU estática+1−custo estático)/2.
A acurácia local de recompensa é de validação quando solicitada; não confundir com
a avaliação global de teste. Tempo total dos clientes é sequencial; uma estimativa
paralela é o máximo dos tempos de clientes por rodada (e deve ser identificada
como estimativa). `Budget` inclui trabalho sequencial e avaliação do servidor.

## Comandos reproduzíveis

Na raiz, use o ambiente que contém PyTorch/torchvision, numpy, sklearn, h5py,
matplotlib e ujson. Neste workspace: `conda activate pfllib`.

Teste CPU pequeno, sem baixar dados, com dois braços e duas rodadas reais do
servidor sobre dados sintéticos:

```bash
python -m unittest discover -s tests -p 'test_semantic_hera.py' -v
```

Preparação MNIST nativa, cache separado, smoke test de duas rodadas (não executado
automaticamente; o dataset completo ainda pode levar tempo):

```bash
python tools/prepare_dataset.py --dataset MNIST --scenario non-iid --num-clients 20 --dirichlet-alpha .5 --partition-seed 42 --output-name MNIST_dir05_seed42
cd system
python main.py -algo HERAFL -data MNIST_dir05_seed42 -m CNN -dev cpu -gr 2 -ls 1 -lbs 64 -jr .1 --hera-semantic --pruning-arms .25 .5 1 --aligned-comparison --wait-for-all --policy-seed 42 --reward-validation-fraction .2 --detailed-evaluation --experiment-id mnist_smoke
cd ..
```

Cifar10 principal com as duas arquiteturas VGG:

```bash
python tools/prepare_dataset.py --dataset Cifar10 --scenario non-iid --num-clients 20 --dirichlet-alpha .2 --partition-seed 42 --output-name Cifar10_dir02_seed42
python tools/semantic_experiments.py --dataset Cifar10_dir02_seed42 --models VGG11 VGG16 --seeds 42 43 44 --wait-for-all
```

O gerador inclui VGG11 e VGG16 por padrão e separa os IDs por modelo, variante
e seed. Para CNN/DNN, passe `--models CNN` ou `--models DNN` explicitamente.
Os modelos VGG usam o learning rate padrão 0.0001 do parser.

Também é possível iniciar os experimentos pelo script (estes comandos treinam):

```bash
DATASET=Cifar10 MODEL=VGG11 ./start.sh --reward-validation-fraction .2 --detailed-evaluation
DATASET=Cifar10 MODEL=VGG16 ./start.sh --reward-validation-fraction .2 --detailed-evaluation
```

`start.sh` usa VGG11 quando MODEL não é informado; use `MODEL=VGG16` para a outra
arquitetura ou `MODEL=CNN` para MNIST. Todos os algoritmos da mesma
execução recebem o mesmo modelo e learning rate. Nenhum experimento longo foi
iniciado automaticamente para validar esse suporte.

O comando do gerador **apenas imprime** a matriz. Acrescentar `--execute` executa-a.
Repita sem `--wait-for-all` para medir efeito dos timeouts. Gerar separadamente
IID (`--scenario iid`) e Dirichlet .5/.2/.05 em nomes distintos e seeds de partição
registradas. `--scenario` no treino não gera dados; `--alpha` do treino não é
alpha de Dirichlet. Novas opções explícitas do preparador rejeitam caches
incompatíveis; escolher outro `--output-name` em vez de sobrescrevê-los.

A matriz inclui FedAvg alinhado, FedLoad, HERAFL sem extensão (controle FedLoad), controle preservado,
heurística, LinUCB, apenas H, sem D, sem Q, contexto permutado, recompensa semântica
e retenções fixas .25/.5/.75/1. Reutiliza partição, inicialização por seed,
perfis, épocas, batches e participação. Três seeds são mínimo científico, não
três execuções de tuning no teste. As opções padrão são pontos iniciais.

H1: comparar generalização final/curva, macro-F1, dispersão e classes com suporte.
H2: comparar acurácia por orçamento realizado de parâmetros, bytes, energia e
tempo; parear curvas no menor orçamento comum, sem extrapolar, e apresentar
fronteiras de custo. Retenções nominais iguais não implicam custos iguais. Rodar
controles fixos para obter envelopes e selecionar orçamentos comuns previamente;
não atribuir ganhos à semântica quando decorrem de maior orçamento. Relatar por
faixa de capacidade/perfil semântico, frequência de retenções, falhas/aceites e
sensibilidade a Z com R fixo. Diferenças de seleção não comprovam utilidade.
Long-tail requer gerador/protocolo próprios e fica para depois.

## Etapa futura e limitações

Ação futura: `SubmodelAction(retention, component_mask, global_parameter_ids,
feature_space_version)`. Máscara deve indicar IDs globais de canais/blocos,
propagação entre camadas e dimensões/validade das features. Estimador proposto
para estudo: EMA da sensibilidade de CE por classe à ativação de cada componente,
computada em minibatches de treino com suporte registrado. Contabilizar passagem,
gradientes e transmissão; comparar máscaras L1/aleatórias/semânticas com mesmo
número real de parâmetros/FLOPs/memória e mesma participação. H3 só poderá ser
testada nessa etapa. A poda L1 atual **não** é seleção estrutural semântica.

Limites de memória/tempo são previsões simplificadas; sem proteção de memória de
pico. Q pode favorecer classes com suporte pequeno. Referências não são embeddings
atemporais. O protocolo transmite agregados, mas não oferece privacidade diferencial.
As perdas opcionais, seleção estrutural, retomada completa e resultados científicos
não fazem parte desta entrega.

## Verificação desta entrega

Executado no ambiente `/home/alfred/miniconda3/envs/pfllib/bin/python`:

* 12 testes semânticos passaram: indicadores calculáveis, bytes, poda/backward,
  ablações, mudança de decisão com R fixo, restrições, schema, duas rodadas CPU,
  contexto congelado, descartes, preservação sem respostas, timeout com overhead
  e salvamento JSON/CSV/plots inclusive sem braço viável.
* Suíte completa após alinhamento ao FedLoad, `python -m unittest discover -s tests -v`: 61 testes,
  58 passaram e 3 terminaram com `FileNotFoundError` por referências externas
  ausentes em `../mininetfed_iwcmc/`. São os testes de igualdade de perfis,
  amostragem do runtime e ordem da seleção estratificada em `test_mininet_fidelity`.
  Não foram desativados nem contabilizados como sucesso.
* Três testes de base passaram: equivalência exata FedLoad/HERAFL sem semântica
  com tempos controlados, seleção/perfis/timeout compartilhados, fallback FedLoad
  e rejeição do schema semântico anterior.
* Quatro testes específicos VGG passaram para ambas as arquiteturas: equivalência
  com o forward legado e chaves de checkpoint, dimensão/coordenadas preservadas,
  backward/agregação/protótipos e duas rodadas CPU com duas retenções.
* Parser `system/main.py --help`, geração da matriz sem execução, compilação Python
  dos arquivos alterados e `git diff --check` verificados.
* O Python padrão do shell não tem `torch`; use o ambiente acima. Os comandos
  MNIST/Cifar10 documentados e a matriz científica não foram executados.
