# HERAFL: generalização de um modelo global em cenários non-IID

## Objetivo e configuração padrão

O objetivo é aprender um único modelo global que generalize entre as distribuições dos clientes. O `start.sh` usa `HERA_COMMUNICATION=parameters`, seleção contextual de submodelos e regularização por protótipos com `HERA_GLOBAL_ALIGN_WEIGHT=1`. Os pesos locais são reinicializados a partir do modelo global a cada participação; o servidor agrega os parâmetros das respostas aceitas. Os protótipos complementam essa agregação e permanecem em RAM.

O modo anterior `HERA_COMMUNICATION=prototypes` continua disponível como variante personalizada: mantém modelos locais persistentes, troca somente protótipos após a inicialização e avalia modelos locais. Seus resultados não representam um modelo global. As opções `HERA_PROTOTYPE_KIND`, RPCL/CPDR, armazenamento de estados e checkpoints dessa variante não se aplicam ao novo modo global.

## Treinamento e colaboração

1. O servidor seleciona participantes e retenções usando os mesmos perfis e protocolo de rodadas do FedLoad. A poda preserva a dimensão e as coordenadas da representação penúltima para tornar os protótipos compatíveis.
2. Cada participante recebe um submodelo derivado dos pesos globais e uma cópia fixa do banco de protótipos anterior. Referências são desconsideradas durante o aquecimento e após a idade máxima configurada.
3. O treinamento local minimiza entropia cruzada mais alinhamento por classe:

   `L = CE(logits, y) + λ mean(||normalize(z) - normalize(p_y)||²)`.

   A média considera apenas exemplos cujas classes têm protótipo válido, com norma não nula. A distância soma as coordenadas de cada vetor normalizado, sem dividir pela dimensão. Os alvos são destacados do grafo; somente o modelo recebe gradientes. Na ausência de referências válidas, usa-se apenas CE.
4. Após o treinamento, uma passagem sem gradientes calcula médias de características por classe usando somente a partição de treinamento. A validação separada alimenta a recompensa da política, sem usar o teste para selecionar retenções.
5. O servidor agrega pesos nas coordenadas ativas, ponderados pelo número de exemplos de treinamento dos clientes. Coordenadas sem atualização preservam seus pesos globais. O banco seguinte combina protótipos aceitos, ponderados pelo suporte de cada classe. Respostas rejeitadas não atualizam pesos nem protótipos.

O banco de uma rodada não orienta outros participantes da mesma rodada. Classes ausentes não recebem protótipos inventados. Se nenhuma resposta é aceita, pesos e banco anteriores são preservados, respeitando o limite de idade para uso posterior. Os protótipos podem continuar sofrendo divergência semântica entre clientes; a inicialização global comum e o alinhamento buscam reduzi-la, sem garantir sua eliminação.

A regularização usa o banco mesmo com `HERA_SEMANTIC_POLICY=legacy`; nesse caso, a seleção usa apenas os recursos do controle, mas a colaboração continua ativa. Com peso zero, a coleta de protótipos ainda pode ocorrer para a política semântica.

## Avaliação e gráficos

Após a agregação, todos os clientes avaliam o mesmo modelo global em suas partições de teste. A acurácia principal é o total de acertos dividido pelo total de exemplos. A AUC micro por cliente é ponderada pelo tamanho do teste; o loss de avaliação no treino é somente CE, permitindo comparação com o framework. Os desvios de acurácia e AUC são calculados entre clientes, sem ponderação.

Os cinco gráficos por rodada (loss, acurácia, AUC e seus dois desvios entre clientes) são exportados em `plots/` para experimentos e comparações. HeraFL usa laranja e FedLoad azul. O HDF5 registra `evaluation_type=global_model` e uma variante que identifica o peso da regularização. Resultados locais antigos mantêm sua identificação original.

O `start.sh` também habilita `--detailed-evaluation`: os relatórios JSON incluem suporte e recall por classe, balanced accuracy, macro F1 e distribuição de acurácia entre clientes para o modelo global. Isso ajuda a distinguir ganhos concentrados em classes frequentes de desempenho mais uniforme. As mesmas partições são usadas por todos os métodos; isso ainda não mede generalização a novos clientes ou mudanças de distribuição.

## Custos e ablações

Este modo transmite parâmetros e protótipos. A contabilidade inclui os bytes dos pesos, coordenadas de entrada quando necessárias, banco recebido e resumo enviado. Treinamento regularizado e extração de protótipos entram no tempo computacional medido; rede e energia são estimativas dos perfis. Avaliação detalhada acrescenta trabalho ao tempo de execução do experimento.

Execute, mantendo partições, modelo, sementes e orçamento iguais:

```bash
# HeraFL global com alinhamento (padrão)
./start.sh

# Ablação: mesma seleção semântica, sem alinhamento
HERA_GLOBAL_ALIGN_WEIGHT=0 ./start.sh

# Controle de seleção por recursos, com alinhamento
HERA_SEMANTIC_POLICY=legacy ./start.sh

# Variante anterior, personalizada e relacional
HERA_COMMUNICATION=prototypes ./start.sh
```

Ao executar `system/main.py` diretamente, habilite `--hera-semantic --hera-communication parameters --hera-global-align-weight 1`. O peso direto tem padrão zero para preservar o controle anterior. Os pesos RPCL/CPDR não são usados no modo global; esta implementação aplica alinhamento normalizado, não uma reprodução literal do FedSC.

Melhora em non-IID é uma hipótese experimental. É necessário comparar com FedLoad, FedAvg e ablações em múltiplas sementes, incluindo custos adicionais. Não se deve interpretar o alinhamento como garantia de convergência, privacidade ou superioridade.
