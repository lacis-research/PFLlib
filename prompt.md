# Implementar seleção adaptativa de submodelos com consciência semântica no HERAFL

## Objetivo

Atue como engenheiro de pesquisa em aprendizado federado e implemente, neste PFLlib, uma evolução do HERAFL baseada em **seleção adaptativa de submodelos com consciência semântica (Semantic-Aware Adaptive Submodel Selection)**. Integre características semânticas dos dados à política que já decide o treinamento parcial, combinando heterogeneidade computacional e estatística. Entregue código, testes, documentação e comandos reproduzíveis.

A pergunta central é: **como selecionar submodelos que respeitem os recursos dos clientes e preservem a qualidade das representações em cenários non-IID?** A política deve considerar recursos `R_k^t` e informação semântica `Z_k^t`, selecionando `S_k^t = pi(R_k^t, Z_k^t)`. Não basta adicionar perdas semânticas ao treino ou registrar entropia: os sinais devem efetivamente participar da escolha do submodelo.

Mescle a proposta ao HERAFL existente: reutilize o LinUCB, a orientação por capacidade, a seleção estratificada, a predição de stragglers, a poda física e a agregação por coordenadas. O objetivo final abrange tamanho e estrutura do submodelo, mas a primeira entrega deve implementar seleção semântica de **tamanho/retenção**; seleção de canais ou blocos fica como etapa posterior, com interface e protocolo documentados. Preserve a seleção de participantes na primeira etapa para isolar o efeito sobre a seleção de submodelos.

Investigue se essa integração melhora a generalização e a relação entre qualidade do aprendizado e uso de recursos. Não presuma ganhos antes dos experimentos, não use redução de drift como objetivo obrigatório e não atribua dificuldade semântica apenas à classe predominante. Demonstre que a política aprende escolhas adequadas aos dados, em vez de simplesmente atribuir modelos maiores a clientes considerados difíceis.

Mantenha `HERAFL` selecionável e acrescente `--hera-semantic` para ativar a extensão. Sem essa opção, preserve o comportamento atual. Separe resultados por configuração e identificador de experimento. Não altere `../FedSC/`, não substitua os dados nativos do PFLlib pelo pipeline daquele projeto e não execute experimentos longos automaticamente.

## 1. Referências e diagnóstico obrigatório

Leia os arquivos antes de editar. Caminhos relativos à raiz de `PFLlib`:

- `system/flcore/clients/clientherafl.py`: atualmente apenas herda `clientFedLoad`.
- `system/flcore/servers/serverherafl.py`: herda `FedLoad`, adiciona orientação de capacidade, seleção estratificada e predição de stragglers.
- `system/flcore/clients/clientfedload.py`: treino, instalação/offload de submodelos e contabilização de recursos.
- `system/flcore/servers/serverfedload.py`: atribuição de retenções, filtragem de respostas, recompensa, agregação e avaliação.
- `system/flcore/heterogeneity/pruning.py`: poda L1 física, coordenadas globais e agregação ponderada por cobertura.
- `system/flcore/heterogeneity/{linucb_hera,straggler,selection,results}.py`.
- `system/flcore/trainmodel/{models,mininet_models,vgg}.py`, `system/main.py`, `tools/prepare_dataset.py`, `dataset/utils/dataset_utils.py` e testes existentes.
- `../FedSC/main_fedsc.py`: `Global.get_re_p`, `get_dis_weights`, `get_co_p` e `Local.local_train`.
- `../FedSC/utils/infonce_lcl.py`, `../FedSC/utils/util_func.py` e `../FedSC/readme.md`.

Referência complementar para protótipos e colaboração, não uma política pronta de seleção de submodelos: Wang et al., **FedSC: Federated Learning with Semantic-Aware Collaboration**, KDD 2025, [DOI](https://doi.org/10.1145/3711896.3736957), [texto completo](https://arxiv.org/html/2506.21012v1). Consulte particularmente as seções 3.3–4.2 e o Algoritmo 1.

Síntese para orientar a leitura: o artigo constrói vizinhanças por diferenças entre similaridades angulares de protótipos locais ao centro da classe, agrega vizinhos em protótipos relacionais e usa discrepâncias das distribuições de rótulos para construir protótipos consistentes. RPCL contrasta classes; CPDR regulariza representações. Verifique as equações 7–16, inclusive o fator de normalização da similaridade, o tratamento de múltiplos positivos e a definição de CPDR. Não apresente temperatura fixa e MSE como reprodução literal dessas equações. A extensão para submodelos e participação parcial descrita abaixo é uma proposta deste projeto, sem transferência automática da garantia de convergência do artigo.

O código local exige revisão, não cópia literal:

1. `get_re_p` calcula uma transformação em `temp`, mas retorna `re_p` sem gravá-la. A distância até cópias da média não implementa a vizinhança angular descrita no artigo.
2. `get_co_p` usa posições de listas de classes como se fossem IDs globais de clientes; classes ausentes quebram essa associação. Também altera `re_p` in-place.
3. A normalização da discrepância pode dividir por zero quando todas as distribuições são uniformes.
4. A montagem de negativos mistura as listas de todo o minibatch: para uma amostra, pode incluir sua própria classe como negativa.
5. Os vetores iniciais criados com `torch.empty` não são protótipos válidos. Não use inicialização não definida nem `except` genérico para esconder inconsistências.

Registre em documentação uma tabela: mecanismo, referência no artigo, comportamento do código original e escolha implementada. Use o código executável como fonte do comportamento atual do PFLlib: há trechos desatualizados em `docs/heterogeneous_algorithms.md`, inclusive sobre a ordem de avaliação e contagem de rodadas.

## 1.1. Política semântica integrada ao HERAFL

Implemente duas políticas comparáveis sobre o mesmo conjunto de retenções: uma heurística explícita como baseline e uma extensão do LinUCB existente como método principal. Não substitua o HERAFL por um controlador independente.

Defina `Z_k^t = [H_k^t, U_k^t, D_k^t, Q_k^t]` com fórmulas, normalização, origem dos dados e validade temporal documentadas:

- `H`: entropia normalizada do histograma de classes do treino; não equivale, sozinha, à dificuldade da tarefa.
- `U`: média da entropia preditiva normalizada em uma passagem determinística de treino, sem usar o teste.
- `D`: divergência cosseno entre protótipos locais e referências globais das classes compartilhadas, com ponderação e tratamento de normas nulas explícitos.
- `Q`: proxy de confiabilidade das representações, por exemplo compactação intraclasse normalizada, acompanhada de suporte por classe. Não a apresente como medida universal de qualidade nem como duplicação de `D`.

Inclua máscaras de disponibilidade, cobertura, idade e retenção que gerou os indicadores. Ausência não significa entropia ou divergência zero observada. Use normalização fixa ou estimada somente com histórico disponível; não ajuste escalas com resultados futuros. Identifique versão e rodada dos protótipos comparados: coordenadas comuns não garantem representações temporalmente idênticas.

A decisão na rodada t deve usar apenas informação disponível antes da escolha, proveniente de respostas aceitas até t−1. Para clientes sem histórico ou com informação expirada, use fallback documentado baseado no HERAFL original. Não treine todos os candidatos para obter a pontuação nem use a perda do submodelo ainda não escolhido. Se houver coleta inicial de histogramas, contabilize-a explicitamente.

Na heurística, defina uma pontuação por braço que combine orientação de recursos, sinais semânticos e custo estimado do candidato, com pesos configuráveis e desempate determinístico. Um termo semântico idêntico somado a todos os braços não altera a escolha e não atende ao objetivo. Não imponha como verdade a regra “maior entropia exige maior modelo”.

No LinUCB, amplie `feature_order` e as matrizes por braço apenas no modo semântico. Preserve os mecanismos existentes de exploração e orientação por capacidade, registrando separadamente bônus de exploração, recompensa estimada, bônus de capacidade e contribuição semântica. Atualize o braço escolhido com o mesmo contexto congelado utilizado na decisão, sem substituir entradas por indicadores extraídos depois do treino. Versione o schema do contexto e rejeite checkpoints incompatíveis claramente.

Mantenha a recompensa original como controle para isolar o efeito do contexto semântico. Ofereça uma ablação separada com recompensa que combine qualidade preditiva, qualidade semântica observada e custo normalizado, definindo sinais, escalas, coeficientes e tratamento de respostas descartadas. Use validação separada do treino para recompensa preditiva e tuning no protocolo científico, aplicando essa correção também aos baselines. A formulação conceitual `L_task + lambda L_sem + mu C` não deve ser confundida com uma perda diferenciável do LinUCB nem pressupor recompensas contrafactuais observadas para todos os braços.

Filtre candidatos por limites explícitos de memória e tempo quando disponíveis. Reconcile esses limites com os pisos de retenção por capacidade: restrições físicas têm prioridade no modo semântico. Se nenhum braço for viável, registre a decisão de não treinar esse cliente; não selecione silenciosamente um braço inviável. Diferencie previsão de custo de garantia real, mantendo o protocolo de timeout. Compare políticas sob os mesmos limites e registre qualquer alteração do conjunto elegível em relação ao legado.

## 1.2. Etapa posterior: seleção estrutural

Documente uma interface futura para ações que combinem retenção e máscara de canais/blocos, mantendo IDs globais de parâmetros. A relevância semântica de componentes deve ter um estimador explícito, custo contabilizado e comparação entre estruturas com orçamento equivalente. Não chame a poda L1 existente de seleção estrutural semântica. A primeira entrega não precisa implementar essa etapa nem comprovar sua hipótese de preservação semântica para classes sub-representadas.

## 2. Compatibilidade entre poda e espaço semântico

O problema central é que a poda atual reduz também a dimensão penúltima. Dois vetores do mesmo comprimento podem representar coordenadas globais diferentes; padding com zeros ou projeções locais independentes não resolve automaticamente esse problema.

Implemente primeiro uma estratégia **de dimensão penúltima preservada**:

- Comece com `FedAvgCNN` e `DNN`, cujas representações antes do classificador são identificáveis. Preserve todos os neurônios de saída da última camada oculta, na ordem global; continue podando suas entradas e as camadas anteriores. Preserve todas as classes do classificador.
- Exponha `forward_features` ou uma interface equivalente, mantendo `forward(x)` retornando apenas logits. O caminho com features deve produzir os mesmos logits que o caminho normal, sem executar duas vezes o backbone ou duplicar augmentation.
- A dimensão semântica é a dimensão penúltima original do modelo e não depende do braço escolhido. Não adicione projetores aleatórios por cliente.
- Acrescente opção explícita à poda, com padrão legado, para preservar a saída da camada de representação. Mantenha os mapas de coordenadas e a agregação dos parâmetros compatíveis.
- Documente que a mesma retenção nominal passa a representar outro custo. Registre fração real de parâmetros treinados, tamanho do classificador preservado e custo total. Ainda deve haver redução física para retenções menores que 1 nos modelos suportados.
- Não estenda por suposição a modelos cuja representação seja um flatten de canais podados, como certas configurações de `MininetCNN`. Suporte adicional a MininetCNN/VGG exige mapeamento e testes próprios. Modelos ainda não suportados devem falhar cedo com mensagem específica quando a opção semântica estiver ativa; devem continuar funcionando no modo legado.
- MLR não oferece camada oculta para essa estratégia: rejeite sua combinação com o modo semântico inicial, sem mudar o comportamento legado.

Deixe a API preparada para uma futura estratégia com coordenadas e máscaras semânticas, mas não implemente dois sistemas incompletos. Não troque features por logits silenciosamente.

## 3. Contrato de dados e ciclo de rodada

Crie um módulo próprio, por exemplo `system/flcore/heterogeneity/semantic.py`, com funções testáveis e estruturas explícitas. Um payload local deve identificar `client_id`, rodada, retenção aplicada, versão do espaço de features, dimensão, contagens por classe, protótipos somente das classes presentes e os indicadores semânticos com suas máscaras de validade. Calcule H/U/Q localmente e D a partir de referências globais explicitamente versionadas; documente quais agregados precisam ser transmitidos e em qual direção. Use tensores destacados do grafo; nunca envie exemplos ou embeddings individuais.

Calcule cada protótipo como a média das features da classe. Faça uma passagem após o treino, sobre a partição de treino, em `eval()` e `no_grad()`, sem augmentation aleatória, acumulando somas e contagens em memória O(classes × dimensão). Essa passagem extra entra no custo medido. Conte cada exemplo uma vez, independentemente do número de épocas. Restaure o modo do modelo quando necessário.

Use um banco global imutável durante cada rodada, com referências por classe para os indicadores da política. Use inicialmente médias por classe ponderadas pelo número de exemplos dos contribuidores aceitos. Grupos relacionais e alvos consistentes são necessários apenas para a extensão opcional de perdas. Todos os clientes da rodada t recebem o banco produzido ao final de t−1, evitando dependência da ordem de execução sequencial.

Sequência exigida:

1. Preparar contextos com recursos e histórico semântico válido; selecionar participantes pelo mecanismo existente e retenções pela política configurada, congelando contexto, candidatos e pontuações da decisão.
2. Instalar submodelos e disponibilizar o mesmo snapshot semântico aos selecionados.
3. Treinar, extrair protótipos e calcular recursos completos antes de aplicar timeout.
4. Filtrar respostas pelo protocolo existente; somente respostas aceitas contribuem tanto para parâmetros quanto para protótipos.
5. Agregar parâmetros com `aggregate_submodels`, preservando coordenadas não cobertas.
6. Construir o próximo banco e atualizar o histórico semântico com payloads aceitos. Atualizar o bandit com recompensa observada e contexto congelado da escolha; manter o tratamento de respostas descartadas explícito e compatível com o protocolo.
7. Avaliar e salvar métricas conforme o ciclo atual.

Comece com banco vazio e uma rodada de aquecimento com CE e seleção pelo HERAFL original. Mantenha histórico por cliente separado do banco global, com rodada, versão, idade e expiração configurável; clientes ausentes não recebem indicadores atuais inventados. Se não houver respostas aceitas, preserve modelo e banco anterior; registre a idade do banco. Se houver respostas, substitua o banco pelo construído naquela rodada: classes não observadas ficam indisponíveis. Não misture silenciosamente protótipos de versões antigas. Mantenha o snapshot independente de referências a `client.model`, que a avaliação substitui pelo modelo global. Libere tensores e otimizadores seguindo o offload existente.

## 4. Extensão opcional: colaboração semântica e perdas

A seleção semântica deve funcionar com CE apenas; esse é o experimento principal. Implemente as perdas abaixo como extensão opcional, desativada por padrão, para avaliar separadamente se a colaboração no treino complementa a política semântica. Não condicione a escolha semântica de retenções à ativação dessas perdas. Os requisitos desta seção e seus testes se aplicam quando essa extensão for implementada.

Implemente separadamente construção relacional, agregação consistente e perdas. Identifique sempre cada contribuição por `(client_id, class_id)`.

Para relações, siga as equações 7–9 da referência. Use somente clientes aceitos que possuem a classe. Limite o número de vizinhos aos disponíveis, exclua o próprio cliente da busca e inclua sua contribuição uma única vez na média. Desempate por ID, trate normas nulas e preserve um protótipo quando houver um único contribuinte. Os grupos relacionais devem ser efetivamente retornados, sem modificar os protótipos originais.

Para agregação consistente, adapte `get_dis_weights` e `get_co_p` às respostas aceitas. Use histogramas de treino, referência uniforme e pesos sigmoid baseados na fração de amostras e discrepância normalizada. Quando a soma de discrepâncias for zero, zere esse termo. Renormalize os pesos sobre os contribuidores reais de cada classe. Não confunda quantidade total do cliente com quantidade de exemplos daquela classe. Não aplique esses pesos à agregação de parâmetros: são pesos do ramo semântico.

O objetivo local será:

```text
loss = CE(logits, y) + lambda_rpcl * loss_rpcl + lambda_cpdr * loss_cpdr
```

Para a primeira implementação, adote e identifique explicitamente uma variante prática: similaridade cosseno com temperatura fixa, InfoNCE com múltiplos positivos e CPDR por MSE, combinando correções do banco com escolhas do código local. Não rotule essa variante como reprodução exata do artigo. Documente as diferenças e deixe isolado o cálculo de similaridade para posterior avaliação do fator U do artigo.

- Em RPCL, positivos são todos os protótipos relacionais disponíveis da classe da amostra; negativos são somente outras classes. Calcule `logsumexp(todos) - logsumexp(positivos)` por amostra, com máscaras por rótulo e estabilidade numérica.
- CPDR usa o alvo consistente correspondente à classe; especifique a redução MSE como média sobre dimensões e amostras válidas. Use features e alvos na mesma escala; a normalização cosseno de RPCL não deve alterar in-place os alvos de CPDR.
- Sem positivo válido ou sem negativos, a contribuição RPCL daquela amostra é zero. Sem alvo consistente, CPDR é zero. Normalize cada perda pelo seu número de amostras válidas, protegendo o caso zero.
- CE continua operando para todas as amostras. Classes ausentes não ganham protótipos artificiais e não provocam descarte do minibatch.
- Gradientes fluem para as features locais; o banco recebido permanece sem gradiente. Perdas desativadas não devem construir grafos ou buffers desnecessários.

Separe experimentalmente três intervenções: contexto semântico na seleção de retenções, termo semântico na recompensa e perdas semânticas no treino. A primeira é o núcleo obrigatório; as demais são ablações independentes.

## 5. Integração, configuração e custos

Concentre a implementação em `clientHERAFL` e `HERAFL`. Se necessário, introduza hooks pequenos no ciclo compartilhado do FedLoad, com implementações padrão sem efeito. Evite duplicar todo o servidor e preserve FedLoad/ComparisonFedAvg.

Proponha e documente as opções abaixo, com validação antecipada:

| Opção nova | Padrão proposto |
| --- | --- |
| `--hera-semantic` | desativado |
| `--hera-semantic-policy` | `linucb`; alternativas `heuristic` e `legacy` para ablações |
| `--hera-semantic-features` | `H,U,D,Q`; subconjuntos para ablação |
| `--hera-semantic-max-age` | 5 rodadas, inteiro não negativo |
| `--hera-semantic-reward-weight` | 0.0, finito e não negativo |
| `--hera-rpcl-weight` | 0.0, finito e não negativo |
| `--hera-cpdr-weight` | 0.0, finito e não negativo |
| `--hera-semantic-temperature` | 0.05, finito e positivo |
| `--hera-semantic-neighbors` | 1, inteiro não negativo |
| `--hera-semantic-warmup-rounds` | 1, inteiro não negativo |

Esses valores são pontos de partida, não resultados de tuning. Pesos RPCL/CPDR zerados mantêm a seleção semântica e os indicadores necessários ativos. Para o controle estrutural, use política `legacy`, recompensa semântica zero e ambas as perdas zero, mantendo a representação preservada. Desativar `--hera-semantic` recupera o legado completo. Evite extração e comunicação somente quando nenhum componente ativo consumir esses dados. Documente também pesos da heurística, limites de recursos, normalização e configurações da recompensa.

Contabilize separadamente download/upload de parâmetros e de payloads semânticos. Defina uma representação lógica determinística dos payloads e conte tensores, contagens, IDs e metadados enviados. O download pode conter vários protótipos por classe: não o estime como apenas uma matriz classes × dimensão. Não conte um snapshot compartilhado no servidor como um único envio para todos os clientes.

Inclua o custo semântico em `round_time`, `energy_consumption` e bytes totais antes de timeout, atualização de contexto e recompensa. Registre tempo de extração, treino e agregação semântica do servidor separadamente; não misture custo do servidor à energia local. Preserve os rótulos de estimativa por perfil e declare o que não inclui overhead real de rede. Clientes descartados continuam com custos registrados conforme o protocolo atual.

Adicione aos resultados configuração completa, versão semântica, contexto semântico e suas máscaras/idades, candidatos viáveis, pontuações por braço, retenção escolhida, decomposição da recompensa, frequência de fallback, perdas CE/RPCL/CPDR quando habilitadas, classes cobertas, contribuições por classe, idade do banco, retenção nominal, fração real de parâmetros e custos adicionais. Atualize `results.py` e ferramentas de comparação apenas no necessário, mantendo leitura de arquivos antigos. Preserve checkpoints existentes e salve o banco/configuração caso ofereça retomada de treinamento.

## 6. Experimentos non-IID e avaliação justa

Organize a avaliação em torno da pergunta: em que condições a consciência/colaboração semântica aprimora o treinamento parcial do HERAFL, e com qual custo adicional? Use o HERAFL legado como referência principal e o controle de representação preservada com política legada para distinguir o efeito da seleção semântica do efeito da mudança estrutural. Avalie a relação entre qualidade do aprendizado e uso de recursos nas diferentes retenções e faixas de capacidade; não condicione a conclusão à demonstração de redução de drift.

`--scenario non-iid` sozinho não gera uma partição. Verifique os geradores e metadados reais. Não confunda `--alpha` de algoritmos em `main.py` com o alpha de Dirichlet dos dados. Se o preparador não expuser esse parâmetro, acrescente uma opção própria, passe-a ao gerador e registre-a no cache/metadados, sem sobrescrever partições incompatíveis silenciosamente.

Prepare comandos para MNIST como smoke test e Cifar10 como experimento principal. Sugira IID e Dirichlet com alpha 0.5, 0.2 e 0.05; long-tail é uma extensão posterior que exige geração e registro próprios. Reutilize exatamente as mesmas partições, inicialização, perfis, épocas, batch size, participação e seeds entre variantes. Use ao menos três seeds nos resultados científicos.

Compare:

1. FedAvg no protocolo alinhado e FedLoad.
2. HERAFL legado.
3. HERAFL com representação preservada, política legada, recompensa semântica zero e CE apenas.
4. Heurística semântica e LinUCB semântico, ambos com CE e recompensa de controle.
5. LinUCB com apenas H, sem D, sem Q e com todos os indicadores; contexto semântico embaralhado entre clientes como controle diagnóstico, respeitando rodadas e disponibilidade.
6. LinUCB semântico com termo semântico na recompensa, isolando sua contribuição.
7. Retenções fixas, incluindo 1.0, e políticas com orçamento realizado equivalente. Não atribua ganhos à semântica quando forem explicados apenas por mais parâmetros treinados.
8. Se as perdas opcionais forem implementadas: política legada com perdas; política semântica com apenas RPCL, apenas CPDR e ambas.

Teste H1 (generalização) e H2 (qualidade por custo) sem antecipar o resultado. H3 (preservação semântica por seleção estrutural) pertence à etapa posterior. Analise escolhas por faixa de capacidade e perfil semântico, distribuição de retenções, orçamento real e sensibilidade da política a mudanças de Z mantendo R fixo. Demonstre influência semântica em casos controlados e investigue sua utilidade nos experimentos; uma mudança de escolha sozinha não comprova melhor generalização.

Use `--aligned-comparison` quando adequado para controlar seleção e perfis; registre diferenças de participantes aceitos causadas pelo custo adicional e pelos timeouts. Inclua comparação com `--wait-for-all` para separar o efeito estatístico da perda de respostas.

Reporte acurácia global, macro-F1/balanced accuracy com suporte por classe, dispersão entre clientes e desempenho por faixa de capacidade, além de bytes, energia estimada, tempo e acurácia por orçamento. Defina claramente média por cliente versus ponderação por amostras e tempo sequencial versus estimativa de rodada paralela. Ausência de classe no teste não deve virar acurácia zero inventada.

O protocolo atual usa acurácia local de teste na recompensa do bandit. Documente essa limitação; para resultados científicos sem vazamento, use validação separada do treino para recompensa e tuning, aplicando o mesmo protocolo a todos os baselines comparados. Não ajuste hiperparâmetros no teste nem apresente o teste já usado pelo bandit como avaliação independente.

## 7. Testes e critérios de aceite

Adicione testes pequenos e determinísticos que comprovem comportamento, não apenas execução. Testes de relações, RPCL e CPDR são exigidos somente se a extensão opcional correspondente for implementada:

- Com recursos idênticos e estado controlado da política, mudanças semânticas podem alterar a retenção; sinais constantes por braço não são aceitos como mecanismo semântico.
- A seleção respeita candidatos viáveis, resolve conflitos com pisos e trata ausência de braço viável; a atualização LinUCB usa o contexto da decisão e o braço efetivamente escolhido.
- Indicadores H/U/D/Q têm valores verificáveis; ausência, expiração, classes únicas e vetores nulos têm tratamento explícito.
- Indicadores da rodada atual não influenciam sua própria seleção; desativar perdas mantém seleção semântica ativa.
- Heurística e LinUCB têm decisões reproduzíveis; ablações desligam apenas o mecanismo declarado.
- Retenções distintas preservam dimensão e identidade das coordenadas semânticas, reduzem parâmetros e permitem backward; retenção 1 preserva o caminho completo.
- Forward normal e caminho com features são equivalentes em avaliação; poda/agregação continuam corretas e posições não cobertas permanecem intactas.
- Protótipos e contagens correspondem a médias conhecidas; não retêm grafo nem multiplicam contagens pelas épocas.
- Relações angulares têm vizinhos esperados em um exemplo calculável; empates e um único contribuinte são determinísticos.
- Classes faltantes, IDs fora de ordem, discrepância zero, vetores nulos e participação parcial não produzem NaN nem associação errada de pesos.
- Construção consistente não modifica o banco relacional; pesos por classe somam um.
- Um minibatch com rótulos diferentes nunca usa positivos como negativos de uma mesma amostra; aumentar similaridade positiva reduz RPCL em exemplo controlado.
- Banco vazio, ausência de negativos, CPDR sem alvo e pesos zero têm fallback correto; gradiente semântico chega ao extrator, não aos alvos.
- Dois clientes com retenções diferentes treinam por pelo menos duas rodadas em CPU, com banco anterior e evolução de parâmetros.
- Respostas descartadas não entram no banco; nenhuma resposta preserva ambos os estados; a ordem de treino não altera o snapshot recebido.
- Bytes e custos incluem o payload semântico nas duas direções e afetam timeout; exemplos pequenos permitem conferir os totais à mão.
- Modo semântico desativado preserva resultados determinísticos e schemas legados; modelos incompatíveis falham antes do treino.

Execute os testes relevantes existentes e os novos, depois `python -m unittest discover -s tests -v` se o ambiente permitir. Registre dependências ausentes e verificações não executadas, sem alegar sucesso. Não é necessário rodar a matriz científica para validar a implementação.

## 8. Entrega

Entregue a seleção semântica de tamanho funcional, integrada ao HERAFL, testes, documentação das escolhas e comandos efetivamente compatíveis com o parser implementado. Inclua um exemplo CPU curto e scripts/configurações para as ablações, com saídas distintas. Ao finalizar, explique o que mudou, quais testes passaram, limitações do suporte a modelos e diferenças em relação ao artigo e ao código FedSC. Diferencie claramente o núcleo entregue, as perdas opcionais e o roteiro futuro de seleção estrutural. Só atribua ganhos ao método após medi-los.
