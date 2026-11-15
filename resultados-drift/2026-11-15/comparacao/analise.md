# HERA-FL: IID versus Dirichlet — 15/11/2026

As 200 rodadas do cenário Dirichlet terminaram com exit_code0.
O HERA-FL aprendeu, mas perdeu acurácia e estabilidade sob heterogeneidade
forte. Seu custo de computação/comunicação permaneceu próximo ao da base.

| Métrica | Base IID | Dirichlet alpha0,05 |
|---|---:|---:|
| Imagens de treino/teste | 45.000/15.000 | 45.000/15.000 |
| Rodadas e respostas aceitas | 200 e1200 | 200 e1200 |
| Acurácia global final | 78,71% | 52,07% |
| Melhor acurácia | 79,27% (rodada197) | 52,53% (rodada189) |
| Média nas últimas20rodadas | 78,20% | 41,73% |
| Desvio nas últimas20rodadas | 0,64 ponto percentual | 8,38 pontos percentuais |
| Primeira rodada com pelo menos50% | 12 | 99 |
| Tempo reportado de execução | 4737,87s (~79min) | 4783,86s (~80min) |
| Retenção média | 72,96% | 73,00% |

A diferença final é26,64pontos percentuais. Média das últimas20rodadas caiu
36,47pontos percentuais; a última rodada é melhor que a média recente e não
resume a instabilidade. O cenário tem entre2e9classes presentes no treino porUE,
sem máscara imposta; as10classes permanecem no teste de cada cliente.

Métricas dos SUBMODELOS LOCAIS nas últimas20rodadas, antes da agregação,
ponderadas pelas amostras (não são a acurácia do modelo global):

- Classes presentes no treino local:13.924/22.696 =61,35%.
- Classes ausentes no treino local:788/37.304 =2,11%.

Esse comportamento é compatível com especialização forte nas classes locais.
Não identifica sozinho se a poda agrava a dificuldade nem demonstra
esquecimento catastrófico como mecanismo causal. Os pesos locais recebem
conhecimento do modelo global; ausência se refere aos dados locais.

Recursos: tempo reportado +0,97%, soma de computação local medida +1,15%,
comunicação em bytes +0,37%, energia ESTIMADA +1,02%. Retenção praticamente
igual. Não há ganho de eficiência demonstrado para compensar a perda de
qualidade neste cenário. Energia e rede são estimativas por perfil; tempo
real e computação foram medidos no processo. wait-for-all aceitou todas as
atualizações e não mede tolerância a deadlines/disconexões.

Conferências realizadas: mesmos hiperparâmetros principais, VGG11,30UEs,
6participantes/rodada,5épocas, batch64, LR0,0001, política/pisos de retenção,
mesmos perfis completos e mesma sequência de participantes em todas as rodadas.
Dataset preserva todos os índices de treino exatamente uma vez e cada arquivo
de teste é byte a byte igual ao original, conforme validação da preparação.
Somente a atribuição das imagens de treino aos UEs foi alterada.

Conclusão permitida: nesta realização, HERA-FL é sensível à redistribuição
fortemente non-IID com classes ausentes localmente; aprendizado mais lento,
instável e com generalização local fraca às classes ausentes. Uma única
execução por configuração não quantifica a variabilidade entre seeds.
Não se deve alterar a poda/otimizador/etc. para tentar melhorar números sem
nova instrução: o usuário exige preservar o HERA-FL padrão e adicionar
apenas o cenário dos dados. Mobilidade continua pendente, não implementada.

A duração IID correta é ~1h19, não2h38 (erro de soma anterior já corrigido).

Todos os números e caminhos de origem estão em metricas.json.
