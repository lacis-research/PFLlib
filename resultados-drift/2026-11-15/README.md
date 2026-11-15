# Resultados HERA-FL — IID e Dirichlet

Resultados completos das duas execuções comparadas. Cada cenário contém o
log das 200 rodadas, configuração, métricas HDF5, policy JSON, CSVs e seis
gráficos PNG. Arquivos copiados do HD UTILITÁRIOS, com documentação e metadados inativos
limpos nesta cópia; hashes locais conferidos por SHA-256
(manifesto.json). Os originais permanecem no HD.

| Cenário | Acurácia final | Melhor acurácia | Duração reportada |
|---|---:|---:|---:|
| IID | 78,71% | 79,27% | ~79min |
| Dirichlet | 52,07% | 52,53% | ~80min |

Ambos: VGG11, 30UEs, 6participantes/rodada, 200rodadas, 5épocas locais,
45mil imagens de treino e 15mil de teste. O cenário Dirichlet usa alpha0,05,
seed de particionamento66 e mantém exatamente os testes da base.
As classes ausentes surgem naturalmente da distribuição do treino entre UEs.


- [Análise comparativa](comparacao/analise.md)
- [Métricas comparativas](comparacao/metricas.json)
- [Gráficos IID](iid/graficos/)
- [Gráficos Dirichlet](dirichlet/graficos/)

As curvas mean_accuracy são avaliações dos submodelos locais, enquanto
as curvas global_accuracy avaliam o modelo global após agregação.
Retenção é fração de canais/neurônios ocultos, não porcentagem total de pesos.
Energia e comunicação são estimativas de perfis; tempo total de clientes
modelado não é tempo real de execução. Há uma realização por cenário.

Esta pasta é versionável no Git (não usa o diretório results ignorado pelo
.gitignore). Nenhum commit ou push foi realizado. Dados de imagens, venv,
caches e pesos dos modelos não foram copiados para evitar ocupar o disco do
Ubuntu. Para reprodução, consultar docs/herafl_class_shift.md na raiz do projeto;
o launcher ainda aponta para a cópia de execução no HD.
