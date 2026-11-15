# Alterações em relação à base HERA-FL

Referência: commit cec3d1cfdf547848cf6c9720710de6bc2947e160, branch herafl.

## Comportamento experimental adicionado

Redistribuição por Dirichlet das mesmas45.000imagens de treino entre30UEs,
com classes ausentes surgindo da amostragem. Mantidos os mesmos15.000exemplos
de teste, pixels, rótulos, normalização e quantidade de imagens por cliente.

## Arquivos existentes que receberam acréscimos

- `system/flcore/clients/clientherafl.py`: leitura das classes reais do treino
  e avaliação local separada em classes presentes/ausentes. A acurácia total
  retornada para a recompensa continua usando todos os exemplos do teste.
- `system/flcore/servers/serverherafl.py`: registro das classes locais e da
  configuração do dataset nos arquivos de resultado. Os métodos existentes
  da classe permanecem iguais ao commit de referência.
- `README.md`: referências à feature e aos resultados locais.

Portanto, arquivos existentes foram ampliados; não se deve afirmar que todos
os arquivos ficaram intocados. O núcleo do algoritmo foi preservado.

## Núcleo preservado

Arquiteturas VGG11/MininetCNN, laço de treinamento, treinamento dos submodelos,
LinUCB, poda estruturada, agregação, perfis e preditor permanecem iguais à base.
Não foram alterados os parâmetros do treinamento completo da referência.

Novos arquivos fornecem o gerador, helpers Dirichlet com atribuição ao FedSC,
testes, launcher de execução, documentação e resultados do experimento.

A comparação registrada usa as mesmas imagens, testes, perfis e participantes
por rodada; o fator experimental é a distribuição dos dados de treino.
