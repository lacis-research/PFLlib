# HERA-FL com Dirichlet e todas as imagens da base

Para o cenário completo do FedSC, com cauda longa exponencial e integração
ao `start.sh`, consulte [Cenários FedSC](fedsc_scenarios.md). O
`examples/run_hera_shift.sh` agora executa esse cenário completo de forma
portátil. O protocolo abaixo continua disponível no preparador ao omitir
`--imb-factor`; os caminhos externos descritos abaixo são históricos.

Protocolo atual: `hera_dirichlet_full_data_v3`. A única mudança nos dados da
base é a atribuição das imagens de treino aos UEs pelo particionador Dirichlet
original do FedSC. Todas as imagens de treino são preservadas e as classes ausentes surgem da amostragem.
Cada teste local é copiado integralmente da base IID, sem alterações.

## Condições preservadas

- Mesmas 45.000 imagens de treino, cada uma utilizada exatamente uma vez.
- Mesmas 15.000 imagens de teste e os mesmos 500 exemplos por UE.
- Mesmos pixels, rótulos, normalização e separação entre treino e teste.
- Mesmos 30 UEs, VGG11, poda, LinUCB, preditor, recompensa e agregação.
- Launcher completo: 200 rodadas, 5 épocas locais, 6/30 participantes,
  batch64, LR0,0001, seed de execução42, perfis paper e wait-for-all.

A seed66 e alpha0,05 controlam apenas a redistribuição dos dados. Cada UE
recebe 1.500 imagens de treino pelo balanceamento final do próprio FedSC.
As proporções de classes locais mudam; classes ausentes são observadas nos
rótulos amostrados, sem uma quantidade obrigatória de classes por UE.
Algum UE pode ter todas as classes, dependendo da realização da distribuição.

O teste original contém as 10 classes em cada UE. Assim, o submodelo local
pode ser avaliado em categorias ausentes de seu treinamento local. Os pesos
baixados podem carregar conhecimento de outros UEs sobre essas categorias.
Não há introdução programada de classes ao longo das rodadas.

## Dados e preparação

Dataset: `/media/bolla/HD UTILITÁRIOS/testesherafl/datasets/Cifar10DirichletFull30`.
O diretório de saída deve ser novo. A base e os resultados anteriores são
preservados.

```bash
cd /home/bolla/artigo/HeraFL_drift
"/media/bolla/HD UTILITÁRIOS/testesherafl/venv/bin/python" tools/prepare_hera_shift.py \
  --source-dataset "/media/bolla/HD UTILITÁRIOS/testesherafl/PFLlib/dataset/Cifar10" \
  --output-dir "/media/bolla/HD UTILITÁRIOS/testesherafl/datasets/Cifar10DirichletFull30" \
  --num-clients 30 --alpha 0.05 --seed 66
```

O gerador preserva os arrays sem normalizá-los novamente. Conserva também
os totais originais por classe, que podem ser ligeiramente diferentes devido
ao split aleatório da base. A entrada é o dataset da base HERA-FL, e a única
transformação é a atribuição dos índices de treino a outros clientes.

## Proveniência e evidência de preservação

`config.json` registra dataset de origem, configuração original, hashes de
cada arquivo de treino/teste, contagens por classe, classes observadas por UE,
totais de amostras e `all_source_training_samples_used_once=true`.
`indices.json` contém a atribuição dos índices do pool de treino e os intervalos
que ligam esse pool aos arquivos originais. O gerador verifica que a lista
redistribuída é exatamente uma permutação de todos os índices de treino.
Os arquivos de teste são copiados sem modificação, permitindo verificar seus
hashes byte a byte.

O particionamento usa os helpers de hwang52/FedSC, commit
`a04dc8e630706a0c3f12f716c45857ead95a4940`, preservados e atribuídos em
`tools/fedsc_partition.py`. Exige ao menos10 clientes. Mantém o critério original
de resorteio e o balanceamento final; não impõe mínimo customizado de64 exemplos
nem máscara de2 classes. Normalização e treinamento são os do HERA-FL.

## Execução na GPU

A cópia atualizada do código e o alias do dataset ficam no HD UTILITÁRIOS.

```bash
bash "/media/bolla/HD UTILITÁRIOS/testesherafl/executar-herafl-shift.sh"
```

O launcher padrão executa200 rodadas; não é disparado ao gerar os dados.
`HERA_ROUNDS=1 HERA_LOCAL_EPOCHS=1` permite uma validação curta.
`SHIFT_GOAL` define o nome; por padrão é criado um identificador próprio.
Não usar start.sh para gerar esse alias: chamar o gerador acima explicitamente.

## Avaliação das classes locais

Após treinar cada participante e antes de agregar, são registrados acertos,
amostras e acurácia nas classes presentes/ausentes do treino real:

- `local_seen_correct`, `local_seen_samples`, `local_seen_accuracy`;
- `local_absent_correct`, `local_absent_samples`, `local_absent_accuracy`;
- `local_train_class_count`.

Esses campos ficam em round_stats no policy JSON e CSV round_clients.
Grupo sem amostras tem acurácia null. Para combinar clientes, somar acertos e
amostras antes de dividir. Não confundir essa avaliação dos submodelos locais
com a avaliação do modelo global após agregação.

O config dos resultados registra as classes reais por UE e a configuração
completa dos dados. A recompensa continua usando a acurácia de todo o teste
local, como na implementação anterior.

## Testes

```bash
python -m unittest discover -s tests -p test_hera_shift.py -v
```

10 testes: fidelidade dos índices ao FedSC em fixtures de10/17/30 clientes,
conservação de imagens mesmo com totais por classe diferentes, igualdade dos
testes originais, ausência natural de classes, não sobrescrita de bases e uma
rodada HERA-FL com gravação de métricas. A comparação futura entre IID e esse
cenário pode usar os mesmos exemplos e parâmetros; repetições independentes
continuam necessárias para avaliar variabilidade.
