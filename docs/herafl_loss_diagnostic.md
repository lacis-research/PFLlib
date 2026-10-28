# Diagnóstico histórico das perdas — 28/10/2026

**Este diagnóstico refere-se à implementação v2, posteriormente substituída.**
Ela usava U diferenciável sobre todo o dataset e CPDR como soma absoluta.
A v3 segue o código oficial local: não calcula U e usa MSE. Os valores abaixo
não descrevem as perdas atuais. Veja [implementação atual](herafl.md).

O script `tools/diagnose_hera_losses.py` agora mede a v3. Para não sobrescrever
os números históricos, use `--output tmp/hera-loss-diagnostic-official.json`.
Os novos logs registram o tempo do forward semântico, sem estatísticas de U.

O processo do experimento anterior já não estava ativo quando este diagnóstico
começou. Não foi interrompido pelo diagnóstico, e seu estado de memória não foi
recuperado. Os pesos e as referências desse processo não foram usados aqui.

## Procedimento

```bash
python tools/diagnose_hera_losses.py --device cuda
```

Usa 128 exemplos de treino de cada um dos clientes 0 e 4 do CIFAR-10 existente,
excluindo o prefixo de validação de 20% com a mesma semente de particionamento.
São 256 exemplos, batch 64, CNN, SGD com LR 0.001, semente 42, float32,
um aquecimento CE compartilhado e duas épocas por variante.
Todos os modelos, perdas, gradientes e normalizadores são calculados em CUDA;
não há fallback CPU. Os dados desse recorte ficam na GPU.

Cada variante começa dos mesmos pesos e usa a mesma ordem de minibatches.
As referências são médias de classe congeladas após o aquecimento, com um
protótipo relacional por classe. **Não há poda, agregação federada nem avaliação
de teste** neste diagnóstico. Ele isola comportamento das perdas e não reproduz
as rodadas observadas pelo usuário. Os tempos incluem instrumentação de gradientes.

## Resultado observado na RTX 4060 Laptop

Norma L2 do gradiente de cada termo no primeiro minibatch:

| Termo | Norma |
|---|---:|
| CE | 1.99 |
| CPDR | 33.70 |
| RPCL | 40.27 |

Os termos semânticos têm normas aproximadamente 17 e 20 vezes maiores que CE.
Isso indica uma diferença de escala com pesos iguais a 1; normas isoladas não
medem a direção ou o efeito combinado dos gradientes.

| Variante | CE de avaliação no treino após 2 épocas |
|---|---:|
| CE | 2.29534 |
| CE + CPDR | 2.30376 |
| CE + RPCL | 2.28729 |
| CE + CPDR + RPCL | 2.30462 |

Com ambos os termos ativos, U ficou aproximadamente entre 0.51 e 1.01, sem
singularidade próxima de zero neste recorte. A norma média das características
foi 1.03 após duas épocas, contra 1.37 no controle CE. Esses resultados justificam
investigar principalmente a escala/efeito da CPDR, mas não provam colapso nem
explicam sozinhos a acurácia global do experimento anterior.

O JSON completo, com versões PyTorch/CUDA, tempos e picos de memória, é salvo em
`tmp/hera-loss-diagnostic.json`. Nenhum peso padrão foi alterado automaticamente.

## Novos logs no treinamento

Cada cliente imprime início e fim de época, e progresso entre minibatches quando
transcorrem pelo menos 30 segundos. Uma operação individual longa ainda precisa
terminar antes da próxima mensagem. São exibidos CE, CPDR, RPCL, perda total,
U médio e duração da época.

O JSON de política salva `clients[].loss_diagnostics`, com médias ponderadas
pelas amostras, contribuições ponderadas das perdas, U mínimo/médio/máximo,
tempo do forward do normalizador e tempo de backward + atualização. A recomputação
do checkpoint do normalizador está incluída no backward. Os logs de terminal
aparecem durante o treino; o JSON continua sendo exportado ao final da execução.

O próximo experimento federado deve comparar CE com a configuração completa
usando o mesmo protocolo e orçamento. Ajustar pesos ou substituir a CPDR impressa
pela MSE do código oficial constitui uma variante que deve ser identificada.
