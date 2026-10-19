# HERAFL: adaptação de modelos locais e colaboração por protótipos em aprendizado federado heterogêneo

## Resumo

O aprendizado federado em ambientes heterogêneos exige conciliar diferenças na capacidade computacional dos participantes, restrições de comunicação e distribuições locais de dados. O HERAFL, na formulação por protótipos apresentada neste documento, combina poda estruturada adaptativa, persistência do aprendizado local e compartilhamento de representações por classe. Cada cliente mantém um modelo mestre e treina um submodelo cuja retenção é escolhida por uma política contextual. Após o treinamento, os participantes compartilham protótipos e metadados, utilizados pelo servidor para construir referências que orientam as rodadas seguintes. A proposta busca ajustar a capacidade ativa dos modelos às condições dos clientes e reduzir a dependência da transmissão recorrente de parâmetros, preservando a continuidade da otimização local. Sua avaliação deve considerar conjuntamente qualidade preditiva, comunicação, tempo e energia, distinguindo desempenho personalizado de generalização global.

**Palavras-chave:** aprendizado federado; heterogeneidade; poda estruturada; protótipos; bandits contextuais; personalização.

## Motivação e hipótese de pesquisa

A adoção de uma única configuração de treinamento pode impor custos incompatíveis com os recursos de parte dos clientes. Paralelamente, distribuições não idênticas de dados podem produzir necessidades distintas de representação e especialização. Adaptar apenas o volume de computação não resolve, por si só, a transferência de conhecimento entre participantes; compartilhar conhecimento também não assegura que o treinamento seja adequado às condições locais.

O HERAFL articula essas duas dimensões. A adaptação computacional é realizada pela escolha da fração de canais ou neurônios mantidos no submodelo ativo. A colaboração ocorre por referências semânticas extraídas dos dados de treinamento. A hipótese de pesquisa é que essa combinação possa oferecer uma relação favorável entre qualidade preditiva e custo operacional, mantendo modelos capazes de se especializar em suas distribuições locais. Essa hipótese requer validação experimental e não pressupõe superioridade em todos os cenários IID e non-IID.

## Formulação do método

Considere um sistema com \(N\) clientes, \(C\) classes e treinamento organizado em rodadas. Cada cliente \(i\) dispõe de conjuntos separados de treinamento, validação e teste, além de um modelo mestre \(W_i\). Após uma distribuição inicial comum de pesos, os modelos evoluem localmente. No protocolo principal, o servidor não mantém um modelo global treinado por agregação de parâmetros: mantém uma política de adaptação e um banco de referências por classe.

A cada participação, uma retenção \(a_i^t\in(0,1]\) determina o submodelo que será otimizado. A poda seleciona saídas de camadas segundo a importância L1 de seus pesos, propagando as coordenadas selecionadas para as camadas seguintes. A dimensão da representação penúltima é preservada para permitir a troca de protótipos compatíveis em tamanho. A retenção nominal, entretanto, não corresponde necessariamente à fração total de parâmetros ativos.

Ao concluir o treinamento, o cliente incorpora ao mestre apenas as coordenadas utilizadas. Pesos e momentos do otimizador associados a coordenadas inativas permanecem preservados e podem ser recuperados em participações posteriores. Desse modo, mudanças no tamanho do submodelo não exigem reiniciar o aprendizado local. A persistência mantém continuidade, mas também implica armazenamento adicional e não equivale à otimização contínua de um modelo denso.

### Adaptação por política contextual

A escolha da retenção utiliza um bandit contextual LinUCB. Cada braço representa uma retenção possível, e o contexto reúne atributos de energia, capacidade computacional, quantidade de dados, custo de processamento e largura de banda. Para um contexto \(x_i^t\), o escore de um braço \(a\) é:

\[
 s(a,x_i^t)= (x_i^t)^\top\widehat{\theta}_a
 +\alpha_a\sqrt{(x_i^t)^\top A_a^{-1}x_i^t},
 \qquad \widehat{\theta}_a=A_a^{-1}b_a.
\]

O primeiro termo estima a recompensa e o segundo representa exploração. Restrições opcionais de memória e tempo previstos podem excluir braços inviáveis. A recompensa observada combina acurácia de validação e utilidades de energia e tempo:

\[
 R_i^t=w_{\mathrm{acc}}U_{\mathrm{acc},i}^t
       +w_EU_{E,i}^t+w_TU_{T,i}^t,
 \qquad w_{\mathrm{acc}}+w_E+w_T=1.
\]

Na implementação, as utilidades de energia e tempo medem proximidade de valores de referência, e não uma redução monotônica desses custos. O contexto utilizado para atualizar o bandit é o mesmo disponível no momento da decisão. Opcionalmente, ele é ampliado com indicadores de diversidade de rótulos, incerteza preditiva, divergência em relação ao banco e concentração das representações por classe. A política padrão utiliza somente recursos; a comunicação por protótipos permanece ativa em ambas as configurações.

### Colaboração por protótipos

Após otimizar seu submodelo, o cliente realiza uma passagem sobre o treinamento, sem atualização de gradientes, para calcular a representação média de cada classe observada. Seja \(z_i(x)\in\mathbb{R}^d\) a representação penúltima e \(D_{i,c}\) o subconjunto de treinamento da classe \(c\):

\[
 p_{i,c}=\frac{1}{|D_{i,c}|}\sum_{x\in D_{i,c}}z_i(x).
\]

Na variante simples, o servidor combina os protótipos das respostas aceitas por uma média ponderada pelo suporte local. Para o conjunto de contribuidores \(\mathcal{A}_{t,c}\) da classe \(c\):

\[
 p_c^t=
 \frac{\sum_{i\in\mathcal{A}_{t,c}}|D_{i,c}|\,p_{i,c}^t}
      {\sum_{i\in\mathcal{A}_{t,c}}|D_{i,c}|}.
\]

Essas referências são utilizadas em treinamentos posteriores. O objetivo local combina entropia cruzada e alinhamento entre representações normalizadas:

\[
 \mathcal{L}_i^t=\mathcal{L}_{\mathrm{CE}}
 +\lambda_{\mathrm{align}}
 \mathbb{E}_{(x,y)\in D_i^{\mathrm{ref}}}
 \left[\frac{1}{d}
 \left\|\widehat{z}_i(x)-\widehat{p}_y^{\mathrm{ref}}\right\|_2^2\right],
\]

em que \(D_i^{\mathrm{ref}}\) contém os exemplos com referência válida no banco recebido, e o acento circunflexo indica normalização L2. Na ausência de referências válidas, o termo colaborativo é zero. Os alvos permanecem fixos durante o treinamento, enquanto as representações locais recebem gradientes. A divisão por \(d\) torna a escala do alinhamento dependente da dimensão, exigindo atenção à calibração de seu peso.

Uma variante relacional estende o banco com relações entre protótipos de diferentes contribuidores e acrescenta objetivos de contraste e consistência. Essa extensão oferece um mecanismo adicional de colaboração, mas pode elevar o tráfego conforme cresce o número de relações compartilhadas. Preservar a dimensão dos vetores não garante, por si só, alinhamento semântico entre modelos locais.

## Dinâmica de uma rodada federada

Uma rodada começa com a seleção uniforme de uma fração dos clientes. O servidor congela os contextos disponíveis e uma referência comum, derivada do banco anterior. Em seguida, escolhe as retenções de todos os participantes viáveis antes de observar os resultados de treinamento daquela rodada. Referências indisponíveis ou antigas são desconsideradas; um período inicial de aquecimento permite treinamento sem alinhamento colaborativo.

Cada participante constrói seu submodelo a partir do próprio mestre, recupera os estados correspondentes do otimizador e executa o treinamento local. Depois, extrai protótipos, mede a acurácia de validação e incorpora as atualizações ao estado persistente. A resposta ao servidor contém representações agregadas e telemetria, sem transmissão recorrente dos pesos treinados.

O servidor verifica as respostas e aplica as regras de aceitação. Somente respostas aceitas alimentam a atualização da política e a construção do novo banco. Entretanto, um cliente cuja resposta seja rejeitada preserva seu aprendizado local e incorre nos custos do trabalho realizado. Os protótipos produzidos na rodada atual não alteram as referências dos participantes que ainda estão executando: a colaboração é incorporada em rodadas posteriores.

Ao fim da rodada, os modelos locais são avaliados quando previsto pelo protocolo experimental. Clientes ausentes mantêm seus estados anteriores e podem voltar a participar sem reinicialização. O processo prossegue até o limite de rodadas ou o critério de parada, produzindo uma coleção de modelos locais e um histórico de decisões, custos e resultados.

## Avaliação e contribuição pretendida

A contribuição central da proposta é integrar seleção contextual de capacidade, poda local persistente e colaboração por representações em um mesmo ciclo federado. Essa integração permite investigar separadamente o efeito da adaptação estrutural, da continuidade do otimizador e da informação compartilhada. Comparações com treinamento local independente, retenção fixa, modelo completo e alinhamento desativado são necessárias para atribuir eventuais ganhos a cada componente.

A avaliação principal corresponde à acurácia dos modelos locais em suas respectivas partições de teste, ponderada pelo número de exemplos. Ela deve ser acompanhada pela distribuição de desempenho entre clientes e pelos custos de comunicação, tempo e energia. Em cenários non-IID, testes locais concentrados em poucas classes podem favorecer modelos especializados; por isso, acurácia local elevada não demonstra automaticamente generalização para uma distribuição global equilibrada.

A implementação mede tempo computacional e volume serializado de mensagens, enquanto custos de rede e energia são estimados por perfis. A inicialização, as respostas rejeitadas e a avaliação devem ser consideradas conforme a contabilidade adotada. O método não estabelece garantias formais de privacidade, convergência ou superioridade. Seu mérito deve ser demonstrado por experimentos controlados em cenários IID e non-IID, com múltiplas sementes, critérios de avaliação explícitos e análise conjunta de qualidade e custo.
