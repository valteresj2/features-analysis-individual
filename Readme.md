### Objetivo Geral
- Comparar características de conjuntos de dados distintos (como teste e produção) ou dentro de um único conjunto de dados (com base em uma variável alvo), com o objetivo de entender como diferentes variáveis se comportam e se relacionam umas com as outras.

### Funções Específicas

#### 1. **multiple_dfs**:
   - Combina múltiplos DataFrames do Pandas em uma única planilha Excel, separados por espaços especificados.
   - Útil para criar um relatório consolidado de várias análises.

#### 2. **transform_table**:
   - Transforma um DataFrame para mostrar a contagem e a porcentagem de cada valor único em uma coluna específica (alvo).
   - Utilizado para resumir dados categóricos.

#### 3. **Classe FAI**:
   - Fornece funcionalidades para analisar a aderência (semelhança) entre dois conjuntos de dados (por exemplo, um conjunto de dados de teste e um conjunto de dados de produção) e realizar análises bivariadas.

   - **Método adherence**:
     - Compara a distribuição de variáveis em dois conjuntos de dados (teste e produção).
     - Calcula estatísticas como Percentual de Informação Populacional (PSI) e coeficiente de correlação de Pearson para variáveis categóricas, e estatística KS para variáveis numéricas.
     - Identifica discrepâncias significativas e gera relatórios detalhados.
     - Pode ser usado para monitorar mudanças no comportamento dos dados ao longo do tempo ou entre diferentes ambientes.

   - **Método perf_features**:
     - Realiza análise bivariada, avaliando a relação entre cada variável independente e uma variável alvo.
     - Fornece contagem, porcentagem, entropia, razão de risco e outras estatísticas para cada valor da variável.
     - Ajuda a entender a importância e o impacto de cada variável independente na variável alvo.

### Resultados e Aplicação
- Os resultados dessas análises podem ser usados para entender melhor os padrões nos dados, identificar variáveis importantes, monitorar a consistência dos dados entre diferentes ambientes (como teste e produção), e guiar a tomada de decisões em ciência de dados.
- Os relatórios gerados podem ser úteis para apresentações, análises mais profundas ou como parte de processos de garantia de qualidade de dados.

### Contexto de Utilização
- Esse script é particularmente valioso em contextos onde a comparação detalhada e a análise de conjuntos de dados são essenciais, como em testes de modelos de machine learning, validação de dados e análise exploratória de dados.

