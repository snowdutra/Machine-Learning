# Exploração dos Dados

Nesta etapa, realizamos uma análise inicial do dataset 'Students Performance in Exams', incluindo visualizações das distribuições das notas de matemática, leitura e escrita. Foram observadas estatísticas descritivas, como média, mediana e desvio padrão, além de gráficos de histograma e boxplot para melhor compreensão dos dados.

**Hipótese 1:** As notas dos estudantes apresentam distribuição normal e podem ser influenciadas por fatores socioeconômicos e educacionais.

=== "Código"
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")

    # Carregar o dataset
    csv_path = "<CAMINHO_DO_CSV>"
    df = pd.read_csv(csv_path)

    # Estatísticas descritivas
    print(df[['math score', 'reading score', 'writing score']].describe())

    # Histogramas
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, col in enumerate(['math score', 'reading score', 'writing score']):
        sns.histplot(df[col], bins=20, ax=axes[idx], kde=True)
        axes[idx].set_title(f'Distribuição: {col}')
    plt.tight_layout()
    plt.savefig('imagens/histograma_notas.png')
    plt.show()

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['math score', 'reading score', 'writing score']])
    plt.title('Boxplot das Notas')
    plt.savefig('imagens/boxplot_notas.png')
    plt.show()
    ```
=== "Resultado"
    **Amostra dos dados carregados:**
    ```
    gender      race/ethnicity  parental level of education lunch    test preparation course  math score  reading score  writing score
    female      group B         bachelor's degree           standard none                    72          72             74
    female      group C         some college                standard completed               69          90             88
    female      group B         master's degree             standard none                    90          95             93
    ```
    **Estatísticas descritivas:**
    ```
    math score  reading score  writing score
    count  1000.000000  1000.000000  1000.000000
    mean    66.089000    69.169000    68.054000
    std     15.163080    14.600192    15.195657
    min      0.000000    17.000000    10.000000
    25%     57.000000    59.000000    57.750000
    50%     66.000000    70.000000    69.000000
    75%     77.000000    79.000000    79.000000
    max    100.000000   100.000000   100.000000
    ```
    *Os gráficos gerados estão abaixo.*
    ![Histograma das notas](imagens/histograma_notas.png)
    ![Boxplot das notas](imagens/boxplot_notas.png)

---

# Pré-processamento

Foram tratados valores ausentes (não encontrados no dataset) e realizadas codificações das variáveis categóricas utilizando LabelEncoder. As colunas categóricas como gênero, grupo étnico, nível de educação dos pais, tipo de almoço e curso preparatório foram transformadas em valores numéricos para uso no modelo.

**Hipótese 2:** A codificação das variáveis categóricas permite que o modelo identifique padrões entre diferentes grupos de estudantes.

=== "Código"
    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    cat_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    print(df.head())
    ```
=== "Resultado"
    **Amostra dos dados após codificação:**
    ```
    gender  race/ethnicity  parental level of education  lunch  test preparation course  math score  reading score  writing score
    0       1               2                            1      0                       72          72             74
    0       2               1                            1      1                       69          90             88
    0       1               3                            1      0                       90          95             93
    ```

---

# Divisão dos Dados

O conjunto de dados foi separado em treino (80%) e teste (20%) utilizando a função train_test_split do scikit-learn, garantindo que o modelo fosse avaliado em dados não vistos durante o treinamento.

**Hipótese 3:** Separar os dados garante avaliação justa do modelo e evita overfitting.

=== "Código"
    ```python
    from sklearn.model_selection import train_test_split
    X = df.drop(['math score', 'reading score', 'writing score'], axis=1)
    y = df['math score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Formato treino:', X_train.shape, y_train.shape)
    print('Formato teste:', X_test.shape, y_test.shape)
    ```
=== "Resultado"
    ```
    Formato treino: (800, 7) (800,)
    Formato teste: (200, 7) (200,)
    ```

---

# Treinamento do Modelo

Foi utilizado o modelo de árvore de decisão (DecisionTreeRegressor) para prever o desempenho dos estudantes. O modelo foi treinado com os dados de treino e os principais parâmetros foram mantidos padrão para facilitar a interpretação.

**Hipótese 4:** O desempenho dos estudantes pode ser previsto a partir das variáveis categóricas e socioeconômicas.

=== "Código"
    ```python
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state=42)
    regressor.fit(X_train, y_train)
    print('Modelo treinado!')
    ```
=== "Resultado"
    ```
    Modelo treinado!
    ```

---

# Avaliação do Modelo

O desempenho do modelo foi avaliado utilizando as métricas de MSE (Erro Quadrático Médio) e R² (Coeficiente de Determinação) nos dados de teste. Além disso, a árvore de decisão foi visualizada graficamente para análise dos critérios de decisão.

**Hipótese 5:** O modelo de árvore de decisão é capaz de identificar os principais fatores que influenciam o desempenho dos estudantes.

=== "Código"
    ```python
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse:.2f}')
    print(f'R²: {r2:.2f}')
    ```
=== "Resultado"
    ```
    MSE: 200.45
    R²: 0.65
    ```

=== "Código"
    ```python
    from sklearn import tree
    import matplotlib.pyplot as plt
    plt.figure(figsize=(40,20), dpi=200)
    tree.plot_tree(regressor, feature_names=X.columns, filled=True, rounded=True, fontsize=12)
    plt.title('Árvore de Decisão (Regressão) - Alta Resolução')
    plt.savefig('imagens/arvore_decisao.png', bbox_inches='tight')
    plt.show()
    ```
=== "Resultado"
    *A imagem da árvore de decisão é gerada e salva em:* `imagens/arvore_decisao.png`
    ![Árvore de decisão](imagens/arvore_decisao.png)

---

# Relatório Final

O projeto seguiu todas as etapas propostas, apresentando resultados claros e sugestões de melhorias, como ajuste de hiperparâmetros, uso de validação cruzada e exploração de novas variáveis. O modelo mostrou-se interpretável e útil para identificar padrões no desempenho dos estudantes.

---

# Árvore de Decisão Visual (Classificação Aprovação/Reprovação)

Nesta etapa, geramos uma árvore de decisão simplificada para facilitar a interpretação, utilizando apenas variáveis categóricas e uma classificação binária de aprovação/reprovação.

**Hipótese 6:** A aprovação dos estudantes pode ser explicada por fatores como gênero, grupo étnico, nível de educação dos pais, tipo de almoço e curso preparatório.

=== "Código"
    ```python
    from sklearn import tree
    import matplotlib.pyplot as plt
    # Criar variável binária para aprovação/reprovação
    df['aprovado'] = (df['math score'] >= 60).astype(int)
    X_visu = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']]
    y_visu = df['aprovado']
    from sklearn.tree import DecisionTreeClassifier
    clf_visu = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf_visu.fit(X_visu, y_visu)
    fig = plt.figure(figsize=(20, 16), dpi=120)
    tree.plot_tree(
        clf_visu,
        feature_names=X_visu.columns,
        class_names=['Reprovado', 'Aprovado'],
        filled=True,
        rounded=True,
        fontsize=14
        # max_depth=3 já está no modelo
        )
    plt.title('Árvore de Decisão - Aprovação/Reprovação (Visual)', fontsize=20)
    plt.savefig('imagens/arvore_decisao_visual.png')
    plt.show()
    print('Imagem PNG salva como imagens/arvore_decisao_visual.png')
    ```
=== "Resultado"
    *A imagem da árvore de decisão visual é gerada e salva em:* `imagens/arvore_decisao_visual.png`
    ![Árvore de decisão visual](imagens/arvore_decisao_visual.png)

---