# 6. Pré-processamento dos Dados

O processo de codificação das variáveis categóricas e normalização das notas prepara os dados para o modelo K-means. Todas as variáveis categóricas foram transformadas em números inteiros e as notas normalizadas.

**🟢 Resultado**

| gender | race/ethnicity | parental level of education | lunch | test preparation course | math score | reading score | writing score |
|--------|----------------|----------------------------|-------|------------------------|------------|---------------|---------------|
| -1.22  | -1.18          | 1.75                       | 0.66  | -0.64                  | 72         | 72            | 74            |
| -1.22  | 0.13           | -0.13                      | 0.66  | 1.56                   | 69         | 90            | 88            |
| -1.22  | -1.18          | 2.63                       | 0.66  | -0.64                  | 90         | 95            | 93            |
| 0.82   | -2.49          | -1.00                      | -1.51 | -0.64                  | 47         | 57            | 44            |
| 0.82   | 0.13           | -0.13                      | 0.66  | -0.64                  | 76         | 78            | 75            |

> As variáveis categóricas foram codificadas e as features normalizadas, garantindo dados prontos para o modelo K-means.