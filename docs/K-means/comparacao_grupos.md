# 10. Comparação de Grupos

A comparação de médias por gênero e grupo étnico mostra diferenças entre desempenho em leitura, escrita e matemática. No contexto do K-means, podemos analisar como os clusters se distribuem entre esses grupos.

**🟢 Resultado**

Média de matemática por cluster:

| cluster | math score |
|---------|------------|
| 0       | 63.37      |
| 1       | 68.73      |

Distribuição dos clusters por grupo étnico:

| race/ethnicity | cluster 0 | cluster 1 |
|---------------|-----------|-----------|
| 0             | 61        | 39        |
| 1             | 63        | 37        |
| 2             | 64        | 44        |
| 3             | 67        | 37        |
| 4             | 74        | 42        |

```python
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x='cluster', y='math score', data=df_encoded, ci=None, ax=ax)
plt.title('Média de Matemática por Cluster')
plt.savefig('imagens/barplot_cluster.png')
plt.close()
fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(x='race/ethnicity', hue='cluster', data=df_encoded, ax=ax)
plt.title('Distribuição dos Clusters por Grupo Étnico')
plt.savefig('imagens/barplot_cluster_etnia.png')
plt.close()
```

![](imagens/barplot_cluster.png)
![](imagens/barplot_cluster_etnia.png)

> 💡 Os gráficos mostram como os clusters formados pelo K-means se distribuem entre os grupos, permitindo interpretações sobre padrões de desempenho.