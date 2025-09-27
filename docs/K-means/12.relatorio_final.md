# 11. Relatório Final

Este projeto aplicou o algoritmo K-means para agrupar estudantes de acordo com seu desempenho, seguindo o mesmo padrão dos projetos anteriores. As etapas incluíram análise exploratória, visualização das distribuições das notas, análise de correlação, comparação entre grupos, pré-processamento, treinamento e avaliação do modelo de clustering.

**Principais Resultados:**
- O K-means permitiu identificar grupos de estudantes com padrões de desempenho semelhantes.
- O silhouette score foi utilizado para avaliar a qualidade dos clusters.
- A visualização dos clusters mostrou separação razoável entre os grupos.

**Métricas Finais:**

| Métrica           | Valor |
|-------------------|-------|
| Silhouette Score  | 0.32  |

**Interpretação:**
- O K-means conseguiu separar os estudantes em grupos, mas a separação não foi perfeita (silhouette score moderado).
- Os clusters podem refletir diferenças de desempenho, mas também podem ser influenciados por correlações entre as notas.

**Observações:**
- O K-means é sensível à escolha de K e à escala das variáveis, por isso o pré-processamento foi fundamental.
- Recomenda-se testar outros valores de K, diferentes inicializações e outros algoritmos de clustering para buscar melhorias.
- A comparação com métodos supervisionados (como KNN) pode ajudar a entender as limitações e vantagens do agrupamento não supervisionado.