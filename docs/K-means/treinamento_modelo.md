# Treinamento do Modelo

O modelo K-means é treinado utilizando todo o conjunto de dados, pois é um método não supervisionado. O número de clusters pode ser definido com base no método do cotovelo (elbow method).

```python
from sklearn.cluster import KMeans
k = 2  # Exemplo
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
```

**🟢 Resultado**

Modelo K-means treinado com sucesso e clusters atribuídos aos dados.

> 🤖 O modelo está pronto para análise dos agrupamentos e avaliação por métricas de clusterização.