# 10. Treinamento do Modelo KNN

O modelo KNN é treinado utilizando os dados de treino. O processo é similar ao da árvore de decisão, mas agora com o algoritmo KNN.

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

**🟢 Resultado**

Modelo KNN treinado com sucesso nos dados de treino.

> 🤖 O modelo está pronto para realizar previsões e ser avaliado.
