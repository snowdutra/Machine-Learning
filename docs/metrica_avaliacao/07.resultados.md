---
hide:
- toc
---

# 03. Resultados e Interpretação

## Avaliação do KNN

=== "Código"
   ```python
   import pandas as pd
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

   y_test = pd.read_csv('knn_y_test.csv').values.ravel()
   y_pred = pd.read_csv('knn_y_pred.csv').values.ravel()

   print(f'Acurácia: {accuracy_score(y_test, y_pred):.2f}')
   print(f'Precisão: {precision_score(y_test, y_pred):.2f}')
   print(f'Recall: {recall_score(y_test, y_pred):.2f}')
   print(f'F1-Score: {f1_score(y_test, y_pred):.2f}')
   print('Matriz de Confusão:')
   print(confusion_matrix(y_test, y_pred))
   print('Relatório de Classificação:')
   print(classification_report(y_test, y_pred))
   ```
=== "Resultado"
   ```
   Acurácia: 0.62
   Precisão: 0.68
   Recall: 0.79
   F1-Score: 0.73
   Matriz de Confusão:
   [[ 31  74]
   [ 40 155]]
   Relatório de Classificação:
              precision    recall  f1-score   support
           0       0.44      0.30      0.35       105
           1       0.68      0.79      0.73       195
    accuracy                           0.62       300
   macro avg       0.56      0.55      0.54       300
   weighted avg    0.59      0.62      0.60       300
   ```

## Avaliação do K-Means

=== "Código"
   ```python
   import pandas as pd
   from sklearn.metrics import silhouette_score

   X = pd.read_csv('kmeans_X.csv').values
   clusters = pd.read_csv('kmeans_clusters.csv').values.ravel()

   print(f'Silhouette Score: {silhouette_score(X, clusters):.2f}')
   ```
=== "Resultado"
   ```
   Silhouette Score: 0.47
   ```

## Interpretação

A partir dessas métricas, é possível identificar se o modelo está adequado ao problema, se há necessidade de ajustes ou se outro algoritmo pode ser mais indicado. Sempre analise os resultados considerando o contexto dos dados e o objetivo do projeto.
