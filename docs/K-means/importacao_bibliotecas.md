# 1. Importação das Bibliotecas

Utilizamos pandas para manipulação de dados, numpy para operações matemáticas, matplotlib e seaborn para visualização gráfica, e scikit-learn para o modelo K-means.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sns.set(style="whitegrid")
```

**🟢 Resultado**

```python
print('Bibliotecas importadas com sucesso!')
```