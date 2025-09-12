
# 1. Importação das Bibliotecas

Utilizamos pandas para manipulação de dados, numpy para operações matemáticas, matplotlib e seaborn para visualização gráfica, e scikit-learn para o modelo KNN.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
sns.set(style="whitegrid")
```

**🟢 Resultado**

```python
print('Bibliotecas importadas com sucesso!')
```
