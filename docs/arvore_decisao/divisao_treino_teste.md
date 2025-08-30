# 9. Divisão dos Dados em Treino e Teste

Dividimos o dataset em treino (80%) e teste (20%) para garantir que o modelo seja avaliado em dados não vistos durante o treinamento, evitando overfitting.

=== "Código"
	```python
	from sklearn.model_selection import train_test_split

	# Selecionar features e target
	X = df.drop(['math score', 'reading score', 'writing score'], axis=1)
	y = df['math score']  # Exemplo: prever nota de matemática (pode ajustar para classificação)

	# Para classificação, pode criar uma coluna de aprovação/reprovação, por exemplo:
	# y = (df['math score'] >= 60).astype(int)

	# Dividir em treino e teste
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	print('Formato treino:', X_train.shape, y_train.shape)
	print('Formato teste:', X_test.shape, y_test.shape)
	```
=== "Resultado"
	O conjunto de treino possui 800 exemplos e o de teste 200, garantindo avaliação justa do modelo.
