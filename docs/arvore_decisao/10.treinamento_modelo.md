# 10. Treinamento do Modelo de Árvore de Decisão

Utilizamos o modelo DecisionTreeRegressor para prever o desempenho dos estudantes. O modelo aprende padrões nos dados de treino para realizar previsões sobre novos exemplos.

=== "Código"
	```python
	from sklearn.model_selection import GridSearchCV
	from sklearn.tree import DecisionTreeRegressor

	param_grid = {
		'max_depth': [3, 5, 10, 20, None],
		'min_samples_split': [2, 5, 10, 20],
		'min_samples_leaf': [1, 2, 4, 8],
		'max_features': [None, 'sqrt', 'log2']  # Removido 'auto' para evitar erro
	}

	grid_search = GridSearchCV(
		DecisionTreeRegressor(random_state=42),
		param_grid,
		cv=5,
		scoring='r2',
		n_jobs=-1
	)
	grid_search.fit(X_train, y_train)
	best_tree = grid_search.best_estimator_
	print('Melhores hiperparâmetros:', grid_search.best_params_)
	```
=== "Resultado"
	O modelo foi treinado com sucesso e está pronto para realizar previsões.
