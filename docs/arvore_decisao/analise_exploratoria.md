# 3. Visualizar Dados Básicos

Aqui verificamos o formato do dataset, os tipos de dados e se há valores nulos. Isso é fundamental para garantir a qualidade dos dados antes de qualquer análise.

=== "Código"
	```python
	print('Formato do dataset:', df.shape)
	df.info()
	print('\nValores nulos por coluna:')
	print(df.isnull().sum())
	```
=== "Resultado"
	O dataset possui 1000 linhas e 8 colunas, sem valores nulos. Isso indica que não é necessário tratamento de dados ausentes.
