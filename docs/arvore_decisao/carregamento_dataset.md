# 2. Carregar o Dataset

O dataset foi obtido do Kaggle e contém informações sobre desempenho de estudantes em exames. As colunas incluem gênero, grupo étnico (representado por rótulos genéricos como "group A", "group B"), nível de educação dos pais, tipo de almoço, curso preparatório e notas em matemática, leitura e escrita.

> **Nota sobre os grupos étnicos:** Os nomes dos grupos (A, B, C, D, E) são fictícios e não correspondem a etnias reais. O Kaggle utiliza esses rótulos para preservar o anonimato dos participantes, portanto não é possível identificar as etnias reais.

=== "Código"
	```python
	import kagglehub

	# Baixar o dataset do Kaggle
	path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
	print("Path to dataset files:", path)

	# Carregar o arquivo CSV
	csv_path = path + "/StudentsPerformance.csv"
	df = pd.read_csv(csv_path)
	df.head()
	```
=== "Resultado"
	**Amostra dos dados carregados:**
	```
	gender      race/ethnicity  parental level of education lunch    test preparation course  math score  reading score  writing score
	female      group B         bachelor's degree           standard none                    72          72             74
	female      group C         some college                standard completed               69          90             88
	female      group B         master's degree             standard none                    90          95             93
	```
