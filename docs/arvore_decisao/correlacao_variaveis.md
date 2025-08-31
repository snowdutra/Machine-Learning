# 6. Correlação entre Variáveis

Calculamos a matriz de correlação entre as notas. Isso permite identificar se há relação entre o desempenho em matemática, leitura e escrita.

=== "Código"
	```python
	corr = df[['math score', 'reading score', 'writing score']].corr()
	print('Matriz de correlação:')
	print(corr)

	plt.figure(figsize=(8, 6))
	sns.heatmap(corr, annot=True, cmap='Blues')
	plt.title('Heatmap de Correlação entre Notas')
	plt.savefig('imagens/heatmap_correlacao.png')
	plt.show()
	from IPython.display import Image, display
	display(Image(filename='imagens/heatmap_correlacao.png'))
	```
=== "Resultado"
	As notas de leitura e escrita têm correlação muito alta (acima de 0.95), indicando que estudantes que vão bem em uma tendem a ir bem na outra. Matemática tem correlação moderada com as demais.
	![Heatmap de Correlação entre Notas](imagens/heatmap_correlacao.png)
