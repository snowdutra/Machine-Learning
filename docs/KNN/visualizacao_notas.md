# 4. Análise Exploratória dos Dados

Análise estatística das notas dos estudantes, incluindo média, desvio padrão, valores mínimos e máximos.


**🟢 Resultado**

|        | math score | reading score | writing score |
|--------|------------|--------------|--------------|
| count  | 1000.00    | 1000.00      | 1000.00      |
| mean   | 66.09      | 69.17        | 68.05        |
| std    | 15.16      | 14.60        | 15.19        |
| min    | 0.00       | 17.00        | 10.00        |
| 25%    | 57.00      | 59.00        | 57.75        |
| 50%    | 66.00      | 70.00        | 69.00        |
| 75%    | 77.00      | 79.00        | 79.00        |
| max    | 100.00     | 100.00       | 100.00       |

---

# 5. Visualização de Distribuições das Notas

Histogramas e boxplots para visualizar a distribuição das notas.

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, col in enumerate(['math score', 'reading score', 'writing score']):
    sns.histplot(df[col], bins=20, ax=axes[idx], kde=True)
    axes[idx].set_title(f'Distribuição: {col}')
plt.tight_layout()
plt.savefig('imagens/histograma_notas.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['math score', 'reading score', 'writing score']])
plt.title('Boxplot das Notas')
plt.savefig('imagens/boxplot_notas.png')
plt.show()
```

![](imagens/histograma_notas.png)
![](imagens/boxplot_notas.png)

> Os histogramas mostram distribuição aproximadamente normal, com alguns outliers evidenciados no boxplot.
=== "Resultado"

<div style="font-family:monospace; font-size:15px; background:#f8f8f8; border-radius:8px; padding:16px; width:max-content;">
<b>Estatísticas das notas</b>
<table>
    <tr><th style="text-align:right;">&nbsp;</th><th style="text-align:center;">math score</th><th style="text-align:center;">reading score</th><th style="text-align:center;">writing score</th></tr>
    <tr><td style="text-align:right;">count</td><td style="text-align:center;">1000</td><td style="text-align:center;">1000</td><td style="text-align:center;">1000</td></tr>
    <tr><td style="text-align:right;">mean</td><td style="text-align:center;">66.09</td><td style="text-align:center;">69.17</td><td style="text-align:center;">68.05</td></tr>
    <tr><td style="text-align:right;">std</td><td style="text-align:center;">15.16</td><td style="text-align:center;">14.60</td><td style="text-align:center;">15.19</td></tr>
    <tr><td style="text-align:right;">min</td><td style="text-align:center;">0</td><td style="text-align:center;">17</td><td style="text-align:center;">10</td></tr>
    <tr><td style="text-align:right;">25%</td><td style="text-align:center;">57</td><td style="text-align:center;">59</td><td style="text-align:center;">57.75</td></tr>
    <tr><td style="text-align:right;">50%</td><td style="text-align:center;">66</td><td style="text-align:center;">70</td><td style="text-align:center;">69</td></tr>
    <tr><td style="text-align:right;">75%</td><td style="text-align:center;">77</td><td style="text-align:center;">79</td><td style="text-align:center;">79</td></tr>
    <tr><td style="text-align:right;">max</td><td style="text-align:center;">100</td><td style="text-align:center;">100</td><td style="text-align:center;">100</td></tr>
</table>
</div>

<div style="font-size:15px; margin-top:12px; background:#f8f8f8; border-radius:8px; padding:16px; width:max-content;">
<b>Interpretação dos gráficos</b>
<ul style="margin-left:16px;">
    <li>As distribuições das notas são aproximadamente normais, com leve assimetria em matemática.</li>
    <li>O boxplot evidencia alguns outliers, principalmente nas notas mais baixas.</li>
    <li>A maioria dos estudantes apresenta desempenho entre 60 e 80 pontos.</li>
</ul>
</div>


# Visualização das Notas

Esta etapa foi detalhada no projeto da árvore de decisão. Consulte:
[Visualização das Notas - Árvore de Decisão](https://snowdutra.github.io/Machine-Learning/arvore_decisao/visualizacao_notas.md)
