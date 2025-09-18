# 2. Carregamento do Dataset

O dataset utilizado é o mesmo do projeto de árvore de decisão e KNN, contendo informações sobre desempenho de estudantes em exames. As colunas incluem gênero, grupo étnico, nível de educação dos pais, tipo de almoço, curso preparatório e notas em matemática, leitura e escrita.

```python
import kagglehub
# Baixar o dataset do Kaggle
path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
csv_path = path + "/StudentsPerformance.csv"
df = pd.read_csv(csv_path)
df.head()
```

**🟢 Resultado**

| gender | race/ethnicity | parental level of education | lunch        | test preparation course | math score | reading score | writing score |
|--------|----------------|----------------------------|--------------|------------------------|------------|---------------|---------------|
| female | group B        | bachelor's degree          | standard     | none                   | 72         | 72            | 74            |
| female | group C        | some college               | standard     | completed              | 69         | 90            | 88            |
| female | group B        | master's degree            | standard     | none                   | 90         | 95            | 93            |
| male   | group A        | associate's degree         | free/reduced | none                   | 47         | 57            | 44            |
| male   | group C        | some college               | standard     | none                   | 76         | 78            | 75            |

> 💡 As primeiras linhas mostram a diversidade de grupos e notas presentes no dataset.