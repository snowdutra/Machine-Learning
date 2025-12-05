---
hide:
- toc
---

# Relatório PySpark - Cluster com Docker Compose

## 1. Configuração do Ambiente
- O cluster Spark foi configurado com Docker Compose, incluindo Spark Master, Spark Worker e Jupyter Notebook.
- As interfaces web do Spark Master, Worker e Jupyter foram acessadas e validadas.

### Interface do Spark Master

![Interface do Spark Master](imagens/master.png)

---

### Interface do Spark Worker

![Interface do Worker](imagens/worker.png)

---

### Interface do Jupyter Notebook

![Jupyter Notebook](imagens/jupyter.png)

---

## 2. Execução de Scripts PySpark
- O arquivo `StudentsPerformance.csv` foi utilizado para análise de dados.
- Foram realizadas operações de leitura, agregação, estatísticas e exportação de resultados.
- Gráficos foram gerados para visualização das médias de matemática por grupo étnico.

### Código 01
![Código](imagens/codigo_01.png)



### Resultado 01
![Resultado](imagens/resultado_01.png)

---

### Código 02
![Código](imagens/codigo_02.png)



### Resultado 02
![Resultado](imagens/resultado_02.png)

---

### Código 03
![Código](imagens/codigo_03.png)



### Resultado 03
![Resultado](imagens/resultado_03.png)

---

### Código 04
![Código](imagens/codigo_04.png)



### Resultado 04
![Resultado](imagens/resultado_04.png)

---

## 3. Principais Resultados
- Médias das notas por gênero e grupo étnico calculadas com PySpark.
- Contagem de alunos por grupo étnico.
- Exportação dos resultados para CSV.
- Visualização gráfica das médias por grupo étnico.

## 4. Desafios e Soluções
- Ajuste das imagens Docker para garantir o funcionamento das interfaces web.
- Configuração correta do ambiente para rodar PySpark no Jupyter Notebook do container.

## 5. Aprendizados
- Experiência prática com cluster Spark em ambiente Docker.
- Integração entre PySpark, Jupyter e análise de dados reais.
- Exportação e visualização de resultados para portfólio.

---