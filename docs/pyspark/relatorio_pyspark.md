---
hide:
- toc
---

# Relatório PySpark - Cluster com Docker Compose

## 1. Configuração do Ambiente
- O cluster Spark foi configurado com Docker Compose, incluindo Spark Master, Spark Worker e Jupyter Notebook.
- As interfaces web do Spark Master, Worker e Jupyter foram acessadas e validadas.

**Insira aqui as imagens das interfaces do Spark Master, Worker e Jupyter Notebook:**

![Interface do Spark Master](imagens/spark_master.png)

![Interface do Worker](imagens/spark_master.png)

![Jupyter Notebook](imagens/spark_master.png)


## 2. Execução de Scripts PySpark
- O arquivo `StudentsPerformance.csv` foi utilizado para análise de dados.
- Foram realizadas operações de leitura, agregação, estatísticas e exportação de resultados.
- Gráficos foram gerados para visualização das médias de matemática por grupo étnico.

**Insira aqui prints dos códigos executados, resultados das células e gráficos gerados:**
![Código](imagens/spark_master.png)
![Resultado](imagens/spark_master.png)

![Código](imagens/spark_master.png)
![Resultado](imagens/spark_master.png)

![Código](imagens/spark_master.png)
![Resultado](imagens/spark_master.png)

![Código](imagens/spark_master.png)
![Resultado](imagens/spark_master.png)



## 3. Principais Resultados
- Médias das notas por gênero e grupo étnico calculadas com PySpark.
- Contagem de alunos por grupo étnico.
- Exportação dos resultados para CSV.
- Visualização gráfica das médias por grupo étnico.

**Insira aqui prints dos resultados finais e arquivos exportados:**
![CSV exportado](imagens/spark_master.png)


## 4. Desafios e Soluções
- Ajuste das imagens Docker para garantir o funcionamento das interfaces web.
- Configuração correta do ambiente para rodar PySpark no Jupyter Notebook do container.

## 5. Aprendizados
- Experiência prática com cluster Spark em ambiente Docker.
- Integração entre PySpark, Jupyter e análise de dados reais.
- Exportação e visualização de resultados para portfólio.

---