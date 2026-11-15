# Classificação de Gênero Vocal · Inteligência Artificial & Machine Learning

Checkpoint 2 · 2026.1 · Engenharia de Software · FIAP

---

## Esta entrega: Inteligência Artificial & Machine Learning

Classificação binária de gênero vocal com foco em regularização e análise crítica de funções de perda.

- **Parte A:** análise exploratória de 20 features acústicas extraídas de vozes humanas, identificação de multicolinearidade e padronização. Busca em grade com validação cruzada k=5 sobre regressão logística com penalização L1 (Lasso) e L2 (Ridge), comparando o efeito de cada regularizador na seleção de variáveis e na estabilidade dos coeficientes.
- **Parte B:** implementação manual de regressão logística treinada com RMSE como função de perda (gradiente descendente), comparação com log-loss via curvas ROC, Precisão–Recall e calibração, discutindo por que RMSE é uma escolha inadequada para classificação probabilística.

**Dados:** [Voice Gender Recognition](https://raw.githubusercontent.com/primaryobjects/voice-gender/master/voice.csv), com 3.168 amostras, 20 features acústicas e alvo binário balanceado (male/female). O notebook lê o arquivo direto do link.

| Entregável | Onde |
|---|---|
| Notebook | [`CP2-AI.ipynb`](CP2-AI.ipynb) |

## Como executar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook CP2-AI.ipynb
```

---

## Tecnologias

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)](https://matplotlib.org)
[![seaborn](https://img.shields.io/badge/seaborn-4C72B0?style=flat-square)](https://seaborn.pydata.org)

---

## Referências

| # | Referência | Aplicação no projeto |
|---|---|---|
| 1 | [primaryobjects — Voice Gender Recognition](https://raw.githubusercontent.com/primaryobjects/voice-gender/master/voice.csv) | Base de dados da atividade |
| 2 | [scikit-learn — `LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | Regressão logística com penalização L1 e L2 |
| 3 | [scikit-learn — `GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) | Busca em grade com validação cruzada |
