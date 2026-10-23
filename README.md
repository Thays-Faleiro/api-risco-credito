# API Preditiva de Análise de Risco de Crédito

Microsserviço RESTful de alta performance desenvolvido com **FastAPI** e **Pydantic** para inferências de Machine Learning em tempo real.

---

##  1. Objetivo e Problema de Negócio
A aplicação automatiza a classificação do perfil de clientes de uma instituição financeira em **Baixo Risco** (crédito aprovado automaticamente) ou **Alto Risco** (análise manual necessária). O objetivo é otimizar o tempo de concessão de crédito e reduzir a inadimplência.

---

## 2. Origem e Arquitetura do Modelo
- **Base de Dados:** Dataset sintético representando variáveis socioeconômicas (`idade`, `renda_mensal`, `score_credito`, `tipo_moradia`).
- **Pré-processamento:** `StandardScaler` para variáveis numéricas e `OneHotEncoder` para variáveis categóricas via `ColumnTransformer`.
- **Modelo:** `RandomForestClassifier` encapsulado num `Pipeline` do Scikit-Learn e serializado via `Joblib`.

---

##  3. Tecnologias Utilizadas
- **Linguagem:** Python 3.10+
- **Framework Web:** FastAPI & Uvicorn
- **Validação de Dados:** Pydantic v2
- **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib

---

## 4. Como Executar o Projeto Localmente

### Pré-requisitos
Certifique-se de ter o Python 3.10+ instalado em sua máquina.

### Passos de Instalação e Execução
1. **Clonar o repositório:**
   ```bash
   git clone [https://github.com/Thays-Faleiro/api-risco-credito.git](https://github.com/Thays-Faleiro/api-risco-credito.git)
   cd api-risco-credito




   pip install -r requirements.txt



   uvicorn app.main:app --reload
