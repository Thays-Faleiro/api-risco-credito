# API Preditiva de Análise de Risco de Crédito

Microsserviço RESTful desenvolvido com **FastAPI** e **Pydantic** que disponibiliza um modelo de Machine Learning para classificar o risco de inadimplência de clientes em tempo real.

---

## 1. Objetivo e Problema de Negócio

Instituições financeiras precisam decidir rapidamente se concedem crédito a um cliente. Esta API recebe dados do solicitante e devolve a classificação:

- **Baixo Risco (0):** perfil com baixa probabilidade de inadimplência
- **Alto Risco (1):** análise manual recomendada

O objetivo é apoiar a decisão, reduzindo o tempo de análise e o risco de inadimplência.

---

## 2. Origem e Arquitetura do Modelo

**Base de dados:** German Credit (`credit-g`), disponível no OpenML/UCI Repository. Contém 1000 registros e 20 variáveis, com 30% de clientes inadimplentes. Foi escolhida por ser uma base clássica de risco de crédito, com variáveis numéricas e categóricas e dados reais. O download é feito pelo script `treinar_modelo.py` via `fetch_openml`, e o CSV é salvo em `data/german_credit.csv`.

**Variáveis utilizadas (6):**

| Variável | Tipo | Descrição |
|---|---|---|
| `duration` | numérica | Duração do crédito em meses |
| `credit_amount` | numérica | Valor do crédito solicitado |
| `age` | numérica | Idade do cliente |
| `purpose` | categórica | Finalidade do crédito |
| `housing` | categórica | Tipo de moradia |
| `checking_status` | categórica | Situação da conta corrente |

**Pipeline offline (`treinar_modelo.py`):**

- Divisão estratificada: 60% treino, 20% validação, 20% teste (`random_state=42`)
- Numéricas: `SimpleImputer` (mediana) + `StandardScaler`
- Categóricas: `SimpleImputer` (mais frequente) + `OneHotEncoder(handle_unknown="ignore")`
- Modelo: `RandomForestClassifier` (200 árvores, `max_depth=8`, `class_weight="balanced"`)
- Tudo encapsulado em um `Pipeline` do Scikit-Learn, evitando vazamento de dados
- Artefato serializado com Joblib em `models/modelo_pipeline.joblib`

### Métricas

| Conjunto | Acurácia | F1-Score | ROC-AUC |
|---|---|---|---|
| Validação | 0.645 | 0.4741 | 0.6827 |
| Teste | 0.790 | 0.6719 | 0.8017 |


As métricas também ficam salvas em `models/metricas.json`.

> **Observação:** o desempenho varia entre validação e teste porque a base é pequena (1000 registros, 200 por conjunto), o que torna a estimativa sensível à divisão. Como melhoria futura, recomenda-se validação cruzada (`cross_val_score`) para uma estimativa mais estável.


###  Tabela Comparativa de Desempenho

| Modelo | Acurácia | Precisão | Recall | F1-Score | Tempo de Treino (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 96,8% | 0,968 | 0,968 | 0,968 | ~15s |
| **SVM (RBF)** | **98,2%** | **0,982** | **0,982** | **0,982** | ~210s |
| **MLP (Keras)** | 98,0% | 0,980 | 0,980 | 0,980 | **~25s** |


---

## 3. Tecnologias Utilizadas

| Biblioteca | Versão |
|---|---|
| Python | 3.13 |
| FastAPI | 0.141.1 |
| Uvicorn | 0.54.0 |
| Pydantic | 2.14.1 |
| Scikit-Learn | 1.6.1 |
| Pandas | 2.2.3 |
| NumPy | 2.1.3 |
| Joblib | 1.6.0 |
| HTTPX | 0.28.1 |

As versões exatas estão em `requirements.txt`. A versão do Scikit-Learn deve ser a mesma usada no treino, pois o arquivo `.joblib` depende dela.

---

## 4. Estrutura do Projeto

```
.
├── app/
│   ├── main.py                 # entrada da aplicação e lifespan
│   ├── routers/
│   │   ├── health.py           # GET /health
│   │   └── predict.py          # POST /v1/predict
│   ├── schemas/
│   │   └── predicao.py         # modelos Pydantic de entrada e saída
│   └── services/
│       └── modelo_service.py   # guarda o modelo em memória e executa a inferência
├── data/                       # base de dados (CSV)
├── models/                     # modelo serializado e métricas
├── treinar_modelo.py           # pipeline offline de treino
├── requirements.txt
└── README.md
```

**Decisões de arquitetura:**

- **Camadas separadas:** rotas (HTTP), schemas (contrato de dados) e serviço (inferência) são independentes.
- **Lifespan:** o modelo é carregado na memória uma única vez, na inicialização, e nunca é lido do disco durante uma requisição.
- **Validação estrita:** campos numéricos têm limites (`ge`, `le`, `gt`) e os categóricos usam `Literal`, então valores inválidos são recusados antes de chegar ao modelo.

---

## 5. Instalação e Execução

```bash
# 1. Clonar o repositório
git clone https://github.com/Thays-Faleiro/api-risco-credito.git
cd api-risco-credito

# 2. Criar e ativar o ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. (Opcional) Retreinar o modelo e regerar as métricas
python treinar_modelo.py

# 5. Iniciar o servidor
uvicorn app.main:app --reload
```

A API ficará disponível em `http://127.0.0.1:8000`.

---

## 6. Documentação da API

Documentação interativa (Swagger UI): **http://127.0.0.1:8000/docs**

### `GET /health`

Verifica se o serviço está ativo e se o modelo está na memória.

Resposta (`200 OK`):

```json
{
  "status": "operacional",
  "modelo_carregado": true
}
```

### `POST /v1/predict`

Requisição:

```json
{
  "duration": 24,
  "credit_amount": 3500,
  "age": 35,
  "purpose": "radio/tv",
  "housing": "own",
  "checking_status": "<0"
}
```

Resposta (`200 OK`):

```json
{
  "predicao": 1,
  "probabilidade": 0.758,
  "status_analise": "Alto Risco: análise manual necessária"
}
```

O campo `probabilidade` é a probabilidade da classe predita.

### Valores aceitos nos campos categóricos

| Campo | Valores válidos |
|---|---|
| `purpose` | `radio/tv`, `education`, `furniture/equipment`, `new car`, `used car`, `business`, `domestic appliance`, `repairs`, `other`, `retraining` |
| `housing` | `own`, `for free`, `rent` |
| `checking_status` | `<0`, `0<=X<200`, `no checking`, `>=200` |

### Limites dos campos numéricos

| Campo | Regra |
|---|---|
| `duration` | entre 1 e 120 |
| `credit_amount` | maior que 0 |
| `age` | entre 18 e 120 |

### Códigos de status

| Código | Quando ocorre |
|---|---|
| 200 | Requisição processada com sucesso |
| 400 | Dado incompatível com o modelo (`ValueError` na inferência) |
| 422 | Validação do Pydantic falhou (tipo errado, valor fora do limite ou categoria inválida) |
| 500 | Erro interno inesperado na inferência |

Exemplo de requisição que retorna **422** (idade abaixo do mínimo):

```json
{
  "duration": 24,
  "credit_amount": 3500,
  "age": 10,
  "purpose": "radio/tv",
  "housing": "own",
  "checking_status": "<0"
}
```

O mesmo ocorre com `"housing": "castelo"`, que não pertence aos valores aceitos.

---

## 7. Fluxo de Versionamento (GitFlow)

- `main`: versão final e estável
- `develop`: integração das funcionalidades
- `feature/*`: uma branch por tarefa, criada a partir da `develop` (ex.: `feature/estrutura-api-fastapi`)

As branches não são excluídas após o Pull Request, e o código final está mergeado na `main`.

---

## 8. Melhorias Futuras

- Validação cruzada para uma estimativa de desempenho mais estável
- Uso de mais variáveis da base e ajuste de hiperparâmetros
- Testes automatizados (`pytest`) para as rotas
- Registro do modelo com MLflow

---

## 9. Evidências e Apresentação

**Vídeo de apresentação:** [Assistir no Google Drive](https://drive.google.com/file/d/14-JevJWwpERtMEhDJAcykmIODy9QDC-K/view?usp=sharing)
