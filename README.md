# 📊 Predictive Markets Pipeline

Coleta dados do **Polymarket** e **Kalshi** toda noite, faz matching semântico entre os mercados e salva os arquivos resultantes no repositório.

## 📁 Saídas (pasta `data/`)

| Arquivo | Descrição |
|---|---|
| `poly_df.parquet` / `.json` | Todos os eventos abertos do Polymarket |
| `df_kalshi_filtered.parquet` / `.json` | Mercados ativos da Kalshi |
| `kalshi_poly_df.parquet` / `.json` | Mercados matched entre Kalshi e Polymarket com preços |

## ⏰ Agendamento

O pipeline roda automaticamente toda noite às **02:00 UTC (23:00 BRT)** via GitHub Actions.

Você também pode disparar manualmente clicando em **Actions → Nightly Predictive Markets Pipeline → Run workflow**.

## 📥 Como baixar os dados na sua máquina local

### Opção 1 — Git pull (recomendado)
```bash
git clone https://github.com/SEU_USUARIO/predictive-markets-pipeline.git
cd predictive-markets-pipeline
# A pasta data/ sempre terá os dados mais recentes
```
Depois de cada execução noturna, basta:
```bash
git pull
```

### Opção 2 — Download do Artifact
1. Vá em **Actions** no GitHub
2. Clique na última execução
3. Baixe o artifact `market-data-*`

### Opção 3 — Download via Python (sem clonar o repositório)
```python
import pandas as pd

REPO   = "SEU_USUARIO/predictive-markets-pipeline"
BRANCH = "main"
BASE   = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data"

kalshi_poly_df      = pd.read_json(f"{BASE}/kalshi_poly_df.json")
df_kalshi_filtered  = pd.read_json(f"{BASE}/df_kalshi_filtered.json")
poly_df             = pd.read_json(f"{BASE}/poly_df.json")
```

## ▶️ Rodar localmente

```bash
pip install -r requirements.txt
python pipeline.py
```

## 📦 Dependências

Ver `requirements.txt`.
