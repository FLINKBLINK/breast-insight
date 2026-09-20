# 🧬 Breast Insight

Plataforma educacional para análise da base Breast Cancer Wisconsin Diagnostic.

## 1. Estrutura da base

Coloque o arquivo `data.csv` dentro da pasta `data/`.

## 2. Instalação

```bash
python -m venv .venv
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

## 3. Configuração

Copie `.env.example` para `.env`.

A aplicação usa SQLite por padrão:

```env
DATABASE_URL=sqlite:///data/breast_insight.db
```

## 4. Executar

```bash
streamlit run app.py
```

Abra a página `Database` e clique em `Importar CSV para SQLite`.

## 5. Observação estatística

O Bootstrap implementado estima a diferença entre as médias do grupo maligno e benigno, usando reamostragem com reposição. O intervalo padrão é calculado por percentis.

O modelo de Machine Learning é educacional e não deve ser usado para diagnóstico clínico.
