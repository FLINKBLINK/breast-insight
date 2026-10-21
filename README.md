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

## 5. Análises disponíveis

- **Machine Learning:** modelos supervisionados para fins educacionais.
- **DBSCAN:** agrupamento não supervisionado com padronização das variáveis, gráfico k-distance para auxiliar a escolha de `eps`, visualização com PCA, contagem de grupos e ruído e comparação exploratória posterior com o diagnóstico original.
- **Bootstrap:** estima a diferença entre as médias dos grupos maligno e benigno por reamostragem com reposição; o intervalo padrão é calculado por percentis.

Na página DBSCAN, a coluna de diagnóstico não é utilizada para formar os grupos. O gráfico k-distance auxilia a selecionar `eps`; ele não é o mesmo que o cotovelo de WSS do K-Means.

## 6. Aviso

O projeto é educacional. Resultados de modelos e agrupamentos não constituem diagnóstico nem validação clínica e não devem ser utilizados para decisões médicas.
