# TransmutPySpark

**Mutation testing para pipelines PySpark com a DataFrame API.**

TransmutPySpark avalia a qualidade dos seus testes unitários aplicando mutações cirúrgicas no código PySpark e verificando se os testes detectam as mudanças. O resultado é um **Mutation Score** — a porcentagem de mutantes que seus testes conseguiram matar.

---

## Como funciona

```
Seu código ETL          Seus testes unitários
     │                         │
     └──────────┬──────────────┘
                ▼
        [ AST Analyzer ]
        Descobre funções elegíveis
        Ignora main(), I/O puro, dunders
        Mapeia test → função
                ▼
        [ Operadores de Mutação ]
        Gera variantes do código
        (negações, substituições, trocas)
                ▼
        [ Test Runner ]
        Executa pytest para cada mutante
        Em paralelo, isolado por subprocess
                ▼
        [ Reporter ]
        Calcula Mutation Score
        Gera report.html com diffs
```

A ferramenta nunca modifica seus arquivos originais. Cada mutante é um arquivo separado executado em sandbox isolado.

---

## Instalação

```bash
# Recomendado: pipx (gerencia o venv automaticamente)
pipx install transmutpyspark

# Ou com pip
pip install transmutpyspark
```

> **Requisito:** Java 8+ instalado (necessário para o PySpark).
> No Ubuntu/Debian: `sudo apt install default-jdk-headless`

---

## Uso rápido

### 1. Inicializar a configuração

Na raiz do seu projeto ETL:

```bash
transmut init --src etl_code/ --tests tests/
```

Isso cria um `transmut.toml`:

```toml
[transmut]
source_dirs   = ["etl_code/"]
tests_dirs    = ["tests/"]
operators     = ["MTR", "NFTP", "ATR", "UTS"]
workspace_dir = "."
```

### 2. Executar

```bash
transmut run
```

Saída no terminal:

```
TransmutPySpark — Mutation Testing para pipelines PySpark
──────────────────────────────────────────────────────────
...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mutation Score : 73%  (97 killed / 133 total)
  Sobreviventes  : 36 mutante(s) não detectados
  Relatório      : TransmutPysparkOutput/report.html
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 3. Ver o relatório

```bash
transmut show
```

Abre o `report.html` no browser com o breakdown completo por arquivo fonte e operador.

---

## Formas de configurar

### Flags diretas (sem arquivo de config)

```bash
transmut run --src etl_code/ --tests tests/
transmut run --src etl_code/transforms.py --tests tests/test_transforms.py
transmut run --src etl_code/ --tests tests/ --operators NFTP MTR
```

### transmut.toml (recomendado para uso contínuo)

```toml
[transmut]
source_dirs   = ["etl_code/", "transformations/"]
tests_dirs    = ["tests/"]
operators     = ["MTR", "NFTP", "ATR", "UTS"]
workspace_dir = "."
```

```bash
transmut run   # detecta o toml automaticamente
```

### config.txt (formato legado)

```
program_path   = etl_code/transforms.py
tests_path     = tests/test_transforms.py
operators_list = MTR, NFTP, ATR, UTS
workspace_dir  = .
```

```bash
transmut run --config config.txt
```

---

## Operadores de mutação

### MTR — Mapping Transformation Replacement

Substitui expressões em transformações de mapeamento por valores limite.

```python
# Original
df.withColumn("revenue", col("price") * col("qty"))

# Mutantes gerados
df.withColumn("revenue", 0)
df.withColumn("revenue", 1)
df.withColumn("revenue", None)
df.withColumn("revenue", "")
```

**Detecta:** falhas em cálculos e transformações de colunas.

---

### NFTP — Negation of Filter/where Transformation Predicate

Inverte ou nega predicados em operações de filtro.

```python
# Original
df.filter(col("status") == "active")

# Mutantes gerados
df.filter(~(col("status") == "active"))   # negação total
df.filter(col("status") != "active")      # inversão de operador
```

**Detecta:** falhas na lógica de filtragem de dados.

---

### ATR — Aggregation Transformation Replacement

Substitui funções de agregação por equivalentes semânticos distintos.

```python
# Original
df.groupBy("region").agg(F.sum("amount"))

# Mutantes gerados
df.groupBy("region").agg(F.avg("amount"))
df.groupBy("region").agg(F.max("amount"))
df.groupBy("region").agg(F.count("amount"))
```

**Detecta:** testes que não verificam a função de agregação correta.

---

### UTS — Utility Transformation Substitution

Troca a ordem de transformações independentes consecutivas.

```python
# Original
df.filter(...).select(...)   # Par independente

# Mutante gerado
df.select(...).filter(...)   # Ordem trocada
```

**Detecta:** dependências ocultas entre transformações na pipeline.

---

## Estrutura de saída

Após executar, o diretório `TransmutPysparkOutput/` é criado:

```
TransmutPysparkOutput/
├── report.html          ← relatório visual completo
├── mutants/
│   ├── atr/             ← mutantes do arquivo atr.py
│   │   ├── ATR/         ← gerados pelo operador ATR
│   │   └── MTR/         ← gerados pelo operador MTR
│   ├── mtr/
│   │   └── MTR/
│   ├── nftp/
│   │   └── NFTP/
│   └── uts/
│       ├── MTR/
│       └── NFTP/
└── sandboxes/           ← diretórios de execução (removidos após uso)
```

---

## Relatório HTML

O relatório organiza os resultados em três níveis:

```
📄 atr.py  (81 mutants · 2 operators)  Score: 65%
  └── ATR  15 mutants
        ├── ATR_1  KILLED    ← diff + código do mutante
        ├── ATR_2  SURVIVED  ← testes que falharam
        └── ...
  └── MTR  66 mutants
        └── ...

📄 mtr.py  (18 mutants · 1 operator)
  └── ...
```

Para cada mutante, o relatório mostra:
- **Status**: killed / survived / timeout / error
- **Diff**: comparação linha a linha com o arquivo original correto
- **Mutant source**: código completo do arquivo mutante
- **Failed tests**: quais testes detectaram o mutante (quando killed)

---

## Como a ferramenta decide o que mutar

O `ast_analyzer` percorre a AST de cada arquivo fonte e aplica os seguintes filtros:

**Incluídas como alvo:**
- Funções de módulo com lógica de transformação DataFrame
- Métodos de classe com transformações

**Excluídas automaticamente:**
- `main()`, `setup()`, `teardown()` e variantes
- Funções de I/O puro (só leitura/escrita, sem transformação)
- Dunders (`__init__`, `__repr__`, etc.)
- Funções privadas de módulo (prefixo `_`)
- Funções com decoradores de orquestração (`@task`, `@dag`, `@flow`)

---

## Mapeamento automático de testes

A ferramenta mapeia automaticamente quais testes cobrem cada função, usando duas estratégias combinadas:

**Estratégia A — imports:** identifica quais arquivos de teste importam o módulo fonte.

**Estratégia B — nomes:** cruza os nomes das funções de teste com os nomes das funções alvo.

```python
# test_transforms.py importa transforms.py  ← estratégia A
from transforms import compute_revenue

def test_compute_revenue_basic():   # casa com compute_revenue  ← estratégia B
    ...
```

Isso permite que o `TestRunner` execute **só os testes relevantes** para cada mutante, reduzindo significativamente o tempo de execução.

---

## Integração com CI

```yaml
# .github/workflows/mutation.yml
name: Mutation Testing

on:
  pull_request:
    paths: ["etl_code/**", "tests/**"]

jobs:
  mutation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Java (required by PySpark)
        run: sudo apt-get install -y default-jdk-headless

      - run: pip install transmutpyspark

      - run: transmut run --src etl_code/ --tests tests/

      - name: Check minimum score
        run: |
          python3 -c "
          import json, sys
          # lê o score do relatório
          # adapte conforme seu threshold
          print('Mutation testing concluído')
          "

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: mutation-report
          path: TransmutPysparkOutput/report.html
```

---

## Publicar no PyPI

```bash
pip install build twine

python -m build
twine upload dist/*
```

Crie sua conta em [pypi.org](https://pypi.org) e gere um API token em *Account Settings → API tokens*.

---
