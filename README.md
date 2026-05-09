# TransmutPySpark

**Mutation testing para pipelines PySpark com DataFrame API.**

TransmutPySpark mede a qualidade dos seus testes unitários aplicando mutações cirúrgicas no código PySpark e verificando se os testes detectam as mudanças. O resultado é um **Mutation Score** — a porcentagem de mutantes que seus testes conseguem "matar".

```
Mutation Score : 73%  (97 killed / 133 total)
Sobreviventes  : 36 mutantes não detectados
Relatório      : TransmutPysparkOutput/report.html
```

Um mutante **sobrevivente** indica um caso de teste que está faltando na sua suite. O relatório mostra o diff exato de cada sobrevivente para guiar o que você precisa testar.

---

## Sumário

- [Instalação via pip](#instalação-via-pip)
- [Instalação via git clone](#instalação-via-git-clone)
- [Início rápido](#início-rápido)
- [Estrutura esperada do projeto](#estrutura-esperada-do-projeto)
- [Configuração](#configuração)
- [Operadores de mutação](#operadores-de-mutação)
- [Como funciona internamente](#como-funciona-internamente)
- [Saída gerada](#saída-gerada)
- [Interpretando o Mutation Score](#interpretando-o-mutation-score)
- [Integração com CI](#integração-com-ci)
- [Contribuindo — adicionando novos operadores](#contribuindo--adicionando-novos-operadores)
- [Referência de comandos](#referência-de-comandos)
- [Requisitos](#requisitos)

---

## Instalação via pip

```bash
pipx install transmutpyspark
```

> **Recomendado:** `pipx` gerencia o ambiente automaticamente e expõe o comando `transmut` no PATH sem precisar ativar nenhum venv.
>
> Se não tiver pipx: `sudo apt install pipx && pipx ensurepath`

Alternativa com pip:

```bash
pip install transmutpyspark
```

---

## Instalação via git clone

Para desenvolver a ferramenta, contribuir com novos operadores ou usá-la diretamente do repositório:

**1. Clone o repositório**

```bash
git clone https://github.com/seu-user/TransmutPySpark.git
cd TransmutPySpark
```

**2. Crie e ative um ambiente virtual**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> No Windows: `.venv\Scripts\activate`

**3. Instale em modo editável**

```bash
pip install -e .
```

O modo `-e` (editable) faz com que qualquer alteração no código fonte seja refletida imediatamente sem reinstalar — ideal para desenvolvimento. O comando `transmut` ficará disponível enquanto o venv estiver ativo.

**4. Confirme a instalação**

```bash
transmut --help
```

**Para ativar o venv automaticamente em novos terminais**, adicione ao seu `~/.bashrc` ou `~/.zshrc`:

```bash
echo "source ~/caminho/para/TransmutPySpark/.venv/bin/activate" >> ~/.bashrc
source ~/.bashrc
```

---

## Início rápido

```bash
# 1. Vai para a raiz do seu projeto PySpark
cd meu_projeto/

# 2. Cria o arquivo de configuração
transmut init --src etl_code/ --tests tests/

# 3. Roda o pipeline de mutação
transmut run

# 4. Abre o relatório no browser
transmut show
```

---

## Estrutura esperada do projeto

TransmutPySpark funciona com qualquer estrutura de projeto PySpark. Os padrões mais comuns:

**Estrutura flat (pipelines simples):**
```
meu_projeto/
├── etl_code/
│   ├── transformations.py
│   ├── aggregations.py
│   └── filters.py
└── tests/
    ├── test_transformations.py
    ├── test_aggregations.py
    └── test_filters.py
```

**Estrutura modular (projetos maiores):**
```
meu_projeto/
├── jobs/
│   ├── sales/
│   │   └── transforms.py
│   └── inventory/
│       └── transforms.py
└── tests/
    ├── sales/
    │   └── test_transforms.py
    └── inventory/
        └── test_transforms.py
```

A ferramenta descobre automaticamente todos os arquivos `.py` recursivamente, ignorando `__init__.py`, `conftest.py`, `setup.py` e pastas como `.venv`, `__pycache__`, `.git`.

---

## Configuração

### Modo A — flags diretas (sem arquivo de config)

Ideal para uso pontual ou scripts de CI:

```bash
transmut run --src etl_code/ --tests tests/
```

```bash
# Arquivo único
transmut run \
  --src etl_code/transformations.py \
  --tests tests/test_transformations.py

# Só alguns operadores
transmut run --src etl_code/ --tests tests/ --operators NFTP MTR

# Com mais workers paralelos
transmut run --src etl_code/ --tests tests/ --workers 8

# Com logs detalhados
transmut run --src etl_code/ --tests tests/ --verbose
```

### Modo B — transmut.toml (recomendado)

Cria uma vez, versiona junto com o projeto:

```bash
transmut init --src etl_code/ --tests tests/
```

Edite o `transmut.toml` gerado:

```toml
[transmut]
source_dirs   = ["etl_code/"]
tests_dirs    = ["tests/"]
operators     = ["MTR", "NFTP", "ATR", "UTS"]
workspace_dir = "."
```

Depois é só:

```bash
transmut run
```

### Modo C — config.txt (compatibilidade com versões anteriores)

```bash
transmut run --config config.txt
```

```ini
program_path   = etl_code/transformations.py
tests_path     = tests/test_transformations.py
operators_list = MTR, NFTP, ATR, UTS
workspace_dir  = .
```

---

## Operadores de mutação

TransmutPySpark implementa quatro operadores específicos para a DataFrame API do PySpark. Cada operador implementa a classe abstrata `Operator` e é responsável por identificar nós elegíveis na AST e gerar os arquivos mutantes.

---

### NFTP — Negation of Filter Transformation Predicate

**Alvo:** operações `.filter()` e `.where()` com predicados booleanos.

```python
# Original
df.filter(col("fare_amount") >= 2.5)
df.where(col("status").isNotNull() & (col("distance") > 0))
```

**Mutações geradas:**

| Tipo | Original | Mutante |
|------|----------|---------|
| Negação total | `pred` | `~pred` |
| Inversão de comparação | `>` | `<=` |
| Inversão de comparação | `==` | `!=` |
| Inversão de comparação | `>=` | `<` |
| Inversão lógica | `&` | `\|` |
| Inversão lógica | `\|` | `&` |
| Swap null check | `isNull` | `isNotNull` |
| Negação de isin | `isin(...)` | `~isin(...)` |

**Detecta:** predicados invertidos, condições de borda ausentes, lógica booleana incorreta.

---

### MTR — Mapping Transformation Replacement

**Alvo:** operações `.withColumn()`, `.select()`, `.map()`, `.mapInPandas()`.

```python
# Original
df.withColumn("valor_por_km", col("fare_amount") / col("trip_distance"))
df.select("trip_id", "region", "fare_amount")
```

**Mutações geradas:**

Cada expressão nas transformações é substituída por valores limite:

| Label | Substituto |
|-------|-----------|
| `zero` | `0` |
| `one` | `1` |
| `neg_one` | `-1` |
| `none` | `None` |
| `empty_str` | `""` |
| `negated` | `-<expr>` |
| `identity` | `col(...)` (quando aplicável) |

**Detecta:** cálculos incorretos, expressões que nunca foram testadas com valores limite, ausência de validação de `None`.

---

### ATR — Aggregation Transformation Replacement

**Alvo:** operações de agregação e funções de janela.

```python
# Original
df.groupBy("region").agg(F.sum("fare_amount").alias("total"))
ranked_df = df.withColumn("rank", F.rank().over(window_spec))
```

**Mutações geradas:**

Substituição de função de agregação:

| Original | Mutantes gerados |
|----------|-----------------|
| `sum` | `avg`, `max`, `min`, `count`, `first`, `last` |
| `count` | `sum`, `avg`, `max`, `min`, `first`, `last` |
| `avg` | `sum`, `max`, `min`, `count`, `first`, `last` |

Substituição de função de janela:

| Original | Mutantes gerados |
|----------|-----------------|
| `rank` | `dense_rank`, `row_number`, `percent_rank`, `cume_dist` |
| `dense_rank` | `rank`, `row_number`, `percent_rank`, `cume_dist` |

Remoção de chave de agrupamento:

```python
# Original
df.groupBy("driver_id", "region").count()

# Mutantes
df.groupBy("region").count()      # remove "driver_id"
df.groupBy("driver_id").count()   # remove "region"
```

**Detecta:** função de agregação errada, granularidade incorreta no `groupBy`, funções de janela inadequadas.

---

### UTS — Unary Transformations Swap

**Alvo:** pares de operações consecutivas sem dependência de coluna entre si.

```python
# Original — filter depois select, sem dependência
df.filter(col("fare_amount") >= 2.5).select("trip_id", "fare_amount")

# Mutante — ordem trocada
df.select("trip_id", "fare_amount").filter(col("fare_amount") >= 2.5)
```

O operador analisa dependências de coluna: se a segunda operação usa uma coluna criada pela primeira (ex: `withColumn` seguido de `filter` nessa coluna), o par **não é elegível** e não gera mutante.

**Detecta:** dependências implícitas de ordem, transformações que produzem resultados diferentes dependendo da sequência de aplicação.

---

## Como funciona internamente

```
transmut run
    │
    ├── ConfigLoader
    │   Detecta o modo de entrada (arquivo, diretório ou transmut.toml)
    │   e normaliza tudo em um ResolvedConfig uniforme.
    │
    ├── AST Analyzer
    │   ├── Descoberta    percorre a AST com ast.walk() buscando
    │   │                 FunctionDef e ClassDef elegíveis
    │   ├── Filtragem     ignora main(), funções de I/O puro, dunders,
    │   │                 funções privadas de módulo, decoradores de
    │   │                 orquestração (@dag, @task, @flow...)
    │   └── Mapeamento    vincula cada função aos seus testes via
    │                     análise de imports (Estratégia B) combinada
    │                     com convenção de nomes (Estratégia A)
    │
    ├── MutationManager
    │   ├── parse_to_ast()
    │   │   Parseia cada arquivo fonte em AST separada.
    │   │   Mantém um dict { source_file → AST } para isolar
    │   │   completamente os arquivos entre si.
    │   │
    │   ├── apply_mutation()
    │   │   Para cada FunctionTarget × Operador:
    │   │     1. operator.analyse_ast(target.node)
    │   │        O operador analisa APENAS o nó da função — não o arquivo
    │   │     2. operator.build_mutant(nodes, file_ast, ...)
    │   │        Gera arquivo .py COMPLETO com a função mutada
    │   │        usando _replace_node() + ast.unparse()
    │   │     3. Propaga test_files e test_functions para o Mutant
    │   │
    │   └── run_tests()
    │       Para cada mutante, em paralelo (ThreadPoolExecutor):
    │         - Cria sandbox isolado com o arquivo mutante
    │         - Injeta etl_code/ no PYTHONPATH do subprocess
    │         - Executa: python -m pytest <test_files> -k <test_functions>
    │         - Classifica: killed / survived / error / timeout
    │
    └── Reporter
        Gera report.html organizado por:
          arquivo fonte → operador → mutante
        com diff correto (original vs mutante) e código fonte completo
```

### Por que escopo cirúrgico por função?

A ferramenta não muta o arquivo inteiro de uma vez. Para cada mutante:

1. O `analyse_ast` recebe apenas o **nó AST da função alvo** — não o arquivo
2. O `build_mutant` recebe a **AST completa do arquivo** para gerar um `.py` válido, mas substitui **apenas o nó da função**
3. O `pytest` roda apenas os **testes mapeados** para aquela função específica

Isso reduz drasticamente o tempo de execução e mantém os resultados precisos — um mutante não contamina a análise de outro.

---

## Saída gerada

```
TransmutPysparkOutput/
├── report.html              relatório visual interativo
└── mutants/
    ├── transformations/     mutantes de transformations.py
    │   ├── MTR/             gerados pelo operador MTR
    │   │   ├── MTR_1_withColumn_expr0_zero.py
    │   │   ├── MTR_2_withColumn_expr0_one.py
    │   │   └── ...
    │   └── NFTP/
    │       └── ...
    └── aggregations/
        └── ATR/
            └── ...
```

### Relatório HTML

O relatório é organizado em três níveis colapsáveis:

```
📄 transformations.py  (42 mutants · 2 operators)  Score: 65%
  └── MTR  36 mutants   24 killed · 10 survived · 2 error
        └── 001  line 14 | withColumn() expr → zero  SURVIVED  0.3s
              ├── Diff           linha exata alterada em vermelho/verde
              └── Mutant source  arquivo .py completo do mutante

  └── NFTP  6 mutants   6 killed
        └── ...
```

Cada entrada mostra o status (`KILLED` / `SURVIVED` / `ERROR` / `TIMEOUT`), o tempo de execução, os testes que falharam (quando killed), o diff e o código fonte completo do mutante.

---

## Interpretando o Mutation Score

```
Mutation Score = mutantes killed / total de mutantes
```

| Score | Classificação | O que significa |
|-------|--------------|----------------|
| ≥ 80% | **STRONG** | Suite robusta — a maioria das mutações é detectada |
| 50–79% | **MODERATE** | Gaps importantes — lógica de negócio com pouca cobertura |
| < 50% | **WEAK** | Testes insuficientes — mutações passam sem ser detectadas |

**Como usar os resultados:** filtre os mutantes com status `SURVIVED` no relatório. O diff mostra exatamente o que foi alterado — escreva um teste que falhe com aquela alteração e passe com o código original.

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

      - uses: actions/setup-java@v4
        with:
          java-version: "11"
          distribution: "temurin"

      - run: pip install transmutpyspark

      - run: transmut run --src etl_code/ --tests tests/

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: mutation-report
          path: TransmutPysparkOutput/report.html
```

---

## Contribuindo — adicionando novos operadores

Toda a lógica de mutação está isolada em operadores independentes. Adicionar um novo operador não requer modificar nenhum outro arquivo da ferramenta — basta criar uma classe que herda de `Operator` e registrá-la.

### Estrutura da classe abstrata

```python
# src/operators/operator.py (resumo)
@dataclass
class Operator(ABC):
    id:               int
    name:             str
    mutant_registers: str | list[str]   # métodos/nós alvo
    mutant_list:      list[Mutant]      # populado pelo build_mutant

    @abstractmethod
    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        """
        Recebe o nó AST da função alvo.
        Retorna a lista de nós elegíveis para mutação.
        """
        ...

    @abstractmethod
    def build_mutant(self,
                     nodes: list[ast.AST],
                     original_ast: ast.AST,
                     original_path: str,
                     mutant_dir: str) -> list[Mutant]:
        """
        Recebe os nós encontrados e a AST completa do arquivo.
        Gera um arquivo .py mutante por nó e popula self.mutant_list.
        """
        ...
```

**Helpers disponíveis na classe base** (não precisam ser reimplementados):

| Método | O que faz |
|--------|-----------|
| `_replace_node(original_ast, target, replacement)` | Clona a AST e substitui `target` por `replacement` — localiza o nó por `lineno + col_offset + unparse` |
| `_write_mutant_file(mutated_ast, mutant_dir, filename)` | Converte AST para código com `ast.unparse()` e grava em disco |
| `_next_mutant_id()` | Retorna o próximo ID sequencial para o mutante |
| `_assert_valid_tree(tree)` | Valida que `tree` é um `ast.AST` |
| `_assert_valid_nodes(nodes)` | Valida que `nodes` é lista de `ast.AST` |
| `_assert_valid_path(path, name)` | Valida que o path é string não vazia |
| `_log_analyse_ast_found(count, description)` | Log padronizado de nós encontrados |
| `_log_mutant_created(id, details)` | Log padronizado de mutante criado |
| `_log_build_mutant_done()` | Log de conclusão com total de mutantes |

### Passo a passo: criando um operador novo

**Exemplo:** operador `JFR` (Join Filter Removal) que remove condições de `.join()`.

**1. Crie o arquivo `src/operators/operator_jfr.py`:**

```python
# src/operators/operator_jfr.py
"""
JFR – Join Filter Removal
==========================
Alvo: operações df.join(other, on=condition, how=...)
Mutação: remove a condição de join, substituindo por join sem filtro
"""

import ast
import copy
from dataclasses import dataclass, field

from src.model.mutant import Mutant
from src.operators.operator import Operator


@dataclass
class OperatorJFR(Operator):
    # Atributos de classe usados pelo factory Operator.create()
    _DEFAULT_ID        = 5
    _DEFAULT_NAME      = "JFR"
    _DEFAULT_REGISTERS = ["join"]

    id:               int             = 5
    name:             str             = "JFR"
    mutant_registers: list[str]       = field(
        default_factory=lambda: ["join"]
    )

    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        """
        Encontra todas as chamadas .join() com condição explícita.
        Retorna a lista de call nodes elegíveis.
        """
        self._assert_valid_tree(tree)
        eligible = []

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "join"
                and len(node.args) >= 2   # join(other, condition, ...)
            ):
                eligible.append(node)

        self._log_analyse_ast_found(
            len(eligible), "join() calls with explicit condition"
        )
        return eligible

    def build_mutant(
        self,
        nodes:        list[ast.AST],
        original_ast: ast.AST,
        original_path: str,
        mutant_dir:   str,
    ) -> list[Mutant]:
        self._assert_valid_nodes(nodes)
        self._assert_valid_path(original_path, "original_path")
        self._assert_valid_path(mutant_dir, "mutant_dir")

        for join_call in nodes:
            # Clona o nó e remove o segundo argumento (a condição)
            mutated_call       = copy.deepcopy(join_call)
            original_condition = ast.unparse(mutated_call.args[1])
            mutated_call.args  = [mutated_call.args[0]]  # só "other"

            # Substitui o nó na AST completa do arquivo
            mutated_ast = self._replace_node(
                original_ast, join_call, mutated_call
            )

            mid      = self._next_mutant_id()
            filename = f"JFR_{mid}_join_remove_condition.py"
            path     = self._write_mutant_file(mutated_ast, mutant_dir, filename)

            mutant = Mutant(
                id=mid,
                operator=self.name,
                original_path=original_path,
                mutant_path=path,
                modified_line=str(getattr(join_call, "lineno", "?")),
            )
            self.mutant_list.append(mutant)
            self._log_mutant_created(
                mid,
                f"line {mutant.modified_line} | "
                f"join() condition removed | "
                f"original: {original_condition} [{filename}]"
            )

        self._log_build_mutant_done()
        return self.mutant_list
```

**2. Registre o operador no `MutationManager`:**

```python
# src/mutation_manager.py
OPERATOR_REGISTRY = {
    "NFTP": "src.operators.operator_nftp.OperatorNFTP",
    "MTR":  "src.operators.operator_mtr.OperatorMTR",
    "ATR":  "src.operators.operator_atr.OperatorATR",
    "UTS":  "src.operators.operator_uts.OperatorUTS",
    "JFR":  "src.operators.operator_jfr.OperatorJFR",  # ← adiciona aqui
}
```

**3. Use o novo operador:**

```bash
transmut run --src etl_code/ --tests tests/ --operators JFR

# Ou adicione ao transmut.toml
operators = ["MTR", "NFTP", "ATR", "UTS", "JFR"]
```

### Regras para um bom operador

- `analyse_ast` deve receber o **nó da função** (não o arquivo inteiro) e retornar apenas nós dentro desse escopo
- `build_mutant` **nunca modifica `original_ast` in place** — use sempre `_replace_node()` que faz `deepcopy` internamente
- Um mutante por substituição — não agrupe múltiplas mutações em um único arquivo
- Use os helpers de log (`_log_analyse_ast_found`, `_log_mutant_created`, `_log_build_mutant_done`) para manter consistência no output
- Nomeie os arquivos com o padrão `OPERADOR_id_descricao.py`

### Abrindo um Pull Request

1. Crie uma branch: `git checkout -b operador/JFR`
2. Implemente o operador seguindo o padrão acima
3. Adicione testes unitários em `tests/operators/test_operator_jfr.py`
4. Registre no `OPERATOR_REGISTRY`
5. Documente no `README.md`
6. Abra o PR descrevendo qual padrão de código o operador muta e quais bugs ele detecta

---

## Referência de comandos

```
transmut init   [--src PATH] [--tests PATH] [--output DIR]
    Cria transmut.toml no diretório atual com configuração padrão.

transmut run    [--src PATH] [--tests PATH]
                [--operators OP [OP ...]]
                [--config FILE]
                [--output DIR]
                [--workers N]
                [--verbose / -v]
    Executa o pipeline completo de mutation testing.
    Se nenhuma flag for passada, detecta transmut.toml ou config.txt
    automaticamente na raiz do diretório atual.

transmut show
    Abre o report.html mais recente no browser padrão.
```

---

## Requisitos

- Python 3.10+
- Java 8+ (necessário para o PySpark inicializar o contexto Spark)
- PySpark 3.3+ (instalado automaticamente como dependência)
- pytest 7.0+

---

## Licença

MIT
