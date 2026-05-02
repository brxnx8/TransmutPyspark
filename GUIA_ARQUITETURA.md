# TransmutPyspark - Guia Completo da Arquitetura

## 📋 Sumário
1. [Visão Geral](#visão-geral)
2. [Arquitetura em Camadas](#arquitetura-em-camadas)
3. [Componentes Principais](#componentes-principais)
4. [Operadores de Mutação](#operadores-de-mutação)
5. [Fluxo de Dados](#fluxo-de-dados)
6. [Padrões de Design](#padrões-de-design)
7. [Estrutura de Diretórios](#estrutura-de-diretórios)
8. [Guia de Extensibilidade](#guia-de-extensibilidade)

---

## Visão Geral

**TransmutPyspark** é uma ferramenta automatizada de **teste de mutação** especializada em aplicações Apache PySpark que utilizam a API DataFrame. 

### O que é Teste de Mutação?

Teste de mutação é uma técnica de garantia de qualidade que avalia a efetividade de um conjunto de testes através da:

1. **Injeção de defeitos intencionais** (mutantes) no código-fonte
2. **Execução dos testes** contra cada mutante
3. **Medição de cobertura** usando a métrica **Mutation Score**:
   ```
   Mutation Score = Mutantes Mortos / (Mortos + Sobreviventes + Timeout + Erros)
   ```

### Por que para PySpark?

Pipelines de dados em PySpark são especialmente propensos a:
- Erros sutis em transformações de dados (`.map()`, `.select()`, `.withColumn()`)
- Predicados incorretos em filtragens (`.filter()`, `.where()`)
- Lógica de negócio complexa nas operações

O teste de mutação detecta se seus testes conseguem pegar esses erros.

---

## Arquitetura em Camadas

```
┌────────────────────────────────────────────────────────────┐
│                  CAMADA DE APRESENTAÇÃO                    │
│  main.py → MutationManager (Entrada do Sistema)             │
└────────┬─────────────────────────────────────────────────┘
         │
┌────────┴─────────────────────────────────────────────────────┐
│          CAMADA DE ORQUESTRAÇÃO                              │
│  MutationManager (Pipeline Coordinator)                      │
│  ├─ load()              → Carrega config                     │
│  ├─ parse_to_ast()      → Converte código em AST             │
│  ├─ apply_mutation()    → Gera mutantes                      │
│  ├─ run_tests()         → Executa testes                     │
│  └─ agregate_results()  → Consolida resultados              │
└────────┬─────────────────────────────────────────────────────┘
         │
    ┌────┴────┬──────────────────┬────────────────────┐
    │          │                  │                    │
┌───▼──┐  ┌────▼─────┐  ┌────────▼────────┐  ┌──────▼──────┐
│ANÁLISE│  │MUTAÇÃO   │  │TESTE & RELATÓRIO│  │MODELOS DADOS│
│(AST)  │  │          │  │                 │  │             │
│       │  │- Ops     │  │- TestRunner     │  │- Mutant     │
│- Parse│  │- Gera    │  │- Reporter       │  │- TestResult │
│- Walk │  │- Escreve │  │- Calcula Score  │  │- ConfigLdr  │
└───────┘  └──────────┘  └─────────────────┘  └─────────────┘
```

### Detalhamento das Camadas

#### **1. Camada de Apresentação (main.py)**
- Ponto de entrada do sistema
- Inicializa `MutationManager` com arquivo de configuração
- Executa o pipeline completo com `.run()`

#### **2. Camada de Orquestração (MutationManager)**
- Coordena o fluxo de execução
- Implementa padrão **Orchestrator**
- Gerencia estado do pipeline
- Carrega dinamicamente operadores de mutação

#### **3. Camada de Análise (AST)**
- Converte código Python em **Abstract Syntax Tree**
- Utiliza módulo `ast` nativo do Python
- Prepara estrutura para análise e mutação

#### **4. Camada de Mutação (Operators)**
- Implementa diferentes estratégias de mutação
- Utiliza padrão **Strategy**
- Gera mutantes no disco

#### **5. Camada de Teste e Relatório**
- Executa testes contra mutantes em paralelo
- Calcula métricas de mutação
- Gera relatório HTML

#### **6. Camada de Modelos (Data Transfer Objects)**
- Define estruturas de dados (dataclasses)
- Fornece validação automática
- Facilita passagem de dados entre componentes

---

## Componentes Principais

### **1. MutationManager** 
**Arquivo:** `src/mutation_manager.py`

**Responsabilidade:** Orquestração do pipeline completo

**Pipeline de Execução:**
```python
manager = MutationManager("config.txt")
manager.load()              # Carrega configuração e código
       .parse_to_ast()      # Converte em AST
       .apply_mutation()    # Gera mutantes
       .run_tests()         # Executa testes
       .agregate_results()  # Consolida resultados
```

**Métodos Principais:**

| Método | Entrada | Saída | Responsabilidade |
|--------|---------|-------|------------------|
| `load()` | config.txt | config, code_original, work_dir | Carrega configuração e código |
| `parse_to_ast()` | code_original | code_ast | Converte código em AST |
| `apply_mutation()` | code_ast | mutant_list | Gera mutantes |
| `run_tests()` | mutant_list | result_list | Executa testes |
| `agregate_results()` | result_list | report.html | Consolida e reporta |

**Registry de Operadores:**
```python
OPERATOR_REGISTRY = {
    "NFTP": "src.operators.operator_nftp.OperatorNFTP",
    "MTR":  "src.operators.operator_mtr.OperatorMTR",
    "ATR":  "src.operators.operator_atr.OperatorATR",
    "UTS":  "src.operators.operator_uts.OperatorUTS",
}
```

---

### **2. Operator (Classe Abstrata)**
**Arquivo:** `src/operators/operator.py`

**Responsabilidade:** Definir interface para operadores de mutação

**Interface Obrigatória:**

```python
@dataclass
class Operator(ABC):
    id: int
    name: str
    mutant_registers: list[str]  # Métodos-alvo
    mutant_list: list[Mutant] = field(default_factory=list)
    
    @abstractmethod
    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        """Identifica nós elegíveis para mutação"""
        
    @abstractmethod
    def build_mutant(self, nodes, original_ast, original_path, mutant_dir) -> list[Mutant]:
        """Gera mutantes a partir dos nós identificados"""
```

**Fluxo de Uso:**
```
1. analyse_ast()   → Encontra nós-alvo na AST
2. build_mutant()  → Cria mutantes baseado nos nós
3. Escreve mutantes em disco
4. Retorna lista de Mutant
```

---

### **3. TestRunner**
**Arquivo:** `src/test_module/test_runner.py`

**Responsabilidade:** Executar testes contra mutantes em paralelo

**Características:**
- **Paralelismo:** Utiliza `ThreadPoolExecutor` com até 4 workers
- **Isolamento:** Cada mutante executado em subprocess próprio
- **Injeção de dependências:** Via `PYTHONPATH`
- **Timeout:** 120 segundos por mutante
- **Classificação de Resultados:**
  - `killed`: Teste falhou (mutante detectado) ✓
  - `survived`: Teste passou (mutante não detectado) ✗
  - `error`: Erro na execução
  - `timeout`: Execução ultrapassou limite

**Classificação de Resultado por Código de Saída:**
```python
exit_code 0   → survived  (testes passaram)
exit_code 1   → killed    (teste falhou)
exit_code 2-5 → error     (erro pytest)
timeout       → timeout   (>120s)
```

---

### **4. Reporter**
**Arquivo:** `src/reporter/reporter.py`

**Responsabilidade:** Consolidar resultados e gerar relatório HTML

**Fluxo de Processamento:**
```python
reporter.calculate()     # Calcula Mutation Score
reporter.make_diff()     # Gera diffs original vs mutantes
reporter.show_results()  # Escreve report.html
```

**Saídas Geradas:**
1. **report.html** - Relatório visual completo
2. **metrics.json** - Métricas estruturadas

**Conteúdo do Relatório:**
- **Summary Cards:** Score de mutação com codificação cromática
- **Tabela de Resultados:** Status, operador, linha modificada
- **Seção de Diffs:** Comparação original vs mutantes
- **Dump JSON:** Métricas para processamento automatizado

---

### **5. ConfigLoader**
**Arquivo:** `src/config/config_loader.py`

**Responsabilidade:** Encapsular configuração da execução

**Estrutura de Configuração (config.txt):**
```plaintext
program_path  = /path/to/program.py        # Arquivo-alvo
tests_path    = /path/to/test_program.py   # Suite de testes
operators_list = NFTP, MTR, ATR, UTS       # Operadores a usar
workspace_dir = /path/to/workspace/        # Diretório de trabalho
```

**Campos Obrigatórios:**
- `program_path` - Caminho do programa PySpark
- `tests_path` - Caminho da suite de testes
- `operators_list` - Lista de operadores separados por vírgula
- `workspace_dir` - Diretório onde saídas serão criadas

---

### **6. Modelos de Dados**

#### **Mutant**
**Arquivo:** `src/model/mutant.py`

```python
@dataclass
class Mutant:
    id: int              # Identificador único
    operator: str        # Nome do operador
    original_path: str   # Caminho do arquivo original
    mutant_path: str     # Caminho do mutante gerado
    modified_line: str   # Código da linha modificada
```

#### **TestResult**
**Arquivo:** `src/model/test_result.py`

```python
@dataclass
class TestResult:
    mutant_id: int       # ID do mutante testado
    status: str          # "killed" | "survived" | "error" | "timeout"
    execution_time: float
    failed_tests: list[str]  # Testes que falharam
```

---

## Operadores de Mutação

### **1. NFTP - Negation of Filter Transformation Predicate**

**Alvo:** Operações de filtragem em DataFrames

```python
df.filter((col("a") > 0) & (col("b") < 10))
df.where(col("status") == "ativo")
```

**Estratégias de Mutação:**
1. **Negação de subcondições** - `~pred`
2. **Inversão de operadores** - `>` → `<=`, `==` → `!=`
3. **Swap null checks** - `isNull` ↔ `isNotNull`
4. **Inversão lógica** - `&` → `|`, `and` → `or`

**Exemplo:**
```python
# Original
df.filter((col("a") > 0) & (col("b") < 10))

# Mutantes NFTP:
df.filter(~(col("a") > 0) & (col("b") < 10))  # Nega subcond 1
df.filter((col("a") > 0) & ~(col("b") < 10))  # Nega subcond 2
df.filter((col("a") <= 0) & (col("b") < 10))  # Inverte >
```

**Vantagem:** Detecta erros em lógica de filtragem

---

### **2. MTR - Mapping Transformation Replacement**

**Alvo:** Operações de transformação de dados

```python
df.map(process_row)
df.select("a", "b")
df.withColumn("new_col", expression)
df.mapInPandas(transform)
```

**Estratégia:** Substitui funções de transformação por lambdas limite

**Substituições Geradas:**
```python
lambda *a, **k: 0           # Zero
lambda *a, **k: 1           # Um
lambda *a, **k: -1          # Negativo
lambda *a, **k: None        # Nulo
lambda *a, **k: ""          # String vazia
lambda *a, **k: []          # Lista vazia
lambda *a, **k: 2**31-1     # Máximo (int32)
lambda *a, **k: -(2**31)    # Mínimo (int32)
```

**Exemplo:**
```python
# Original
df.map(lambda x: x * 2).collect()

# Mutantes MTR:
df.map(lambda *a, **k: 0).collect()
df.map(lambda *a, **k: 1).collect()
df.map(lambda *a, **k: None).collect()
# ... (mais substituições)
```

**Vantagem:** Detecta falhas em cálculos e transformações

---

### **3. ATR - Aggregation Transformation Replacement**

**Alvo:** Operações de agregação

```python
df.agg(F.sum("amount"))
df.groupBy("category").agg(F.avg("price"))
```

**Estratégia:** Substitui funções de agregação

---

### **4. UTS - Utility Transformation Substitution**

**Alvo:** Operações auxiliares

```python
df.cast(new_type)
df.coalesce(n)
```

**Estratégia:** Substitui valores utilitários

---

## Fluxo de Dados

```
┌─────────────────────────────┐
│  config.txt                 │
│  (program_path, tests_path) │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ MutationManager.load()              │
│ → ConfigLoader                      │
│ → Lê código-fonte original          │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ MutationManager.parse_to_ast()       │
│ → ast.parse(code_original)           │
│ → ast.fix_missing_locations()        │
│ → AST (Abstract Syntax Tree)         │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ MutationManager.apply_mutation()         │
│                                          │
│ Para cada Operator:                      │
│ 1. Operator.analyse_ast(AST)             │
│    → Lista de nós elegíveis              │
│ 2. Operator.build_mutant(nodes)          │
│    → Deep-copy da AST                    │
│    → Substitui nó-alvo                   │
│    → Unparse para código-fonte           │
│    → Escreve em disco                    │
│    → Retorna Mutant                      │
│                                          │
│ Resultado: mutant_list[]                 │
└──────────────┬───────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ MutationManager.run_tests()             │
│                                         │
│ Para cada mutante (paralelo):           │
│ 1. Prepara subprocess isolado           │
│ 2. Executa: pytest <tests_path>         │
│ 3. Classifica resultado:                │
│    - killed (exit=1)                    │
│    - survived (exit=0)                  │
│    - error (exit=2-5)                   │
│    - timeout (>120s)                    │
│                                         │
│ Resultado: result_list[]                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ MutationManager.agregate_results()       │
│                                          │
│ 1. Reporter.calculate()                  │
│    → Mutation Score = killed/(total)     │
│ 2. Reporter.make_diff()                  │
│    → Diffs original vs mutantes          │
│ 3. Reporter.show_results()               │
│    → Gera report.html                    │
│    → Gera metrics.json                   │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ TransmutPysparkOutput/               │
│ ├─ report.html (relatório visual)    │
│ ├─ metrics.json (dados estruturados) │
│ ├─ mutants/ (código dos mutantes)    │
│ └─ sandboxes/ (diretórios exec)      │
└──────────────────────────────────────┘
```

---

## Padrões de Design

### **1. Strategy Pattern (Operadores)**
```
┌──────────────────┐
│    Operator      │
│   (abstrato)     │
├──────────────────┤
│ analyse_ast()    │
│ build_mutant()   │
└──────────────────┘
       ▲
    ┌──┴──┬──────────┬──────────┐
    │     │          │          │
┌───┴──┐ ┌─┴────┐ ┌──┴────┐ ┌──┴────┐
│NFTP  │ │ MTR  │ │ ATR   │ │ UTS   │
└──────┘ └──────┘ └───────┘ └───────┘
```
Permite trocar estratégias de mutação em tempo de execução.

### **2. Registry Pattern (OPERATOR_REGISTRY)**
```python
OPERATOR_REGISTRY = {
    "NFTP": "src.operators.operator_nftp.OperatorNFTP",
    "MTR":  "src.operators.operator_mtr.OperatorMTR",
    ...
}
```
Possibilita carregamento dinâmico sem dependências estáticas.

### **3. Orchestrator Pattern (MutationManager)**
Coordena componentes especializados em fluxo bem definido:
```
load() → parse_to_ast() → apply_mutation() → run_tests() → agregate_results()
```

### **4. Data Transfer Object (DTO)**
Classes com `@dataclass`:
- `Mutant` - Metadados do mutante
- `TestResult` - Resultado de teste
- `ConfigLoader` - Configuração

Benefícios:
- Type checking automático
- Serialização facilitada
- Documentação clara

### **5. Factory Pattern (loadOperator)**
```python
def _load_operator(op_name):
    dotted_path = OPERATOR_REGISTRY[op_name]
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()
```
Desacopla criação de instâncias da lógica principal.

---

## Estrutura de Diretórios

### **Saída Gerada (TransmutPysparkOutput)**

```
TransmutPysparkOutput/
├── report.html                      # Relatório visual
├── metrics.json                     # Métricas estruturadas
├── mutants/                         # Mutantes gerados
│   ├── nftp_1_filter_line42/
│   │   └── nftp.py                 # Código mutante
│   ├── nftp_2_where_line15/
│   │   └── nftp.py
│   ├── mtr_1_map_zero/
│   │   └── mtr.py
│   ├── mtr_2_withColumn_one/
│   │   └── mtr.py
│   └── ...
└── sandboxes/                       # Diretórios de execução
    ├── nftp_1_filter_line42/
    │   ├── nftp.py
    │   ├── __init__.py
    │   └── (dependências)
    ├── nftp_2_where_line15/
    │   └── ...
    └── ...
```

### **Nomenclatura de Mutantes**

```
{operador}_{id}_{metodo|param}_{linha}/
└─ Exemplo: nftp_1_filter_line42/
             │      │ │      │    │
             │      │ │      │    └─ Linha no código original
             │      │ │      └─────── Método/Parâmetro-alvo
             │      │ └──────────── Identificador sequencial
             │      └──────────── Código do operador
             └─────────────────── Nome do operador
```

---

## Guia de Extensibilidade

### **1. Adicionar Novo Operador**

**Passo 1:** Criar classe herdando de `Operator`

```python
# src/operators/operator_novo.py
from dataclasses import dataclass
from src.operators.operator import Operator
import ast

@dataclass
class OperatorNOVO(Operator):
    def __init__(self):
        super().__init__(
            id=5,
            name="NOVO",
            mutant_registers=["seu_alvo"]
        )
    
    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        """
        Identifica nós elegíveis para mutação.
        
        Deve traversar a AST e retornar lista de nós
        que serão mutados em build_mutant().
        """
        nodes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.SeuAlvo):
                nodes.append(node)
        return nodes
    
    def build_mutant(self, nodes, original_ast, original_path, mutant_dir) -> list:
        """
        Gera mutantes a partir dos nós identificados.
        
        Para cada nó:
        1. Deep-copy da AST original
        2. Localiza nó na cópia usando coordenadas
        3. Substitui nó pela mutação
        4. Unparse para código-fonte
        5. Escreve em disco
        6. Retorna Mutant com metadados
        """
        from src.model.mutant import Mutant
        import copy
        
        mutant_list = []
        
        for node in nodes:
            # Deep-copy
            mutated_ast = copy.deepcopy(original_ast)
            
            # Sua lógica de mutação aqui
            # Exemplo: substituição de operador
            
            # Unparse
            mutant_code = ast.unparse(mutated_ast)
            
            # Escrever em disco
            mutant_dir_path = f"{mutant_dir}/novo_{len(mutant_list)}"
            # ... criar diretório e escrever arquivo
            
            # Criar Mutant
            m = Mutant(
                id=-1,  # Será renumerado pelo MutationManager
                operator="NOVO",
                original_path=original_path,
                mutant_path=mutant_dir_path,
                modified_line="..."
            )
            mutant_list.append(m)
        
        self.mutant_list = mutant_list
        return mutant_list
```

**Passo 2:** Registrar no OPERATOR_REGISTRY

```python
# src/mutation_manager.py
OPERATOR_REGISTRY = {
    ...
    "NOVO": "src.operators.operator_novo.OperatorNOVO",
}
```

**Passo 3:** Usar na configuração

```plaintext
# config.txt
operators_list = NFTP, MTR, NOVO
```

---

### **2. Customizar Comportamento de Operador**

#### **Alterar Métodos-Alvo (MTR)**

```python
# Em config.txt ou via código:
operator = OperatorMTR()
operator.mutant_registers = ["map", "select"]  # Apenas estes
```

#### **Adicionar Novas Substituições (MTR)**

```python
# src/operators/operator_mtr.py
_LITERAL_REPLACEMENTS = {
    "zero":      ast.Constant(value=0),
    "one":       ast.Constant(value=1),
    "custom":    ast.Constant(value="seu_valor"),  # ← Nova
}
```

---

### **3. Estender Reporter**

```python
# src/reporter/reporter.py

def calculate(self) -> "Reporter":
    # Lógica existente...
    
    # Adicionar métrica customizada:
    survived_by_operator = defaultdict(int)
    for result in self.result_list:
        if result.status == "survived":
            mutant = self._find_mutant(result.mutant_id)
            survived_by_operator[mutant.operator] += 1
    
    self.result_calculate["survived_by_operator"] = dict(survived_by_operator)
    return self
```

---

### **4. Adicionar Novo Critério de Classificação (TestRunner)**

```python
# src/test_module/test_runner.py

def _classify_result(self, exit_code, timeout=False) -> str:
    if timeout:
        return "timeout"
    elif exit_code == 0:
        return "survived"
    elif exit_code == 1:
        return "killed"
    elif exit_code in [2, 3, 4, 5]:
        return "error"
    else:
        return "unknown"  # ← Novo critério
```

---

## Fluxo de Execução Prático

### **Exemplo Passo-a-Passo**

**1. Configuração (config.txt)**
```plaintext
program_path  = etl_project_example/etl_code/uts.py
tests_path    = etl_project_example/etl_test/test_uts.py
operators_list = NFTP, MTR
workspace_dir = etl_project_example/
```

**2. Execução**
```bash
python main.py
```

**3. Logs da Execução**
```
[INFO] Config carregado: etl_project_example/etl_code/uts.py
[INFO] AST gerado com sucesso
[INFO] Operador 'NFTP': 3 mutante(s) gerado(s)
[INFO] Operador 'MTR': 8 mutante(s) gerado(s)
[INFO] Total de mutantes: 11
[INFO] Testes executados: 11 resultado(s)
[INFO] Report gerado: TransmutPysparkOutput/report.html
```

**4. Saída Gerada**
```
TransmutPysparkOutput/
├── report.html
├── metrics.json
├── mutants/
│   ├── nftp_1_filter_line42/
│   ├── nftp_2_where_line8/
│   ├── nftp_3_filter_line101/
│   ├── mtr_1_map_zero/
│   ├── mtr_2_map_one/
│   ├── mtr_3_withColumn_none/
│   ├── mtr_4_select_empty_str/
│   ├── mtr_5_flatMap_empty_list/
│   └── ...
└── sandboxes/
```

**5. Interpretação do Report**

```
┌─────────────────────────────────────┐
│ MUTATION SCORE: 72.73% (BONS!)      │ (Verde: ≥80%) 
├─────────────────────────────────────┤
│ ✓ Killed: 8                         │
│ ✗ Survived: 3                       │
│ ⏱ Timeout: 0                        │
│ ⚠ Error: 0                          │
├─────────────────────────────────────┤
│ Mutantes Sobreviventes:             │
│ • mtr_2_map_one (linha 45)          │
│ • nftp_1_filter_line42 (linha 42)   │
│ • mtr_8_select_max_int32 (linha 78) │
└─────────────────────────────────────┘
```

---

## Tecnologias e Dependências

### **Python Core**
- **`ast`** - Parsing e manipulação de árvores sintáticas
- **`subprocess`** - Execução de pytest em subprocessos
- **`concurrent.futures`** - Paralelismo (`ThreadPoolExecutor`)
- **`pathlib`** - Manipulação de caminhos (multiplataforma)
- **`dataclasses`** - DTOs com validação automática
- **`importlib`** - Carregamento dinâmico de módulos

### **Dependências Externas**
- **`pytest`** - Framework de testes
- **`pyspark`** - Framework para processamento distribuído
- **`difflib`** - Geração de diffs

### **Ambiente**
- Python 3.7+
- Apache Spark (para executar código PySpark)
- PySpark API

---

## Resumo da Arquitetura

| Componente | Responsabilidade | Padrão |
|-----------|-----------------|--------|
| `MutationManager` | Orquestração do pipeline | Orchestrator |
| `Operator` | Interface de mutação | Strategy |
| `NFTP/MTR/ATR/UTS` | Estratégias específicas | Strategy |
| `TestRunner` | Execução paralela de testes | Executor Pattern |
| `Reporter` | Consolidação e relatório | Builder Pattern |
| `ConfigLoader` | Encapsulação de config | DTO |
| `Mutant/TestResult` | Modelos de dados | DTO |

---

## Próximas Etapas

Para aprofundar seu entendimento:

1. **Explore o código-fonte:**
   - Comece em `main.py`
   - Navegue por `mutation_manager.py`
   - Examine operadores em `src/operators/`

2. **Executar com exemplos:**
   - Use `etl_project_example/` como referência
   - Modifique `config.txt` com seus próprios programas

3. **Estender a ferramenta:**
   - Crie novo operador seguindo padrão
   - Adicione métricas customizadas ao Reporter

4. **Integrar em CI/CD:**
   - Use saída `metrics.json` para automação
   - Configure thresholds de Mutation Score

---

**Versão:** 1.0  
**Última atualização:** 2026-05-01  
**Autor:** TransmutPyspark Team
