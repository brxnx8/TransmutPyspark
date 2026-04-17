# Arquitetura da Ferramenta TransmutPyspark

## 1. Introdução

TransmutPyspark é uma ferramenta de teste de mutação especializada em aplicações Apache PySpark. O teste de mutação é uma técnica de avaliação de qualidade que mede a efetividade de um conjunto de testes injetando defeitos intencionais (mutantes) no código-fonte e verificando se os testes conseguem detectá-los. Esta técnica é particularmente relevante para pipelines de dados em PySpark, onde operações de transformação complexas podem conter erros sutis que passam despercebidos.

## 2. Visão Geral Arquitetural

A arquitetura de TransmutPyspark segue um padrão de **pipeline em camadas**, onde cada componente possui responsabilidades bem definidas e integra-se aos demais através de interfaces claras. O sistema é dividido em cinco camadas principais:

```
┌─────────────────────────────────────────────────────┐
│          Camada de Apresentação                     │
│              (main.py)                              │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│     Camada de Orquestração (MutationManager)        │
│  - Coordenação do pipeline                          │
│  - Carregamento de configurações                    │
│  - Gerenciamento de estados                         │
└─────────────────────────────────────────────────────┘
              ↙            ↓            ↘
┌──────────────────┐  ┌──────────────┐  ┌────────────────────┐
│ Camada de        │  │ Camada de    │  │ Camada de          │
│ Análise (AST)    │  │ Mutação      │  │ Teste e Relatório  │
│                  │  │              │  │                    │
│ - Parse Python   │  │ - Operadores │  │ - TestRunner       │
│ - AST Traversal  │  │ - Geração    │  │ - Reporter         │
└──────────────────┘  │ - Escrita    │  │ - Cálculo Score    │
                      │   Mutantes   │  └────────────────────┘
                      └──────────────┘
                             ↓
┌─────────────────────────────────────────────────────┐
│        Camada de Modelos de Dados                   │
│  - Mutant, TestResult, ConfigLoader                │
└─────────────────────────────────────────────────────┘
```

## 3. Componentes Principais

### 3.1 MutationManager (Orquestrador Central)

**Responsabilidade:** Coordenar toda a execução do pipeline de teste de mutação.

O `MutationManager` implementa o padrão **Orchestrator** e segue o princípio de **separação de responsabilidades**, delegando tarefas especializadas a componentes específicos. O pipeline completo é executado através do método `run()`, que executa as etapas em sequência:

```
load() → parse_to_ast() → apply_mutation() → run_tests() → agregate_results()
```

**Características principais:**

- **Carregamento de configuração:** Lê arquivo de configuração em formato `key = value`, valida chaves obrigatórias (`program_path`, `tests_path`, `operators_list`)
- **Análise sintática:** Converte código-fonte em Abstract Syntax Tree (AST) utilizando o módulo `ast` do Python
- **Gerenciamento de operadores:** Implementa registro dinâmico de operadores de mutação via `OPERATOR_REGISTRY` (padrão Registry)
- **Carregamento dinâmico:** Utiliza `importlib` para carregar operadores em tempo de execução sem dependências estáticas

### 3.2 Camada de Análise Sintática

#### 3.2.1 Análise AST (Abstract Syntax Tree)

A transformação de código-fonte em AST é fundamental para a estratégia de mutação. O processo segue estas etapas:

1. **Parse:** `ast.parse(code_original)` converte código-fonte em árvore sintática
2. **Validação:** `ast.fix_missing_locations()` garante que todos os nós possuem informações de localização (lineno, col_offset)
3. **Traversal:** Subclasses de `Operator` utilizam `ast.walk()` para percorrer a árvore e identificar nós elegíveis para mutação

**Vantagens dessa abordagem:**

- Análise estruturada e confiável do código Python
- Abstração dos detalhes sintáticos
- Acesso a metadados de localização (linha, coluna) para rastreamento de mudanças
- Facilita manipulação e transformação de código sem parsing manual

### 3.3 Camada de Mutação

#### 3.3.1 Padrão Operator (Padrão Strategy)

A arquitetura implementa o **padrão Strategy** através da classe abstrata `Operator`, que define a interface que todos os operadores de mutação devem implementar:

```python
class Operator(ABC):
    @abstractmethod
    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        """Identifica nós elegíveis para mutação"""

    @abstractmethod
    def build_mutant(self, nodes, original_ast, original_path, mutant_dir) -> list[Mutant]:
        """Gera mutantes a partir dos nós identificados"""
```

**Responsabilidades:**

1. **analyse_ast:** Travessa a AST e retorna lista de nós que devem sofrer mutação
2. **build_mutant:** Para cada nó elegível, cria um ou mais mutantes:
   - Deep-copy da AST original
   - Substituição do nó-alvo pela mutação
   - Unparse de volta para código-fonte
   - Escrita em disco
   - Registro em `mutant_list`

**Validação em pós-inicialização:**

`Operator` utiliza `@dataclass` com `__post_init__` para validar todos os atributos imediatamente após construção, garantindo invariantes:

- `id`: inteiro não-negativo
- `name`: string não-vazia (normalizada para maiúsculas)
- `mutant_registers`: string ou lista não-vazia de strings
- `mutant_list`: lista de instâncias `Mutant`

#### 3.3.2 OperatorNFTP (Negation of Filter Transformation Predicate)

**Objetivo:** Testar a robustez de predicados em operações de filtragem.

**Estratégia de mutação:**

1. Identifica chamadas `.filter()` ou `.where()` com predicados
2. Decompõe predicados compostos em sub-condições atômicas:
   - Atravessa `ast.BoolOp` (operadores Python `and`/`or`)
   - Atravessa `ast.BinOp` com `ast.BitAnd`/`ast.BitOr` (operadores PySpark `&`/`|`)
   - Retorna folhas como candidatas à negação
3. Para cada sub-condição, gera mutante com negação `not (sub_condition)`
4. Evita dupla negação (detecta e ignora nós já envolvidos em `UnaryOp(ast.Not)`)

**Exemplo:**
```python
# Original
df.filter((col("a") > 0) & (col("b") < 10))

# Mutantes gerados:
df.filter(not (col("a") > 0) & (col("b") < 10))  # Mutante 1
df.filter((col("a") > 0) & not (col("b") < 10))  # Mutante 2
```

**Técnica de relocalização de nós:**

Utiliza coordenadas `(lineno, col_offset)` como chave de identidade para localizar nós em cópias profundas da AST. Essa estratégia permite:

- Mapear nós da AST original para cópias
- Realizar mutações isoladas em cada cópia
- Evitar efeitos colaterais de mutações anteriores

#### 3.3.3 OperatorMTR (Mapping Transformation Replacement)

**Objetivo:** Testar efetividade against falhas em operações de transformação de dados.

**Estratégia de mutação:**

1. Identifica chamadas de mapeamento: `.map()`, `.flatMap()`, `.withColumn()`, `.select()`, `.mapValues()`, `.foreach()`, `.foreachPartition()`
2. Para cada chamada elegível, gera até 8 mutantes, substituindo a função pelo resultado de lambdas de limite/fronteira:

| Substituição | Rationale |
|---|---|
| `lambda *a, **k: 0` | Valor típico (falso em contexto booleano) |
| `lambda *a, **k: 1` | Unidade (verdadeiro em contexto booleano) |
| `lambda *a, **k: -1` | Negativo / sentinela |
| `lambda *a, **k: None` | Nulo (testa segurança contra None) |
| `lambda *a, **k: ""` | String vazia (testa segurança de strings) |
| `lambda *a, **k: []` | Lista vazia (testa segurança de coleções) |
| `lambda *a, **k: 2**31-1` | Limite máximo (inteiro 32-bit) |
| `lambda *a, **k: -(2**31)` | Limite mínimo (inteiro 32-bit) |

**Caso especial - withColumn:**

Para `.withColumn(col_name, expression)`, a função reside no segundo argumento (índice 1), enquanto para outros métodos habita o primeiro (índice 0). O operador detecta automaticamente:

```python
func_arg_idx = 1 if method_name == "withColumn" else 0
```

**Geração de lambdas AST:**

Constrói lambdas AST que aceitam `*args`/`**kwargs` para máxima flexibilidade:

```python
def _lambda_returning(value_node: ast.expr) -> ast.Lambda:
    return ast.Lambda(
        args=ast.arguments(
            posonlyargs=[], args=[], vararg=ast.arg(arg="a"),
            kwonlyargs=[], kw_defaults=[], kwarg=ast.arg(arg="k"),
            defaults=[]
        ),
        body=value_node,
    )
```

### 3.4 Camada de Teste

#### 3.4.1 TestRunner

**Responsabilidade:** Executar testes contra todos os mutantes de forma paralela e coletando resultados.

**Estratégia de parallelismo:**

Utiliza `ThreadPoolExecutor` com `max_workers = min(4, cpu_count or 1)`, limitando concorrência para:
- Evitar contenção com SparkSession potencialmente ativa
- Manter overhead de memória controlado
- Garantir subsistemas auxiliares não ficarem sobrecarregados

**Execução de testes:**

Para cada mutante:

1. Prepara variáveis de ambiente:
   - Extrai diretório do mutante
   - Prepara `PYTHONPATH` com diretório do mutante no início
   - Inclui caminho PySpark (`/opt/bitnami/spark/python`)
   - Preserva `PYTHONPATH` original para resolver dependências

2. Executa pytest em subprocess isolado:
   ```
   pytest <tests_path> -x -q --tb=short
   ```
   - `-x`: para no primeiro erro (fail-fast)
   - `-q`: modo quiet (menos verbosidade)
   - `--tb=short`: traceback reduzido
   - `timeout=120`: limite de 2 minutos por mutante

3. Classifica resultado pelo código de saída:
   - `0`: **survived** (todos testes passaram)
   - `1`: **killed** (teste falhou)
   - `2-5`: **error** (erro pytest)
   - timeout: **timeout** (exceção `TimeoutExpired`)

**Extração de testes falhados:**

Parser de saída do pytest identifica linhas `FAILED path::test_name` e extrai identificadores de teste:

```python
for line in stdout.splitlines():
    if line.strip().startswith("FAILED "):
        test_id = line.strip()[len("FAILED "):].strip()
        failed.append(test_id)
```

### 3.5 Camada de Relatório

#### 3.5.1 Reporter

**Responsabilidade:** Agregar resultados, calcular métricas e gerar relatórios HTML.

**Fluxo de processamento (método chaining):**

```python
reporter.calculate()    # Calcula mutation score
reporter.make_diff()    # Gera diffs entre original e mutantes
reporter.show_results() # Escreve relatório HTML
```

**Cálculo de Mutation Score:**

```
Mutation Score = killed / (killed + survived + timeout + error)
```

Mensuração de efetividade dos testes: denota fração de mutantes que os testes conseguiram detectar.

**Geração de diffs:**

Utiliza `difflib.unified_diff()` para comparar:
- Código original
- Código de cada mutante

Formato: unified diff padrão (compatível com `patch` e ferramentas de versionamento)

**Geração de relatório HTML:**

Estrutura do relatório:

1. **Summary Cards:**
   - Mutation Score (com codificação cromática: verde ≥80%, laranja ≥50%, vermelho <50%)
   - Contadores: Total, Killed, Survived, Timeout, Error

2. **Tabela de Resultados:**
   - Identificador do mutante
   - Operador que o gerou
   - Linha de código modificada
   - Status (com estilo visual)
   - Testes que falharam
   - Tempo de execução

3. **Seção de Diffs:**
   - Secções expansíveis (`<details>`) por mutante
   - Colorização: verde para adições, vermelho para remoções
   - Syntax highlighting para código

4. **Dump JSON:**
   - Métricas estruturadas em JSON
   - Compatível com ferramentas de processamento

## 4. Padrões de Design Empregados

### 4.1 Padrão Strategy

Implementado através da classe abstrata `Operator` com subclasses concretas `OperatorNFTP` e `OperatorMTR`. Permite:

- Encapsular diferentes estratégias de mutação
- Trocar estratégias em tempo de execução
- Adicionar novas estratégias sem modificar código existente (princípio Open/Closed)

### 4.2 Padrão Registry

`OPERATOR_REGISTRY` mapeia identificadores de string para dotted paths de módulos:

```python
OPERATOR_REGISTRY = {
    "NFTP": "src.operators.operator_nftp.OperatorNFTP",
    "MTR": "src.operators.operator_mtr.OperatorMTR",
}
```

Possibilita:

- Registro dinâmico de novos operadores
- Carregamento lazy via `importlib`
- Desacoplamento entre componentes

### 4.3 Padrão Orchestrator

`MutationManager` coordena componentes `Operator`, `TestRunner` e `Reporter` através de um fluxo bem definido, implementando o padrão **Chain of Responsibility** onde cada etapa depende da anterior.

### 4.4 Padrão Data Transfer Object (DTO)

Classes `@dataclass`:
- `Mutant`: Encapsula metadados de um mutante
- `TestResult`: Encapsula resultado de teste de um mutante
- `ConfigLoader`: Encapsula configuração da execução

Benefícios:

- Type checking automático
- Serialização/deserialização facilitada
- Representação clara de contrato de dados
- `__repr__` automático para debugging

### 4.5 Padrão Factory com Injeção de Dependência

`loadOperator()` do `MutationManager` implementa factory pattern:

```python
def _load_operator(self, op_name):
    dotted_path = OPERATOR_REGISTRY[op_name]
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    operator_class = getattr(module, class_name)
    return operator_class()
```

Desacopla criação de instâncias da lógica principal.

## 5. Fluxo de Dados

```
┌────────────────────────────┐
│   Arquivo de Configuração  │
│      (config.txt)          │
└────────────┬───────────────┘
             │
             ↓
┌────────────────────────────┐
│  MutationManager.load()    │
│  → ConfigLoader            │
└────────────┬───────────────┘
             │
             ↓
┌────────────────────────────┐
│ Código-fonte PySpark       │
│      (*.py)                │
└────────────┬───────────────┘
             │
             ↓
┌────────────────────────────────────────┐
│ MutationManager.parse_to_ast()         │
│ → AST (Abstract Syntax Tree)           │
└────────────┬────────────────────────────┘
             │
             ↓
┌────────────────────────────────────────────┐
│ MutationManager.apply_mutation()           │
│ → Operator.analyse_ast(AST)                │
│   → Lista de nós elegíveis                 │
│ → Operator.build_mutant(nodes)             │
│   → Mutantes gerados em disco              │
│   → Lista[Mutant]                          │
└────────────┬───────────────────────────────┘
             │
             ↓
┌──────────────────────────────────────────────┐
│ MutationManager.run_tests()                  │
│ → TestRunner.run_test()                      │
│   → Executa pytest para cada mutante         │
│   → Lista[TestResult]                        │
└────────────┬─────────────────────────────────┘
             │
             ↓
┌────────────────────────────────────────────────┐
│ MutationManager.agregate_results()             │
│ → Reporter.calculate()                         │
│   → Calcula Mutation Score                     │
│ → Reporter.make_diff()                         │
│   → Gera diffs original vs mutantes            │
│ → Reporter.show_results()                      │
│   → Escreve report.html                        │
└────────────────────────────────────────────────┘
```

## 6. Estrutura de Diretórios para Mutantes

Os mutantes gerados seguem uma estrutura hierárquica:

```
mutants/
├── nftp_1_filter_line42/
│   └── nftp.py           # Mutante gerado por NFTP
├── nftp_2_where_line15/
│   └── nftp.py
├── mtr_1_map_zero/
│   └── mtr.py            # Mutante gerado por MTR
├── mtr_2_withColumn_one/
│   └── mtr.py
└── mtr_3_foreach_none/
    └── mtr.py
```

**Nomenclatura:**
- `{operador}_{id}_{metodo|parametro}_{modificadores}/`: nome do diretório
- `{operador}.py`: arquivo de código mutante

**Vantagens:**

- Rastreabilidade: nome identifica operador e linha modificada
- Isolamento: cada mutante em subdiretório próprio
- Importabilidade: estrutura permite injetar mutante via `PYTHONPATH`

## 7. Mecanismos de Extensibilidade

### 7.1 Adição de Novos Operadores

Para adicionar novo operador de mutação:

1. Criar classe herdando de `Operator`:
   ```python
   class OperatorNOVO(Operator):
       def __init__(self):
           super().__init__(id=3, name="NOVO", mutant_registers=["target"])

       def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
           # Implementar lógica de análise
           pass

       def build_mutant(self, nodes, original_ast, original_path, mutant_dir) -> list[Mutant]:
           # Implementar lógica de geração
           pass
   ```

2. Registrar no `OPERATOR_REGISTRY`:
   ```python
   OPERATOR_REGISTRY = {
       ...
       "NOVO": "src.operators.operator_novo.OperatorNOVO",
   }
   ```

3. Referenciar na configuração:
   ```
   operators_list = NFTP, MTR, NOVO
   ```

### 7.2 Customização de Comportamento

**Substituições em MTR:**

A lista `_REPLACEMENTS` em `operator_mtr.py` pode ser estendida:

```python
_REPLACEMENTS: list[tuple[str, ast.expr]] = [
    ("zero", ast.Constant(value=0)),
    # ... adicionar novas substituições aqui
]
```

**Métodos alvo em MTR:**

`mutant_registers` pode ser customizado na instância:

```python
operator = OperatorMTR(mutant_registers=["map", "flatMap"])  # Apenas estes métodos
```

## 8. Tecnologias e Dependências

### 8.1 Python Core

- **ast:** Parsing e manipulação de árvores sintáticas
- **subprocess:** Execução de pytest em subprocessos isolados
- **concurrent.futures:** Paralelismo com `ThreadPoolExecutor`
- **pathlib:** Manipulação de caminhos (multiplataforma)
- **dataclasses:** Definição de DTOs com validação automática
- **importlib:** Carregamento dinâmico de módulos
- **difflib:** Geração de unified diffs
- **json:** Serialização de resultados
- **logging:** Rastreamento de execução

### 8.2 Dependências Externas

- **pytest:** Framework de testes (executado em subprocessos)
- **PySpark:** Framework de processamento paralelo (não é dependência direta, mas alvo de teste)

## 9. Invariantes Arquiteturais

### 9.1 Imutabilidade da AST Original

A AST original nunca é mutada in-place. Cada operação de mutação trabalha sobre uma deep-copy:

```python
tree_copy = copy.deepcopy(original_ast)  # Cópia fresca
# ... aplicar mutações em tree_copy
mutant_source = ast.unparse(tree_copy)   # Unparse da cópia
```

**Rationale:** Evita efeitos colaterais entre mutantes e garante independência.

### 9.2 Isolamento de Testes

Cada teste é executado em subprocess separado com `PYTHONPATH` customizado:

```python
env["PYTHONPATH"] = os.pathsep.join([mutant_dir, ...])
proc = subprocess.run([pytest, ...], env=env, timeout=120)
```

**Rationale:** Evita importações em cache, simula ambiente real, detecta side effects.

### 9.3 Rastreabilidade Completa

Cada entidade mantém:
- Mutante: `id`, `original_path`, `mutant_path`, `operator`, `modified_line`
- Resultado: `mutant` (seu id), `status`, `failed_tests`, `execution_time`

**Rationale:** Permite auditoria completa e debugging.

## 10. Considerações de Performance

### 10.1 Paralelismo

- Testes executados em `min(4, cpu_count)` workers
- Limite evita contenção com SparkSession
- ThreadPoolExecutor bem-suited para I/O-bound tasks (subprocess)

### 10.2 Timeout

- Cada mutante tem timeout de 120 segundos
- Previne travamentos de testes ruins
- Mutantes timeout são registrados como status "timeout"

### 10.3 Limpeza de Mutantes

Código gerado é escrito em disco e reutilizado:
- Valida AST antes de escrever
- Utiliza `ast.fix_missing_locations()` para garantir validez
- Subprocesses carregam de disco sem duplication em memória

## 11. Limitações e Melhorias Futuras

### 11.1 Limitações Atuais

1. **Suporte a linguagem:** Atualmente opera apenas em Python (PySpark)
2. **Granularidade de mutação:** Limitada ao nível de subcondições (NFTP) e funções (MTR)
3. **Detecção de comportamento equivalente:** Não detecta mutantes que mantêm semântica idêntica
4. **Customização de predicados de teste:** Presume toda suite de testes é relevante

### 11.2 Extensões Potenciais

1. **Novos operadores:** AOR (Arithmetic Operator Replacement), ROR (Relational Operator Replacement)
2. **Machine learning:** Priorizar mutantes por probabilidade de serem vivos
3. **Integração CI/CD:** Webhooks, relatórios em pull requests
4. **Otimização de execução:** Caching de resultados, análise incremental

## 12. Conclusão

A arquitetura de TransmutPyspark demonstra separação clara de responsabilidades, extensibilidade através de padrões consolidados e robustez através de validações em múltiplos níveis. O design em camadas permite que mudanças em um componente tenham impacto mínimo em outros, enquanto o uso extensivo de dataclasses e type hints facilita debugging e manutenção.

A ferramenta fornece uma base sólida para pesquisa em teste de mutação em aplicações de processamento paralelo de dados, com a capacidade de evolução para suportar novos operadores e estratégias de análise conforme a pesquisa avança.
