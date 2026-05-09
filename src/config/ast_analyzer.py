"""
ast_analyzer.py
===============
Três responsabilidades em um único módulo:

  1. Descoberta  — encontra FunctionDef / métodos de ClassDef elegíveis
  2. Filtragem   — ignora main(), I/O puro, dunders, decoradores de orquestração
  3. Mapeamento  — test → função via convenção de nomes + análise de imports

O resultado é uma lista de FunctionTarget, cada um com os test_files e
test_functions que o cobrem — informação usada pelo TestRunner para rodar
só os testes relevantes para cada mutante.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────── #
# Constantes de filtragem                                                  #
# ─────────────────────────────────────────────────────────────────────── #

# Nomes que nunca devem ser mutados
_IGNORED_NAMES: frozenset[str] = frozenset({
    "main", "setup", "teardown", "setUp", "tearDown",
    "__init__", "__repr__", "__str__", "__eq__",
    "__hash__", "__len__", "__iter__", "__next__",
})

# Decoradores que indicam ponto de entrada / orquestração
_ORCHESTRATION_DECORATORS: frozenset[str] = frozenset({
    "task", "dag",              # Airflow
    "flow", "step",             # Prefect / Metaflow
    "pytest.fixture", "fixture",
    "property",
    "classmethod",
    "abstractmethod",
})

# Chamadas que caracterizam leitura/escrita sem lógica de transformação
_IO_METHOD_NAMES: frozenset[str] = frozenset({
    # leitura
    "read", "read_csv", "read_parquet", "read_json", "read_orc",
    "read_delta", "read_table", "load", "read_text",
    # escrita
    "write", "save", "to_csv", "to_parquet", "to_json",
    "insertInto", "saveAsTable", "write_text",
    # contexto Spark
    "getOrCreate", "builder", "enableHiveSupport",
})

# Chamadas que indicam lógica de transformação PySpark
_TRANSFORM_METHOD_NAMES: frozenset[str] = frozenset({
    "filter", "where", "select", "withColumn", "withColumnRenamed",
    "groupBy", "agg", "join", "union", "unionAll", "unionByName",
    "drop", "distinct", "orderBy", "sort", "limit",
    "map", "flatMap", "mapInPandas", "mapInArrow",
    "cast", "coalesce", "repartition",
    "fillna", "dropna", "replace",
})


# ─────────────────────────────────────────────────────────────────────── #
# Modelo de dados                                                          #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class FunctionTarget:
    """
    Unidade de trabalho consumida pelo MutationManager.
    Representa uma função/método elegível para mutação,
    já com os testes que a cobrem mapeados.
    """
    name:           str             # "compute_revenue"
    qualified_name: str             # "ClassName.method" ou só "name"
    source_file:    Path
    class_name:     str | None      # None se for função de módulo
    lineno:         int
    node:           ast.FunctionDef # nó original — usado pelos operadores
    test_functions: list[str] = field(default_factory=list)
    test_files:     list[Path] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"FunctionTarget("
            f"qualified_name={self.qualified_name!r}, "
            f"file={self.source_file.name!r}, "
            f"line={self.lineno}, "
            f"tests={len(self.test_functions)})"
        )


# ─────────────────────────────────────────────────────────────────────── #
# 1 + 2 — Descoberta e filtragem                                           #
# ─────────────────────────────────────────────────────────────────────── #

def extract_targets(source_file: Path) -> list[FunctionTarget]:
    """
    Lê source_file, percorre sua AST e devolve todas as funções/métodos
    elegíveis para mutação (já filtrados).
    """
    # Sempre trabalha com path absoluto para garantir unicidade como chave
    source_file = source_file.resolve()
    try:
        tree = ast.parse(source_file.read_text(encoding="utf-8"),
                         filename=str(source_file))
    except SyntaxError as e:
        logger.warning(f"[ast_analyzer] SyntaxError em {source_file}: {e} — ignorado.")
        return []

    targets: list[FunctionTarget] = []

    for node in ast.walk(tree):

        # ── Métodos de classe ───────────────────────────────────────────
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _is_eligible(item, inside_class=True):
                        targets.append(FunctionTarget(
                            name=item.name,
                            qualified_name=f"{node.name}.{item.name}",
                            source_file=source_file,
                            class_name=node.name,
                            lineno=item.lineno,
                            node=item,
                        ))

        # ── Funções de módulo (top-level) ───────────────────────────────
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_top_level(node, tree) and _is_eligible(node):
                targets.append(FunctionTarget(
                    name=node.name,
                    qualified_name=node.name,
                    source_file=source_file,
                    class_name=None,
                    lineno=node.lineno,
                    node=node,
                ))

    logger.info(
        f"[ast_analyzer] {source_file.name}: "
        f"{len(targets)} target(s) elegíveis encontrados."
    )
    return targets


def _is_top_level(
    node:  ast.FunctionDef | ast.AsyncFunctionDef,
    tree:  ast.Module,
) -> bool:
    """Confirma que a função é filha direta do módulo (não de uma classe)."""
    return isinstance(tree, ast.Module) and node in tree.body


def _is_eligible(
    node:         ast.FunctionDef | ast.AsyncFunctionDef,
    inside_class: bool = False,
) -> bool:
    """
    Retorna True se a função deve ser incluída como alvo de mutação.
    """
    name = node.name

    # Ignora nomes reservados explícitos
    if name in _IGNORED_NAMES:
        return False

    # Ignora todos os dunders
    if name.startswith("__") and name.endswith("__"):
        return False

    # Ignora funções privadas de módulo (prefixo _)
    # Métodos privados de classe podem conter lógica relevante — mantidos
    if not inside_class and name.startswith("_"):
        return False

    # Verifica decoradores problemáticos
    for deco in node.decorator_list:
        if _deco_name(deco) in _ORCHESTRATION_DECORATORS:
            return False

    # Ignora funções que são puro I/O sem nenhuma transformação
    if _is_pure_io(node):
        return False

    return True


def _deco_name(deco: ast.expr) -> str:
    """Extrai o nome textual de um decorador (simples ou com atributo)."""
    if isinstance(deco, ast.Name):
        return deco.id
    if isinstance(deco, ast.Attribute):
        return f"{_deco_name(deco.value)}.{deco.attr}"
    return ""


def _is_pure_io(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """
    Heurística: se o corpo tem chamadas de I/O e *nenhuma* chamada de
    transformação PySpark, a função não tem o que mutar de forma útil.
    """
    io_count        = 0
    transform_count = 0

    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            if child.attr in _IO_METHOD_NAMES:
                io_count += 1
            elif child.attr in _TRANSFORM_METHOD_NAMES:
                transform_count += 1

    return io_count > 0 and transform_count == 0


# ─────────────────────────────────────────────────────────────────────── #
# 3 — Mapeamento test → função                                            #
# ─────────────────────────────────────────────────────────────────────── #

def map_tests_to_targets(
    targets:    list[FunctionTarget],
    test_files: list[Path],
) -> list[FunctionTarget]:
    """
    Para cada FunctionTarget, descobre quais test_files e test_functions
    o cobrem.  Estratégias combinadas:
      A) análise de imports  — filtra quais arquivos de teste importam o fonte
      B) convenção de nomes  — filtra quais funções de teste batem com o alvo
    """
    # Pré-computa imports e nomes de teste por arquivo (evita re-parse)
    file_imports:   dict[Path, set[str]]  = {}
    file_testfuncs: dict[Path, list[str]] = {}

    for tf in test_files:
        file_imports[tf]   = _extract_imports(tf)
        file_testfuncs[tf] = _extract_test_function_names(tf)

    for target in targets:
        source_module = target.source_file.stem   # "uts" de "uts.py"
        matched_files: list[Path] = []
        matched_funcs: list[str]  = []

        for tf in test_files:
            # Estratégia A: o arquivo de teste importa o módulo fonte?
            if source_module not in file_imports[tf]:
                continue

            matched_files.append(tf)

            # Estratégia B: quais funções de teste batem com este target?
            for test_func in file_testfuncs[tf]:
                if _names_match(test_func, target.qualified_name):
                    matched_funcs.append(test_func)

        target.test_files     = matched_files
        target.test_functions = matched_funcs

        if not matched_files:
            logger.warning(
                f"[ast_analyzer] Nenhum teste mapeado para "
                f"'{target.qualified_name}' em {target.source_file.name}."
            )

    return targets


def _extract_imports(test_file: Path) -> set[str]:
    """
    Extrai os nomes de módulo importados no arquivo de teste.
    Ex.: 'from etl_code.uts import ...' → {'uts', 'etl_code.uts', 'etl_code'}
    """
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            # Adiciona cada segmento do caminho pontilhado
            for part in node.module.split("."):
                modules.add(part)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
                for part in alias.name.split("."):
                    modules.add(part)

    return modules


def _extract_test_function_names(test_file: Path) -> list[str]:
    """
    Retorna os nomes de todas as funções/métodos que começam com 'test_'
    (ou 'Test') dentro do arquivo.
    """
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.lower().startswith("test"):
                names.append(node.name)
    return names


def _names_match(test_func_name: str, target_qualified: str) -> bool:
    """
    Heurísticas de correspondência nome de teste ↔ função alvo.

    Exemplos que casam:
      test_compute_revenue    ↔  compute_revenue
      test_SalesJob_transform ↔  SalesJob.transform
      test_transform          ↔  SalesJob.transform   (match parcial)
      TestComputeRevenue      ↔  compute_revenue       (case-insensitive)
    """
    candidate = test_func_name
    for prefix in ("test_", "Test"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
            break

    # Match exato (função de módulo)
    if candidate == target_qualified:
        return True

    # Match case-insensitive
    if candidate.lower() == target_qualified.lower():
        return True

    # Match com classe: "SalesJob_transform" → "SalesJob.transform"
    if "_" in candidate:
        dot_version = candidate.replace("_", ".", 1)
        if dot_version == target_qualified:
            return True

    # Match parcial: "transform" cobre "SalesJob.transform"
    if "." in target_qualified:
        method_name = target_qualified.split(".")[-1]
        if candidate == method_name or candidate.lower() == method_name.lower():
            return True

    # Match por prefixo: "compute_revenue_basic" começa com "compute_revenue"
    # Cobre padrões como test_compute_revenue_basic → compute_revenue
    func_name = target_qualified.split(".")[-1]   # método ou função simples
    if candidate.startswith(func_name) or candidate.lower().startswith(func_name.lower()):
        return True

    return False


# ─────────────────────────────────────────────────────────────────────── #
# Entry point público                                                      #
# ─────────────────────────────────────────────────────────────────────── #

def analyze(
    source_files: list[Path],
    test_files:   list[Path],
) -> list[FunctionTarget]:
    """
    Função pública única.  Recebe listas de arquivos e devolve
    FunctionTargets já filtrados e com testes mapeados.
    """
    all_targets: list[FunctionTarget] = []
    for sf in source_files:
        all_targets.extend(extract_targets(sf))

    all_targets = map_tests_to_targets(all_targets, test_files)

    total_with_tests = sum(1 for t in all_targets if t.test_files)
    logger.info(
        f"[ast_analyzer] Total: {len(all_targets)} targets, "
        f"{total_with_tests} com testes mapeados."
    )
    return all_targets
