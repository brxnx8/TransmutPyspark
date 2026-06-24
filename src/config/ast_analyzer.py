from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


_IGNORED_NAMES: frozenset[str] = frozenset({
    "main", "setup", "teardown", "setUp", "tearDown",
    "__init__", "__repr__", "__str__", "__eq__",
    "__hash__", "__len__", "__iter__", "__next__",
})

_ORCHESTRATION_DECORATORS: frozenset[str] = frozenset({
    "task", "dag",
    "flow", "step",
    "pytest.fixture", "fixture",
    "property",
    "classmethod",
    "abstractmethod",
})

_IO_METHOD_NAMES: frozenset[str] = frozenset({
    "read", "read_csv", "read_parquet", "read_json", "read_orc",
    "read_delta", "read_table", "load", "read_text",
    "write", "save", "to_csv", "to_parquet", "to_json",
    "insertInto", "saveAsTable", "write_text",
    "getOrCreate", "builder", "enableHiveSupport",
})

_TRANSFORM_METHOD_NAMES: frozenset[str] = frozenset({
    "filter", "where", "select", "withColumn", "withColumnRenamed",
    "groupBy", "agg", "join", "union", "unionAll", "unionByName",
    "drop", "distinct", "orderBy", "sort", "limit",
    "map", "flatMap", "mapInPandas", "mapInArrow",
    "cast", "coalesce", "repartition",
    "fillna", "dropna", "replace",
})



@dataclass
class FunctionTarget:
    name:           str
    qualified_name: str
    source_file:    Path
    class_name:     str | None
    lineno:         int
    node:           ast.FunctionDef
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


def extract_targets(source_file: Path) -> list[FunctionTarget]:
    source_file = source_file.resolve()
    try:
        tree = ast.parse(source_file.read_text(encoding="utf-8"),
                         filename=str(source_file))
    except SyntaxError as e:
        logger.warning(f"[ast_analyzer] SyntaxError em {source_file}: {e} — ignorado.")
        return []

    targets: list[FunctionTarget] = []

    for node in ast.walk(tree):

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
    return isinstance(tree, ast.Module) and node in tree.body


def _is_eligible(
    node:         ast.FunctionDef | ast.AsyncFunctionDef,
    inside_class: bool = False,
) -> bool:
    name = node.name

    if name in _IGNORED_NAMES:
        return False

    if name.startswith("__") and name.endswith("__"):
        return False

    if not inside_class and name.startswith("_"):
        return False

    for deco in node.decorator_list:
        if _deco_name(deco) in _ORCHESTRATION_DECORATORS:
            return False

    if _is_pure_io(node):
        return False

    return True


def _deco_name(deco: ast.expr) -> str:
    if isinstance(deco, ast.Name):
        return deco.id
    if isinstance(deco, ast.Attribute):
        return f"{_deco_name(deco.value)}.{deco.attr}"
    return ""


def _is_pure_io(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    io_count        = 0
    transform_count = 0

    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            if child.attr in _IO_METHOD_NAMES:
                io_count += 1
            elif child.attr in _TRANSFORM_METHOD_NAMES:
                transform_count += 1

    return io_count > 0 and transform_count == 0



def map_tests_to_targets(
    targets:    list[FunctionTarget],
    test_files: list[Path],
) -> list[FunctionTarget]:
    file_imports:   dict[Path, set[str]]  = {}
    file_testfuncs: dict[Path, list[str]] = {}

    for tf in test_files:
        file_imports[tf]   = _extract_imports(tf)
        file_testfuncs[tf] = _extract_test_function_names(tf)

    for target in targets:
        source_module = target.source_file.stem
        matched_files: list[Path] = []
        matched_funcs: list[str]  = []

        for tf in test_files:
            if source_module not in file_imports[tf]:
                continue

            matched_files.append(tf)

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
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            for part in node.module.split("."):
                modules.add(part)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
                for part in alias.name.split("."):
                    modules.add(part)

    return modules


def _extract_test_function_names(test_file: Path) -> list[str]:
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
    candidate = test_func_name
    for prefix in ("test_", "Test"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
            break

    if candidate == target_qualified:
        return True

    if candidate.lower() == target_qualified.lower():
        return True

    if "_" in candidate:
        dot_version = candidate.replace("_", ".", 1)
        if dot_version == target_qualified:
            return True

    if "." in target_qualified:
        method_name = target_qualified.split(".")[-1]
        if candidate == method_name or candidate.lower() == method_name.lower():
            return True

    func_name = target_qualified.split(".")[-1]   # método ou função simples
    if candidate.startswith(func_name) or candidate.lower().startswith(func_name.lower()):
        return True

    return False


def analyze(
    source_files: list[Path],
    test_files:   list[Path],
) -> list[FunctionTarget]:
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
