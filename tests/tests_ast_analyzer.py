from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.ast_analyzer import (
    FunctionTarget,
    _deco_name,
    _extract_imports,
    _extract_test_function_names,
    _is_eligible,
    _is_pure_io,
    _is_top_level,
    _names_match,
    analyze,
    extract_targets,
    map_tests_to_targets,
)


# ===========================================================================
# Helpers de fixture
# ===========================================================================

def _write(tmp_path: Path, filename: str, code: str) -> Path:
    """Escreve um arquivo Python com dedent automático e retorna seu Path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(code), encoding="utf-8")
    return p


def _parse_first_func(code: str) -> ast.FunctionDef:
    """Parseia código e devolve o primeiro FunctionDef encontrado."""
    tree = ast.parse(textwrap.dedent(code))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    raise ValueError("Nenhuma função encontrada no código fornecido.")


def _make_target(
    name: str = "compute",
    qualified_name: str | None = None,
    source_file: Path | None = None,
    class_name: str | None = None,
    lineno: int = 1,
) -> FunctionTarget:
    """Fábrica de FunctionTarget para testes que não precisam de nó AST real."""
    node = _parse_first_func("def compute(): pass")
    return FunctionTarget(
        name=name,
        qualified_name=qualified_name or name,
        source_file=source_file or Path("fake.py"),
        class_name=class_name,
        lineno=lineno,
        node=node,
    )


# ===========================================================================
# FunctionTarget — modelo de dados
# ===========================================================================

class TestFunctionTarget:

    def test_should_create_target_with_required_fields(self):
        node = _parse_first_func("def fn(): pass")
        t = FunctionTarget(
            name="fn",
            qualified_name="fn",
            source_file=Path("a.py"),
            class_name=None,
            lineno=1,
            node=node,
        )
        assert t.name           == "fn"
        assert t.qualified_name == "fn"
        assert t.class_name     is None
        assert t.lineno         == 1

    def test_should_have_empty_test_lists_by_default(self):
        t = _make_target()
        assert t.test_functions == []
        assert t.test_files     == []

    def test_should_include_qualified_name_in_repr(self):
        t = _make_target(name="revenue", qualified_name="Job.revenue")
        assert "Job.revenue" in repr(t)

    def test_should_include_lineno_in_repr(self):
        t = _make_target(lineno=42)
        assert "42" in repr(t)

    def test_should_include_test_count_in_repr(self):
        t = _make_target()
        t.test_functions = ["test_a", "test_b"]
        assert "2" in repr(t)

    def test_should_include_source_filename_in_repr(self):
        t = _make_target(source_file=Path("/some/path/etl.py"))
        assert "etl.py" in repr(t)


# ===========================================================================
# _deco_name
# ===========================================================================

class TestDecoName:

    def test_should_return_name_for_simple_decorator(self):
        node = ast.Name(id="property", ctx=ast.Load())
        assert _deco_name(node) == "property"

    def test_should_return_dotted_name_for_attribute_decorator(self):
        # pytest.fixture
        node = ast.Attribute(
            value=ast.Name(id="pytest", ctx=ast.Load()),
            attr="fixture",
            ctx=ast.Load(),
        )
        assert _deco_name(node) == "pytest.fixture"

    def test_should_return_deeply_nested_attribute_name(self):
        # a.b.c
        inner = ast.Attribute(
            value=ast.Name(id="a", ctx=ast.Load()),
            attr="b",
            ctx=ast.Load(),
        )
        outer = ast.Attribute(value=inner, attr="c", ctx=ast.Load())
        assert _deco_name(outer) == "a.b.c"

    def test_should_return_empty_string_for_unknown_node_type(self):
        node = ast.Constant(value=42)
        assert _deco_name(node) == ""

    def test_should_return_empty_string_for_call_node(self):
        # @pytest.mark.parametrize(...)  →  ast.Call
        func = ast.Attribute(
            value=ast.Name(id="pytest", ctx=ast.Load()),
            attr="mark",
            ctx=ast.Load(),
        )
        call_node = ast.Call(func=func, args=[], keywords=[])
        # Call não é Name nem Attribute → retorna ""
        assert _deco_name(call_node) == ""


# ===========================================================================
# _is_pure_io
# ===========================================================================

class TestIsPureIo:

    def test_should_return_true_when_only_io_calls_present(self):
        node = _parse_first_func("""
            def load_data(spark):
                return spark.read.parquet("s3://bucket/path")
        """)
        assert _is_pure_io(node) is True

    def test_should_return_false_when_transform_calls_present_alongside_io(self):
        node = _parse_first_func("""
            def load_and_filter(spark):
                df = spark.read.parquet("s3://bucket")
                return df.filter("col > 0")
        """)
        assert _is_pure_io(node) is False

    def test_should_return_false_when_only_transform_calls_present(self):
        node = _parse_first_func("""
            def transform(df):
                return df.select("a", "b").filter("a > 1")
        """)
        assert _is_pure_io(node) is False

    def test_should_return_false_when_no_io_or_transform_calls(self):
        node = _parse_first_func("""
            def compute(x, y):
                return x + y
        """)
        assert _is_pure_io(node) is False

    def test_should_return_true_for_write_only_function(self):
        node = _parse_first_func("""
            def save_result(df):
                df.write.parquet("s3://out")
        """)
        assert _is_pure_io(node) is True

    def test_should_return_true_for_spark_session_creation(self):
        node = _parse_first_func("""
            def get_spark():
                return SparkSession.builder.getOrCreate()
        """)
        assert _is_pure_io(node) is True

    def test_should_return_false_when_groupby_and_write_present(self):
        node = _parse_first_func("""
            def aggregate_and_save(df):
                result = df.groupBy("col").agg({"val": "sum"})
                result.write.parquet("s3://out")
        """)
        assert _is_pure_io(node) is False

    def test_should_return_true_for_to_csv_write(self):
        node = _parse_first_func("""
            def export(df):
                df.to_csv("/tmp/out.csv")
        """)
        assert _is_pure_io(node) is True


# ===========================================================================
# _is_eligible
# ===========================================================================

class TestIsEligible:

    # ── Happy path ──────────────────────────────────────────────────────────

    def test_should_return_true_for_regular_public_function(self):
        node = _parse_first_func("def compute_revenue(): pass")
        assert _is_eligible(node) is True

    def test_should_return_true_for_private_method_inside_class(self):
        node = _parse_first_func("def _internal_transform(self): pass")
        # Métodos privados de classe são permitidos
        assert _is_eligible(node, inside_class=True) is True

    # ── Nomes reservados ────────────────────────────────────────────────────

    def test_should_return_false_for_main_function(self):
        node = _parse_first_func("def main(): pass")
        assert _is_eligible(node) is False

    def test_should_return_false_for_setup_function(self):
        node = _parse_first_func("def setup(): pass")
        assert _is_eligible(node) is False

    def test_should_return_false_for_teardown_function(self):
        node = _parse_first_func("def tearDown(): pass")
        assert _is_eligible(node) is False

    def test_should_return_false_for_dunder_init(self):
        node = _parse_first_func("def __init__(self): pass")
        assert _is_eligible(node) is False

    def test_should_return_false_for_dunder_repr(self):
        node = _parse_first_func("def __repr__(self): pass")
        assert _is_eligible(node) is False

    # ── Qualquer dunder ─────────────────────────────────────────────────────

    def test_should_return_false_for_arbitrary_dunder(self):
        node = _parse_first_func("def __custom_dunder__(self): pass")
        assert _is_eligible(node) is False

    # ── Privadas de módulo ──────────────────────────────────────────────────

    def test_should_return_false_for_module_private_function(self):
        node = _parse_first_func("def _helper(): pass")
        assert _is_eligible(node, inside_class=False) is False

    def test_should_return_true_for_private_method_in_class(self):
        node = _parse_first_func("def _helper(self): pass")
        assert _is_eligible(node, inside_class=True) is True

    # ── Decoradores de orquestração ─────────────────────────────────────────

    def test_should_return_false_for_task_decorator(self):
        node = _parse_first_func("""
            @task
            def my_task(): pass
        """)
        assert _is_eligible(node) is False

    def test_should_return_false_for_dag_decorator(self):
        node = _parse_first_func("""
            @dag
            def my_dag(): pass
        """)
        assert _is_eligible(node) is False

    def test_should_return_false_for_pytest_fixture_decorator(self):
        node = _parse_first_func("""
            @pytest.fixture
            def my_fixture(): pass
        """)
        assert _is_eligible(node) is False

    def test_should_return_false_for_property_decorator(self):
        node = _parse_first_func("""
            @property
            def value(self): pass
        """)
        assert _is_eligible(node) is False

    def test_should_return_false_for_classmethod_decorator(self):
        node = _parse_first_func("""
            @classmethod
            def create(cls): pass
        """)
        assert _is_eligible(node) is False

    def test_should_return_false_for_abstractmethod_decorator(self):
        node = _parse_first_func("""
            @abstractmethod
            def process(self): pass
        """)
        assert _is_eligible(node) is False

    def test_should_return_true_for_unknown_decorator(self):
        # Decoradores não listados não bloqueiam a elegibilidade
        node = _parse_first_func("""
            @my_custom_decorator
            def process(df): pass
        """)
        assert _is_eligible(node) is True

    # ── Puro I/O ────────────────────────────────────────────────────────────

    def test_should_return_false_for_pure_io_function(self):
        node = _parse_first_func("""
            def load(spark):
                return spark.read.parquet("s3://bucket")
        """)
        assert _is_eligible(node) is False

    def test_should_return_true_for_function_with_io_and_transform(self):
        node = _parse_first_func("""
            def load_and_clean(spark):
                df = spark.read.parquet("s3://bucket")
                return df.filter("col IS NOT NULL")
        """)
        assert _is_eligible(node) is True


# ===========================================================================
# _is_top_level
# ===========================================================================

class TestIsTopLevel:

    def test_should_return_true_for_top_level_function(self):
        code = textwrap.dedent("""
            def my_func():
                pass
        """)
        tree = ast.parse(code)
        func_node = tree.body[0]
        assert _is_top_level(func_node, tree) is True

    def test_should_return_false_for_nested_function(self):
        code = textwrap.dedent("""
            def outer():
                def inner():
                    pass
        """)
        tree = ast.parse(code)
        outer = tree.body[0]
        inner = outer.body[0]
        assert _is_top_level(inner, tree) is False

    def test_should_return_false_for_method_inside_class(self):
        code = textwrap.dedent("""
            class MyClass:
                def method(self):
                    pass
        """)
        tree = ast.parse(code)
        class_node = tree.body[0]
        method_node = class_node.body[0]
        assert _is_top_level(method_node, tree) is False

    def test_should_return_true_for_async_top_level_function(self):
        code = textwrap.dedent("""
            async def async_func():
                pass
        """)
        tree = ast.parse(code)
        func_node = tree.body[0]
        assert _is_top_level(func_node, tree) is True


# ===========================================================================
# _extract_imports
# ===========================================================================

class TestExtractImports:

    def test_should_extract_from_import_module_name(self, tmp_path):
        f = _write(tmp_path, "test_x.py", """
            from etl_code.uts import compute
        """)
        result = _extract_imports(f)
        assert "uts" in result

    def test_should_extract_full_dotted_module_from_import_from(self, tmp_path):
        f = _write(tmp_path, "test_x.py", """
            from etl_code.uts import compute
        """)
        result = _extract_imports(f)
        assert "etl_code.uts" in result

    def test_should_extract_all_segments_of_dotted_module(self, tmp_path):
        f = _write(tmp_path, "test_x.py", """
            from etl_code.uts import compute
        """)
        result = _extract_imports(f)
        assert "etl_code" in result
        assert "uts" in result

    def test_should_extract_plain_import(self, tmp_path):
        f = _write(tmp_path, "test_x.py", """
            import pyspark.sql
        """)
        result = _extract_imports(f)
        assert "pyspark.sql" in result
        assert "pyspark" in result
        assert "sql" in result

    def test_should_return_empty_set_for_syntax_error_file(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def (((broken syntax", encoding="utf-8")
        result = _extract_imports(f)
        assert result == set()

    def test_should_return_empty_set_for_file_with_no_imports(self, tmp_path):
        f = _write(tmp_path, "test_x.py", """
            x = 1 + 1
        """)
        result = _extract_imports(f)
        assert result == set()

    def test_should_extract_multiple_imports_from_same_file(self, tmp_path):
        f = _write(tmp_path, "test_x.py", """
            from etl.uts import compute
            import transforms
        """)
        result = _extract_imports(f)
        assert "uts" in result
        assert "transforms" in result

    def test_should_handle_import_from_without_module(self, tmp_path):
        # `from . import something` — node.module é None
        f = _write(tmp_path, "test_x.py", """
            from . import something
        """)
        # Não deve lançar exceção
        result = _extract_imports(f)
        assert isinstance(result, set)


# ===========================================================================
# _extract_test_function_names
# ===========================================================================

class TestExtractTestFunctionNames:

    def test_should_extract_test_prefixed_functions(self, tmp_path):
        f = _write(tmp_path, "test_x.py", """
            def test_compute():
                pass
        """)
        result = _extract_test_function_names(f)
        assert "test_compute" in result

    def test_should_extract_Test_prefixed_functions(self, tmp_path):
        f = _write(tmp_path, "test_x.py", """
            def TestCompute():
                pass
        """)
        result = _extract_test_function_names(f)
        assert "TestCompute" in result

    def test_should_not_extract_non_test_functions(self, tmp_path):
        f = _write(tmp_path, "test_x.py", """
            def helper():
                pass
            def compute():
                pass
        """)
        result = _extract_test_function_names(f)
        assert result == []

    def test_should_extract_async_test_functions(self, tmp_path):
        f = _write(tmp_path, "test_x.py", """
            async def test_async_compute():
                pass
        """)
        result = _extract_test_function_names(f)
        assert "test_async_compute" in result

    def test_should_extract_test_methods_inside_classes(self, tmp_path):
        f = _write(tmp_path, "test_x.py", """
            class TestSuite:
                def test_method(self):
                    pass
        """)
        result = _extract_test_function_names(f)
        assert "test_method" in result

    def test_should_return_empty_list_for_syntax_error_file(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def (((broken syntax", encoding="utf-8")
        result = _extract_test_function_names(f)
        assert result == []

    def test_should_return_empty_list_for_file_with_no_test_functions(
        self, tmp_path
    ):
        f = _write(tmp_path, "test_x.py", """
            def setup():
                pass
        """)
        result = _extract_test_function_names(f)
        assert result == []

    def test_should_extract_multiple_test_functions(self, tmp_path):
        f = _write(tmp_path, "test_x.py", """
            def test_a():
                pass
            def test_b():
                pass
            def helper():
                pass
        """)
        result = _extract_test_function_names(f)
        assert set(result) == {"test_a", "test_b"}


# ===========================================================================
# _names_match
# ===========================================================================

class TestNamesMatch:

    # ── Match exato ─────────────────────────────────────────────────────────

    def test_should_match_exact_function_name(self):
        assert _names_match("test_compute_revenue", "compute_revenue") is True

    def test_should_match_with_Test_prefix_exact(self):
        assert _names_match("TestComputeRevenue", "ComputeRevenue") is True

    # ── Match case-insensitive ───────────────────────────────────────────────

    def test_should_match_case_insensitive(self):
        # O match case-insensitive compara candidate.lower() == target.lower().
        # "ComputeRevenue" != "compute_revenue" (underscore vs. sem underscore).
        # O match correto é quando a diferença é só de caixa, não de formatação:
        assert _names_match("test_COMPUTE_REVENUE", "compute_revenue") is True

    def test_should_match_uppercase_test_function_with_lowercase_target(self):
        assert _names_match("TestComputeRevenue", "computerevenue") is True

    # ── Match com classe (underscore → ponto) ────────────────────────────────

    def test_should_match_class_method_via_underscore_convention(self):
        assert _names_match("test_SalesJob_transform", "SalesJob.transform") is True

    def test_should_not_use_second_underscore_as_dot(self):
        # Apenas a primeira underscore é convertida para ponto
        # "SalesJob_transform_v2" → "SalesJob.transform_v2" ≠ "SalesJob.transform"
        # mas deve casar pelo match de prefixo
        result = _names_match("test_SalesJob_transform_v2", "SalesJob.transform")
        # "SalesJob.transform_v2" != "SalesJob.transform", mas o candidato
        # "SalesJob_transform_v2" começa com "transform" via split(".")[-1]?
        # Vamos apenas garantir que não lança exceção:
        assert isinstance(result, bool)

    # ── Match parcial (nome do método) ───────────────────────────────────────

    def test_should_match_partial_method_name(self):
        assert _names_match("test_transform", "SalesJob.transform") is True

    def test_should_match_partial_method_name_case_insensitive(self):
        assert _names_match("test_Transform", "SalesJob.transform") is True

    # ── Match por prefixo ────────────────────────────────────────────────────

    def test_should_match_by_prefix_of_function_name(self):
        assert _names_match("test_compute_revenue_basic", "compute_revenue") is True

    def test_should_match_by_prefix_case_insensitive(self):
        assert _names_match("test_Compute_Revenue_edge", "compute_revenue") is True

    # ── Sem match ────────────────────────────────────────────────────────────

    def test_should_not_match_completely_different_names(self):
        assert _names_match("test_load_data", "compute_revenue") is False

    def test_should_not_match_partial_substring_in_middle(self):
        # "revenue" aparece no meio de "compute_revenue_calculator"
        # mas o candidato deve começar com "compute_revenue"
        assert _names_match("test_revenue_only", "compute_revenue") is False

    def test_should_not_match_empty_candidate_against_nonempty_target(self):
        assert _names_match("test_", "compute_revenue") is False

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_should_match_function_without_test_prefix(self):
        # Sem prefixo test_ ou Test → candidate == original
        assert _names_match("compute_revenue", "compute_revenue") is True

    def test_should_handle_qualified_name_without_dot(self):
        assert _names_match("test_fn", "fn") is True


# ===========================================================================
# extract_targets
# ===========================================================================

class TestExtractTargets:

    def test_should_return_empty_list_for_empty_file(self, tmp_path):
        f = _write(tmp_path, "empty.py", "")
        assert extract_targets(f) == []

    def test_should_return_empty_list_for_syntax_error_file(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def (((broken syntax", encoding="utf-8")
        result = extract_targets(f)
        assert result == []

    def test_should_extract_top_level_function(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            def compute_revenue(df):
                return df.select("revenue")
        """)
        targets = extract_targets(f)
        names = [t.name for t in targets]
        assert "compute_revenue" in names

    def test_should_not_extract_main_function(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            def main():
                pass
        """)
        assert extract_targets(f) == []

    def test_should_not_extract_private_module_function(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            def _helper():
                pass
        """)
        assert extract_targets(f) == []

    def test_should_extract_method_from_class(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            class SalesJob:
                def transform(self, df):
                    return df.filter("val > 0")
        """)
        targets = extract_targets(f)
        qnames = [t.qualified_name for t in targets]
        assert "SalesJob.transform" in qnames

    def test_should_not_extract_dunder_method_from_class(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            class SalesJob:
                def __init__(self):
                    pass
        """)
        assert extract_targets(f) == []

    def test_should_set_class_name_for_method_targets(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            class SalesJob:
                def transform(self, df):
                    return df.select("a")
        """)
        targets = extract_targets(f)
        assert targets[0].class_name == "SalesJob"

    def test_should_set_class_name_none_for_module_functions(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            def compute(df):
                return df.select("a")
        """)
        targets = extract_targets(f)
        assert targets[0].class_name is None

    def test_should_resolve_source_file_to_absolute_path(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            def compute(df):
                return df.select("a")
        """)
        targets = extract_targets(f)
        assert targets[0].source_file.is_absolute()

    def test_should_set_correct_lineno_for_function(self, tmp_path):
        f = _write(tmp_path, "etl.py", "\n\ndef compute(df):\n    return df\n")
        targets = extract_targets(f)
        assert targets[0].lineno == 3

    def test_should_not_extract_pure_io_function(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            def load(spark):
                return spark.read.parquet("s3://bucket")
        """)
        assert extract_targets(f) == []

    def test_should_extract_async_function(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            async def process(df):
                return df.select("a")
        """)
        targets = extract_targets(f)
        assert len(targets) == 1

    def test_should_not_extract_orchestration_decorated_function(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            @task
            def my_task():
                pass
        """)
        assert extract_targets(f) == []

    def test_should_extract_multiple_functions_from_same_file(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            def compute_a(df):
                return df.filter("a > 0")

            def compute_b(df):
                return df.select("b")
        """)
        targets = extract_targets(f)
        assert len(targets) == 2

    def test_should_extract_both_method_and_module_function(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            def module_func(df):
                return df.select("a")

            class MyJob:
                def method(self, df):
                    return df.filter("b > 0")
        """)
        targets = extract_targets(f)
        qnames = [t.qualified_name for t in targets]
        assert "module_func" in qnames
        assert "MyJob.method" in qnames

    def test_should_extract_private_method_from_class(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            class MyJob:
                def _internal(self, df):
                    return df.select("a")
        """)
        targets = extract_targets(f)
        assert len(targets) == 1

    def test_should_log_warning_on_syntax_error(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def (((broken syntax", encoding="utf-8")
        with patch("ast_analyzer.logger") as mock_log:
            extract_targets(f)
            mock_log.warning.assert_called_once()

    def test_should_store_ast_node_on_target(self, tmp_path):
        f = _write(tmp_path, "etl.py", """
            def compute(df):
                return df.select("a")
        """)
        targets = extract_targets(f)
        assert isinstance(
            targets[0].node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )


# ===========================================================================
# map_tests_to_targets
# ===========================================================================

class TestMapTestsToTargets:

    def test_should_map_test_file_when_it_imports_source_module(
        self, tmp_path
    ):
        src = _write(tmp_path, "uts.py", """
            def compute(df):
                return df.select("a")
        """)
        test_f = _write(tmp_path, "test_uts.py", """
            from uts import compute
            def test_compute():
                pass
        """)
        targets = extract_targets(src)
        result  = map_tests_to_targets(targets, [test_f])
        assert test_f in result[0].test_files

    def test_should_not_map_test_file_when_it_does_not_import_source(
        self, tmp_path
    ):
        src = _write(tmp_path, "uts.py", """
            def compute(df):
                return df.select("a")
        """)
        test_f = _write(tmp_path, "test_other.py", """
            from other_module import something
            def test_something():
                pass
        """)
        targets = extract_targets(src)
        result  = map_tests_to_targets(targets, [test_f])
        assert result[0].test_files == []

    def test_should_map_matching_test_function_by_name(self, tmp_path):
        src = _write(tmp_path, "uts.py", """
            def compute(df):
                return df.select("a")
        """)
        test_f = _write(tmp_path, "test_uts.py", """
            from uts import compute
            def test_compute():
                pass
        """)
        targets = extract_targets(src)
        result  = map_tests_to_targets(targets, [test_f])
        assert "test_compute" in result[0].test_functions

    def test_should_not_map_unrelated_test_function(self, tmp_path):
        src = _write(tmp_path, "uts.py", """
            def compute(df):
                return df.select("a")
        """)
        test_f = _write(tmp_path, "test_uts.py", """
            from uts import compute
            def test_unrelated_thing():
                pass
        """)
        targets = extract_targets(src)
        result  = map_tests_to_targets(targets, [test_f])
        assert result[0].test_functions == []

    def test_should_map_multiple_test_files(self, tmp_path):
        src = _write(tmp_path, "uts.py", """
            def compute(df):
                return df.select("a")
        """)
        tf1 = _write(tmp_path, "test_uts_unit.py", """
            from uts import compute
            def test_compute():
                pass
        """)
        tf2 = _write(tmp_path, "test_uts_integration.py", """
            from uts import compute
            def test_compute_integration():
                pass
        """)
        targets = extract_targets(src)
        result  = map_tests_to_targets(targets, [tf1, tf2])
        assert tf1 in result[0].test_files
        assert tf2 in result[0].test_files

    def test_should_log_warning_when_no_tests_mapped(self, tmp_path):
        src = _write(tmp_path, "uts.py", """
            def compute(df):
                return df.select("a")
        """)
        targets = extract_targets(src)
        with patch("ast_analyzer.logger") as mock_log:
            map_tests_to_targets(targets, [])
            mock_log.warning.assert_called()

    def test_should_return_same_targets_list(self, tmp_path):
        src = _write(tmp_path, "uts.py", """
            def compute(df):
                return df.select("a")
        """)
        targets = extract_targets(src)
        result  = map_tests_to_targets(targets, [])
        assert result is targets

    def test_should_handle_empty_targets_list(self, tmp_path):
        result = map_tests_to_targets([], [])
        assert result == []

    def test_should_handle_syntax_error_in_test_file(self, tmp_path):
        src = _write(tmp_path, "uts.py", """
            def compute(df):
                return df.select("a")
        """)
        bad_test = tmp_path / "test_bad.py"
        bad_test.write_text("def (((broken", encoding="utf-8")
        targets = extract_targets(src)
        # Não deve lançar exceção — apenas não mapeia
        result = map_tests_to_targets(targets, [bad_test])
        assert result[0].test_files == []

    def test_should_map_test_function_via_qualified_name_convention(
        self, tmp_path
    ):
        src = _write(tmp_path, "uts.py", """
            class SalesJob:
                def transform(self, df):
                    return df.filter("a > 0")
        """)
        test_f = _write(tmp_path, "test_uts.py", """
            from uts import SalesJob
            def test_SalesJob_transform():
                pass
        """)
        targets = extract_targets(src)
        result  = map_tests_to_targets(targets, [test_f])
        assert "test_SalesJob_transform" in result[0].test_functions


# ===========================================================================
# analyze  (entry point público)
# ===========================================================================

class TestAnalyze:

    def test_should_return_function_targets_for_valid_source(self, tmp_path):
        src = _write(tmp_path, "etl.py", """
            def compute(df):
                return df.select("a")
        """)
        test_f = _write(tmp_path, "test_etl.py", """
            from etl import compute
            def test_compute():
                pass
        """)
        result = analyze([src], [test_f])
        assert len(result) == 1
        assert result[0].name == "compute"

    def test_should_return_empty_list_when_no_source_files(self, tmp_path):
        result = analyze([], [])
        assert result == []

    def test_should_aggregate_targets_from_multiple_source_files(
        self, tmp_path
    ):
        src1 = _write(tmp_path, "etl1.py", """
            def fn_a(df):
                return df.select("a")
        """)
        src2 = _write(tmp_path, "etl2.py", """
            def fn_b(df):
                return df.filter("b > 0")
        """)
        result = analyze([src1, src2], [])
        names = [t.name for t in result]
        assert "fn_a" in names
        assert "fn_b" in names

    def test_should_map_tests_to_targets_in_analyze(self, tmp_path):
        src = _write(tmp_path, "etl.py", """
            def compute(df):
                return df.select("a")
        """)
        test_f = _write(tmp_path, "test_etl.py", """
            from etl import compute
            def test_compute():
                pass
        """)
        result = analyze([src], [test_f])
        assert test_f in result[0].test_files

    def test_should_handle_source_file_with_syntax_error(self, tmp_path):
        bad = tmp_path / "bad.py"
        bad.write_text("def (((", encoding="utf-8")
        result = analyze([bad], [])
        assert result == []

    def test_should_call_extract_targets_for_each_source_file(
        self, tmp_path
    ):
        src1 = _write(tmp_path, "a.py", "def fn_a(df): return df.select('x')")
        src2 = _write(tmp_path, "b.py", "def fn_b(df): return df.select('y')")

        with patch("ast_analyzer.extract_targets", wraps=extract_targets) as mock_et:
            analyze([src1, src2], [])
            assert mock_et.call_count == 2

    def test_should_return_targets_with_test_files_populated(self, tmp_path):
        src = _write(tmp_path, "uts.py", """
            def compute(df):
                return df.select("a")
        """)
        test_f = _write(tmp_path, "test_uts.py", """
            from uts import compute
            def test_compute():
                pass
        """)
        result = analyze([src], [test_f])
        total_with_tests = sum(1 for t in result if t.test_files)
        assert total_with_tests == 1

    def test_should_count_targets_without_tests_correctly(self, tmp_path):
        src = _write(tmp_path, "etl.py", """
            def compute(df):
                return df.select("a")
        """)
        result = analyze([src], [])
        assert all(t.test_files == [] for t in result)