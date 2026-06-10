from __future__ import annotations

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Ajuste de import path (estrutura src/model/...)
# ---------------------------------------------------------------------------
import sys, types

# Cria módulos stub para src.model.mutant e src.model.mutant_id_manager
# caso o projeto não esteja no PYTHONPATH durante os testes.
def _ensure_stubs():
    # src
    if "src" not in sys.modules:
        sys.modules["src"] = types.ModuleType("src")
    # src.model
    if "src.model" not in sys.modules:
        mod = types.ModuleType("src.model")
        sys.modules["src.model"] = mod
        sys.modules["src"].model = mod  # type: ignore[attr-defined]
    # src.model.mutant  →  usa a classe real se disponível, senão stub
    if "src.model.mutant" not in sys.modules:
        try:
            from mutant import Mutant  # arquivo local
        except ImportError:
            @dataclass_stub
            class Mutant: ...          # pragma: no cover
        mod_mutant = types.ModuleType("src.model.mutant")
        mod_mutant.Mutant = Mutant     # type: ignore[attr-defined]
        sys.modules["src.model.mutant"] = mod_mutant
    # src.model.mutant_id_manager  →  usa a classe real se disponível
    if "src.model.mutant_id_manager" not in sys.modules:
        try:
            from mutant_id_manager import MutantIDManager
        except ImportError:
            class MutantIDManager:     # pragma: no cover
                _counter = 0
                def next_id(self): self._counter += 1; return self._counter
                def reset(self): self._counter = 0
        mod_mgr = types.ModuleType("src.model.mutant_id_manager")
        mod_mgr.MutantIDManager = MutantIDManager  # type: ignore[attr-defined]
        sys.modules["src.model.mutant_id_manager"] = mod_mgr

_ensure_stubs()

from src.operators.operator import Operator 
from src.model.mutant import Mutant
from src.model.mutant_id_manager import MutantIDManager


# ===========================================================================
# Subclasses concretas para teste
# ===========================================================================

class ConcreteOperator(Operator):
    """Implementação mínima — retorna listas vazias."""
    _DEFAULT_ID        = 0
    _DEFAULT_NAME      = "CONCRETE"
    _DEFAULT_REGISTERS = "SomeNode"

    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        self._assert_valid_tree(tree)
        return []

    def build_mutant(self, nodes, original_ast, original_path, mutant_dir):
        self._assert_valid_nodes(nodes)
        self._assert_valid_path(original_path, "original_path")
        self._assert_valid_path(mutant_dir,    "mutant_dir")
        return self.mutant_list


class SpyOperator(Operator):
    """Registra os argumentos recebidos para testes de integração."""
    _DEFAULT_ID        = 1
    _DEFAULT_NAME      = "SPY"
    _DEFAULT_REGISTERS = ["NodeA", "NodeB"]

    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        self._assert_valid_tree(tree)
        return list(ast.walk(tree))[:3]   # devolve até 3 nós reais

    def build_mutant(self, nodes, original_ast, original_path, mutant_dir):
        self._assert_valid_nodes(nodes)
        self._assert_valid_path(original_path, "original_path")
        self._assert_valid_path(mutant_dir,    "mutant_dir")
        return self.mutant_list


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def reset_id_manager():
    """Isola o Singleton MutantIDManager entre testes."""
    MutantIDManager._instance = None
    MutantIDManager._counter  = 0
    yield
    MutantIDManager._instance = None
    MutantIDManager._counter  = 0


@pytest.fixture()
def op() -> ConcreteOperator:
    return ConcreteOperator(id=1, name="AOR", mutant_registers="Add")


@pytest.fixture()
def simple_ast() -> ast.AST:
    return ast.parse("x = 1 + 2")


@pytest.fixture()
def sample_mutant(tmp_path) -> Mutant:
    return Mutant(
        id=1,
        operator="AOR",
        original_path=str(tmp_path / "src.py"),
        mutant_path=str(tmp_path / "mutant_1.py"),
        modified_line="line 1: Add → Sub",
    )


# ===========================================================================
# Instanciação e __post_init__
# ===========================================================================

class TestOperatorInstantiation:

    def test_should_create_instance_with_valid_arguments(self, op):
        assert op.id == 1
        assert op.name == "AOR"
        assert op.mutant_registers == "Add"
        assert op.mutant_list == []

    def test_should_normalize_name_to_uppercase_on_creation(self):
        o = ConcreteOperator(id=1, name="aor", mutant_registers="Add")
        assert o.name == "AOR"

    def test_should_strip_whitespace_from_name_on_creation(self):
        o = ConcreteOperator(id=1, name="  ror  ", mutant_registers="Add")
        assert o.name == "ROR"

    def test_should_accept_list_for_mutant_registers(self):
        o = ConcreteOperator(id=1, name="ROR", mutant_registers=["Lt", "Gt"])
        assert o.mutant_registers == ["Lt", "Gt"]

    def test_should_accept_zero_as_valid_id(self):
        o = ConcreteOperator(id=0, name="LCR", mutant_registers="Node")
        assert o.id == 0

    def test_should_initialize_mutant_list_as_empty_by_default(self, op):
        assert op.mutant_list == []

    def test_should_accept_pre_populated_mutant_list(self, sample_mutant):
        o = ConcreteOperator(
            id=1, name="AOR", mutant_registers="Add",
            mutant_list=[sample_mutant],
        )
        assert len(o.mutant_list) == 1

    def test_should_raise_type_error_when_id_is_negative(self):
        with pytest.raises(TypeError, match="non-negative integer"):
            ConcreteOperator(id=-1, name="AOR", mutant_registers="Add")

    def test_should_raise_type_error_when_id_is_string(self):
        with pytest.raises(TypeError, match="non-negative integer"):
            ConcreteOperator(id="1", name="AOR", mutant_registers="Add")  # type: ignore

    def test_should_raise_type_error_when_name_is_not_string(self):
        with pytest.raises(TypeError, match="non-empty string"):
            ConcreteOperator(id=1, name=123, mutant_registers="Add")  # type: ignore

    def test_should_raise_type_error_when_name_is_empty_string(self):
        with pytest.raises(TypeError, match="non-empty string"):
            ConcreteOperator(id=1, name="", mutant_registers="Add")

    def test_should_raise_type_error_when_name_is_whitespace_only(self):
        with pytest.raises(TypeError, match="non-empty string"):
            ConcreteOperator(id=1, name="   ", mutant_registers="Add")

    def test_should_raise_value_error_when_mutant_registers_is_empty_string(self):
        with pytest.raises(ValueError, match="must not be an empty string"):
            ConcreteOperator(id=1, name="AOR", mutant_registers="")

    def test_should_raise_value_error_when_mutant_registers_is_empty_list(self):
        with pytest.raises(ValueError, match="must not be empty"):
            ConcreteOperator(id=1, name="AOR", mutant_registers=[])

    def test_should_raise_value_error_when_mutant_registers_list_has_empty_string(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            ConcreteOperator(id=1, name="AOR", mutant_registers=["Lt", ""])

    def test_should_raise_type_error_when_mutant_registers_is_wrong_type(self):
        with pytest.raises(TypeError, match="str or list"):
            ConcreteOperator(id=1, name="AOR", mutant_registers=42)  # type: ignore

    def test_should_raise_type_error_when_mutant_list_is_not_a_list(self):
        with pytest.raises(TypeError, match="must be a list"):
            ConcreteOperator(id=1, name="AOR", mutant_registers="Add",
                             mutant_list="wrong")  # type: ignore

    def test_should_raise_type_error_when_mutant_list_contains_non_mutant(self):
        with pytest.raises(TypeError, match="Mutant instances"):
            ConcreteOperator(id=1, name="AOR", mutant_registers="Add",
                             mutant_list=["not_a_mutant"])  # type: ignore

    def test_should_not_allow_direct_instantiation_of_abstract_class(self):
        with pytest.raises(TypeError):
            Operator(id=1, name="AOR", mutant_registers="Add")  # type: ignore


# ===========================================================================
# Factory method create()
# ===========================================================================

class TestOperatorCreate:

    def test_should_create_instance_via_factory_with_default_values(self):
        o = ConcreteOperator.create()
        assert o.id   == ConcreteOperator._DEFAULT_ID
        assert o.name == ConcreteOperator._DEFAULT_NAME.upper()
        assert o.mutant_registers == ConcreteOperator._DEFAULT_REGISTERS

    def test_should_create_instance_with_list_registers_via_factory(self):
        o = SpyOperator.create()
        assert o.mutant_registers == ["NodeA", "NodeB"]

    def test_should_return_correct_subclass_type_from_factory(self):
        assert isinstance(ConcreteOperator.create(), ConcreteOperator)
        assert isinstance(SpyOperator.create(), SpyOperator)


# ===========================================================================
# analyse_ast (interface + guards)
# ===========================================================================

class TestAnalyseAst:

    def test_should_return_empty_list_for_concrete_operator(self, op, simple_ast):
        result = op.analyse_ast(simple_ast)
        assert result == []

    def test_should_return_list_of_ast_nodes(self, simple_ast):
        o = SpyOperator.create()
        result = o.analyse_ast(simple_ast)
        assert isinstance(result, list)
        assert all(isinstance(n, ast.AST) for n in result)

    def test_should_raise_type_error_when_tree_is_not_ast(self, op):
        with pytest.raises(TypeError, match="ast.AST instance"):
            op.analyse_ast("not an ast")  # type: ignore

    def test_should_raise_type_error_when_tree_is_none(self, op):
        with pytest.raises(TypeError, match="ast.AST instance"):
            op.analyse_ast(None)  # type: ignore

    def test_should_raise_type_error_when_tree_is_integer(self, op):
        with pytest.raises(TypeError, match="ast.AST instance"):
            op.analyse_ast(42)  # type: ignore

    def test_should_accept_any_valid_ast_module(self, op):
        tree = ast.parse("def f(): pass\nf()")
        assert op.analyse_ast(tree) == []

    def test_should_accept_empty_module_ast(self, op):
        tree = ast.parse("")
        assert op.analyse_ast(tree) == []


# ===========================================================================
# build_mutant (interface + guards)
# ===========================================================================

class TestBuildMutant:

    def test_should_return_empty_list_when_no_nodes_given(self, op, simple_ast, tmp_path):
        result = op.build_mutant([], simple_ast, "/src/f.py", str(tmp_path))
        assert result == []

    def test_should_raise_type_error_when_nodes_is_not_list(self, op, simple_ast, tmp_path):
        with pytest.raises(TypeError, match="must be a list"):
            op.build_mutant(None, simple_ast, "/src/f.py", str(tmp_path))  # type: ignore

    def test_should_raise_type_error_when_nodes_contains_non_ast(self, op, simple_ast, tmp_path):
        with pytest.raises(TypeError, match="ast.AST instances"):
            op.build_mutant(["not_ast"], simple_ast, "/src/f.py", str(tmp_path))  # type: ignore

    def test_should_raise_value_error_when_original_path_is_empty(self, op, simple_ast, tmp_path):
        with pytest.raises(ValueError, match="original_path"):
            op.build_mutant([], simple_ast, "", str(tmp_path))

    def test_should_raise_value_error_when_original_path_is_whitespace(self, op, simple_ast, tmp_path):
        with pytest.raises(ValueError, match="original_path"):
            op.build_mutant([], simple_ast, "   ", str(tmp_path))

    def test_should_raise_value_error_when_mutant_dir_is_empty(self, op, simple_ast):
        with pytest.raises(ValueError, match="mutant_dir"):
            op.build_mutant([], simple_ast, "/src/f.py", "")

    def test_should_raise_value_error_when_mutant_dir_is_whitespace(self, op, simple_ast):
        with pytest.raises(ValueError, match="mutant_dir"):
            op.build_mutant([], simple_ast, "/src/f.py", "   ")


# ===========================================================================
# clear_mutant_list
# ===========================================================================

class TestClearMutantList:

    def test_should_empty_mutant_list_when_populated(self, op, sample_mutant):
        op.mutant_list.append(sample_mutant)
        op.clear_mutant_list()
        assert op.mutant_list == []

    def test_should_not_raise_when_list_is_already_empty(self, op):
        op.clear_mutant_list()  # não deve lançar exceção
        assert op.mutant_list == []

    def test_should_allow_new_mutants_after_clearing(self, op, sample_mutant):
        op.mutant_list.append(sample_mutant)
        op.clear_mutant_list()
        op.mutant_list.append(sample_mutant)
        assert len(op.mutant_list) == 1

    def test_should_return_none(self, op):
        assert op.clear_mutant_list() is None

    def test_should_log_info_on_clear(self, op, caplog):
        with caplog.at_level(logging.INFO, logger="operator_module"):
            op.clear_mutant_list()
        assert "cleared" in caplog.text


# ===========================================================================
# _replace_node
# ===========================================================================

class TestReplaceNode:

    def test_should_replace_target_node_in_cloned_tree(self, op):
        tree = ast.parse("x = 1 + 2")
        # Encontra o nó Add
        add_node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.Add)
        )
        result = op._replace_node(tree, add_node, ast.Sub())
        # O unparse deve conter subtração
        assert "1 - 2" in ast.unparse(result)

    def test_should_not_modify_original_tree(self, op):
        tree = ast.parse("x = 1 + 2")
        original_src = ast.unparse(tree)
        add_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Add))
        op._replace_node(tree, add_node, ast.Sub())
        assert ast.unparse(tree) == original_src

    def test_should_return_ast_instance(self, op):
        tree = ast.parse("x = 1 + 2")
        add_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Add))
        result = op._replace_node(tree, add_node, ast.Sub())
        assert isinstance(result, ast.AST)

    def test_should_return_unchanged_tree_when_target_not_found(self, op):
        tree   = ast.parse("x = 1 + 2")
        other  = ast.parse("y = 3 * 4")
        mult   = next(n for n in ast.walk(other) if isinstance(n, ast.Mult))
        result = op._replace_node(tree, mult, ast.Add())
        # nenhuma substituição deve ocorrer
        assert "1 + 2" in ast.unparse(result)

    def test_should_fix_missing_locations_in_result(self, op):
        tree = ast.parse("a = True")
        name = next(n for n in ast.walk(tree) if isinstance(n, ast.Constant))
        result = op._replace_node(tree, name, ast.Constant(value=False))
        # fix_missing_locations garante que lineno existe nos nós
        for node in ast.walk(result):
            if hasattr(node, "lineno"):
                assert node.lineno is not None


# ===========================================================================
# _write_mutant_file
# ===========================================================================

class TestWriteMutantFile:

    def test_should_write_py_file_to_disk(self, op, tmp_path):
        tree = ast.parse("x = 99")
        path = op._write_mutant_file(tree, str(tmp_path), "mutant_1.py")
        assert Path(path).exists()

    def test_should_return_absolute_path_string(self, op, tmp_path):
        tree = ast.parse("x = 1")
        path = op._write_mutant_file(tree, str(tmp_path), "m.py")
        assert Path(path).is_absolute()

    def test_should_write_valid_python_source(self, op, tmp_path):
        tree = ast.parse("result = 2 + 3")
        path = op._write_mutant_file(tree, str(tmp_path), "m.py")
        content = Path(path).read_text(encoding="utf-8")
        # deve ser parseable novamente
        parsed = ast.parse(content)
        assert isinstance(parsed, ast.Module)

    def test_should_create_parent_directories_if_missing(self, op, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c")
        tree = ast.parse("pass")
        path = op._write_mutant_file(tree, nested, "m.py")
        assert Path(path).exists()

    def test_should_overwrite_existing_file(self, op, tmp_path):
        tree1 = ast.parse("x = 1")
        tree2 = ast.parse("x = 999")
        op._write_mutant_file(tree1, str(tmp_path), "m.py")
        path = op._write_mutant_file(tree2, str(tmp_path), "m.py")
        assert "999" in Path(path).read_text()


# ===========================================================================
# _next_mutant_id
# ===========================================================================

class TestNextMutantId:

    def test_should_return_sequential_integers(self, op):
        ids = [op._next_mutant_id() for _ in range(5)]
        assert ids == list(range(1, 6))

    def test_should_return_integer_type(self, op):
        assert isinstance(op._next_mutant_id(), int)

    def test_should_share_counter_across_operator_instances(self):
        o1 = ConcreteOperator(id=1, name="AOR", mutant_registers="Add")
        o2 = ConcreteOperator(id=2, name="ROR", mutant_registers="Lt")
        o1._next_mutant_id()   # 1
        o1._next_mutant_id()   # 2
        assert o2._next_mutant_id() == 3  # Singleton compartilhado


# ===========================================================================
# Guards protegidos (_assert_*)
# ===========================================================================

class TestAssertGuards:

    # _assert_valid_tree
    def test_should_pass_when_tree_is_ast_instance(self, op, simple_ast):
        op._assert_valid_tree(simple_ast)  # sem exceção

    def test_should_raise_type_error_for_non_ast_tree(self, op):
        with pytest.raises(TypeError, match="ast.AST instance"):
            op._assert_valid_tree({"key": "value"})  # type: ignore

    # _assert_valid_nodes
    def test_should_pass_when_nodes_is_empty_list(self, op):
        op._assert_valid_nodes([])

    def test_should_pass_when_nodes_contains_ast_items(self, op, simple_ast):
        nodes = list(ast.walk(simple_ast))[:2]
        op._assert_valid_nodes(nodes)

    def test_should_raise_type_error_when_nodes_is_not_list(self, op):
        with pytest.raises(TypeError, match="must be a list"):
            op._assert_valid_nodes(("a",))  # type: ignore

    def test_should_raise_type_error_when_nodes_has_non_ast_item(self, op):
        with pytest.raises(TypeError, match="ast.AST instances"):
            op._assert_valid_nodes([ast.parse("x=1"), "bad"])  # type: ignore

    # _assert_valid_path
    def test_should_pass_when_path_is_non_empty_string(self, op):
        op._assert_valid_path("/some/path.py", "original_path")

    def test_should_raise_value_error_when_path_is_empty_string(self, op):
        with pytest.raises(ValueError, match="original_path"):
            op._assert_valid_path("", "original_path")

    def test_should_raise_value_error_when_path_is_whitespace(self, op):
        with pytest.raises(ValueError, match="mutant_dir"):
            op._assert_valid_path("   ", "mutant_dir")

    def test_should_raise_value_error_when_path_is_not_string(self, op):
        with pytest.raises(ValueError, match="original_path"):
            op._assert_valid_path(None, "original_path")  # type: ignore


# ===========================================================================
# Logging helpers
# ===========================================================================

class TestLoggingHelpers:

    def test_should_log_info_with_count_and_description_in_analyse_ast_found(
        self, op, caplog
    ):
        with caplog.at_level(logging.INFO, logger="operator_module"):
            op._log_analyse_ast_found(3, "binary operations")
        assert "3" in caplog.text
        assert "binary operations" in caplog.text

    def test_should_log_warning_with_reason_in_skipping_node(self, op, caplog):
        with caplog.at_level(logging.WARNING, logger="operator_module"):
            op._log_skipping_node("no eligible sub-conditions found")
        assert "no eligible sub-conditions found" in caplog.text

    def test_should_log_info_with_mutant_id_and_details_in_mutant_created(
        self, op, caplog
    ):
        with caplog.at_level(logging.INFO, logger="operator_module"):
            op._log_mutant_created(42, "line 7 Add→Sub")
        assert "42" in caplog.text
        assert "line 7 Add→Sub" in caplog.text

    def test_should_log_total_mutant_count_in_build_mutant_done(
        self, op, sample_mutant, caplog
    ):
        op.mutant_list.append(sample_mutant)
        with caplog.at_level(logging.INFO, logger="operator_module"):
            op._log_build_mutant_done()
        assert "1" in caplog.text

    def test_should_log_zero_when_mutant_list_is_empty_in_build_mutant_done(
        self, op, caplog
    ):
        with caplog.at_level(logging.INFO, logger="operator_module"):
            op._log_build_mutant_done()
        assert "0" in caplog.text


# ===========================================================================
# __repr__
# ===========================================================================

class TestOperatorRepr:

    def test_should_start_with_class_name(self, op):
        assert repr(op).startswith("ConcreteOperator(")

    def test_should_include_id_in_repr(self, op):
        assert "id=1" in repr(op)

    def test_should_include_name_in_repr(self, op):
        assert "name='AOR'" in repr(op)

    def test_should_include_mutant_registers_in_repr(self, op):
        assert "mutant_registers='Add'" in repr(op)

    def test_should_show_zero_mutants_when_list_is_empty(self, op):
        assert "mutants=0" in repr(op)

    def test_should_show_correct_mutant_count_when_list_is_populated(
        self, op, sample_mutant
    ):
        op.mutant_list.append(sample_mutant)
        assert "mutants=1" in repr(op)

    def test_should_return_string_type(self, op):
        assert isinstance(repr(op), str)

    def test_should_use_subclass_name_not_operator(self):
        o = SpyOperator.create()
        assert repr(o).startswith("SpyOperator(")