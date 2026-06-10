import ast
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.operators.operator_atr import (
    OperatorATR,
    _method_name,
    _func_name,
    _is_agg_call,
    _is_window_call,
    _swap_func,
    _find_agg_calls,
    _find_string_constants,
    _modified_line_desc,
)
from src.model.mutant import Mutant


@pytest.fixture
def operator_atr():
    """Fixture que fornece uma instância de OperatorATR com os métodos da superclasse mockados."""
    op = OperatorATR()
    op._assert_valid_tree = MagicMock()
    op._log_analyse_ast_found = MagicMock()
    op._assert_valid_nodes = MagicMock()
    op._assert_valid_path = MagicMock()
    op._log_build_mutant_done = MagicMock()
    op._next_mutant_id = MagicMock(return_value=1)
    op._replace_node = MagicMock(return_value=ast.parse("pass"))
    op._write_mutant_file = MagicMock(return_value="/tmp/mutant.py")
    op._log_mutant_created = MagicMock()
    op.mutant_list = []
    return op


def test_should_return_attribute_name_when_call_is_attribute():
    tree = ast.parse("df.show()")
    call_node = tree.body[0].value
    assert _method_name(call_node) == "show"


def test_should_return_none_when_call_is_not_attribute():
    tree = ast.parse("print('hello')")
    call_node = tree.body[0].value
    assert _method_name(call_node) is None


def test_should_return_func_name_when_call_is_name():
    tree = ast.parse("sum(1, 2)")
    call_node = tree.body[0].value
    assert _func_name(call_node) == "sum"


def test_should_return_true_when_call_is_agg_function():
    tree = ast.parse("F.sum('col')")
    call_node = tree.body[0].value
    assert _is_agg_call(call_node) is True


def test_should_return_true_when_call_is_window_function():
    tree = ast.parse("F.rank().over(w)")
    # O nó mais externo é 'over', o interno é 'rank'
    rank_call = tree.body[0].value.func.value
    assert _is_window_call(rank_call) is True


def test_should_swap_function_name_when_attribute_is_provided():
    tree = ast.parse("F.sum('col')")
    call_node = tree.body[0].value
    new_call = _swap_func(call_node, "max")
    assert _func_name(new_call) == "max"
    assert _func_name(call_node) == "sum"  # Garante que não mutou o original


def test_should_extract_agg_calls_when_grouped_by_is_present():
    tree = ast.parse("df.groupBy('id').agg(F.sum('val'))")
    agg_calls = _find_agg_calls(tree)
    assert len(agg_calls) == 1
    assert _method_name(agg_calls[0]) == "agg"


def test_should_extract_unique_string_constants_when_ast_is_parsed():
    tree = ast.parse("df.select('col_a', 'col_b').withColumn('col_a', F.lit('val'))")
    constants = _find_string_constants(tree)
    assert sorted(constants) == sorted(["col_a", "col_b", "val"])


def test_should_return_eligible_nodes_when_analyse_ast_is_called(operator_atr):
    tree = ast.parse("df.groupBy('id').agg(F.sum('val'))\ndf.withColumn('r', F.rank().over(w))")
    eligible = operator_atr.analyse_ast(tree)
    
    assert len(eligible) == 2
    operator_atr._assert_valid_tree.assert_called_once_with(tree)
    operator_atr._log_analyse_ast_found.assert_called_once_with(2, "aggregation and window function calls")


def test_should_mutate_agg_functions_and_columns_when_build_mutant_receives_agg_call(operator_atr):
    tree = ast.parse("df.groupBy('id').agg(F.sum('val'), F.max('other'))")
    nodes = _find_agg_calls(tree)
    
    # Executa o build_mutant interceptando a emissão
    operator_atr._emit = MagicMock()
    operator_atr.build_mutant(nodes, tree, "orig.py", "/tmp")
    
    # 7 funções de agregação restantes para trocar * 2 funções originais = 14 mutantes de função
    # Troca de colunas: 'id', 'val', 'other'. O F.sum('val') vai testar 'id' e 'other' (2).
    # O F.max('other') vai testar 'id' e 'val' (2). Total = 4 mutantes de coluna.
    # Total de chamadas ao _emit no bloco de A1 e A2: 18 chamadas
    assert operator_atr._emit.call_count >= 18


def test_should_mutate_groupby_shorthands_when_build_mutant_receives_shorthand(operator_atr):
    tree = ast.parse("df.groupBy('id').sum('val')")
    nodes = _find_agg_calls(tree)
    
    operator_atr._emit = MagicMock()
    operator_atr.build_mutant(nodes, tree, "orig.py", "/tmp")
    
    # Deve trocar 'sum' pelas outras 7 funções do _AGG_FUNCTIONS
    assert operator_atr._emit.call_count == 7


def test_should_mutate_window_functions_when_build_mutant_receives_window_call(operator_atr):
    tree = ast.parse("F.rank().over(w)")
    # Coletando manualmente o nó elegível (como feito em analyse_ast)
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _is_window_call(node)]
    
    operator_atr._emit = MagicMock()
    operator_atr.build_mutant(nodes, tree, "orig.py", "/tmp")
    
    # Deve trocar 'rank' pelas outras 4 funções do _WINDOW_FUNCTIONS
    assert operator_atr._emit.call_count == 4


def test_should_drop_groupby_keys_when_mutate_groupby_keys_is_called(operator_atr):
    tree = ast.parse("df.groupBy('k1', 'k2', 'k3').sum('val')")
    operator_atr._emit = MagicMock()
    
    operator_atr._mutate_groupby_keys(tree, "orig.py", "/tmp")
    
    # O nó tem 3 chaves. O loop removerá uma de cada vez gerando 3 mutantes.
    assert operator_atr._emit.call_count == 3


def test_should_create_mutant_and_log_when_emit_is_called(operator_atr):
    tree = ast.parse("F.sum('val')")
    target = tree.body[0].value
    replacement = ast.parse("F.max('val')").body[0].value
    
    operator_atr._emit(tree, target, replacement, "orig.py", "/tmp", target, "sum→max")
    
    assert len(operator_atr.mutant_list) == 1
    mutant = operator_atr.mutant_list[0]
    
    assert mutant.id == 1
    assert mutant.operator == "ATR"
    assert mutant.original_path == "orig.py"
    assert mutant.mutant_path == "/tmp/mutant.py"
    assert "sum→max" in mutant.modified_line
    
    operator_atr._replace_node.assert_called_once_with(tree, target, replacement)
    operator_atr._write_mutant_file.assert_called_once()
    operator_atr._log_mutant_created.assert_called_once()