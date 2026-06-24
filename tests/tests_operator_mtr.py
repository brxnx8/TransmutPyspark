import ast
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from src.operators.operator_mtr import (
    OperatorMTR,
    _method_name,
    _is_col_call,
    _make_identity,
    _negate_expr,
    _collect_target_expressions,
    _modified_line_desc,
)
from src.model.mutant import Mutant


@pytest.fixture
def operator_mtr():
    """Fixture que fornece uma instância de OperatorMTR com os métodos da superclasse mockados."""
    op = OperatorMTR()
    op._assert_valid_tree = MagicMock()
    op._log_analyse_ast_found = MagicMock()
    op._assert_valid_nodes = MagicMock()
    op._assert_valid_path = MagicMock()
    op._log_build_mutant_done = MagicMock()
    op._next_mutant_id = MagicMock(return_value=1)
    op._replace_node = MagicMock(return_value=ast.parse("pass"))
    op._write_mutant_file = MagicMock(return_value="/tmp/mutant_mtr.py")
    op._log_mutant_created = MagicMock()
    op.mutant_list = []
    return op


# --- Testes de Funções Auxiliares ---

def test_should_return_attribute_name_when_call_is_attribute():
    tree = ast.parse("df.withColumn('a', F.lit(1))")
    call_node = tree.body[0].value
    assert _method_name(call_node) == "withColumn"


def test_should_return_none_when_call_is_not_attribute():
    tree = ast.parse("print('hello')")
    call_node = tree.body[0].value
    assert _method_name(call_node) is None


def test_should_return_true_when_node_is_col_function_call():
    tree = ast.parse("col('idade')")
    node = tree.body[0].value
    assert _is_col_call(node) is True


def test_should_return_true_when_node_is_attribute_col_call():
    tree = ast.parse("F.col('idade')")
    node = tree.body[0].value
    assert _is_col_call(node) is True


def test_should_return_false_when_node_is_not_col_call():
    tree_sum = ast.parse("F.sum('idade')")
    assert _is_col_call(tree_sum.body[0].value) is False
    
    tree_const = ast.parse("'idade'")
    assert _is_col_call(tree_const.body[0].value) is False


def test_should_return_left_operand_when_binop_has_col_on_left():
    tree = ast.parse("col('a') + 1")
    binop = tree.body[0].value
    identity = _make_identity(binop)
    assert identity is binop.left


def test_should_return_right_operand_when_binop_has_col_on_right():
    tree = ast.parse("1 + F.col('a')")
    binop = tree.body[0].value
    identity = _make_identity(binop)
    assert identity is binop.right


def test_should_return_none_when_binop_does_not_contain_col():
    tree = ast.parse("lit(1) + lit(2)")
    binop = tree.body[0].value
    assert _make_identity(binop) is None


def test_should_return_none_when_expression_is_not_binop():
    tree = ast.parse("col('a')")
    expr = tree.body[0].value
    assert _make_identity(expr) is None


def test_should_return_negated_expression_when_negate_expr_is_called():
    tree = ast.parse("col('a')")
    expr = tree.body[0].value
    negated = _negate_expr(expr)
    assert isinstance(negated, ast.UnaryOp)
    assert isinstance(negated.op, ast.USub)
    # Verifica se operand é uma cópia profunda garantindo que tem os mesmos atributos
    assert ast.dump(negated.operand) == ast.dump(expr)


def test_should_collect_second_arg_when_method_is_withcolumn():
    tree = ast.parse("df.withColumn('a', col('b') + 1)")
    call_node = tree.body[0].value
    target = _collect_target_expressions(call_node)
    assert len(target) == 1
    assert ast.unparse(target[0]) == "col('b') + 1"


def test_should_return_empty_list_when_withcolumn_has_insufficient_args():
    tree = ast.parse("df.withColumn('a')")
    call_node = tree.body[0].value
    assert _collect_target_expressions(call_node) == []


def test_should_collect_all_args_when_method_is_select():
    tree = ast.parse("df.select(col('a'), col('b'))")
    call_node = tree.body[0].value
    target = _collect_target_expressions(call_node)
    assert len(target) == 2


def test_should_collect_first_arg_when_method_is_map_family():
    tree = ast.parse("df.mapInPandas(my_func)")
    call_node = tree.body[0].value
    target = _collect_target_expressions(call_node)
    assert len(target) == 1
    assert ast.unparse(target[0]) == "my_func"


def test_should_return_empty_list_when_map_family_has_no_args():
    tree = ast.parse("df.mapInPandas()")
    call_node = tree.body[0].value
    assert _collect_target_expressions(call_node) == []


def test_should_return_empty_list_when_method_is_unknown():
    tree = ast.parse("df.filter(col('a') > 1)")
    call_node = tree.body[0].value
    assert _collect_target_expressions(call_node) == []


def test_should_return_formatted_string_when_modified_line_desc_is_called():
    tree = ast.parse("df.withColumn('a', col('b'))")
    call_node = tree.body[0].value
    expr = call_node.args[1]
    
    desc = _modified_line_desc(call_node, expr, "zero")
    assert "withColumn() expr → zero" in desc
    assert "original: col('b')" in desc


# --- Testes do OperatorMTR ---

def test_should_return_eligible_nodes_when_analyse_ast_finds_valid_mapping_methods(operator_mtr):
    code = """
df.withColumn('a', col('b') + 1)
df.select('a', 'b')
df.filter('a > 0')
"""
    tree = ast.parse(code)
    eligible = operator_mtr.analyse_ast(tree)
    
    # filter deve ser ignorado. Apenas withColumn e select são elegíveis.
    assert len(eligible) == 2
    operator_mtr._log_analyse_ast_found.assert_called_once_with(2, "mapping transformation calls")


def test_should_ignore_nodes_when_analyse_ast_finds_invalid_methods_or_no_args(operator_mtr):
    tree = ast.parse("df.withColumn('a')\nprint('hello')\ndf.map()")
    eligible = operator_mtr.analyse_ast(tree)
    
    # Todos são ignorados pois não têm argumentos suficientes ou não são chamadas de métodos elegíveis
    assert len(eligible) == 0


def test_should_emit_standard_mutants_when_build_mutant_receives_simple_expression(operator_mtr):
    tree = ast.parse("df.withColumn('a', F.lit(10))")
    nodes = operator_mtr.analyse_ast(tree)
    
    operator_mtr._emit = MagicMock()
    operator_mtr.build_mutant(nodes, tree, "orig.py", "/tmp")
    
    # Substituições padrão: zero, one, neg_one, none, empty_str (5) + negated (1) = 6 mutantes
    assert operator_mtr._emit.call_count == 6


def test_should_emit_identity_mutant_when_build_mutant_receives_binop_with_col(operator_mtr):
    tree = ast.parse("df.withColumn('a', col('b') + 10)")
    nodes = operator_mtr.analyse_ast(tree)
    
    operator_mtr._emit = MagicMock()
    operator_mtr.build_mutant(nodes, tree, "orig.py", "/tmp")
    
    # Substituições: 5 literais + 1 negated + 1 identity = 7 mutantes
    assert operator_mtr._emit.call_count == 7
    
    # Verifica se a label "identity" foi passada para o _emit
    emitted_labels = [call.args[-1] for call in operator_mtr._emit.call_args_list]
    assert "identity" in emitted_labels


def test_should_create_mutant_and_log_when_emit_is_called(operator_mtr):
    tree = ast.parse("df.withColumn('a', col('b'))")
    call_node = tree.body[0].value
    original_expr = call_node.args[1]
    replacement = ast.Constant(value=0)
    
    operator_mtr._emit(
        original_ast=tree,
        call_node=call_node,
        method="withColumn",
        expr_idx=0,
        replacement=replacement,
        original_expr=original_expr,
        original_path="orig.py",
        mutant_dir="/tmp",
        label="zero"
    )
    
    assert len(operator_mtr.mutant_list) == 1
    mutant = operator_mtr.mutant_list[0]
    
    assert mutant.id == 1
    assert mutant.operator == "MTR"
    assert mutant.original_path == "orig.py"
    assert mutant.mutant_path == "/tmp/mutant_mtr.py"
    assert "withColumn() expr → zero" in mutant.modified_line
    
    operator_mtr._replace_node.assert_called_once_with(tree, original_expr, replacement)
    operator_mtr._write_mutant_file.assert_called_once()
    operator_mtr._log_mutant_created.assert_called_once()