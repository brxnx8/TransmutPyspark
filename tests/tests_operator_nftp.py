import ast
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from src.operators.operator_nftp import (
    OperatorNFTP,
    _method_name,
    _get_predicate,
    _build_negation,
    _collect_operator_mutations,
    _modified_line_desc,
)
from src.model.mutant import Mutant


@pytest.fixture
def operator_nftp():
    """Fixture que fornece uma instância de OperatorNFTP com os métodos da superclasse mockados."""
    op = OperatorNFTP()
    op._assert_valid_tree = MagicMock()
    op._log_analyse_ast_found = MagicMock()
    op._assert_valid_nodes = MagicMock()
    op._assert_valid_path = MagicMock()
    op._log_build_mutant_done = MagicMock()
    op._log_skipping_node = MagicMock()
    op._next_mutant_id = MagicMock(return_value=1)
    op._replace_node = MagicMock(return_value=ast.parse("pass"))
    op._write_mutant_file = MagicMock(return_value="/tmp/mutant_nftp.py")
    op._log_mutant_created = MagicMock()
    op.mutant_list = []
    return op


# --- Testes de Funções Auxiliares ---

def test_should_return_attribute_name_when_call_is_attribute():
    tree = ast.parse("df.filter(col('a') > 1)")
    call_node = tree.body[0].value
    assert _method_name(call_node) == "filter"


def test_should_return_none_when_call_is_not_attribute():
    tree = ast.parse("print('hello')")
    call_node = tree.body[0].value
    assert _method_name(call_node) is None


def test_should_return_first_argument_when_get_predicate_receives_args():
    tree = ast.parse("df.where(col('a') > 1)")
    call_node = tree.body[0].value
    pred = _get_predicate(call_node)
    assert pred is not None
    assert ast.unparse(pred) == "col('a') > 1"


def test_should_return_none_when_get_predicate_receives_no_args():
    tree = ast.parse("df.filter()")
    call_node = tree.body[0].value
    assert _get_predicate(call_node) is None


def test_should_wrap_with_invert_unary_op_when_build_negation_is_called():
    tree = ast.parse("col('a') > 1")
    pred = tree.body[0].value
    negated = _build_negation(pred)
    
    assert isinstance(negated, ast.UnaryOp)
    assert isinstance(negated.op, ast.Invert)
    assert ast.dump(negated.operand) == ast.dump(pred)


def test_should_collect_compare_inversions_when_ast_has_comparisons():
    # Testa < que vira >= e == que vira !=
    tree = ast.parse("(col('a') < 1) & (col('b') == 2)")
    pred = tree.body[0].value
    
    pairs = _collect_operator_mutations(pred)
    
    # 1 BinOp (&) + 2 Compares (<, ==). Total de 3 mutações.
    assert len(pairs) == 3
    
    # Valida se a inversão do `<` para `>=` ocorreu
    compare_lt_pair = [p for p in pairs if isinstance(p[0], ast.Compare) and isinstance(p[0].ops[0], ast.Lt)][0]
    assert isinstance(compare_lt_pair[1].ops[0], ast.GtE)
    
    # Valida se a inversão do `==` para `!=` ocorreu
    compare_eq_pair = [p for p in pairs if isinstance(p[0], ast.Compare) and isinstance(p[0].ops[0], ast.Eq)][0]
    assert isinstance(compare_eq_pair[1].ops[0], ast.NotEq)


def test_should_collect_binop_inversions_when_ast_has_bitwise_operators():
    # Testa & virando |
    tree = ast.parse("col('a') & col('b')")
    pred = tree.body[0].value
    
    pairs = _collect_operator_mutations(pred)
    assert len(pairs) == 1
    
    orig, repl = pairs[0]
    assert isinstance(orig.op, ast.BitAnd)
    assert isinstance(repl.op, ast.BitOr)


def test_should_collect_isnull_isnotnull_inversions_when_ast_has_null_checks():
    tree = ast.parse("col('a').isNull() | col('b').isNotNull()")
    pred = tree.body[0].value
    
    pairs = _collect_operator_mutations(pred)
    
    # 1 BinOp (|) + 2 Calls (isNull, isNotNull) = 3 mutações
    assert len(pairs) == 3
    
    calls = [p for p in pairs if isinstance(p[0], ast.Call)]
    assert len(calls) == 2
    
    # Verifica a troca de isNull para isNotNull
    isnull_orig = calls[0][0]
    isnull_repl = calls[0][1]
    assert _method_name(isnull_orig) == "isNull"
    assert _method_name(isnull_repl) == "isNotNull"
    
    # Verifica a troca de isNotNull para isNull
    isnotnull_orig = calls[1][0]
    isnotnull_repl = calls[1][1]
    assert _method_name(isnotnull_orig) == "isNotNull"
    assert _method_name(isnotnull_repl) == "isNull"


def test_should_collect_negated_isin_when_ast_has_isin_call():
    tree = ast.parse("col('a').isin([1, 2])")
    pred = tree.body[0].value
    
    pairs = _collect_operator_mutations(pred)
    assert len(pairs) == 1
    
    orig, repl = pairs[0]
    assert _method_name(orig) == "isin"
    assert isinstance(repl, ast.UnaryOp)
    assert isinstance(repl.op, ast.Invert)


def test_should_return_formatted_string_when_modified_line_desc_is_called():
    tree = ast.parse("df.filter(col('a') > 1)")
    call_node = tree.body[0].value
    pred = call_node.args[0]
    
    desc = _modified_line_desc(call_node, pred, "full_negation")
    assert "filter() predicate → full_negation" in desc
    assert "original: col('a') > 1" in desc


# --- Testes do OperatorNFTP ---

def test_should_return_eligible_nodes_when_analyse_ast_finds_filter_or_where(operator_nftp):
    code = """
df.filter(col('a') > 1)
df.where(col('b') == 2)
df.select('a')
"""
    tree = ast.parse(code)
    eligible = operator_nftp.analyse_ast(tree)
    
    # select deve ser ignorado. Apenas filter e where são elegíveis.
    assert len(eligible) == 2
    operator_nftp._log_analyse_ast_found.assert_called_once_with(2, "filter/where calls with predicate")


def test_should_ignore_nodes_when_analyse_ast_finds_invalid_methods_or_no_args(operator_nftp):
    tree = ast.parse("df.filter()\ndf.where()\ndf.map()")
    eligible = operator_nftp.analyse_ast(tree)
    
    # Os métodos existem, mas faltam predicados ou são inválidos.
    assert len(eligible) == 0


def test_should_emit_full_negation_and_sub_mutations_when_build_mutant_is_called(operator_nftp):
    tree = ast.parse("df.filter(col('a') > 1)")
    nodes = operator_nftp.analyse_ast(tree)
    
    operator_nftp._emit = MagicMock()
    operator_nftp.build_mutant(nodes, tree, "orig.py", "/tmp")
    
    # Ocorre 1 emissão para "full_negation" e 1 emissão para inversão de operador (> para <=). Total = 2.
    assert operator_nftp._emit.call_count == 2
    
    emitted_labels = [call.args[-1] for call in operator_nftp._emit.call_args_list]
    assert "full_negation" in emitted_labels
    assert "op_inv_Compare" in emitted_labels


def test_should_log_skipping_node_when_build_mutant_finds_no_operator_mutations(operator_nftp):
    # Passa uma coluna diretamente, sem operadores relacionais ou de método (nenhum par extraído)
    tree = ast.parse("df.filter(col('is_active'))")
    nodes = operator_nftp.analyse_ast(tree)
    
    operator_nftp._emit = MagicMock()
    operator_nftp.build_mutant(nodes, tree, "orig.py", "/tmp")
    
    # A negação total sempre acontece (1). Como não há operadores inversíveis, _collect_operator_mutations retorna [].
    assert operator_nftp._emit.call_count == 1
    
    # Deve acionar o log informando que não há sub-condições inversíveis
    operator_nftp._log_skipping_node.assert_called_once()
    assert "no invertible sub-conditions found" in operator_nftp._log_skipping_node.call_args[0][0]


def test_should_create_mutant_and_log_when_emit_is_called(operator_nftp):
    tree = ast.parse("df.filter(col('a') > 1)")
    call_node = tree.body[0].value
    pred = call_node.args[0]
    replacement = _build_negation(pred)
    
    operator_nftp._emit(
        original_ast=tree,
        target=pred,
        replacement=replacement,
        original_path="orig.py",
        mutant_dir="/tmp",
        call_node=call_node,
        pred=pred,
        label="full_negation"
    )
    
    assert len(operator_nftp.mutant_list) == 1
    mutant = operator_nftp.mutant_list[0]
    
    assert mutant.id == 1
    assert mutant.operator == "NFTP"
    assert mutant.original_path == "orig.py"
    assert mutant.mutant_path == "/tmp/mutant_nftp.py"
    assert "filter() predicate → full_negation" in mutant.modified_line
    
    operator_nftp._replace_node.assert_called_once_with(tree, pred, replacement)
    operator_nftp._write_mutant_file.assert_called_once()
    operator_nftp._log_mutant_created.assert_called_once()