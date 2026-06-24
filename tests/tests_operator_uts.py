import ast
import pytest
from unittest.mock import MagicMock, Mock
from pathlib import Path

# Ajuste o import conforme a estrutura real do projeto
from src.operators.operator_uts import (
    OperatorUTS,
    _Pair,
    _node_key,
    _pair_key,
    _method_name,
    _inner_call,
    _columns_created,
    _columns_referenced,
    _has_dependency,
    _extract_pipeline,
    _any_transitive_dependency,
    _find_pairs,
    _build_swapped,
    _modified_line_desc,
)
from src.model.mutant import Mutant


@pytest.fixture
def operator_uts():
    """Fixture que fornece uma instância de OperatorUTS com os métodos da superclasse mockados."""
    op = OperatorUTS()
    op._assert_valid_tree = MagicMock()
    op._log_analyse_ast_found = MagicMock()
    op._assert_valid_nodes = MagicMock()
    op._assert_valid_path = MagicMock()
    op._log_build_mutant_done = MagicMock()
    op._next_mutant_id = MagicMock(return_value=1)
    op._write_mutant_file = MagicMock(return_value="/tmp/mutant_uts.py")
    op._log_mutant_created = MagicMock()
    op.mutant_list = []
    return op


# --- Testes de Funções Auxiliares Básicas ---

def test_should_return_positional_tuple_when_node_key_is_called():
    node = Mock(lineno=10, col_offset=2, end_lineno=10, end_col_offset=15)
    assert _node_key(node) == (10, 2, 10, 15)


def test_should_return_default_tuple_when_node_lacks_position_attributes():
    node = Mock(spec=[]) # Mock vazio sem lineno etc.
    assert _node_key(node) == (-1, -1, -1, -1)


def test_should_return_tuple_of_node_keys_when_pair_key_is_called():
    inner = Mock(lineno=1, col_offset=0, end_lineno=1, end_col_offset=5)
    outer = Mock(lineno=2, col_offset=0, end_lineno=2, end_col_offset=5)
    pair = _Pair(outer=outer, inner=inner, outer_method="a", inner_method="b", distance=1)
    
    expected = ((1, 0, 1, 5), (2, 0, 2, 5))
    assert _pair_key(pair) == expected


def test_should_return_attr_name_when_call_is_attribute():
    tree = ast.parse("df.filter()")
    call_node = tree.body[0].value
    assert _method_name(call_node) == "filter"


def test_should_return_none_when_call_is_not_attribute():
    tree = ast.parse("print('hello')")
    call_node = tree.body[0].value
    assert _method_name(call_node) is None


def test_should_return_receiver_call_when_inner_call_exists():
    tree = ast.parse("df.filter('a').select('b')")
    outer_call = tree.body[0].value
    inner_call = _inner_call(outer_call)
    
    assert inner_call is not None
    assert _method_name(inner_call) == "filter"


def test_should_return_none_when_inner_call_does_not_exist():
    tree = ast.parse("df.filter('a')")
    outer_call = tree.body[0].value
    inner_call = _inner_call(outer_call)
    
    # 'df' não é um ast.Call, então retorna None
    assert inner_call is None


# --- Testes de Análise de Dependências de Colunas ---

def test_should_return_column_name_when_withcolumn_is_used_with_string_constant():
    tree = ast.parse("df.withColumn('nova_coluna', lit(1))")
    call_node = tree.body[0].value
    cols = _columns_created(call_node)
    assert cols == {"nova_coluna"}


def test_should_return_empty_set_when_method_is_not_column_creator():
    tree = ast.parse("df.select('coluna_x')")
    call_node = tree.body[0].value
    cols = _columns_created(call_node)
    assert cols == set()


def test_should_return_referenced_columns_when_extracting_from_ast():
    # Testa os 4 casos: col(), F.col(), df['col'], e df.col, além de ignorar nomes de métodos
    code = "df.filter(col('a') == F.col('b') + df['c'] + df.d)"
    tree = ast.parse(code)
    call_node = tree.body[0].value
    
    cols = _columns_referenced(call_node)
    assert cols == {"a", "b", "c", "d"}
    assert "filter" not in cols # Garante que métodos são ignorados


def test_should_return_true_when_outer_references_inner_created_column():
    tree = ast.parse("df.withColumn('a', lit(1)).filter(col('a') > 0)")
    pipeline = _extract_pipeline(tree)
    inner, outer = pipeline[0], pipeline[1]
    
    assert _has_dependency(inner, outer) is True


def test_should_return_false_when_no_dependency_exists():
    tree = ast.parse("df.withColumn('a', lit(1)).filter(col('b') > 0)")
    pipeline = _extract_pipeline(tree)
    inner, outer = pipeline[0], pipeline[1]
    
    assert _has_dependency(inner, outer) is False


def test_should_return_true_when_transitive_dependency_exists():
    # pipeline: withColumn("a") -> withColumn("b", col("a")) -> filter(col("b"))
    # filter(col("b")) depende de withColumn("a") transitivamente por causa do "b" no meio
    tree = ast.parse("df.withColumn('a', lit(1)).withColumn('b', col('a')).filter(col('b') > 0)")
    pipeline = _extract_pipeline(tree)
    
    assert _any_transitive_dependency(pipeline, 0, 2) is True


def test_should_return_false_when_no_transitive_dependency_exists():
    tree = ast.parse("df.withColumn('a', lit(1)).withColumn('b', lit(2)).filter(col('c') > 0)")
    pipeline = _extract_pipeline(tree)
    
    assert _any_transitive_dependency(pipeline, 0, 2) is False


# --- Testes de Pipeline e Pares ---

def test_should_extract_and_sort_pipeline_nodes_by_execution_order():
    tree = ast.parse("df.filter(1).select(2).drop(3)")
    nodes = _extract_pipeline(tree)
    
    assert len(nodes) == 3
    assert _method_name(nodes[0]) == "filter"
    assert _method_name(nodes[1]) == "select"
    assert _method_name(nodes[2]) == "drop"


def test_should_find_adjacent_pairs_when_max_distance_is_one():
    tree = ast.parse("df.filter(1).select(2).drop(3)")
    pairs = _find_pairs(tree, max_distance=1)
    
    assert len(pairs) == 2
    # filter(1) e select(2)
    assert pairs[0].inner_method == "filter" and pairs[0].outer_method == "select"
    # select(2) e drop(3)
    assert pairs[1].inner_method == "select" and pairs[1].outer_method == "drop"


def test_should_find_distant_pairs_when_max_distance_is_greater_than_one():
    tree = ast.parse("df.filter(1).select(2).drop(3)")
    pairs = _find_pairs(tree, max_distance=2)
    
    # Todos os pares adjacentes + o par com distância 2
    assert len(pairs) == 3
    distant_pair = [p for p in pairs if p.distance == 2][0]
    assert distant_pair.inner_method == "filter" and distant_pair.outer_method == "drop"


def test_should_find_all_independent_pairs_when_max_distance_is_minus_one():
    tree = ast.parse("df.filter(1).select(2).drop(3).limit(4)")
    pairs = _find_pairs(tree, max_distance=-1)
    
    # Com 4 elementos sem dependência, comb(4, 2) = 6 pares possíveis
    assert len(pairs) == 6


def test_should_ignore_pairs_with_same_method_name():
    tree = ast.parse("df.filter(1).filter(2)")
    pairs = _find_pairs(tree, max_distance=1)
    assert len(pairs) == 0


def test_should_ignore_pairs_with_dependencies():
    tree = ast.parse("df.withColumn('a', lit(1)).select('a')")
    pairs = _find_pairs(tree, max_distance=1)
    assert len(pairs) == 0


# --- Testes de Mutação AST ---

def test_should_swap_method_attributes_and_arguments_when_build_swapped_is_called():
    code = "df.filter(col('a')).drop('b')"
    tree = ast.parse(code)
    pairs = _find_pairs(tree)
    
    swapped_tree = _build_swapped(pairs[0], tree)
    swapped_code = ast.unparse(swapped_tree)
    
    # A ordem original era filter -> drop. O swap inverte.
    assert "df.drop('b').filter(col('a'))" in swapped_code


def test_should_raise_runtime_error_when_nodes_are_not_found_in_cloned_ast():
    code = "df.filter(col('a')).drop('b')"
    tree = ast.parse(code)
    pairs = _find_pairs(tree)
    
    # Criamos uma árvore totalmente nova, com coordenadas (lineno) diferentes
    different_tree = ast.parse("df.select(1).limit(2)")
    
    with pytest.raises(RuntimeError) as exc_info:
        _build_swapped(pairs[0], different_tree)
    
    assert "Nós não encontrados na AST clonada" in str(exc_info.value)


def test_should_return_formatted_string_when_modified_line_desc_is_called():
    tree = ast.parse("df.filter(col('a')).drop('b')")
    pairs = _find_pairs(tree)
    desc = _modified_line_desc(pairs[0])
    
    assert "swap filter↔drop" in desc
    assert "distance 1" in desc
    assert "original: filter→drop" in desc


# --- Testes de OperatorUTS ---

def test_should_return_eligible_outer_nodes_when_analyse_ast_is_called(operator_uts):
    tree = ast.parse("df.filter(1).select(2).drop(3)")
    operator_uts.max_distance = 1
    eligible = operator_uts.analyse_ast(tree)
    
    # max_distance=1 para 3 nós adjacentes gera 2 outer nodes elegíveis (select e drop)
    assert len(eligible) == 2
    operator_uts._log_analyse_ast_found.assert_called_once_with(2, "swappable transform pairs")


def test_should_emit_swapped_mutants_when_build_mutant_is_called(operator_uts):
    tree = ast.parse("df.filter(1).select(2).drop(3)")
    operator_uts.max_distance = 2
    
    # Extrai o pipeline para simular a seleção de 'nodes' a ser enviada pro build_mutant
    pipeline = _extract_pipeline(tree)
    outer_node = pipeline[2] # O drop
    
    operator_uts._emit = MagicMock()
    # Enviamos apenas o nó outer drop para o construtor. Isso simula o comportamento
    # onde o usuário ou o framework limitou os nós elegíveis.
    operator_uts.build_mutant([outer_node], tree, "orig.py", "/tmp")
    
    # O nó 'drop' está envolvido em 2 pares quando max_distance=2 (com filter e com select)
    assert operator_uts._emit.call_count == 2


def test_should_create_mutant_and_log_when_emit_is_called(operator_uts):
    tree = ast.parse("df.filter(1).select(2)")
    pairs = _find_pairs(tree)
    pair = pairs[0]
    
    operator_uts._emit(tree, pair, "orig.py", "/tmp")
    
    assert len(operator_uts.mutant_list) == 1
    mutant = operator_uts.mutant_list[0]
    
    assert mutant.id == 1
    assert mutant.operator == "UTS"
    assert mutant.original_path == "orig.py"
    assert mutant.mutant_path == "/tmp/mutant_uts.py"
    assert "swap filter↔select" in mutant.modified_line
    
    # Valida se delegou as responsabilidades à superclasse
    operator_uts._write_mutant_file.assert_called_once()
    operator_uts._log_mutant_created.assert_called_once()