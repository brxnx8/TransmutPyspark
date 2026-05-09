# src/operators/operator_uts.py
"""
UTS – Unary Transformations Swap
==================================
Troca a ordem de dois métodos DataFrame dentro de um pipeline encadeado
quando não há dependência de coluna entre eles.

O parâmetro ``max_distance`` controla até quantas posições de distância
no pipeline um par pode estar para ser elegível ao swap:
  - max_distance=1  → apenas pares adjacentes (comportamento original)
  - max_distance=2  → adjacentes + distância 2
  - max_distance=-1 → sem limite (todos os pares independentes)
"""

import ast
import copy
from dataclasses import dataclass, field
from itertools import combinations
from typing import NamedTuple

from src.model.mutant import Mutant
from src.operators.operator import Operator

_UNARY_TRANSFORMS = {
    "filter", "where",
    "withColumn", "withColumnRenamed",
    "select", "drop",
    "distinct", "dropDuplicates",
    "orderBy", "sort",
    "limit",
    "cache", "persist",
}

_COLUMN_CREATORS = {"withColumn"}


def _node_key(node: ast.AST) -> tuple:
    return (
        getattr(node, "lineno", -1),
        getattr(node, "col_offset", -1),
        getattr(node, "end_lineno", -1),
        getattr(node, "end_col_offset", -1),
    )


def _pair_key(pair: "_Pair") -> tuple:
    """Identifica um par de forma única por ambos os nós.

    Necessário porque com max_distance > 1 o mesmo nó outer pode aparecer
    em múltiplos pares com inners diferentes — _node_key(outer) sozinho
    causaria colisão na filtragem de build_mutant.
    """
    return (_node_key(pair.inner), _node_key(pair.outer))


class _Pair(NamedTuple):
    outer: ast.Call        # nó de maior índice no pipeline (será swappado)
    inner: ast.Call        # nó de menor índice no pipeline (será swappado)
    outer_method: str
    inner_method: str
    distance: int          # distância entre os dois no pipeline (1 = adjacente)


def _method_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _inner_call(call: ast.Call) -> ast.Call | None:
    if not isinstance(call.func, ast.Attribute):
        return None
    receiver = call.func.value
    return receiver if isinstance(receiver, ast.Call) else None


def _columns_created(call: ast.Call) -> set[str]:
    if _method_name(call) in _COLUMN_CREATORS and call.args:
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return {first.value}
    return set()


def _columns_referenced(call: ast.Call) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(call):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_col = (
            (isinstance(func, ast.Name) and func.id == "col") or
            (isinstance(func, ast.Attribute) and func.attr == "col")
        )
        if is_col and node.args and isinstance(node.args[0], ast.Constant):
            names.add(node.args[0].value)
    return names


def _has_dependency(inner: ast.Call, outer: ast.Call) -> bool:
    return bool(_columns_created(inner) & _columns_referenced(outer))


def _extract_pipeline(tree: ast.AST) -> list[ast.Call]:
    """
    Retorna todos os nós ast.Call de transformações unárias encontrados
    na árvore, ordenados pela posição de fim no código-fonte — o que
    corresponde à ordem de execução do pipeline encadeado.
    """
    nodes: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _method_name(node) in _UNARY_TRANSFORMS:
            nodes.append(node)
    nodes.sort(key=lambda n: (n.end_lineno, n.end_col_offset))
    return nodes


def _any_transitive_dependency(
    pipeline: list[ast.Call], i: int, j: int
) -> bool:
    """
    Retorna True se existir qualquer dependência de coluna que impeça
    o swap entre os nós de índice i e j no pipeline, incluindo
    dependências transitivas através dos nós intermediários entre eles.
    """
    node_i = pipeline[i]
    for k in range(i, j):
        node_k = pipeline[k]
        node_k1 = pipeline[k + 1]
        # nó k cria coluna que nó k+1 referencia
        if _has_dependency(node_k, node_k1):
            return True
    # dependência direta entre i e j (pode não ter intermediários)
    if _has_dependency(node_i, pipeline[j]):
        return True
    return False


def _find_pairs(tree: ast.AST, max_distance: int = 1) -> list[_Pair]:
    """
    Encontra todos os pares elegíveis para swap dentro do pipeline,
    respeitando max_distance e verificando dependências transitivas.

    max_distance=-1 desativa o limite de distância.
    """
    pipeline = _extract_pipeline(tree)
    pairs: list[_Pair] = []

    for i, j in combinations(range(len(pipeline)), 2):
        distance = j - i
        if max_distance != -1 and distance > max_distance:
            continue

        node_i = pipeline[i]
        node_j = pipeline[j]

        m_i = _method_name(node_i)
        m_j = _method_name(node_j)

        if m_i == m_j:
            continue
        if _any_transitive_dependency(pipeline, i, j):
            continue

        # inner = nó de menor índice (executa primeiro), outer = maior índice
        pairs.append(_Pair(
            outer=node_j,
            inner=node_i,
            outer_method=m_j,
            inner_method=m_i,
            distance=distance,
        ))

    return pairs


def _build_swapped(pair: _Pair, original_ast: ast.AST) -> ast.AST:
    """
    Troca o conteúdo (attr + args + keywords) dos dois nós no pipeline,
    preservando a estrutura de encadeamento do AST intacta.
    Funciona tanto para pares adjacentes quanto não-adjacentes.
    """
    mutated = copy.deepcopy(original_ast)

    key_inner = _node_key(pair.inner)
    key_outer = _node_key(pair.outer)

    node_a = node_b = None
    for node in ast.walk(mutated):
        if not isinstance(node, ast.Call):
            continue
        k = _node_key(node)
        if k == key_inner:
            node_a = node
        elif k == key_outer:
            node_b = node

    if node_a is None or node_b is None:
        raise RuntimeError(
            f"Nós não encontrados na AST clonada: "
            f"inner={pair.inner_method} outer={pair.outer_method}"
        )

    # Trocar attr (nome do método) + args + keywords entre os dois nós
    node_a.func.attr, node_b.func.attr = node_b.func.attr, node_a.func.attr
    node_a.args,      node_b.args      = node_b.args,      node_a.args
    node_a.keywords,  node_b.keywords  = node_b.keywords,  node_a.keywords

    ast.fix_missing_locations(mutated)
    return mutated


def _modified_line_desc(pair: _Pair) -> str:
    line = getattr(pair.outer, "lineno", "?")
    return (
        f"line {line} | swap {pair.inner_method}↔{pair.outer_method} "
        f"| distance {pair.distance} "
        f"| original: {pair.inner_method}→{pair.outer_method}"
    )


@dataclass
class OperatorUTS(Operator):
    _DEFAULT_ID        = 4
    _DEFAULT_NAME      = "UTS"
    _DEFAULT_REGISTERS = ["filter", "withColumn", "select"]

    id:               int             = 4
    name:             str             = "UTS"
    mutant_registers: str | list[str] = field(
        default_factory=lambda: ["filter", "withColumn", "select"]
    )
    max_distance:     int             = 1

    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        self._assert_valid_tree(tree)
        pairs = _find_pairs(tree, self.max_distance)
        eligible = [p.outer for p in pairs]
        self._log_analyse_ast_found(len(eligible), "swappable transform pairs")
        return eligible

    def build_mutant(
        self,
        nodes: list[ast.AST],
        original_ast: ast.AST,
        original_path: str,
        mutant_dir: str,
    ) -> list[Mutant]:
        self._assert_valid_nodes(nodes)
        self._assert_valid_path(original_path, "original_path")
        self._assert_valid_path(mutant_dir, "mutant_dir")

        eligible_node_keys = {_node_key(n) for n in nodes}
        pairs = [
            p for p in _find_pairs(original_ast, self.max_distance)
            if _node_key(p.outer) in eligible_node_keys
        ]

        for pair in pairs:
            modified_line = _modified_line_desc(pair)

            mid = self._next_mutant_id()
            filename = f"UTS_{mid}_{pair.inner_method}_{pair.outer_method}_d{pair.distance}_swap.py"

            mutated_ast = _build_swapped(pair, original_ast)
            mutant_path = self._write_mutant_file(mutated_ast, mutant_dir, filename)

            mutant = Mutant(
                id=mid,
                operator=self.name,
                original_path=original_path,
                mutant_path=mutant_path,
                modified_line=modified_line,
            )
            self.mutant_list.append(mutant)
            self._log_mutant_created(mid, f"{modified_line} [{filename}]")

        self._log_build_mutant_done()
        return self.mutant_list