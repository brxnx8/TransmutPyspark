# src/operators/operator_uts.py
"""
UTS – Unary Transformations Swap
==================================
Troca a ordem de dois métodos DataFrame consecutivos
quando não há dependência de coluna entre eles.
"""

import ast
import copy
from dataclasses import dataclass, field
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


class _Pair(NamedTuple):
    outer: ast.Call
    inner: ast.Call
    outer_method: str
    inner_method: str


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


def _find_pairs(tree: ast.AST) -> list[_Pair]:
    pairs: list[_Pair] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        outer_m = _method_name(node)
        if outer_m not in _UNARY_TRANSFORMS:
            continue
        inner = _inner_call(node)
        if inner is None:
            continue
        inner_m = _method_name(inner)
        if inner_m not in _UNARY_TRANSFORMS:
            continue
        if outer_m == inner_m:
            continue
        if _has_dependency(inner, node):
            continue
        pairs.append(_Pair(node, inner, outer_m, inner_m))
    return pairs


def _build_swapped(pair: _Pair) -> ast.Call:
    """
    Original:  base  .inner(inner_args)  .outer(outer_args)
    Mutante:   base  .outer(outer_args)  .inner(inner_args)
    """
    base = pair.inner.func.value  # type: ignore[union-attr]

    new_outer = copy.deepcopy(pair.outer)
    new_outer.func = ast.Attribute(
        value=copy.deepcopy(base),
        attr=pair.outer_method,
        ctx=ast.Load(),
    )

    new_inner = copy.deepcopy(pair.inner)
    new_inner.func = ast.Attribute(
        value=new_outer,
        attr=pair.inner_method,
        ctx=ast.Load(),
    )
    return new_inner


def _modified_line_desc(pair: _Pair) -> str:
    line = getattr(pair.outer, "lineno", "?")
    return (
        f"line {line} | swap {pair.inner_method}↔{pair.outer_method} "
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

    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        self._assert_valid_tree(tree)
        pairs = _find_pairs(tree)
        eligible = [p.outer for p in pairs]
        self._log_analyse_ast_found(len(eligible), "swappable consecutive transform pairs")
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

        eligible_ids = {id(n) for n in nodes}
        pairs = [p for p in _find_pairs(original_ast) if id(p.outer) in eligible_ids]

        for pair in pairs:
            swapped = _build_swapped(pair)
            ast.fix_missing_locations(swapped)
            label = f"{pair.inner_method}↔{pair.outer_method}"
            modified_line = _modified_line_desc(pair)

            mid = self._next_mutant_id()
            filename = f"UTS_{mid}_{pair.inner_method}_{pair.outer_method}_swap.py"

            mutated_ast = self._replace_node(original_ast, pair.outer, swapped)
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