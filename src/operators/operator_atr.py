# src/operators/operator_atr.py
"""
ATR – Aggregation Transformation Replacement
=============================================
Alvo (DataFrame API):
  - df.groupBy(...).agg(F.sum("x"))   → troca função de agregação
  - df.groupBy(...).sum("x")          → shorthand
  - F.rank().over(window)             → troca função de janela
  - groupBy key                       → remove uma das chaves
"""

import ast
import copy
from dataclasses import dataclass, field
from pathlib import Path

from src.model.mutant import Mutant
from src.operators.operator import Operator

_AGG_FUNCTIONS   = ["sum", "count", "avg", "mean", "max", "min", "first", "last"]
_WINDOW_FUNCTIONS = ["rank", "dense_rank", "row_number", "percent_rank", "cume_dist"]
_GROUPBY_SHORTHANDS = {"sum", "count", "avg", "mean", "max", "min"}
_GROUPBY_METHODS    = {"groupBy", "groupby"}


def _method_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _func_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_agg_call(node: ast.Call) -> bool:
    return _func_name(node) in _AGG_FUNCTIONS


def _is_window_call(node: ast.Call) -> bool:
    return _func_name(node) in _WINDOW_FUNCTIONS


def _swap_func(call: ast.Call, new_name: str) -> ast.Call:
    new = copy.deepcopy(call)
    if isinstance(new.func, ast.Attribute):
        new.func.attr = new_name
    elif isinstance(new.func, ast.Name):
        new.func.id = new_name
    return new


def _find_agg_calls(tree: ast.AST) -> list[ast.Call]:
    """Retorna chamadas .agg() ou shorthand precedidas de .groupBy()."""
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        mname = _method_name(node)
        if mname not in {"agg"} | _GROUPBY_SHORTHANDS:
            continue
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        if isinstance(receiver, ast.Call) and _method_name(receiver) in _GROUPBY_METHODS:
            result.append(node)
    return result


def _find_string_constants(tree: ast.AST) -> list[str]:
    seen: set[str] = set()
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in seen:
                seen.add(node.value)
                result.append(node.value)
    return result


def _modified_line_desc(call: ast.Call, label: str) -> str:
    line = getattr(call, "lineno", "?")
    method = _method_name(call) or _func_name(call) or "?"
    return f"line {line} | {method}() → {label} | original: {ast.unparse(call)}"


@dataclass
class OperatorATR(Operator):
    _DEFAULT_ID        = 3
    _DEFAULT_NAME      = "ATR"
    _DEFAULT_REGISTERS = ["agg", "groupBy", "rank"]

    id:               int             = 3
    name:             str             = "ATR"
    mutant_registers: str | list[str] = field(
        default_factory=lambda: ["agg", "groupBy", "rank"]
    )

    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        self._assert_valid_tree(tree)
        eligible: list[ast.AST] = []

        eligible.extend(_find_agg_calls(tree))

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_window_call(node):
                eligible.append(node)

        self._log_analyse_ast_found(len(eligible), "aggregation and window function calls")
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

        all_strings = _find_string_constants(original_ast)

        for call_node in nodes:
            mname = _method_name(call_node)

            # ── Caso A: .agg(F.sum("col"), ...) ──────────────────────────
            if mname == "agg":
                for arg_idx, arg in enumerate(call_node.args):
                    if not isinstance(arg, ast.Call) or not _is_agg_call(arg):
                        continue
                    current_fn = _func_name(arg)

                    # A1: troca a função de agregação
                    for new_fn in _AGG_FUNCTIONS:
                        if new_fn == current_fn:
                            continue
                        new_arg = _swap_func(arg, new_fn)
                        new_agg = copy.deepcopy(call_node)
                        new_agg.args[arg_idx] = new_arg
                        label = f"agg{arg_idx}_{current_fn}→{new_fn}"
                        self._emit(original_ast, call_node, new_agg,
                                   original_path, mutant_dir, call_node, label)

                    # A2: troca a coluna de entrada
                    if arg.args and isinstance(arg.args[0], ast.Constant):
                        col_name = arg.args[0].value
                        for candidate in all_strings:
                            if candidate == col_name:
                                continue
                            new_arg = copy.deepcopy(arg)
                            new_arg.args[0] = ast.Constant(value=candidate)
                            new_agg = copy.deepcopy(call_node)
                            new_agg.args[arg_idx] = new_arg
                            label = f"agg{arg_idx}_col_{col_name}→{candidate}"
                            self._emit(original_ast, call_node, new_agg,
                                       original_path, mutant_dir, call_node, label)

            # ── Caso B: .groupBy().sum() shorthand ────────────────────────
            elif mname in _GROUPBY_SHORTHANDS:
                for new_fn in _AGG_FUNCTIONS:
                    if new_fn == mname:
                        continue
                    replacement = _swap_func(call_node, new_fn)
                    label = f"{mname}→{new_fn}"
                    self._emit(original_ast, call_node, replacement,
                               original_path, mutant_dir, call_node, label)

            # ── Caso C: funções de janela ─────────────────────────────────
            elif _is_window_call(call_node):
                current = _func_name(call_node)
                for new_fn in _WINDOW_FUNCTIONS:
                    if new_fn == current:
                        continue
                    replacement = _swap_func(call_node, new_fn)
                    label = f"{current}→{new_fn}"
                    self._emit(original_ast, call_node, replacement,
                               original_path, mutant_dir, call_node, label)

        # ── Caso D: remoção de chave do groupBy ───────────────────────────
        self._mutate_groupby_keys(original_ast, original_path, mutant_dir)

        self._log_build_mutant_done()
        return self.mutant_list

    def _mutate_groupby_keys(
        self,
        original_ast: ast.AST,
        original_path: str,
        mutant_dir: str,
    ) -> None:
        for node in ast.walk(original_ast):
            if not isinstance(node, ast.Call):
                continue
            if _method_name(node) not in _GROUPBY_METHODS:
                continue
            if len(node.args) < 2:
                continue
            for i in range(len(node.args)):
                new_node = copy.deepcopy(node)
                removed = ast.unparse(node.args[i])
                new_node.args = [a for j, a in enumerate(node.args) if j != i]
                label = f"groupby_drop_key{i}_{removed}"
                self._emit(original_ast, node, new_node,
                           original_path, mutant_dir, node, label)

    def _emit(
        self,
        original_ast: ast.AST,
        target: ast.AST,
        replacement: ast.AST,
        original_path: str,
        mutant_dir: str,
        ref_call: ast.Call,
        label: str,
    ) -> None:
        mid = self._next_mutant_id()
        filename = f"ATR_{mid}_{label}.py"
        modified_line = _modified_line_desc(ref_call, label)

        mutated_ast = self._replace_node(original_ast, target, replacement)
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