# src/operators/operator_mtr.py
"""
MTR – Mapping Transformation Replacement
=========================================
Alvo (DataFrame API):
  - df.withColumn("col", <expr>)   → muta <expr>
  - df.select(<expr>, ...)         → muta cada <expr> individualmente
  - df.map / df.mapInPandas        → muta a função passada
"""

import ast
import copy
from dataclasses import dataclass, field
from pathlib import Path

from src.model.mutant import Mutant
from src.operators.operator import Operator

_MAPPING_METHODS = {"withColumn", "select", "map", "mapInPandas", "mapInArrow"}

_LITERAL_REPLACEMENTS: dict[str, ast.expr] = {
    "zero":      ast.Constant(value=0),
    "one":       ast.Constant(value=1),
    "neg_one":   ast.Constant(value=-1),
    "none":      ast.Constant(value=None),
    "empty_str": ast.Constant(value=""),
}


def _method_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_col_call(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "col":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "col":
            return True
    return False


def _make_identity(expr: ast.expr) -> ast.expr | None:
    if isinstance(expr, ast.BinOp):
        if _is_col_call(expr.left):
            return expr.left
        if _is_col_call(expr.right):
            return expr.right
    return None


def _negate_expr(expr: ast.expr) -> ast.expr:
    return ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(expr))


def _collect_target_expressions(call_node: ast.Call) -> list[ast.expr]:
    method = _method_name(call_node)
    if method == "withColumn" and len(call_node.args) >= 2:
        return [call_node.args[1]]
    if method == "select":
        return list(call_node.args)
    if method in {"map", "mapInPandas", "mapInArrow"} and call_node.args:
        return [call_node.args[0]]
    return []


def _modified_line_desc(call_node: ast.Call, expr: ast.expr, label: str) -> str:
    method = _method_name(call_node) or "?"
    line = getattr(expr, "lineno", "?")
    return f"line {line} | {method}() expr → {label} | original: {ast.unparse(expr)}"


@dataclass
class OperatorMTR(Operator):
    _DEFAULT_ID        = 1
    _DEFAULT_NAME      = "MTR"
    _DEFAULT_REGISTERS = ["withColumn", "select", "map"]

    # Campos do dataclass com defaults
    id:               int             = 1
    name:             str             = "MTR"
    mutant_registers: str | list[str] = field(
        default_factory=lambda: ["withColumn", "select", "map"]
    )

    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        self._assert_valid_tree(tree)
        eligible: list[ast.AST] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _method_name(node) not in _MAPPING_METHODS:
                continue
            if _collect_target_expressions(node):
                eligible.append(node)

        self._log_analyse_ast_found(len(eligible), "mapping transformation calls")
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

        for call_node in nodes:
            target_exprs = _collect_target_expressions(call_node)
            method = _method_name(call_node) or "unknown"

            for expr_idx, original_expr in enumerate(target_exprs):
                substitutes: dict[str, ast.expr] = dict(_LITERAL_REPLACEMENTS)

                identity = _make_identity(original_expr)
                if identity is not None:
                    substitutes["identity"] = identity

                substitutes["negated"] = _negate_expr(original_expr)

                for label, replacement in substitutes.items():
                    mid = self._next_mutant_id()
                    filename = f"MTR_{mid}_{method}_expr{expr_idx}_{label}.py"
                    modified_line = _modified_line_desc(call_node, original_expr, label)

                    mutated_ast = self._replace_node(original_ast, original_expr, replacement)
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