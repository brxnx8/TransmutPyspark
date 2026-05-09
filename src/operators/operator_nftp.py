# src/operators/operator_nftp.py
"""
NFTP – Negation of Filter Transformation Predicate
====================================================
Alvo (DataFrame API):
  - df.filter(<pred>) / df.where(<pred>)

Mutações:
  1. Negação total:         ~pred
  2. Inversão de operador:  > → <=,  == → !=,  & → |
  3. isNull ↔ isNotNull
  4. isin  → ~isin
"""

import ast
import copy
from dataclasses import dataclass, field

from src.model.mutant import Mutant
from src.operators.operator import Operator

_FILTER_METHODS = {"filter", "where"}

_COMPARE_INVERSIONS: dict[type, type] = {
    ast.Lt:    ast.GtE,
    ast.GtE:   ast.Lt,
    ast.Gt:    ast.LtE,
    ast.LtE:   ast.Gt,
    ast.Eq:    ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In:    ast.NotIn,
    ast.NotIn: ast.In,
}

_BOOL_INVERSIONS: dict[type, type] = {
    ast.BitAnd: ast.BitOr,
    ast.BitOr:  ast.BitAnd,
    ast.And:    ast.Or,
    ast.Or:     ast.And,
}

_ISNULL_SWAP = {"isNull": "isNotNull", "isNotNull": "isNull"}


def _method_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _get_predicate(call: ast.Call) -> ast.expr | None:
    if call.args:
        return call.args[0]
    return None


def _build_negation(pred: ast.expr) -> ast.expr:
    return ast.UnaryOp(op=ast.Invert(), operand=copy.deepcopy(pred))


def _collect_operator_mutations(pred: ast.expr) -> list[tuple[ast.AST, ast.AST]]:
    pairs: list[tuple[ast.AST, ast.AST]] = []

    for node in ast.walk(pred):
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                inv_cls = _COMPARE_INVERSIONS.get(type(op))
                if inv_cls:
                    new_node = copy.deepcopy(node)
                    new_node.ops[i] = inv_cls()
                    pairs.append((node, new_node))

        elif isinstance(node, ast.BinOp):
            inv_cls = _BOOL_INVERSIONS.get(type(node.op))
            if inv_cls:
                new_node = copy.deepcopy(node)
                new_node.op = inv_cls()
                pairs.append((node, new_node))

        elif isinstance(node, ast.Call):
            mname = _method_name(node)
            if mname in _ISNULL_SWAP:
                new_node = copy.deepcopy(node)
                new_node.func.attr = _ISNULL_SWAP[mname]
                pairs.append((node, new_node))
            elif mname == "isin":
                negated = ast.UnaryOp(op=ast.Invert(), operand=copy.deepcopy(node))
                pairs.append((node, negated))

    return pairs


def _modified_line_desc(call_node: ast.Call, pred: ast.expr, label: str) -> str:
    method = _method_name(call_node) or "?"
    line = getattr(pred, "lineno", "?")
    return f"line {line} | {method}() predicate → {label} | original: {ast.unparse(pred)}"


@dataclass
class OperatorNFTP(Operator):
    _DEFAULT_ID        = 2
    _DEFAULT_NAME      = "NFTP"
    _DEFAULT_REGISTERS = ["filter", "where"]

    id:               int             = 2
    name:             str             = "NFTP"
    mutant_registers: str | list[str] = field(
        default_factory=lambda: ["filter", "where"]
    )

    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        self._assert_valid_tree(tree)
        eligible: list[ast.AST] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _method_name(node) not in _FILTER_METHODS:
                continue
            if _get_predicate(node) is not None:
                eligible.append(node)

        self._log_analyse_ast_found(len(eligible), "filter/where calls with predicate")
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
            pred = _get_predicate(call_node)
            method = _method_name(call_node) or "unknown"
            line = getattr(pred, "lineno", "?")

            # Mutante 1 — negação total
            self._emit(
                original_ast, pred, _build_negation(pred),
                original_path, mutant_dir,
                call_node, pred, "full_negation",
            )

            # Mutantes 2+ — inversão pontual de operadores
            op_pairs = _collect_operator_mutations(pred)
            if not op_pairs:
                self._log_skipping_node(
                    f"Call at line {line}: no invertible sub-conditions found"
                )

            for sub_orig, sub_repl in op_pairs:
                label = f"op_inv_{type(sub_orig).__name__}"
                self._emit(
                    original_ast, sub_orig, sub_repl,
                    original_path, mutant_dir,
                    call_node, pred, label,
                )

        self._log_build_mutant_done()
        return self.mutant_list

    def _emit(
        self,
        original_ast: ast.AST,
        target: ast.AST,
        replacement: ast.AST,
        original_path: str,
        mutant_dir: str,
        call_node: ast.Call,
        pred: ast.expr,
        label: str,
    ) -> None:
        mid = self._next_mutant_id()
        filename = f"NFTP_{mid}_{_method_name(call_node)}_{label}.py"
        modified_line = _modified_line_desc(call_node, pred, label)

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