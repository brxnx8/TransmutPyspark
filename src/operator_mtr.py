"""
MTROperator — Mapping Transformation Replacement
=================================================
Concrete subclass of ``Operator`` that targets mapping transformations in
PySpark programs.

What MTR does
-------------
PySpark mapping operations — ``rdd.map(...)``, ``rdd.flatMap(...)``,
``df.withColumn(...)``, ``df.select(F.expr(...))`` and similar — are the
backbone of any data pipeline.  A subtle bug in the lambda or UDF passed to
these operations can silently corrupt every row in a dataset without raising
any exception.

MTR attacks this class of defects by replacing the **user-supplied mapping
function** with a set of boundary / atypical replacement functions:

    +----------------------------+------------------------------------------+
    | Replacement                | Rationale                                 |
    +============================+==========================================+
    | ``lambda *a, **k: 0``      | Typical zero / falsy value               |
    +----------------------------+------------------------------------------+
    | ``lambda *a, **k: 1``      | Typical unity value                      |
    +----------------------------+------------------------------------------+
    | ``lambda *a, **k: -1``     | Negative / sentinel                      |
    +----------------------------+------------------------------------------+
    | ``lambda *a, **k: None``   | Null — tests None-safety                 |
    +----------------------------+------------------------------------------+
    | ``lambda *a, **k: ""``     | Empty string — tests string-safety       |
    +----------------------------+------------------------------------------+
    | ``lambda *a, **k: []``     | Empty list — tests collection-safety     |
    +----------------------------+------------------------------------------+
    | ``lambda *a, **k: 2**31-1``| MAX_INT (32-bit) boundary                |
    +----------------------------+------------------------------------------+
    | ``lambda *a, **k:-(2**31)``| MIN_INT (32-bit) boundary                |
    +----------------------------+------------------------------------------+

For every eligible call node discovered by ``analyse_ast``, one mutant is
generated per replacement function, yielding up to 8 mutants per occurrence.

Why it adds value
-----------------
* Mapping operations are ubiquitous — MTR is typically the highest-yield
  operator in PySpark mutation suites (as observed in João Batista's study).
* It exercises the test suite's ability to detect both type errors (``None``,
  ``[]``) and numeric boundary errors (``MAX_INT``, ``MIN_INT``).
* Replacing the entire function body makes the mutation independent of the
  original lambda's complexity — even a trivial test should catch it.

Detected patterns
-----------------
``analyse_ast`` recognises the following AST patterns as mapping calls:

    1. ``<expr>.map(<func>)``       — RDD .map()
    2. ``<expr>.flatMap(<func>)``   — RDD .flatMap()
    3. ``<expr>.withColumn(<col>, <expr>)``   — DataFrame column transform
    4. ``<expr>.select(<exprs>)``   — DataFrame projection (first arg only)
    5. ``<expr>.mapValues(<func>)`` — PairRDD .mapValues()
    6. ``<expr>.foreach(<func>)``   — RDD .foreach() / .foreachPartition()
    7. ``<expr>.foreachPartition(<func>)``

Only call nodes whose method name appears in ``mutant_registers`` are
returned, keeping the operator focused and extensible.
"""

import ast
import copy
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.operator import Operator
from src.mutant import Mutant

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────── #
# Boundary replacement lambdas                                                 #
#                                                                              #
# Each entry is a tuple (label, ast.expr) where:                              #
#   label   — short string used in the mutant filename and modified_line      #
#   ast_node — the AST expression to splice in place of the original function #
# ─────────────────────────────────────────────────────────────────────────── #

def _lambda_returning(value_node: ast.expr) -> ast.Lambda:
    """Return an ``ast.Lambda`` that accepts *args/**kwargs and returns value_node."""
    return ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=ast.arg(arg="a"),
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=ast.arg(arg="k"),
            defaults=[],
        ),
        body=value_node,
    )


def _int_constant(n: int) -> ast.Constant:
    return ast.Constant(value=n)


def _unary_minus(n: int) -> ast.UnaryOp:
    return ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=n))


def _power_expr(base: int, exp: int) -> ast.BinOp:
    return ast.BinOp(
        left=ast.Constant(value=base),
        op=ast.Pow(),
        right=ast.Constant(value=exp),
    )


def _max_int_expr() -> ast.BinOp:
    # 2**31 - 1
    return ast.BinOp(
        left=_power_expr(2, 31),
        op=ast.Sub(),
        right=ast.Constant(value=1),
    )


def _min_int_expr() -> ast.UnaryOp:
    # -(2**31)
    return ast.UnaryOp(op=ast.USub(), operand=_power_expr(2, 31))


_REPLACEMENTS: list[tuple[str, ast.expr]] = [
    ("zero",    ast.Constant(value=0)),
    ("one",     ast.Constant(value=1)),
    ("neg_one", ast.Constant(value=-1)),
    ("none",    ast.Constant(value=None)),
    ("empty_str", ast.Constant(value="")),
    ("empty_list", ast.List(elts=[], ctx=ast.Load())),
    ("max_int", _max_int_expr()),
    ("min_int", _min_int_expr()),
]


# ─────────────────────────────────────────────────────────────────────────── #
# MTROperator                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class OperatorMTR(Operator):
    """
    Mapping Transformation Replacement operator.

    Targets every RDD/DataFrame mapping call whose method name is listed in
    ``mutant_registers`` and replaces the function argument with each of the
    boundary replacement lambdas defined in ``_REPLACEMENTS``.

    Default ``mutant_registers``::

        ["map", "flatMap", "withColumn", "select",
         "mapValues", "foreach", "foreachPartition"]

    You may override this list to narrow or widen the operator's scope.
    """

    def __init__(
        self,
        operator_id: int = 1,
        mutant_registers: list[str] | None = None,
    ) -> None:
        super().__init__(
            id=operator_id,
            name="MTR",
            mutant_registers=mutant_registers or [
                "map",
                "flatMap",
                "withColumn",
                "select",
                "mapValues",
                "foreach",
                "foreachPartition",
            ],
        )

    # ------------------------------------------------------------------ #
    # analyse_ast                                                          #
    # ------------------------------------------------------------------ #

    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        """
        Walk ``tree`` and return every ``ast.Call`` node that represents a
        PySpark mapping operation.

        A call qualifies when:
        * It is a method call (``ast.Attribute``).
        * The method name is in ``self.mutant_registers``.
        * It has at least one positional argument (the function to replace).

        Parameters
        ----------
        tree : ast.AST
            Parsed AST from ``MutationManager``.

        Returns
        -------
        list[ast.AST]
            Eligible ``ast.Call`` nodes; empty if none found.

        Raises
        ------
        TypeError
            If ``tree`` is not an ``ast.AST`` instance.
        """
        self._assert_valid_tree(tree)

        eligible: list[ast.AST] = []
        registered = set(self.mutant_registers)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in registered:
                continue
            if not node.args:
                continue
            eligible.append(node)

        self._log_analyse_ast_found(
            len(eligible),
            "mapping transformation calls with function arguments"
        )
        return eligible

    # ------------------------------------------------------------------ #
    # build_mutant                                                         #
    # ------------------------------------------------------------------ #

    def build_mutant(
        self,
        nodes: list[ast.AST],
        original_ast: ast.AST,
        original_path: str,
        mutant_dir: str,
    ) -> list[Mutant]:
        """
        For every eligible call node and every replacement function, generate
        one mutant, write its source to disk and append a ``Mutant`` instance
        to ``self.mutant_list``.

        Directory structure for mutant files::

            mutants/
            ├── mtr_1_map_zero/
            │   └── mtr.py
            ├── mtr_2_map_one/
            │   └── mtr.py
            ├── mtr_3_withColumn_neg_one/
            │   └── mtr.py
            ...

        Each subdirectory is named ``mtr_<mutant_id>_<method>_<replacement_label>``
        and contains the mutated source code in a file named ``mtr.py``.

        The ``modified_line`` stored in the ``Mutant`` dataclass is the
        unparsed line of the mutated call::

            "df.map(lambda *a, **k: 0)"

        Parameters
        ----------
        nodes : list[ast.AST]
            Eligible call nodes from ``analyse_ast``.
        original_ast : ast.AST
            Unmodified program AST.
        original_path : str
            Path to the original PySpark file.
        mutant_dir : str
            Directory where mutant ``.py`` files will be written.

        Returns
        -------
        list[Mutant]
            ``self.mutant_list`` after all new mutants have been appended.

        Raises
        ------
        TypeError
            If ``nodes`` is not a valid list of AST nodes.
        ValueError
            If ``original_path`` or ``mutant_dir`` is not a non-empty string.
        """
        self._assert_valid_nodes(nodes)
        self._assert_valid_path(original_path, "original_path")
        self._assert_valid_path(mutant_dir,    "mutant_dir")

        mutant_dir_path = Path(mutant_dir)
        mutant_dir_path.mkdir(parents=True, exist_ok=True)

        original_nodes = list(ast.walk(original_ast))

        for call_node in nodes:
            method_name = call_node.func.attr
            call_lineno = getattr(call_node, "lineno", "?")

            # Determine which positional argument index holds the function.
            # For withColumn the function lives at index 1; for all others
            # it is always index 0.
            func_arg_idx = 1 if method_name == "withColumn" else 0

            # Guard: ensure the target argument index exists
            if len(call_node.args) <= func_arg_idx:
                self._log_skipping_node(
                    f"Call at line {call_lineno} (.{method_name}): argument index "
                    f"{func_arg_idx} does not exist (only {len(call_node.args)} arg(s))"
                )
                continue

            # Locate the node's position in the original walk order so we
            # can find the corresponding node in each deep-copied tree.
            try:
                node_idx = original_nodes.index(call_node)
            except ValueError:
                self._log_skipping_node(
                    f"Call at line {call_lineno} (.{method_name}): could not "
                    f"locate node in AST walk order"
                )
                continue

            for label, replacement_value in _REPLACEMENTS:
                mutant_id   = self._next_mutant_id()
                replacement = _lambda_returning(copy.deepcopy(replacement_value))
                ast.fix_missing_locations(replacement)

                # Deep-copy the full AST and locate the target node by index
                tree_copy    = copy.deepcopy(original_ast)
                copied_nodes = list(ast.walk(tree_copy))

                if node_idx >= len(copied_nodes):
                    self._log_skipping_node(
                        f"Mutant {mutant_id} (.{method_name}, {label}): node index "
                        f"{node_idx} out of bounds in copied AST"
                    )
                    continue

                target_in_copy = copied_nodes[node_idx]

                if not isinstance(target_in_copy, ast.Call):
                    self._log_skipping_node(
                        f"Mutant {mutant_id} (.{method_name}, {label}): relocated "
                        f"node is not ast.Call"
                    )
                    continue

                # Splice the replacement lambda in place of the original arg
                target_in_copy.args[func_arg_idx] = replacement
                ast.fix_missing_locations(tree_copy)

                mutant_source = ast.unparse(tree_copy)

                # Derive the single modified line for the Mutant record
                modified_line = ast.unparse(target_in_copy)

                # Create a subdirectory for each mutant and write mtr.py inside it
                subdir_name = f"mtr_{mutant_id}_{method_name}_{label}"
                subdir_path = mutant_dir_path / subdir_name
                subdir_path.mkdir(parents=True, exist_ok=True)

                filename    = "mtr.py"
                mutant_path = subdir_path / filename
                mutant_path.write_text(mutant_source, encoding="utf-8")

                mutant = Mutant(
                    id=mutant_id,
                    operator=self.name,
                    original_path=original_path,
                    mutant_path=str(mutant_path),
                    modified_line=modified_line,
                )
                self.mutant_list.append(mutant)

                self._log_mutant_created(
                    mutant_id,
                    f"call line {call_lineno} (.{method_name}), replacement "
                    f"'{label}': {modified_line[:60]}... [{subdir_name}/mtr.py]"
                )

        self._log_build_mutant_done()
        return self.mutant_list