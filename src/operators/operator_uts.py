"""
UTSOperator
===========
UTS — Unary Transformations Swap
(Troca de Transformações Unárias)

What it does
------------
Targets sequential chains of unary transformations in PySpark programs —
both RDD and DataFrame APIs.  A *unary transformation* is a method call that
receives a single DataFrame/RDD as input and produces a single DataFrame/RDD
as output (e.g. ``filter``, ``map``, ``select``, ``distinct``, ``flatMap``).

For every adjacent pair ``(A, B)`` of unary transformations found in the
program AST, the operator produces one mutant in which the two transformations
are swapped so that ``B`` is executed before ``A``.

Swap strategy
-------------
The operator looks for *method-chaining* patterns where one call's receiver
is itself a call:

::

    df.filter(pred).select(cols)   →   df.select(cols).filter(pred)

A pair ``(outer, inner)`` is eligible when **both** method names belong to
``UNARY_TRANSFORMATIONS`` — the flat registry of all known unary
transformations.  No same-group restriction is enforced: even a swap that
produces semantically invalid code is a valid mutant, because it will simply
fail to run and be killed by the test suite.

Registered transformations
--------------------------
``UNARY_TRANSFORMATIONS``::

    filter / where / select / drop / distinct / dropDuplicates /
    limit / orderBy / sort / cache / persist /
    map / flatMap / sortBy

Mutation example
----------------
Source::

    result = df.filter(col("age") > 18).select("name", "age")

Mutant::

    result = df.select("name", "age").filter(col("age") > 18)

Relationship with MutationManager
-----------------------------------
::

    nodes   = operator.analyse_ast(manager.code_ast)
    mutants = operator.build_mutant(
                  nodes, manager.code_ast,
                  manager.config.program_path, mutant_dir)
"""

import ast
import copy
import logging
from pathlib import Path

from src.operators.operator import Operator
from src.model.mutant import Mutant

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────── #
# Registered unary transformations                                             #
# ─────────────────────────────────────────────────────────────────────────── #

UNARY_TRANSFORMATIONS: frozenset[str] = frozenset({
    # DataFrame
    "filter", "where", "select", "drop", "distinct",
    "dropDuplicates", "limit", "orderBy", "sort",
    "cache", "persist",
    # RDD
    "map", "flatMap", "sortBy",
})

# Sorted list used as mutant_registers
_ALL_UNARY: list[str] = sorted(UNARY_TRANSFORMATIONS)


# ─────────────────────────────────────────────────────────────────────────── #
# UTSOperator                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class UTSOperator(Operator):
    """
    Mutation operator that swaps adjacent unary transformations in a
    PySpark method chain.

    One mutant is generated per eligible adjacent pair ``(outer, inner)``
    where both method names appear in ``UNARY_TRANSFORMATIONS``.  No
    same-group restriction is enforced — semantically invalid swaps are
    acceptable mutants and will be killed by the test suite.  Each mutant
    is derived from a fresh deep-copy of the original AST so the original
    tree is never modified.

    Inherits all validation helpers and the ``mutant_list`` accumulator from
    ``Operator``.
    """

    def __init__(self) -> None:
        super().__init__(
            id=3,
            name="UTS",
            mutant_registers=_ALL_UNARY,
        )

    # ------------------------------------------------------------------ #
    # analyse_ast                                                          #
    # ------------------------------------------------------------------ #

    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        """
        Walk ``tree`` and return every ``ast.Call`` node that is an eligible
        *outer* call in an adjacent unary-transformation pair.

        A node is eligible when:

        1. It is an ``ast.Call`` whose ``func`` is an ``ast.Attribute``.
        2. The attribute name belongs to ``UNARY_TRANSFORMATIONS``.
        3. Its receiver (``node.func.value``) is itself an ``ast.Call`` whose
           attribute name also belongs to ``UNARY_TRANSFORMATIONS``.

        No same-group restriction is applied — any two registered unary
        transformations are swappable regardless of the API (DataFrame or RDD)
        they belong to.  Semantically invalid swaps will simply be killed by
        the test suite.

        Only the *outer* call is returned; the inner call is accessible via
        ``outer.func.value``.

        Parameters
        ----------
        tree : ast.AST
            The parsed AST of the PySpark program, obtained from
            ``MutationManager.code_ast``.

        Returns
        -------
        list[ast.Call]
            Eligible outer call nodes, in the order visited by ``ast.walk``.

        Raises
        ------
        TypeError
            If ``tree`` is not an ``ast.AST`` instance.
        """
        self._assert_valid_tree(tree)

        eligible: list[ast.AST] = []

        for node in ast.walk(tree):
            # Outer call must be a method call
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue

            outer_method = node.func.attr
            if outer_method not in UNARY_TRANSFORMATIONS:
                continue

            # Inner call — the receiver of the outer call
            inner_call = node.func.value
            if not isinstance(inner_call, ast.Call):
                continue
            if not isinstance(inner_call.func, ast.Attribute):
                continue

            inner_method = inner_call.func.attr
            if inner_method not in UNARY_TRANSFORMATIONS:
                continue

            eligible.append(node)

        self._log_analyse_ast_found(
            len(eligible),
            "adjacent unary-transformation pairs in method chains"
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
        For each eligible outer call node, generate one mutant by swapping
        the outer and inner transformations in the method chain.

        Directory structure for mutant files::

            mutants/
            ├── uts_1_filter_select_line10/
            │   └── uts.py
            ├── uts_2_select_filter_line22/
            │   └── uts.py
            ...

        Each subdirectory is named
        ``uts_<mutant_id>_<outer>_<inner>_line<lineno>``, where ``<outer>``
        and ``<inner>`` are the original method names and ``<lineno>`` is the
        line of the outer call.

        Swap mechanics
        --------------
        Given a chain ``base.inner(inner_args).outer(outer_args)`` the mutant
        becomes ``base.outer(outer_args).inner(inner_args)``.

        Concretely, in the AST copy:

        1. The *outer* call keeps its position (``lineno``, ``col_offset``) as
           the outermost node.
        2. Its ``func.attr`` is replaced with the inner method name.
        3. Its ``args`` / ``keywords`` are replaced with the inner call's
           arguments.
        4. The *inner* call copy's ``func.attr`` is replaced with the outer
           method name.
        5. The inner call copy's ``args`` / ``keywords`` are replaced with the
           outer call's original arguments.
        6. The receiver of the inner copy (``inner_copy.func.value``) is
           preserved — it is the original base expression.

        This produces a well-formed AST without requiring any parent-pointer
        manipulation.

        Parameters
        ----------
        nodes : list[ast.Call]
            Eligible outer call nodes returned by ``analyse_ast``.
        original_ast : ast.AST
            The unmodified program AST — never mutated in place.
        original_path : str
            Absolute path to the original PySpark source file.
        mutant_dir : str
            Directory where mutant ``.py`` files will be written.

        Returns
        -------
        list[Mutant]
            The full ``self.mutant_list`` after appending the new mutants.

        Raises
        ------
        TypeError
            If ``nodes`` is not a list of ``ast.AST`` instances.
        ValueError
            If ``original_path`` or ``mutant_dir`` is not a non-empty string.
        """
        self._assert_valid_nodes(nodes)
        self._assert_valid_path(original_path, "original_path")
        self._assert_valid_path(mutant_dir, "mutant_dir")

        mutant_dir_path = Path(mutant_dir)
        mutant_dir_path.mkdir(parents=True, exist_ok=True)

        original_source_lines = self._read_source_lines(original_path)

        for outer_node in nodes:
            outer_lineno:     int = outer_node.lineno      # type: ignore[attr-defined]
            outer_col_offset: int = outer_node.col_offset  # type: ignore[attr-defined]

            outer_method: str      = outer_node.func.attr        # type: ignore[attr-defined]
            inner_node:   ast.Call = outer_node.func.value       # type: ignore[attr-defined]
            inner_method: str      = inner_node.func.attr        # type: ignore[attr-defined]
            inner_lineno: int      = inner_node.lineno           # type: ignore[attr-defined]

            # ── Fresh copy per mutant ─────────────────────────────────────
            tree_copy = copy.deepcopy(original_ast)

            outer_copy = self._find_call_in_copy(
                tree_copy, outer_lineno, outer_col_offset
            )
            if outer_copy is None:
                self._log_skipping_node(
                    f"Could not relocate outer call '{outer_method}' "
                    f"at line {outer_lineno} in AST copy"
                )
                continue

            inner_copy: ast.Call = outer_copy.func.value  # type: ignore[attr-defined]

            # ── Perform the swap ──────────────────────────────────────────
            #
            # Original:  base . inner(inner_args) . outer(outer_args)
            # Mutant:    base . outer(outer_args) . inner(inner_args)
            #
            # We rewrite in-place on the copy:
            #   1. outer_copy keeps its position as the outermost node
            #      but now carries the inner method name + inner args.
            #   2. inner_copy now carries the outer method name + outer args.
            #   3. inner_copy.func.value (the base) stays untouched.

            # Stash originals before overwriting
            outer_args_copy    = copy.deepcopy(outer_copy.args)
            outer_kwargs_copy  = copy.deepcopy(outer_copy.keywords)
            inner_args_copy    = copy.deepcopy(inner_copy.args)
            inner_kwargs_copy  = copy.deepcopy(inner_copy.keywords)

            # Rewrite outer node → becomes the old inner transformation
            outer_copy.func.attr  = inner_method   # type: ignore[attr-defined]
            outer_copy.args       = inner_args_copy
            outer_copy.keywords   = inner_kwargs_copy

            # Rewrite inner node → becomes the old outer transformation
            inner_copy.func.attr  = outer_method   # type: ignore[attr-defined]
            inner_copy.args       = outer_args_copy
            inner_copy.keywords   = outer_kwargs_copy

            # ── Unparse, write, record ────────────────────────────────────
            ast.fix_missing_locations(tree_copy)
            mutant_source = ast.unparse(tree_copy)

            mutant_id    = self._next_mutant_id()
            subdir_name  = (
                f"uts_{mutant_id}_{outer_method}_{inner_method}"
                f"_line{outer_lineno}"
            )
            mutant_subdir = mutant_dir_path / subdir_name
            mutant_subdir.mkdir(parents=True, exist_ok=True)
            mutant_path   = mutant_subdir / "uts.py"
            mutant_path.write_text(mutant_source, encoding="utf-8")

            modified_line = self._get_source_line(
                original_source_lines, outer_lineno
            )

            mutant = Mutant(
                id            = mutant_id,
                operator      = self.name,
                original_path = original_path,
                mutant_path   = str(mutant_path),
                modified_line = modified_line,
            )
            self.mutant_list.append(mutant)

            self._log_mutant_created(
                mutant_id,
                f"call line {outer_lineno} ({outer_method} ↔ {inner_method}), "
                f"inner line {inner_lineno}: "
                f"{modified_line.strip()!r} [{mutant_path.name}]"
            )

        self._log_build_mutant_done()
        return self.mutant_list

    # ------------------------------------------------------------------ #
    # Node location helpers                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_call_in_copy(
        tree_copy:  ast.AST,
        lineno:     int,
        col_offset: int,
    ) -> ast.Call | None:
        """
        Return the ``ast.Call`` node in ``tree_copy`` whose ``(lineno,
        col_offset)`` matches, or ``None`` if not found.

        The pair ``(lineno, col_offset)`` is a stable identity key because
        the copy is produced from the same unmodified source tree before any
        mutation is applied.
        """
        for node in ast.walk(tree_copy):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and getattr(node, "lineno",     None) == lineno
                and getattr(node, "col_offset", None) == col_offset
            ):
                return node
        return None

    # ------------------------------------------------------------------ #
    # Source file helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_source_lines(original_path: str) -> list[str]:
        """
        Read the original source file and return its lines (empty list on
        I/O error so a bad path does not abort the entire mutation run).
        """
        try:
            return Path(original_path).read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, OSError) as exc:
            logger.warning(
                f"[UTSOperator] Warning: could not read source file "
                f"'{original_path}': {exc}. modified_line will be empty."
            )
            return []

    @staticmethod
    def _get_source_line(lines: list[str], lineno: int) -> str:
        """Return the source line at ``lineno`` (1-based); empty on miss."""
        idx = lineno - 1
        if 0 <= idx < len(lines):
            return lines[idx]
        return ""

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.id}, "
            f"name={self.name!r}, "
            f"mutant_registers={self.mutant_registers!r}, "
            f"mutants={len(self.mutant_list)}"
            f")"
        )
