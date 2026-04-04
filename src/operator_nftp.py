"""
NFTPOperator
============
NFTP — Negation of Filter Transformation Predicate
(Negação do Predicado de Transformação de Filtragem)

What it does
------------
Targets filtering transformations in PySpark DataFrames.  For every call to
``.filter()`` or ``.where()`` found in the program AST, the operator produces
one mutant per atomic sub-condition by wrapping that sub-condition in
``not (...)``, leaving the rest of the predicate unchanged.

Negation strategy
-----------------
**UnaryNot only** — the target sub-condition is wrapped in
``ast.UnaryOp(op=ast.Not(), operand=<sub-condition>)``.

Sub-condition decomposition
---------------------------
The predicate argument is decomposed into its leaf sub-conditions by
``_collect_subconditions``.  A node is a *leaf* when it is not a transparent
compound:

- ``ast.BoolOp``  — Python ``and`` / ``or`` → descend into all values
- ``ast.BinOp`` with ``ast.BitAnd`` / ``ast.BitOr`` — PySpark ``&`` / ``|``
  → descend left and right

Any other node is returned as a leaf and becomes a negation candidate.
Nodes that are already a ``not (...)`` (``ast.UnaryOp`` with ``ast.Not``)
are **skipped** — negating them would produce a trivial double-negation.

Mutation examples
-----------------
**Simple predicate** ``df.filter(col("age") > 18)``::

    1 mutant:
        df.filter(not (col("age") > 18))

**Compound predicate** ``df.filter((col("a") > 0) & (col("b") < 10))``::

    2 mutants:
        df.filter(not (col("a") > 0) & (col("b") < 10))   # mutant 1
        df.filter((col("a") > 0) & not (col("b") < 10))   # mutant 2

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

from src.operator import Operator
from src.mutant import Mutant

logger = logging.getLogger(__name__)

class OperatorNFTP(Operator):
    """
    Mutation operator that negates one sub-condition at a time inside every
    ``.filter()`` / ``.where()`` call using ``not (...)``.

    One mutant is generated per atomic leaf sub-condition found in the
    predicate.  Each mutant is independent: it is derived from a fresh
    deep-copy of the original AST.

    Inherits all validation helpers and the ``mutant_list`` accumulator from
    ``Operator``.
    """

    def __init__(self) -> None:
        super().__init__(
            id=2,
            name="NFTP",
            mutant_registers=["filter", "where"],
        )

    # ------------------------------------------------------------------ #
    # analyse_ast                                                          #
    # ------------------------------------------------------------------ #

    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        """
        Walk ``tree`` and return every ``ast.Call`` node that represents a
        ``.filter(predicate)`` or ``.where(predicate)`` method call with at
        least one positional argument.

        Eligibility is intentionally coarse here — we only confirm that a
        filter/where call *has* a predicate.  Sub-condition decomposition and
        the ``not``-already check happen in ``build_mutant``, where a fresh
        AST copy is available for safe manipulation.

        Parameters
        ----------
        tree : ast.AST
            The parsed AST of the PySpark program, obtained from
            ``MutationManager.code_ast``.

        Returns
        -------
        list[ast.Call]
            Eligible call nodes, in the order they are visited by
            ``ast.walk``.

        Raises
        ------
        TypeError
            If ``tree`` is not an ``ast.AST`` instance.
        """
        self._assert_valid_tree(tree)

        eligible: list[ast.AST] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Must be a method call: <expr>.filter(...) or <expr>.where(...)
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in self.mutant_registers:
                continue
            # Must carry at least one positional argument (the predicate)
            if not node.args:
                continue
            eligible.append(node)

        self._log_analyse_ast_found(
            len(eligible),
            ".filter / .where with a predicate argument"
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
        For each eligible call node, decompose its predicate into atomic
        sub-conditions and generate one mutant per sub-condition by wrapping
        it in ``not (...)``.

        Directory structure for mutant files::

            mutants/
            ├── nftp_1_filter_line42/
            │   └── nftp.py
            ├── nftp_2_where_line15/
            │   └── nftp.py
            ├── nftp_3_filter_line42/
            │   └── nftp.py
            ...

        Each subdirectory is named ``nftp_<mutant_id>_<method>_line<lineno>``
        where ``<method>`` is "filter" or "where", and ``<lineno>`` indicates
        the line of the negated sub-condition.

        Steps per call node
        -------------------
        1. Read the predicate (first positional argument) from the *original*
           call node to collect sub-condition ``(lineno, col_offset)``
           coordinates — the original AST is never modified.
        2. For each leaf sub-condition:

           a. Deep-copy the original AST.
           b. Relocate the parent ``ast.Call`` in the copy via
              ``(lineno, col_offset)``.
           c. Build ``not (sub-condition)`` as an ``ast.UnaryOp``.
           d. Splice the negated node into the predicate subtree:

              - If the sub-condition *is* the entire predicate (simple case),
                replace ``call_copy.args[0]`` directly.
              - Otherwise, walk the predicate subtree and replace the child
                that matches the sub-condition's coordinates.

           e. Unparse the mutated AST copy to source code.
           f. Write the source to ``<mutant_dir>/nftp_<id>.py``.
           g. Record ``modified_line`` from the original source file.
           h. Append a ``Mutant`` instance to ``self.mutant_list``.

        Parameters
        ----------
        nodes : list[ast.Call]
            Eligible nodes returned by ``analyse_ast``.
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

        for call_node in nodes:
            call_lineno     = call_node.lineno       # type: ignore[attr-defined]
            call_col_offset = call_node.col_offset   # type: ignore[attr-defined]

            # Collect leaf sub-conditions from the original (read-only) node
            # to capture their stable (lineno, col_offset) coordinates.
            predicate      = call_node.args[0]       # type: ignore[attr-defined]
            sub_conditions = self._collect_subconditions(predicate)

            if not sub_conditions:
                self._log_skipping_node(
                    f"Call at line {call_lineno}: no eligible sub-conditions found "
                    f"(all already negated?)"
                )
                continue

            for sub_node in sub_conditions:
                sub_lineno     = sub_node.lineno
                sub_col_offset = sub_node.col_offset

                # ── Fresh copy per mutant ─────────────────────────────────
                tree_copy = copy.deepcopy(original_ast)

                call_copy = self._find_call_in_copy(
                    tree_copy, call_lineno, call_col_offset
                )
                if call_copy is None:
                    self._log_skipping_node(
                        f"Could not relocate call at line {call_lineno} in AST copy"
                    )
                    continue

                predicate_copy = call_copy.args[0]

                # Locate the sub-condition inside the copied predicate subtree
                sub_copy = self._find_node_in_subtree(
                    predicate_copy, sub_lineno, sub_col_offset
                )
                if sub_copy is None:
                    self._log_skipping_node(
                        f"Could not relocate sub-condition at line {sub_lineno}"
                    )
                    continue

                # ── Build not (sub-condition) ─────────────────────────────
                negated = ast.UnaryOp(
                    op      = ast.Not(),
                    operand = copy.deepcopy(sub_copy),
                )
                ast.copy_location(negated, sub_copy)
                ast.fix_missing_locations(negated)

                # ── Splice into the predicate subtree ─────────────────────
                replaced = self._replace_node_in_subtree(
                    predicate_copy, sub_lineno, sub_col_offset, negated
                )
                if not replaced:
                    # The sub-condition IS the whole predicate (simple case)
                    call_copy.args[0] = negated

                # ── Unparse, write, record ────────────────────────────────
                ast.fix_missing_locations(tree_copy)
                mutant_source = ast.unparse(tree_copy)

                mutant_id     = self._next_mutant_id()
                method_name   = call_node.func.attr
                subdir_name   = f"nftp_{mutant_id}_{method_name}_line{sub_lineno}"
                mutant_subdir = mutant_dir_path / subdir_name
                mutant_subdir.mkdir(parents=True, exist_ok=True)
                mutant_path   = mutant_subdir / "nftp.py"
                mutant_path.write_text(mutant_source, encoding="utf-8")

                modified_line = self._get_source_line(
                    original_source_lines, sub_lineno
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
                    f"call line {call_lineno}, sub-condition line {sub_lineno}: "
                    f"{modified_line.strip()!r} [{mutant_path.name}]"
                )

        self._log_build_mutant_done()
        return self.mutant_list

    # ------------------------------------------------------------------ #
    # Sub-condition decomposition                                          #
    # ------------------------------------------------------------------ #

    def _collect_subconditions(self, node: ast.AST) -> list[ast.expr]:
        """
        Recursively collect atomic (leaf) sub-conditions from a predicate.

        Transparent compound nodes are traversed without becoming candidates:

        - ``ast.BoolOp``  — Python ``and`` / ``or``
        - ``ast.BinOp`` with ``ast.BitAnd`` or ``ast.BitOr``  — PySpark
          ``&`` / ``|``

        Any other node is treated as a leaf and returned as a candidate.

        Nodes that are already a ``not (...)`` (``ast.UnaryOp`` with
        ``ast.Not``) are silently skipped to prevent double-negation mutants.

        Parameters
        ----------
        node : ast.expr
            Root of the predicate subtree.

        Returns
        -------
        list[ast.expr]
            Leaf sub-condition nodes, carrying their original ``lineno`` /
            ``col_offset`` for safe relocation inside AST copies.
        """
        # Already negated — skip to avoid not (not (x))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return []

        # Python boolean compound: and / or
        if isinstance(node, ast.BoolOp):
            result: list[ast.expr] = []
            for value in node.values:
                result.extend(self._collect_subconditions(value))
            return result

        # PySpark bitwise boolean compound: & / |
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.BitAnd, ast.BitOr)
        ):
            return (
                self._collect_subconditions(node.left)
                + self._collect_subconditions(node.right)
            )

        # Leaf — return as negation candidate
        return [node]  # type: ignore[return-value]

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

    @staticmethod
    def _find_node_in_subtree(
        subtree:    ast.AST,
        lineno:     int,
        col_offset: int,
    ) -> ast.expr | None:
        """
        Walk ``subtree`` and return the first node whose ``(lineno,
        col_offset)`` matches, or ``None`` if not found.
        """
        for node in ast.walk(subtree):
            if (
                getattr(node, "lineno",     None) == lineno
                and getattr(node, "col_offset", None) == col_offset
            ):
                return node  # type: ignore[return-value]
        return None

    @staticmethod
    def _replace_node_in_subtree(
        subtree:     ast.AST,
        lineno:      int,
        col_offset:  int,
        replacement: ast.expr,
    ) -> bool:
        """
        Walk ``subtree`` and replace the first **child** whose
        ``(lineno, col_offset)`` matches with ``replacement``.

        The parent's field is patched in place, so the change is immediately
        visible in the enclosing ``tree_copy``.

        Returns ``True`` if a replacement was made.
        Returns ``False`` when no matching child is found — which means the
        target node is the root of ``subtree`` itself (i.e. the predicate is
        a simple, non-compound expression).  The caller must then replace
        ``call_copy.args[0]`` directly.
        """
        for node in ast.walk(subtree):
            for field_name, field_value in ast.iter_fields(node):

                # Direct child node
                if (
                    isinstance(field_value, ast.AST)
                    and getattr(field_value, "lineno",     None) == lineno
                    and getattr(field_value, "col_offset", None) == col_offset
                ):
                    setattr(node, field_name, replacement)
                    return True

                # Child inside a list field (e.g. ast.BoolOp.values)
                if isinstance(field_value, list):
                    for i, item in enumerate(field_value):
                        if (
                            isinstance(item, ast.AST)
                            and getattr(item, "lineno",     None) == lineno
                            and getattr(item, "col_offset", None) == col_offset
                        ):
                            field_value[i] = replacement
                            return True
        return False

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
            logger.warnig(
                f"[NFTPOperator] Warning: could not read source file "
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