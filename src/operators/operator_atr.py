"""
ATROperator
===========
ATR — Aggregation Transformation Replacement
(Substituição de Transformação de Agregação)

What it does
------------
Targets aggregation transformations in PySpark programs.  For every call to
``.reduce()`` or ``.reduceByKey()`` found in the program AST, the operator
produces three mutants per eligible call by modifying the aggregation function:

1. **Return first parameter only** — the lambda body is replaced with its
   first argument (e.g. ``lambda a, b: a``).
2. **Return second parameter only** — the lambda body is replaced with its
   second argument (e.g. ``lambda a, b: b``).
3. **Swap parameter order** — the lambda arguments are swapped so that what
   was ``lambda a, b: f(a, b)`` becomes ``lambda b, a: f(a, b)``, effectively
   passing arguments in reverse order to the original body.

Mutation strategy
-----------------
The operator uses ``(lineno, col_offset)`` coordinates as stable identity
keys — the same technique used by ``OperatorNFTP`` — to locate the target
call inside each fresh deep-copy of the original AST.

The aggregation function argument is expected to be an ``ast.Lambda`` node
with exactly **two parameters**.  Calls whose first argument is not a
two-parameter lambda are skipped with a warning log.

Mutation examples
-----------------
**Simple reduce** ``rdd.reduce(lambda a, b: a + b)``::

    3 mutants:
        rdd.reduce(lambda a, b: a)          # return first only
        rdd.reduce(lambda a, b: b)          # return second only
        rdd.reduce(lambda b, a: a + b)      # swap parameter order

**reduceByKey** ``rdd.reduceByKey(lambda x, y: x + y)``::

    3 mutants:
        rdd.reduceByKey(lambda x, y: x)     # return first only
        rdd.reduceByKey(lambda x, y: y)     # return second only
        rdd.reduceByKey(lambda y, x: x + y) # swap parameter order

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


class OperatorATR(Operator):
    """
    Mutation operator that replaces the aggregation function in every
    ``.reduce()`` / ``.reduceByKey()`` call with three boundary variants:
    return-first, return-second, and swap-parameters.

    Three mutants are generated per eligible call, each derived from a
    fresh deep-copy of the original AST.

    Inherits all validation helpers and the ``mutant_list`` accumulator from
    ``Operator``.
    """

    # Human-readable labels for each mutation variant
    _VARIANTS: list[str] = ["return_first", "return_second", "swap_params"]

    def __init__(self) -> None:
        super().__init__(
            id=3,
            name="ATR",
            mutant_registers=["reduce", "reduceByKey"],
        )

    # ------------------------------------------------------------------ #
    # analyse_ast                                                          #
    # ------------------------------------------------------------------ #

    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        """
        Walk ``tree`` and return every ``ast.Call`` node that represents a
        ``.reduce(func)`` or ``.reduceByKey(func)`` method call with at
        least one positional argument.

        Eligibility is intentionally coarse at this stage — we only confirm
        that the call *has* a function argument.  Validation that the argument
        is a two-parameter lambda is deferred to ``build_mutant``, where a
        fresh AST copy is available for safe manipulation.

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
            # Must be a method call: <expr>.reduce(...) or <expr>.reduceByKey(...)
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in self.mutant_registers:
                continue
            # Must carry at least one positional argument (the aggregation function)
            if not node.args:
                continue
            eligible.append(node)

        self._log_analyse_ast_found(
            len(eligible),
            ".reduce / .reduceByKey with a function argument"
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
        For each eligible call node, generate three mutants by modifying
        the aggregation lambda: return-first, return-second, swap-params.

        Directory structure for mutant files::

            mutants/
            ├── atr_1_reduce_line5_return_first/
            │   └── atr.py
            ├── atr_2_reduce_line5_return_second/
            │   └── atr.py
            ├── atr_3_reduce_line5_swap_params/
            │   └── atr.py
            ├── atr_4_reduceByKey_line12_return_first/
            │   └── atr.py
            ...

        Each subdirectory is named
        ``atr_<mutant_id>_<method>_line<lineno>_<variant>``.

        Steps per call node
        -------------------
        1. Read the aggregation function argument (first positional arg) from
           the *original* call node and validate it is a two-parameter lambda.
           Non-lambda or wrong-arity arguments are skipped.
        2. Record the ``(lineno, col_offset)`` of the call for stable
           relocation inside deep-copied trees.
        3. For each of the three mutation variants:

           a. Deep-copy the original AST.
           b. Locate the parent ``ast.Call`` in the copy via
              ``(lineno, col_offset)``.
           c. Build the mutated lambda from the variant strategy.
           d. Replace ``call_copy.args[0]`` with the mutated lambda.
           e. Unparse the mutated AST copy to source code.
           f. Write the source to
              ``<mutant_dir>/atr_<id>_<method>_line<lineno>_<variant>/atr.py``.
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
            If ``nodes`` is not a valid list of AST nodes.
        ValueError
            If ``original_path`` or ``mutant_dir`` is not a non-empty string.
        """
        self._assert_valid_nodes(nodes)
        self._assert_valid_path(original_path, "original_path")
        self._assert_valid_path(mutant_dir,    "mutant_dir")

        mutant_dir_path      = Path(mutant_dir)
        mutant_dir_path.mkdir(parents=True, exist_ok=True)
        original_source_lines = self._read_source_lines(original_path)

        for call_node in nodes:
            call_lineno    = getattr(call_node, "lineno",     None)
            call_col_offset = getattr(call_node, "col_offset", None)
            method_name    = call_node.func.attr

            # ── Validate that the first argument is a two-parameter lambda ──
            func_arg = call_node.args[0]
            if not self._is_two_param_lambda(func_arg):
                self._log_skipping_node(
                    f"Call at line {call_lineno} (.{method_name}): "
                    f"first argument is not a two-parameter lambda"
                )
                continue

            # Extract original parameter names for swap variant
            param_names = [arg.arg for arg in func_arg.args.args]

            # ── Generate one mutant per variant ──────────────────────────
            for variant in self._VARIANTS:
                tree_copy = copy.deepcopy(original_ast)

                call_copy = self._find_call_in_copy(
                    tree_copy, call_lineno, call_col_offset
                )
                if call_copy is None:
                    self._log_skipping_node(
                        f"Call at line {call_lineno} (.{method_name}): "
                        f"could not locate node in AST copy for variant "
                        f"'{variant}'"
                    )
                    continue

                # Build the mutated lambda according to the current variant
                mutated_lambda = self._build_mutated_lambda(
                    call_copy.args[0], param_names, variant
                )
                ast.fix_missing_locations(mutated_lambda)
                call_copy.args[0] = mutated_lambda
                ast.fix_missing_locations(tree_copy)

                mutant_source = ast.unparse(tree_copy)
                mutant_id     = self._next_mutant_id()
                subdir_name   = (
                    f"atr_{mutant_id}_{method_name}"
                    f"_line{call_lineno}_{variant}"
                )
                mutant_subdir = mutant_dir_path / subdir_name
                mutant_subdir.mkdir(parents=True, exist_ok=True)
                mutant_path   = mutant_subdir / "atr.py"
                mutant_path.write_text(mutant_source, encoding="utf-8")

                modified_line = self._get_source_line(
                    original_source_lines, call_lineno
                )

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
                    f"call line {call_lineno} (.{method_name}), "
                    f"variant '{variant}': "
                    f"{modified_line.strip()!r} [{mutant_path.name}]"
                )

        self._log_build_mutant_done()
        return self.mutant_list

    # ------------------------------------------------------------------ #
    # Lambda mutation builders                                             #
    # ------------------------------------------------------------------ #

    def _build_mutated_lambda(
        self,
        original_lambda: ast.Lambda,
        param_names: list[str],
        variant: str,
    ) -> ast.Lambda:
        """
        Return a new ``ast.Lambda`` according to ``variant``.

        Strategies
        ----------
        ``return_first``
            Keep the original parameters; replace the body with the first
            parameter name as an ``ast.Name`` node.
            Result: ``lambda a, b: a``

        ``return_second``
            Keep the original parameters; replace the body with the second
            parameter name as an ``ast.Name`` node.
            Result: ``lambda a, b: b``

        ``swap_params``
            Keep the original body unchanged; swap the order of the two
            parameter names in the argument list.
            Result: ``lambda b, a: <original body>``
            This makes the function receive arguments in reversed order,
            testing whether the aggregation is commutative.

        Parameters
        ----------
        original_lambda : ast.Lambda
            The lambda from the deep-copied AST (safe to modify in place).
        param_names : list[str]
            The two parameter names extracted from the *original* lambda
            (used to build ``ast.Name`` return nodes and for swap).
        variant : str
            One of ``"return_first"``, ``"return_second"``,
            ``"swap_params"``.

        Returns
        -------
        ast.Lambda
            A new lambda node implementing the requested mutation.
        """
        first_param  = param_names[0]
        second_param = param_names[1]

        if variant == "return_first":
            return ast.Lambda(
                args=copy.deepcopy(original_lambda.args),
                body=ast.Name(id=first_param, ctx=ast.Load()),
            )

        if variant == "return_second":
            return ast.Lambda(
                args=copy.deepcopy(original_lambda.args),
                body=ast.Name(id=second_param, ctx=ast.Load()),
            )

        if variant == "swap_params":
            # Deep-copy the entire lambda and then swap the two arg names
            swapped = copy.deepcopy(original_lambda)
            args    = swapped.args.args
            # Swap the arg identifiers while leaving defaults/annotations intact
            args[0].arg, args[1].arg = args[1].arg, args[0].arg
            return swapped

        raise ValueError(f"[ATROperator] Unknown variant: {variant!r}")

    # ------------------------------------------------------------------ #
    # Lambda validation helper                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_two_param_lambda(node: ast.AST) -> bool:
        """
        Return ``True`` when ``node`` is an ``ast.Lambda`` with exactly two
        positional parameters (no *args, **kwargs, or keyword-only args).
        """
        if not isinstance(node, ast.Lambda):
            return False
        args = node.args
        return (
            len(args.args) == 2
            and args.vararg is None
            and args.kwarg is None
            and len(args.kwonlyargs) == 0
            and len(args.posonlyargs) == 0
        )

    # ------------------------------------------------------------------ #
    # Node location helpers (same pattern as OperatorNFTP)                #
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
    # Source file helpers (same pattern as OperatorNFTP)                  #
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
                f"[ATROperator] Warning: could not read source file "
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