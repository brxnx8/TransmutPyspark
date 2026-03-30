"""
MutationManager
===============
Responsible for:
  1. Parsing the PySpark program source (obtained from ConfigLoader) into an AST.
  2. Coordinating with Operator objects to identify eligible nodes and obtain
     replacement nodes, then reconstructing the mutated source code.
  3. Storing every generated mutant (as source code) in ``mutateList``.

Deliberately out of scope:
  - File I/O of source programs        → ConfigLoader
  - Deciding *what* nodes are eligible → Operator.analyseAST()
  - Deciding *how* to replace a node   → Operator.buildMutate()
  - Running tests against mutants      → TestRunner (future)

Flow
----
::

    cfg     = ConfigLoader(...).load()
    manager = MutationManager(configLoader=cfg)
    manager.parseToAST(cfg.program_source)   # 1. parse → internal AST

    operator.set_code_ast(manager.program_ast)
    operator.analyseAST()                    # 2. operator finds eligible nodes

    manager.applyMutation(operator)          # 3. manager generates mutants
                                             #    using operator.buildMutate()
"""

import ast
import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from code.config_loader import ConfigLoader
    from code.operator import Operator


# ─────────────────────────────────────────────────────────────────────────── #
# Mutant dataclass                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class Mutant:
    """
    Represents a single generated mutant.

    Attributes
    ----------
    operator_name    : Name of the operator that produced this mutant
                       (e.g. ``"AOR"``).
    occurrence_index : Zero-based index of the occurrence that was replaced
                       within the operator's ``registers`` list.
    source_code      : The full mutated program as a source-code string,
                       ready to be written to disk or executed.
    """

    operator_name: str
    occurrence_index: int
    source_code: str

    def __repr__(self) -> str:
        preview = self.source_code[:60].replace("\n", "\\n")
        return (
            f"Mutant(operator={self.operator_name!r}, "
            f"occurrence={self.occurrence_index}, "
            f"source_preview={preview!r}...)"
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Internal helper — NodeReplacer                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class _NodeReplacer(ast.NodeTransformer):
    """
    Replaces **one specific node instance** in the AST with a replacement node.

    Unlike a type-based replacer, this one matches by object identity so it
    works correctly even when the same node type appears multiple times —
    the Operator already identified exactly which instance to replace.

    Parameters
    ----------
    target_node      : The exact node instance to replace (identity match).
    replacement_node : The node to insert in its place (deep-copied on use).
    """

    def __init__(self, target_node: ast.AST, replacement_node: ast.AST) -> None:
        self._target_node = target_node
        self._replacement_node = replacement_node

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if node is self._target_node:
            return copy.deepcopy(self._replacement_node)
        return super().generic_visit(node)


# ─────────────────────────────────────────────────────────────────────────── #
# MutationManager                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class MutationManager:
    """
    Orchestrates the mutation-testing pipeline for a PySpark program.

    Parameters
    ----------
    configLoader : ConfigLoader
        A fully loaded ``ConfigLoader`` instance that provides the raw
        program source and workspace metadata.
    mutateList : list[Mutant]
        Accumulator for every mutant generated across all operators.
        Starts empty and grows with each ``applyMutation`` call.
    """

    configLoader: "ConfigLoader"
    mutateList: list[Mutant] = field(default_factory=list)

    # Internal state — populated by parseToAST()
    _program_ast: ast.AST = field(init=False, default=None, repr=False)
    _program_source: str = field(init=False, default="", repr=False)

    # ------------------------------------------------------------------ #
    # Post-init validation                                                 #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        self._validate_config_loader()
        self._validate_mutate_list()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def parseToAST(self, source: str) -> ast.AST:
        """
        Parse a PySpark program source string into an AST.

        The resulting tree is stored internally and exposed via
        ``program_ast`` so that Operator instances can receive it through
        ``operator.set_code_ast(manager.program_ast)``.

        Calling ``parseToAST`` again with new source replaces the current
        tree and clears ``mutateList`` to avoid stale mutants.

        Parameters
        ----------
        source : str
            Raw Python / PySpark source code to parse.

        Returns
        -------
        ast.AST
            The parsed and location-fixed syntax tree.

        Raises
        ------
        TypeError
            If ``source`` is not a non-empty string.
        ValueError
            If ``source`` contains a syntax error.
        """
        self._validate_source(source)

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValueError(
                f"[MutationManager] Syntax error in source at line "
                f"{exc.lineno}: {exc.msg}"
            ) from exc

        ast.fix_missing_locations(tree)

        self._program_source = source
        self._program_ast = tree

        if self.mutateList:
            print("[MutationManager] New source — clearing previous mutants.")
            self.mutateList.clear()

        print(
            f"[MutationManager] Source parsed successfully "
            f"({sum(1 for _ in ast.walk(tree))} AST nodes)."
        )
        return self._program_ast

    def applyMutation(self, operator: "Operator") -> list[Mutant]:
        """
        Generate one mutant per node registered by ``operator.analyseAST()``.

        For each node in ``operator.registers``:
          1. Calls ``operator.buildMutate(node)`` to obtain the replacement.
          2. Deep-copies the original AST.
          3. Substitutes that exact node instance with the replacement.
          4. Unparses the mutated tree back to source code.
          5. Wraps the result in a ``Mutant`` and appends it to
             ``self.mutateList``.

        The original ``_program_ast`` is never modified.

        Parameters
        ----------
        operator : Operator
            A concrete ``Operator`` subclass that has already had
            ``analyseAST()`` called, so that ``operator.registers`` is
            populated with the eligible nodes.

        Returns
        -------
        list[Mutant]
            The mutants generated by this call (also stored in
            ``self.mutateList``).

        Raises
        ------
        RuntimeError
            If ``parseToAST`` has not been called yet.
        TypeError
            If ``operator`` does not expose the required interface
            (``name``, ``registers``, ``analyseAST``, ``buildMutate``).
        ValueError
            If ``operator.registers`` is empty (``analyseAST`` was not
            called or found no eligible nodes).
        """
        self._assert_ast_ready()
        self._validate_operator(operator)

        if not operator.registers:
            print(
                f"[MutationManager] Operator '{operator.name}': "
                f"registers is empty — call analyseAST() first or no "
                f"eligible nodes exist in the current AST."
            )
            return []

        new_mutants: list[Mutant] = []

        for idx, target_node in enumerate(operator.registers):
            # Ask the operator how to replace this specific node
            replacement_node = operator.buildMutate(target_node)

            # Work on a deep copy so the original tree is never mutated
            tree_copy = copy.deepcopy(self._program_ast)

            # After deep copy, the identity of target_node is gone — find
            # the equivalent node in the copy by position/index
            original_nodes = list(ast.walk(self._program_ast))
            copied_nodes = list(ast.walk(tree_copy))

            try:
                node_index = original_nodes.index(target_node)
                target_in_copy = copied_nodes[node_index]
            except (ValueError, IndexError):
                print(
                    f"[MutationManager] Operator '{operator.name}': "
                    f"could not locate node at occurrence {idx} in the "
                    f"copied tree — skipping."
                )
                continue

            replacer = _NodeReplacer(
                target_node=target_in_copy,
                replacement_node=replacement_node,
            )
            mutated_tree = replacer.visit(tree_copy)
            ast.fix_missing_locations(mutated_tree)

            mutant_source = ast.unparse(mutated_tree)

            mutant = Mutant(
                operator_name=operator.name,
                occurrence_index=idx,
                source_code=mutant_source,
            )
            new_mutants.append(mutant)
            self.mutateList.append(mutant)

        print(
            f"[MutationManager] Operator '{operator.name}': "
            f"{len(new_mutants)} mutant(s) generated from "
            f"{len(operator.registers)} registered node(s)."
        )
        return new_mutants

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def program_ast(self) -> ast.AST:
        """The parsed AST. Available after ``parseToAST()``."""
        self._assert_ast_ready()
        return self._program_ast

    @property
    def program_source(self) -> str:
        """The raw source that was parsed. Available after ``parseToAST()``."""
        self._assert_ast_ready()
        return self._program_source

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _validate_config_loader(self) -> None:
        """Duck-type check: configLoader must expose the required attributes."""
        required_attrs = ("program_source", "workspace_path", "operatorsList")
        for attr in required_attrs:
            if not hasattr(self.configLoader, attr):
                raise TypeError(
                    f"[MutationManager] configLoader is missing attribute "
                    f"'{attr}'. Pass a loaded ConfigLoader instance."
                )
        try:
            _ = self.configLoader.program_source
        except RuntimeError as exc:
            raise RuntimeError(
                "[MutationManager] The provided ConfigLoader has not been "
                "loaded yet. Call configLoader.load() first."
            ) from exc

    def _validate_mutate_list(self) -> None:
        if not isinstance(self.mutateList, list):
            raise TypeError(
                f"[MutationManager] mutateList must be a list, "
                f"got: {type(self.mutateList)}"
            )

    def _validate_source(self, source: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise TypeError(
                f"[MutationManager] source must be a non-empty string, "
                f"got: {type(source)}"
            )

    def _validate_operator(self, operator: object) -> None:
        """Duck-type check: operator must expose name, registers, and methods."""
        required_attrs = ("name", "registers", "analyseAST", "buildMutate")
        for attr in required_attrs:
            if not hasattr(operator, attr):
                raise TypeError(
                    f"[MutationManager] operator is missing attribute / method "
                    f"'{attr}'. Pass a concrete Operator subclass instance."
                )
        if not isinstance(operator.name, str) or not operator.name.strip():
            raise TypeError(
                f"[MutationManager] operator.name must be a non-empty string, "
                f"got: {operator.name!r}"
            )
        if not isinstance(operator.registers, list):
            raise TypeError(
                f"[MutationManager] operator.registers must be a list, "
                f"got: {type(operator.registers)}"
            )

    def _assert_ast_ready(self) -> None:
        if self._program_ast is None:
            raise RuntimeError(
                "[MutationManager] AST is not available. "
                "Call parseToAST(source) first."
            )

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"MutationManager("
            f"ast_ready={self._program_ast is not None}, "
            f"mutants={len(self.mutateList)}"
            f")"
        )