"""
MutationManager
===============
Responsible for:
  1. Parsing the PySpark program source (obtained from ConfigLoader) into an AST.
  2. Receiving Operator objects and applying their mutations to the AST,
     generating one mutant per occurrence found in the tree.
  3. Storing every generated mutant (as source code) in ``mutateList``.

Deliberately out of scope:
  - File I/O of source programs   → ConfigLoader
  - Deciding *what* to mutate     → Operator
  - Running tests against mutants → TestRunner (future)
"""

import ast
import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

# Forward reference only — avoids a hard circular import when Operator is
# defined in a separate module.  At runtime we use the Protocol below.
if TYPE_CHECKING:
    from config_loader import ConfigLoader


# ─────────────────────────────────────────────────────────────────────────── #
# Operator Protocol                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

@runtime_checkable
class OperatorProtocol(Protocol):
    """
    Structural interface that every Operator must satisfy.

    Attributes
    ----------
    target_node_type : type[ast.AST]
        The AST node class this operator targets (e.g. ``ast.Add``).
    replacement_node : ast.AST
        A *template* node that will replace each matched target node.
        MutationManager deep-copies it before each substitution so the
        original template is never mutated.
    operator_id : str
        Human-readable identifier used in log messages and mutant metadata
        (e.g. ``"AOR"``, ``"ROR"``).
    """

    target_node_type: type
    replacement_node: ast.AST
    operator_id: str


# ─────────────────────────────────────────────────────────────────────────── #
# Internal helper — NodeReplacer                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class _NodeReplacer(ast.NodeTransformer):
    """
    AST NodeTransformer that replaces **one specific occurrence** of a target
    node type with a replacement node, leaving all other occurrences intact.

    Parameters
    ----------
    target_type      : type[ast.AST]  — node class to match.
    replacement_node : ast.AST        — node to insert (deep-copied internally).
    occurrence_index : int            — zero-based index of the occurrence to replace.
    """

    def __init__(
        self,
        target_type: type,
        replacement_node: ast.AST,
        occurrence_index: int,
    ) -> None:
        self._target_type = target_type
        self._replacement_node = replacement_node
        self._occurrence_index = occurrence_index
        self._current_count = 0

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, self._target_type):
            if self._current_count == self._occurrence_index:
                self._current_count += 1
                return copy.deepcopy(self._replacement_node)
            self._current_count += 1
        return super().generic_visit(node)


# ─────────────────────────────────────────────────────────────────────────── #
# Mutant dataclass                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class Mutant:
    """
    Represents a single generated mutant.

    Attributes
    ----------
    operator_id      : Identifier of the operator that produced this mutant.
    occurrence_index : Zero-based index of the occurrence that was replaced.
    source_code      : The full mutated program as a source-code string.
    """

    operator_id: str
    occurrence_index: int
    source_code: str

    def __repr__(self) -> str:
        preview = self.source_code[:60].replace("\n", "\\n")
        return (
            f"Mutant(operator={self.operator_id!r}, "
            f"occurrence={self.occurrence_index}, "
            f"source_preview={preview!r}...)"
        )


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
        A fully loaded ConfigLoader instance that provides the program
        source code and workspace metadata.
    mutateList : list[Mutant]
        Accumulator for every mutant generated across all operators.
        Starts empty and is populated by successive ``applyMutation`` calls.

    Typical usage
    -------------
    ::

        cfg     = ConfigLoader(...).load()
        manager = MutationManager(configLoader=cfg)
        tree    = manager.parseToAST(cfg.program_source)

        for operator in operators:
            manager.applyMutation(operator)

        for mutant in manager.mutateList:
            print(mutant)
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

        The resulting tree is stored internally and used by subsequent
        ``applyMutation`` calls. Calling ``parseToAST`` again replaces the
        current tree and clears ``mutateList`` to avoid stale mutants.

        Parameters
        ----------
        source : str
            Raw Python/PySpark source code to parse.

        Returns
        -------
        ast.AST
            The parsed syntax tree.

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
                f"[MutationManager] Syntax error in source at line {exc.lineno}: {exc.msg}"
            ) from exc

        # Fix missing line numbers so ast.unparse works correctly
        ast.fix_missing_locations(tree)

        self._program_source = source
        self._program_ast = tree

        # Reset mutants — new source means previous mutants are stale
        if self.mutateList:
            print(
                "[MutationManager] New source provided — clearing previous mutants."
            )
            self.mutateList.clear()

        print(
            f"[MutationManager] Source parsed successfully "
            f"({sum(1 for _ in ast.walk(tree))} AST nodes)."
        )
        return self._program_ast

    def applyMutation(self, operator: OperatorProtocol) -> list[Mutant]:
        """
        Apply a single mutation operator to the parsed AST.

        For each occurrence of ``operator.target_node_type`` found in the
        tree, one independent mutant is generated (all other occurrences
        remain untouched in that mutant).  Every generated ``Mutant`` is
        appended to ``self.mutateList``.

        Parameters
        ----------
        operator : OperatorProtocol
            An object exposing ``target_node_type``, ``replacement_node``
            and ``operator_id``.

        Returns
        -------
        list[Mutant]
            The subset of mutants generated by *this* operator call
            (also available in ``self.mutateList``).

        Raises
        ------
        RuntimeError
            If ``parseToAST`` has not been called yet.
        TypeError
            If ``operator`` does not satisfy ``OperatorProtocol``.
        """
        self._assert_ast_ready()
        self._validate_operator(operator)

        # Count how many occurrences of the target node exist in the tree
        occurrences = self._count_occurrences(operator.target_node_type)

        if occurrences == 0:
            print(
                f"[MutationManager] Operator '{operator.operator_id}': "
                f"no occurrences of {operator.target_node_type.__name__} found — "
                f"skipping."
            )
            return []

        new_mutants: list[Mutant] = []

        for idx in range(occurrences):
            # Work on a deep copy of the original tree for every mutant
            tree_copy = copy.deepcopy(self._program_ast)

            replacer = _NodeReplacer(
                target_type=operator.target_node_type,
                replacement_node=operator.replacement_node,
                occurrence_index=idx,
            )
            mutated_tree = replacer.visit(tree_copy)
            ast.fix_missing_locations(mutated_tree)

            mutant_source = ast.unparse(mutated_tree)

            mutant = Mutant(
                operator_id=operator.operator_id,
                occurrence_index=idx,
                source_code=mutant_source,
            )
            new_mutants.append(mutant)
            self.mutateList.append(mutant)

        print(
            f"[MutationManager] Operator '{operator.operator_id}': "
            f"{len(new_mutants)} mutant(s) generated "
            f"({occurrences} occurrence(s) of "
            f"{operator.target_node_type.__name__})."
        )
        return new_mutants

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def program_ast(self) -> ast.AST:
        """The parsed AST of the program. Available after parseToAST()."""
        self._assert_ast_ready()
        return self._program_ast

    @property
    def program_source(self) -> str:
        """The raw source that was parsed. Available after parseToAST()."""
        self._assert_ast_ready()
        return self._program_source

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _validate_config_loader(self) -> None:
            """Ensure configLoader is a loaded ConfigLoader instance."""
            
            # 1. Checa apenas os atributos que não são propriedades dinâmicas perigosas
            required_attrs = ("workspace_path", "operatorsList")
            for attr in required_attrs:
                if not hasattr(self.configLoader, attr):
                    raise TypeError(
                        f"[MutationManager] configLoader must be a loaded ConfigLoader "
                        f"instance (missing attribute '{attr}')."
                    )

            # 2. Testa a existência e o estado do program_source no mesmo bloco
            try:
                _ = self.configLoader.program_source
            except AttributeError:
                # Caso a classe nem tenha o atributo implementado
                raise TypeError(
                    "[MutationManager] configLoader must be a loaded ConfigLoader "
                    "instance (missing attribute 'program_source')."
                )
            except RuntimeError as exc:
                # Caso o atributo exista, mas indique que não foi carregado
                raise RuntimeError(
                    f"[MutationManager] The provided ConfigLoader has not been loaded yet. "
                    f"Call configLoader.load() before passing it to MutationManager."
                ) from exc

    def _validate_mutate_list(self) -> None:
        """Ensure mutateList is a list."""
        if not isinstance(self.mutateList, list):
            raise TypeError(
                f"[MutationManager] mutateList must be a list, "
                f"got: {type(self.mutateList)}"
            )

    def _validate_source(self, source: str) -> None:
        """Ensure source is a non-empty string."""
        if not isinstance(source, str) or not source.strip():
            raise TypeError(
                f"[MutationManager] source must be a non-empty string, "
                f"got: {type(source)}"
            )

    def _validate_operator(self, operator: object) -> None:
        """Ensure operator satisfies OperatorProtocol."""
        if not isinstance(operator, OperatorProtocol):
            raise TypeError(
                f"[MutationManager] operator must satisfy OperatorProtocol "
                f"(needs 'target_node_type', 'replacement_node', 'operator_id'). "
                f"Got: {type(operator)}"
            )
        if not isinstance(operator.target_node_type, type) or not issubclass(
            operator.target_node_type, ast.AST
        ):
            raise TypeError(
                f"[MutationManager] operator.target_node_type must be a subclass "
                f"of ast.AST, got: {operator.target_node_type!r}"
            )
        if not isinstance(operator.replacement_node, ast.AST):
            raise TypeError(
                f"[MutationManager] operator.replacement_node must be an ast.AST "
                f"instance, got: {type(operator.replacement_node)}"
            )
        if not isinstance(operator.operator_id, str) or not operator.operator_id.strip():
            raise TypeError(
                f"[MutationManager] operator.operator_id must be a non-empty string, "
                f"got: {operator.operator_id!r}"
            )

    def _count_occurrences(self, target_type: type) -> int:
        """Count how many nodes of target_type exist in the current AST."""
        return sum(
            1 for node in ast.walk(self._program_ast)
            if isinstance(node, target_type)
        )

    def _assert_ast_ready(self) -> None:
        """Raise if parseToAST() has not been called yet."""
        if self._program_ast is None:
            raise RuntimeError(
                "[MutationManager] AST is not available. "
                "Call parseToAST(source) before using this method."
            )

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        ast_ready = self._program_ast is not None
        return (
            f"MutationManager("
            f"ast_ready={ast_ready}, "
            f"mutants={len(self.mutateList)}"
            f")"
        )