"""
Operator
========
Abstract base class that defines the interface every mutation operator
must implement.

Design decisions
----------------
- Uses ``ABC`` + ``@abstractmethod`` so that instantiating a subclass that
  has not implemented ``analyse_ast`` or ``build_mutant`` raises a clear
  ``TypeError`` immediately.
- Attributes (``id``, ``name``, ``mutant_registers``, ``mutant_list``) are
  shared by every concrete operator and validated in ``__post_init__``.
- ``analyse_ast``  → receives the AST tree directly as a parameter,
                     analyses it and returns the eligible nodes for mutation.
- ``build_mutant`` → receives the nodes found by ``analyse_ast``, applies
                     substitutions on the original tree, generates the mutant
                     source files, wraps each result in a ``Mutant`` instance,
                     stores them in ``mutant_list`` and returns the full list.

Relationship with MutationManager
----------------------------------
::

    manager.parseToAST(source)                         # builds the AST
    nodes = operator.analyse_ast(manager.program_ast)  # finds eligible nodes
    mutants = operator.build_mutant(nodes)             # generates mutants
    # mutants are also stored in operator.mutant_list

Deliberately out of scope
--------------------------
- Parsing source code into AST  → MutationManager.parseToAST()
- Running tests against mutants → TestRunner
- Reporting results             → Reporter (future)
"""

import ast
import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from src.model.mutant import Mutant

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────── #
# Abstract Operator                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class Operator(ABC):
    """
    Abstract base class for all mutation operators.

    Attributes
    ----------
    id               : int
        Unique numeric identifier for this operator instance.
    name             : str
        Human-readable name for the operator (e.g. ``"AOR"``, ``"ROR"``).
        Normalised to uppercase on construction.
    mutant_registers : str | list[str]
        Metadata describing where/how this operator applies — e.g. a node
        type name (``"Add"``) or a list of them (``["Lt", "Gt", "LtE"]``).
        Used by concrete subclasses to guide ``analyse_ast``.
    mutant_list      : list[Mutant]
        Accumulator populated by ``build_mutant``.  Empty on construction;
        grows with each call to ``build_mutant``.
    """

    id:               int
    name:             str
    mutant_registers: str | list[str]
    mutant_list:      list[Mutant] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Post-init validation                                                 #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        self._validate_id()
        self._validate_name()
        self._validate_mutant_registers()
        self._validate_mutant_list()

    # ------------------------------------------------------------------ #
    # Abstract interface — subclasses MUST implement both methods          #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        """
        Analyse ``tree`` and return every node eligible for mutation.

        Implementations must:
          1. Walk ``tree`` looking for nodes that match this operator's
             criteria (guided by ``self.mutant_registers``).
          2. Return the list of matching nodes — these are passed directly
             to ``build_mutant``.

        Parameters
        ----------
        tree : ast.AST
            The parsed AST of the PySpark program, obtained from
            ``MutationManager.program_ast``.

        Returns
        -------
        list[ast.AST]
            Nodes eligible for mutation.  Empty list if none found.

        Raises
        ------
        TypeError
            If ``tree`` is not an ``ast.AST`` instance.
        """
        ...

    @abstractmethod
    def build_mutant(self,
                     nodes: list[ast.AST],
                     original_ast: ast.AST,
                     original_path: str,
                     mutant_dir: str) -> list[Mutant]:
        """
        Generate one ``Mutant`` per node, write each to disk and populate
        ``self.mutant_list``.

        For each node in ``nodes``, implementations must:
          1. Deep-copy ``original_ast``.
          2. Replace the target node with the appropriate substitute.
          3. Unparse the modified tree back to source code.
          4. Write the mutated source to a ``.py`` file inside
             ``mutant_dir``.
          5. Record the modified line.
          6. Create a ``Mutant`` instance and append it to
             ``self.mutant_list``.

        At the end return ``self.mutant_list``.

        Parameters
        ----------
        nodes : list[ast.AST]
            Eligible nodes returned by ``analyse_ast``.
        original_ast : ast.AST
            The unmodified program AST (must not be mutated in place).
        original_path : str
            Absolute path to the original PySpark source file — stored in
            each ``Mutant.original_path``.
        mutant_dir : str
            Directory where mutant ``.py`` files will be written.

        Returns
        -------
        list[Mutant]
            The full ``self.mutant_list`` after appending the new mutants.

        Raises
        ------
        TypeError
            If ``nodes`` is not a list or contains non-``ast.AST`` items.
        ValueError
            If ``original_path`` or ``mutant_dir`` is not a non-empty string.
        """
        ...

    # ------------------------------------------------------------------ #
    # Concrete helpers available to all subclasses                         #
    # ------------------------------------------------------------------ #

    def clear_mutant_list(self) -> None:
        """
        Empty ``mutant_list`` without touching any other attribute.

        Useful when reusing the same operator instance across multiple
        mutation rounds.
        """
        self.mutant_list.clear()
        logger.info(f"[Operator:{self.name}] mutant_list cleared.")

    # ------------------------------------------------------------------ #
    # Logging helpers — centralized log output for all operators           #
    # ------------------------------------------------------------------ #

    def _log_analyse_ast_found(self, count: int, description: str) -> None:
        """
        Log the number of eligible nodes found by ``analyse_ast``.

        Parameters
        ----------
        count : int
            Number of eligible nodes found.
        description : str
            Human-readable description of what was searched for.
            Example: "mapping transformation calls with function arguments"
        """
        logger.info(
            f"[{self.__class__.__name__}.analyse_ast] Found {count} eligible "
            f"call site(s) ({description})."
        )

    def _log_skipping_node(self, reason: str) -> None:
        """
        Log that a node or sub-node is being skipped due to validation failure.

        Parameters
        ----------
        reason : str
            Human-readable reason why this node is being skipped.
            Example: "Call at line 42: no eligible sub-conditions found"
        """
        logger.warning(f"[{self.__class__.__name__}.build_mutant] {reason} — skipping.")

    def _log_mutant_created(self, mutant_id: int, details: str) -> None:
        """
        Log that a mutant was successfully created.

        Parameters
        ----------
        mutant_id : int
            The numeric ID of the mutant.
        details : str
            Extra details about the mutation.
            Example: "call line 9 (.withColumn), replacement 'zero': ...[subdir/file.py]"
        """
        logger.info(
            f"[{self.__class__.__name__}.build_mutant] Mutant {mutant_id} created "
            f"— {details}"
        )

    def _log_build_mutant_done(self) -> None:
        """Log the completion of ``build_mutant`` with the final mutant count."""
        logger.info(
            f"[{self.__class__.__name__}.build_mutant] Done — "
            f"{len(self.mutant_list)} total mutant(s) generated."
        )

    # ------------------------------------------------------------------ #
    # Protected guards — call these inside subclass implementations        #
    # ------------------------------------------------------------------ #

    def _assert_valid_tree(self, tree: ast.AST) -> None:
        """Raise ``TypeError`` if ``tree`` is not an ``ast.AST`` instance."""
        if not isinstance(tree, ast.AST):
            raise TypeError(
                f"[Operator:{self.name}] tree must be an ast.AST instance, "
                f"got: {type(tree)}"
            )

    def _assert_valid_nodes(self, nodes: list[ast.AST]) -> None:
        """Raise ``TypeError`` if ``nodes`` is not a list of AST nodes."""
        if not isinstance(nodes, list):
            raise TypeError(
                f"[Operator:{self.name}] nodes must be a list, "
                f"got: {type(nodes)}"
            )
        invalid = [n for n in nodes if not isinstance(n, ast.AST)]
        if invalid:
            raise TypeError(
                f"[Operator:{self.name}] All items in nodes must be "
                f"ast.AST instances. Invalid entries: {invalid}"
            )

    def _assert_valid_path(self, path: str, param_name: str) -> None:
        """Raise ``ValueError`` if ``path`` is not a non-empty string."""
        if not isinstance(path, str) or not path.strip():
            raise ValueError(
                f"[Operator:{self.name}] {param_name} must be a non-empty "
                f"string, got: {path!r}"
            )

    def _next_mutant_id(self) -> int:
        """Return the next available mutant id (1-based, sequential)."""
        return len(self.mutant_list) + 1

    # ------------------------------------------------------------------ #
    # Private validators                                                   #
    # ------------------------------------------------------------------ #

    def _validate_id(self) -> None:
        if not isinstance(self.id, int) or self.id < 0:
            raise TypeError(
                f"[Operator] id must be a non-negative integer, "
                f"got: {self.id!r}"
            )

    def _validate_name(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError(
                f"[Operator] name must be a non-empty string, "
                f"got: {self.name!r}"
            )
        self.name = self.name.strip().upper()

    def _validate_mutant_registers(self) -> None:
        if isinstance(self.mutant_registers, str):
            if not self.mutant_registers.strip():
                raise ValueError(
                    f"[Operator:{self.name}] mutant_registers must not be "
                    f"an empty string."
                )
        elif isinstance(self.mutant_registers, list):
            if not self.mutant_registers:
                raise ValueError(
                    f"[Operator:{self.name}] mutant_registers list must not "
                    f"be empty."
                )
            invalid = [
                r for r in self.mutant_registers
                if not isinstance(r, str) or not r.strip()
            ]
            if invalid:
                raise ValueError(
                    f"[Operator:{self.name}] All items in mutant_registers "
                    f"must be non-empty strings. Invalid: {invalid}"
                )
        else:
            raise TypeError(
                f"[Operator:{self.name}] mutant_registers must be a str or "
                f"list[str], got: {type(self.mutant_registers)}"
            )

    def _validate_mutant_list(self) -> None:
        if not isinstance(self.mutant_list, list):
            raise TypeError(
                f"[Operator:{self.name}] mutant_list must be a list, "
                f"got: {type(self.mutant_list)}"
            )
        invalid = [m for m in self.mutant_list if not isinstance(m, Mutant)]
        if invalid:
            raise TypeError(
                f"[Operator:{self.name}] All items in mutant_list must be "
                f"Mutant instances. Invalid entries: {invalid}"
            )

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