"""
Operator
========
Abstract base class that defines the interface every mutation operator
must implement.

Design decisions
----------------
- Uses ``ABC`` + ``@abstractmethod`` so that instantiating a subclass that
  has not implemented ``analyseAST`` or ``buildMutate`` raises a clear
  ``TypeError`` immediately.
- Attributes (``name``, ``registers``, ``codeAST``) are shared by every
  concrete operator and are validated in ``__post_init__``.
- ``analyseAST``  → inspects the AST and records which nodes are eligible
                    for mutation, storing them in ``registers``.
- ``buildMutate`` → constructs and returns the replacement AST node for a
                    given target node.  The caller (MutationManager) is
                    responsible for applying the node to the full tree.

Relationship with MutationManager
----------------------------------
::

    manager.parseToAST(source)          # builds the AST
    operator.codeAST = manager.program_ast
    operator.analyseAST()               # populates operator.registers
    for node in operator.registers:
        replacement = operator.buildMutate(node)
        # MutationManager uses replacement to generate each mutant

Deliberately out of scope
--------------------------
- Applying the mutant to the full tree → MutationManager.applyMutation()
- Running tests against mutants        → TestRunner (future)
"""

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Operator(ABC):
    """
    Abstract base class for all mutation operators.

    Attributes
    ----------
    name : str
        Human-readable identifier for the operator (e.g. ``"AOR"``,
        ``"ROR"``, ``"LCR"``).  Must be a non-empty string.
    registers : list[ast.AST]
        Metadata store populated by ``analyseAST()``.  Each entry is an
        AST node that this operator is eligible to mutate.  Starts empty
        and is repopulated every time ``analyseAST()`` is called.
    codeAST : ast.AST | None
        The parsed AST of the program under mutation.  Provided by
        MutationManager after ``parseToAST()`` is called.  Must be set
        before calling ``analyseAST()``.
    """

    name: str
    registers: list[ast.AST] = field(default_factory=list)
    codeAST: ast.AST | None = field(default=None)

    # ------------------------------------------------------------------ #
    # Post-init validation                                                 #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        self._validate_name()
        self._validate_registers()
        self._validate_code_ast()

    # ------------------------------------------------------------------ #
    # Abstract interface — subclasses MUST implement these                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def analyseAST(self) -> list[ast.AST]:
        """
        Inspect ``self.codeAST`` and identify every node that this operator
        can mutate.

        Implementations must:
          1. Walk ``self.codeAST`` looking for eligible nodes.
          2. Store the found nodes in ``self.registers`` (replacing any
             previous content).
          3. Return ``self.registers`` for convenience.

        Returns
        -------
        list[ast.AST]
            The nodes eligible for mutation (same object as
            ``self.registers``).

        Raises
        ------
        RuntimeError
            If ``self.codeAST`` has not been set yet.
        """
        ...

    @abstractmethod
    def buildMutate(self, target_node: ast.AST) -> ast.AST:
        """
        Construct and return the replacement node for ``target_node``.

        The returned node is handed to MutationManager, which deep-copies
        it and splices it into the full AST to produce a mutant.

        Parameters
        ----------
        target_node : ast.AST
            The original AST node to be replaced.  Implementations may
            inspect it to decide the exact replacement (e.g. for ROR an
            ``ast.Lt`` becomes ``ast.Gt``).

        Returns
        -------
        ast.AST
            A *new* AST node that will replace ``target_node`` in the
            mutated program.

        Raises
        ------
        TypeError
            If ``target_node`` is not an ``ast.AST`` instance.
        ValueError
            If ``target_node`` is not a node type this operator handles.
        """
        ...

    # ------------------------------------------------------------------ #
    # Concrete helpers available to all subclasses                         #
    # ------------------------------------------------------------------ #

    def set_code_ast(self, code_ast: ast.AST) -> None:
        """
        Set (or replace) the AST to be analysed.

        Clears ``registers`` automatically so stale data from a previous
        analysis is never mixed with results from the new tree.

        Parameters
        ----------
        code_ast : ast.AST
            A parsed AST, typically obtained from
            ``MutationManager.program_ast``.

        Raises
        ------
        TypeError
            If ``code_ast`` is not an ``ast.AST`` instance.
        """
        if not isinstance(code_ast, ast.AST):
            raise TypeError(
                f"[Operator:{self.name}] codeAST must be an ast.AST instance, "
                f"got: {type(code_ast)}"
            )
        self.codeAST = code_ast
        self.registers.clear()
        print(f"[Operator:{self.name}] codeAST updated — registers cleared.")

    def clear_registers(self) -> None:
        """
        Empty ``registers`` without replacing ``codeAST``.

        Useful when the same operator instance is reused across multiple
        mutation rounds.
        """
        self.registers.clear()
        print(f"[Operator:{self.name}] Registers cleared.")

    # ------------------------------------------------------------------ #
    # Protected helpers for use inside subclass implementations            #
    # ------------------------------------------------------------------ #

    def _assert_code_ast_ready(self) -> None:
        """
        Raise ``RuntimeError`` if ``codeAST`` has not been set.

        Subclasses should call this at the top of ``analyseAST()``.
        """
        if self.codeAST is None:
            raise RuntimeError(
                f"[Operator:{self.name}] codeAST is not set. "
                f"Call set_code_ast() or assign codeAST before analyseAST()."
            )

    def _assert_node_in_registers(self, node: ast.AST) -> None:
        """
        Raise ``ValueError`` if ``node`` was not registered by ``analyseAST()``.

        Subclasses may call this inside ``buildMutate()`` to guard against
        nodes that this operator does not handle.
        """
        if node not in self.registers:
            raise ValueError(
                f"[Operator:{self.name}] The provided node is not in registers. "
                f"Ensure analyseAST() has been called and the node is eligible."
            )

    # ------------------------------------------------------------------ #
    # Private validators                                                   #
    # ------------------------------------------------------------------ #

    def _validate_name(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError(
                f"[Operator] name must be a non-empty string, got: {self.name!r}"
            )
        self.name = self.name.strip().upper()

    def _validate_registers(self) -> None:
        if not isinstance(self.registers, list):
            raise TypeError(
                f"[Operator:{self.name}] registers must be a list, "
                f"got: {type(self.registers)}"
            )
        invalid = [r for r in self.registers if not isinstance(r, ast.AST)]
        if invalid:
            raise TypeError(
                f"[Operator:{self.name}] All items in registers must be ast.AST "
                f"instances. Invalid entries: {invalid}"
            )

    def _validate_code_ast(self) -> None:
        if self.codeAST is not None and not isinstance(self.codeAST, ast.AST):
            raise TypeError(
                f"[Operator:{self.name}] codeAST must be an ast.AST instance or None, "
                f"got: {type(self.codeAST)}"
            )

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"registers={len(self.registers)} node(s), "
            f"codeAST={'set' if self.codeAST is not None else 'not set'}"
            f")"
        )