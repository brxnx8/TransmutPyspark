import ast
import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import copy

from src.model.mutant import Mutant
from src.model.mutant_id_manager import MutantIDManager

logger = logging.getLogger(__name__)


@dataclass
class Operator(ABC):

    id:               int
    name:             str
    mutant_registers: str | list[str]
    mutant_list:      list[Mutant] = field(default_factory=list)


    _id_manager = MutantIDManager()

    def __post_init__(self) -> None:
        self._validate_id()
        self._validate_name()
        self._validate_mutant_registers()
        self._validate_mutant_list()

    @classmethod
    def create(cls) -> "Operator":
        return cls(
            id=cls._DEFAULT_ID,
            name=cls._DEFAULT_NAME,
            mutant_registers=cls._DEFAULT_REGISTERS,
        )


    @abstractmethod
    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        pass


    @abstractmethod
    def build_mutant(self,
                     nodes: list[ast.AST],
                     original_ast: ast.AST,
                     original_path: str,
                     mutant_dir: str) -> list[Mutant]:
        pass


    def clear_mutant_list(self) -> None:
        self.mutant_list.clear()
        logger.info(f"[Operator:{self.name}] mutant_list cleared.")


    def _replace_node(
        self,
        original_ast: ast.AST,
        target: ast.AST,
        replacement: ast.AST,
    ) -> ast.AST:
        target_type    = type(target)
        target_lineno  = getattr(target, "lineno",     None)
        target_col     = getattr(target, "col_offset", None)
        target_source  = ast.unparse(target)

        _replacement = copy.deepcopy(replacement)

        class _Replacer(ast.NodeTransformer):
            def generic_visit(self, node: ast.AST) -> ast.AST:
                if (
                    type(node) is target_type
                    and getattr(node, "lineno",     None) == target_lineno
                    and getattr(node, "col_offset", None) == target_col
                    and ast.unparse(node) == target_source
                ):
                    return _replacement
                return super().generic_visit(node)

        mutated = _Replacer().visit(copy.deepcopy(original_ast))
        ast.fix_missing_locations(mutated)
        return mutated

    def _write_mutant_file(
        self,
        mutated_ast: ast.AST,
        mutant_dir: str,
        filename: str,
    ) -> str:
        source = ast.unparse(mutated_ast)
        path = Path(mutant_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return str(path.resolve())

    def _log_analyse_ast_found(self, count: int, description: str) -> None:
        logger.info(
            f"[{self.__class__.__name__}.analyse_ast] Found {count} eligible "
            f"call site(s) ({description})."
        )

    def _log_skipping_node(self, reason: str) -> None:
        logger.warning(f"[{self.__class__.__name__}.build_mutant] {reason} — skipping.")

    def _log_mutant_created(self, mutant_id: int, details: str) -> None:
        logger.info(
            f"[{self.__class__.__name__}.build_mutant] Mutant {mutant_id} created "
            f"— {details}"
        )

    def _log_build_mutant_done(self) -> None:
        logger.info(
            f"[{self.__class__.__name__}.build_mutant] Done — "
            f"{len(self.mutant_list)} total mutant(s) generated."
        )

    def _assert_valid_tree(self, tree: ast.AST) -> None:
        if not isinstance(tree, ast.AST):
            raise TypeError(
                f"[Operator:{self.name}] tree must be an ast.AST instance, "
                f"got: {type(tree)}"
            )

    def _assert_valid_nodes(self, nodes: list[ast.AST]) -> None:
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
        if not isinstance(path, str) or not path.strip():
            raise ValueError(
                f"[Operator:{self.name}] {param_name} must be a non-empty "
                f"string, got: {path!r}"
            )

    def _next_mutant_id(self) -> int:
        return self._id_manager.next_id()

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


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.id}, "
            f"name={self.name!r}, "
            f"mutant_registers={self.mutant_registers!r}, "
            f"mutants={len(self.mutant_list)}"
            f")"
        )