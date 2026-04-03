"""
MutationManager
===============
Orchestrator of the full mutation-testing pipeline.

Responsibilities
----------------
1. ``load``             — build ConfigLoader from config file, read program source.
2. ``parse_to_ast``     — parse source into an AST.
3. ``apply_mutation``   — resolve each operator identifier to its concrete
                          Operator subclass, run analyse_ast + build_mutant
                          and accumulate all generated mutants.
4. ``run_tests``        — hand the mutant list to TestRunner and collect
                          TestResult instances.
5. ``agregate_results`` — hand results to Report for scoring, display and
                          diff generation.

Every method populates its corresponding attribute and returns ``self``,
allowing full method chaining::

    MutationManager("config.txt")
        .load()
        .parse_to_ast()
        .apply_mutation()
        .run_tests()
        .agregate_results()

Error policy
------------
Failures in individual operators (apply_mutation) or individual mutants
(run_tests) are caught, logged and skipped — the pipeline always continues
with the remaining items.

Deliberately out of scope
--------------------------
- Deciding *what* nodes are eligible    → Operator.analyse_ast()
- Deciding *how* to replace a node      → Operator.build_mutant()
- Running the pytest subprocess         → TestRunner
- Computing scores and rendering output → Report
"""

import ast
import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .config_loader import ConfigLoader

if TYPE_CHECKING:
    from .operator import Operator
    from .mutant import Mutant
    from .test_runner import TestRunner
    from .test_result import TestResult
    from .reporter import Reporter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────── #
# Operator registry                                                            #
#                                                                              #
# Maps each operator identifier (as it appears in config.operators_list) to   #
# the dotted import path of its concrete Operator subclass.                   #
# Add new operators here as they are implemented.                              #
# ─────────────────────────────────────────────────────────────────────────── #

_OPERATOR_REGISTRY: dict[str, str] = {
     "NFTP": "src.operator_nftp.OperatorNFTP",
    # "ROR": "code.operators.ror_operator.ROROperator",
    # "LCR": "code.operators.lcr_operator.LCROperator",
}


@dataclass
class MutationManager:
    """
    Orchestrates the full mutation-testing pipeline.

    Parameters
    ----------
    config_path : str
        Path to the plain-text configuration file.  ``load()`` reads it,
        builds the ``ConfigLoader`` instance and populates ``code_original``.

    Attributes
    ----------
    config        : ConfigLoader | None
        Populated by ``load()``.
    code_original : str
        Raw source code of the PySpark application.  Populated by ``load()``.
    code_ast      : ast.AST | None
        Parsed AST of the original program.  Populated by ``parse_to_ast()``.
    mutant_list   : list[Mutant]
        All mutants generated across every operator.  Populated by
        ``apply_mutation()``.
    result_list   : list[TestResult]
        Test outcomes for every mutant.  Populated by ``run_tests()``.
    """

    config_path:   str
    config:        ConfigLoader | None = field(default=None,           init=False)
    code_original: str                 = field(default="",             init=False)
    code_ast:      ast.AST | None      = field(default=None,           init=False)
    mutant_list:   list                = field(default_factory=list,   init=False)
    result_list:   list                = field(default_factory=list,   init=False)

    # ------------------------------------------------------------------ #
    # Post-init validation                                                 #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        if not isinstance(self.config_path, str) or not self.config_path.strip():
            raise ValueError(
                "[MutationManager] config_path must be a non-empty string."
            )

    # ------------------------------------------------------------------ #
    # Pipeline steps                                                       #
    # ------------------------------------------------------------------ #

    def load(self) -> "MutationManager":
        """
        Build the ``ConfigLoader`` instance and read the program source.

        Reads ``config_path`` as a plain-text file where each non-blank,
        non-comment line follows ``key = value``.  Recognised keys:

        - ``program_path``   — path to the PySpark .py file
        - ``tests_path``     — path to the pytest .py file
        - ``operators_list`` — comma-separated operator identifiers

        Returns
        -------
        self — for method chaining.

        Raises
        ------
        FileNotFoundError
            If ``config_path`` or ``program_path`` do not exist.
        ValueError
            If required config keys are missing or program source is empty.
        """
        self._validate_config_path()

        raw    = Path(self.config_path).read_text(encoding="utf-8")
        parsed = self._parse_config_file(raw)

        required = ("program_path", "tests_path", "operators_list")
        missing  = [k for k in required if k not in parsed]
        if missing:
            raise ValueError(
                f"[MutationManager.load] Missing required config keys: "
                f"{missing}"
            )

        operators_list = [
            op.strip().upper()
            for op in parsed["operators_list"].split(",")
            if op.strip()
        ]

        self.config = ConfigLoader(
            program_path=parsed["program_path"].strip(),
            tests_path=parsed["tests_path"].strip(),
            operators_list=operators_list,
        )

        program_file = Path(self.config.program_path)
        if not program_file.exists():
            raise FileNotFoundError(
                f"[MutationManager.load] PySpark program not found: "
                f"{program_file}"
            )

        self.code_original = program_file.read_text(encoding="utf-8")

        if not self.code_original.strip():
            raise ValueError(
                f"[MutationManager.load] Program file is empty: "
                f"{program_file}"
            )

        logger.info(
            "[MutationManager.load] Config loaded — program: %s | "
            "tests: %s | operators: %s",
            self.config.program_path,
            self.config.tests_path,
            self.config.operators_list,
        )
        return self

    def parse_to_ast(self) -> "MutationManager":
        """
        Parse ``code_original`` into an AST and store in ``code_ast``.

        Returns
        -------
        self — for method chaining.

        Raises
        ------
        RuntimeError
            If ``load()`` has not been called yet.
        ValueError
            If ``code_original`` contains a syntax error.
        """
        self._assert_loaded()

        try:
            tree = ast.parse(self.code_original)
        except SyntaxError as exc:
            raise ValueError(
                f"[MutationManager.parse_to_ast] Syntax error at line "
                f"{exc.lineno}: {exc.msg}"
            ) from exc

        ast.fix_missing_locations(tree)
        self.code_ast = tree

        logger.info(
            "[MutationManager.parse_to_ast] AST ready — %d nodes.",
            sum(1 for _ in ast.walk(tree)),
        )
        return self

    def apply_mutation(self) -> "MutationManager":
        """
        For each operator identifier in ``config.operators_list``:

        1. Resolve it to its concrete ``Operator`` subclass via the registry.
        2. Call ``operator.analyse_ast(self.code_ast)`` to find eligible nodes.
        3. Call ``operator.build_mutant(nodes, ...)`` to generate mutants and
           write them to disk.
        4. Extend ``self.mutant_list`` with the returned mutants.

        Failures in individual operators are logged and skipped; the pipeline
        continues with the remaining operators.

        Returns
        -------
        self — for method chaining.

        Raises
        ------
        RuntimeError
            If ``parse_to_ast()`` has not been called yet.
        """
        self._assert_ast_ready()

        mutant_dir = self._ensure_mutant_dir()

        for op_name in self.config.operators_list:
            try:
                operator = self._resolve_operator(op_name)
            except (KeyError, ImportError, AttributeError) as exc:
                logger.warning(
                    "[MutationManager.apply_mutation] Could not load "
                    "operator '%s': %s — skipping.", op_name, exc
                )
                continue

            try:
                nodes = operator.analyse_ast(self.code_ast)

                if not nodes:
                    logger.info(
                        "[MutationManager.apply_mutation] Operator '%s': "
                        "no eligible nodes found — skipping.", op_name
                    )
                    continue

                mutants = operator.build_mutant(
                    nodes=nodes,
                    original_ast=self.code_ast,
                    original_path=self.config.program_path,
                    mutant_dir=str(mutant_dir),
                )
                self.mutant_list.extend(mutants)

                logger.info(
                    "[MutationManager.apply_mutation] Operator '%s': "
                    "%d mutant(s) generated.", op_name, len(mutants)
                )

            except Exception as exc:          # noqa: BLE001
                logger.warning(
                    "[MutationManager.apply_mutation] Operator '%s' "
                    "raised an error: %s — skipping.", op_name, exc
                )

        logger.info(
            "[MutationManager.apply_mutation] Total mutants generated: %d.",
            len(self.mutant_list),
        )
        return self

    def run_tests(self) -> "MutationManager":
        """
        Pass ``mutant_list`` and ``config`` to ``TestRunner`` and collect
        ``TestResult`` instances into ``result_list``.

        Returns
        -------
        self — for method chaining.

        Raises
        ------
        RuntimeError
            If ``apply_mutation()`` has not been called yet (empty
            ``mutant_list``).
        """
        self._assert_mutants_ready()

        from .test_runner import TestRunner

        runner = TestRunner(
            mutant_list=self.mutant_list,
            config=self.config,
        )
        self.result_list = runner.run_test()

        logger.info(
            "[MutationManager.run_tests] %d result(s) collected.",
            len(self.result_list),
        )
        return self

    def agregate_results(self) -> "MutationManager":
        """
        Pass ``result_list``, ``code_original`` and ``mutant_list`` to
        ``Report`` and execute the full reporting sequence:
        ``calculate() → show_results() → make_diff()``.

        Returns
        -------
        self — for method chaining.

        Raises
        ------
        RuntimeError
            If ``run_tests()`` has not been called yet (empty
            ``result_list``).
        """
        self._assert_results_ready()

        from .reporter import Reporter

        reporter = Reporter(
            result_list=self.result_list,
            code_original=self.code_original,
            mutant_list=self.mutant_list,
        )
        reporter.calculate()
        reporter.make_diff()
        reporter.show_results()

        logger.info("[MutationManager.agregate_results] Reporting complete.")
        return self

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _resolve_operator(self, op_name: str) -> "Operator":
        """
        Dynamically import and instantiate the concrete Operator subclass
        registered under ``op_name``.

        Raises
        ------
        KeyError
            If ``op_name`` is not in ``_OPERATOR_REGISTRY``.
        ImportError / AttributeError
            If the module or class cannot be loaded.
        """
        key = op_name.strip().upper()
        if key not in _OPERATOR_REGISTRY:
            raise KeyError(
                f"[MutationManager] Operator '{key}' is not registered. "
                f"Add it to _OPERATOR_REGISTRY before running the pipeline."
            )
        dotted_path          = _OPERATOR_REGISTRY[key]
        module_path, cls_name = dotted_path.rsplit(".", 1)
        module               = importlib.import_module(module_path)
        operator_cls         = getattr(module, cls_name)
        return operator_cls()

    def _ensure_mutant_dir(self) -> Path:
        """Create ``<program_path parent>/mutants/`` and return its Path."""
        mutant_dir = Path(self.config.program_path).parent / "mutants"
        mutant_dir.mkdir(parents=True, exist_ok=True)
        return mutant_dir

    @staticmethod
    def _parse_config_file(raw: str) -> dict[str, str]:
        """
        Parse a ``key = value`` text file into a dict.
        Blank lines and lines starting with ``#`` are ignored.
        """
        result: dict[str, str] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
        return result

    # ------------------------------------------------------------------ #
    # Guards                                                               #
    # ------------------------------------------------------------------ #

    def _validate_config_path(self) -> None:
        path = Path(self.config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"[MutationManager.load] Config file not found: {path}"
            )
        if not path.is_file():
            raise ValueError(
                f"[MutationManager.load] config_path must point to a file: "
                f"{path}"
            )

    def _assert_loaded(self) -> None:
        if self.config is None or not self.code_original:
            raise RuntimeError(
                "[MutationManager] config is not set. Call load() first."
            )

    def _assert_ast_ready(self) -> None:
        if self.code_ast is None:
            raise RuntimeError(
                "[MutationManager] code_ast is not set. "
                "Call parse_to_ast() first."
            )

    def _assert_mutants_ready(self) -> None:
        if not self.mutant_list:
            raise RuntimeError(
                "[MutationManager] mutant_list is empty. "
                "Call apply_mutation() first."
            )

    def _assert_results_ready(self) -> None:
        if not self.result_list:
            raise RuntimeError(
                "[MutationManager] result_list is empty. "
                "Call run_tests() first."
            )

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"MutationManager("
            f"config={'set' if self.config else 'not set'}, "
            f"ast={'set' if self.code_ast else 'not set'}, "
            f"mutants={len(self.mutant_list)}, "
            f"results={len(self.result_list)}"
            f")"
        )