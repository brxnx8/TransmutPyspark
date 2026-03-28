import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession


@dataclass
class ConfigLoader:
    """
    Loads and validates all configurations required for the PySpark mutation testing tool.

    Responsibilities:
      - Validate and read the source files (program + tests) as raw text.
      - Set up the workspace directory structure.
      - Validate the operators list and the SparkSession.

    Explicitly out of scope:
      - Parsing source code into AST/ASN — that is the responsibility of MutationManager.

    Attributes:
        programPath   : Path to the .py file containing the PySpark application.
        testsPath     : Path to the .py file containing the pytest unit tests.
        workspaceDir  : Working directory where mutants and results will be stored.
        sparkSession  : An active SparkSession instance used when running the test suite.
        operatorsList : List of mutation operator identifiers to be applied
                        (e.g. ['AOR', 'ROR', 'LCR']).
    """

    programPath: str
    testsPath: str
    workspaceDir: str
    sparkSession: SparkSession
    operatorsList: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Internal state (populated by load())                                 #
    # ------------------------------------------------------------------ #
    _program_source: str = field(init=False, default="", repr=False)
    _tests_source: str = field(init=False, default="", repr=False)
    _workspace_path: Optional[Path] = field(init=False, default=None, repr=False)
    _loaded: bool = field(init=False, default=False, repr=False)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(self) -> "ConfigLoader":
        """
        Validates all parameters and loads every resource.

        Steps performed:
          1. Validate and read the PySpark program file as raw source.
          2. Validate and read the pytest test file as raw source.
          3. Ensure the workspace directory exists (creates it if necessary).
          4. Validate the operatorsList (must be non-empty list of strings).
          5. Validate the SparkSession.

        Returns:
            self – allows method chaining: ``cfg = ConfigLoader(...).load()``

        Raises:
            FileNotFoundError : If programPath or testsPath do not exist.
            ValueError        : If any parameter fails validation.
            TypeError         : If parameter types are wrong.
        """
        self._validate_program_path()
        self._validate_tests_path()
        self._setup_workspace()
        self._validate_operators_list()
        self._validate_spark_session()

        self._loaded = True
        print("[ConfigLoader] All configurations loaded successfully.")
        return self

    # ------------------------------------------------------------------ #
    # Properties (available after load())                                  #
    # ------------------------------------------------------------------ #

    @property
    def program_source(self) -> str:
        """Raw source code of the PySpark application."""
        self._assert_loaded()
        return self._program_source

    @property
    def tests_source(self) -> str:
        """Raw source code of the pytest test file."""
        self._assert_loaded()
        return self._tests_source

    @property
    def workspace_path(self) -> Path:
        """Resolved Path object for the workspace directory."""
        self._assert_loaded()
        return self._workspace_path

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _validate_program_path(self) -> None:
        """Validate and read the PySpark program file as raw source text."""
        if not isinstance(self.programPath, str) or not self.programPath.strip():
            raise TypeError(
                f"programPath must be a non-empty string, got: {type(self.programPath)}"
            )

        path = Path(self.programPath).resolve()

        if not path.exists():
            raise FileNotFoundError(f"PySpark program not found: {path}")

        if not path.is_file():
            raise ValueError(
                f"programPath must point to a file, not a directory: {path}"
            )

        if path.suffix.lower() != ".py":
            raise ValueError(
                f"programPath must point to a .py file, "
                f"got extension '{path.suffix}': {path}"
            )

        self._program_source = path.read_text(encoding="utf-8")
        print(f"[ConfigLoader] Program source loaded: {path}")

    def _validate_tests_path(self) -> None:
        """Validate and read the pytest test file as raw source text."""
        if not isinstance(self.testsPath, str) or not self.testsPath.strip():
            raise TypeError(
                f"testsPath must be a non-empty string, got: {type(self.testsPath)}"
            )

        path = Path(self.testsPath).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Tests file not found: {path}")

        if not path.is_file():
            raise ValueError(
                f"testsPath must point to a file, not a directory: {path}"
            )

        if path.suffix.lower() != ".py":
            raise ValueError(
                f"testsPath must point to a .py file, "
                f"got extension '{path.suffix}': {path}"
            )

        source = path.read_text(encoding="utf-8")

        # Lightweight check via regex — no AST parsing here
        if not re.search(r"^\s*def\s+test_", source, re.MULTILINE):
            raise ValueError(
                f"No pytest test functions (test_*) found in tests file: {path}"
            )

        self._tests_source = source
        print(f"[ConfigLoader] Tests source loaded: {path}")

    def _setup_workspace(self) -> None:
        """Resolve and create the workspace directory if it does not exist."""
        if not isinstance(self.workspaceDir, str) or not self.workspaceDir.strip():
            raise TypeError(
                f"workspaceDir must be a non-empty string, got: {type(self.workspaceDir)}"
            )

        workspace = Path(self.workspaceDir).resolve()

        if workspace.exists() and not workspace.is_dir():
            raise ValueError(
                f"workspaceDir path exists but is not a directory: {workspace}"
            )

        workspace.mkdir(parents=True, exist_ok=True)

        # Standard subdirectories consumed by the rest of the tool
        (workspace / "mutants").mkdir(exist_ok=True)
        (workspace / "results").mkdir(exist_ok=True)
        (workspace / "logs").mkdir(exist_ok=True)

        self._workspace_path = workspace
        print(f"[ConfigLoader] Workspace ready: {workspace}")

    def _validate_operators_list(self) -> None:
        """Validate the list of mutation operators."""
        if not isinstance(self.operatorsList, list):
            raise TypeError(
                f"operatorsList must be a list, got: {type(self.operatorsList)}"
            )

        if not self.operatorsList:
            raise ValueError(
                "operatorsList must contain at least one mutation operator."
            )

        invalid = [
            op for op in self.operatorsList
            if not isinstance(op, str) or not op.strip()
        ]
        if invalid:
            raise ValueError(
                f"All items in operatorsList must be non-empty strings. "
                f"Invalid entries: {invalid}"
            )

        self.operatorsList = [op.strip().upper() for op in self.operatorsList]
        print(f"[ConfigLoader] Operators: {self.operatorsList}")

    def _validate_spark_session(self) -> None:
        """Ensure the SparkSession is active."""
        if not isinstance(self.sparkSession, SparkSession):
            raise TypeError(
                f"sparkSession must be a pyspark.sql.SparkSession instance, "
                f"got: {type(self.sparkSession)}"
            )

        if self.sparkSession.sparkContext._jsc is None:
            raise ValueError("The provided SparkSession appears to have been stopped.")

        print(
            f"[ConfigLoader] SparkSession validated "
            f"(app: '{self.sparkSession.sparkContext.appName}')."
        )

    def _assert_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "ConfigLoader has not been initialised yet. Call .load() first."
            )

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"ConfigLoader("
            f"programPath={self.programPath!r}, "
            f"testsPath={self.testsPath!r}, "
            f"workspaceDir={self.workspaceDir!r}, "
            f"operatorsList={self.operatorsList!r}, "
            f"loaded={self._loaded}"
            f")"
        )