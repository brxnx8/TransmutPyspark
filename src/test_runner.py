"""
TestRunner
==========
Responsible for:
  1. Receiving a list of mutants (from MutationManager) and a loaded
     ConfigLoader instance.
  2. For each mutant: writing the mutated source to a temporary .py file,
     running the project's pytest suite against it via subprocess, and
     recording the outcome.
  3. Storing every result in ``self.results``.

Execution strategy
------------------
Mutants are tested **in parallel** using ``ThreadPoolExecutor``.  Each
worker spawns an isolated ``pytest`` subprocess, so there is no shared
state between runs.  The degree of parallelism is capped by
``max_workers`` (default: ``min(4, cpu_count)``) to avoid saturating the
machine that also hosts the SparkSession.

Mutant status
-------------
- **killed**   — at least one test failed (pytest exit code 1).
                 This is the desired outcome: the test suite detected the
                 mutation.
- **survived** — all tests passed (pytest exit code 0).
                 The mutation was NOT caught; the test suite may need to
                 be strengthened.
- **error**    — pytest could not run at all (exit codes 2–5, or the
                 subprocess itself crashed).  Typically caused by an
                 import error in the mutated file or an environment issue.

Deliberately out of scope
--------------------------
- Generating mutants              → MutationManager
- Deciding what/how to mutate     → Operator
- Reporting / aggregating results → Reporter (future)
"""

import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from code.config_loader import ConfigLoader
    from code.mutation_manager import Mutant


# ─────────────────────────────────────────────────────────────────────────── #
# TestResult dataclass                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class TestResult:
    """
    Outcome of running the test suite against a single mutant.

    Attributes
    ----------
    mutant_operator  : Name of the operator that generated the mutant.
    occurrence_index : Which occurrence within the operator's registers
                       this mutant corresponds to.
    status           : ``"killed"``, ``"survived"``, or ``"error"``.
    stdout           : Captured standard output from the pytest subprocess.
    stderr           : Captured standard error from the pytest subprocess.
    exit_code        : Raw pytest process exit code.
    duration_seconds : Wall-clock time taken to run the test subprocess.
    """

    mutant_operator:  str
    occurrence_index: int
    status:           str
    stdout:           str
    stderr:           str
    exit_code:        int
    duration_seconds: float

    # Pytest exit-code semantics
    _EXIT_KILLED   = 1   # tests ran, some failed  → mutation killed
    _EXIT_SURVIVED = 0   # tests ran, all passed   → mutation survived

    def is_killed(self) -> bool:
        """Return True if the mutation was detected by the test suite."""
        return self.status == "killed"

    def is_survived(self) -> bool:
        """Return True if the mutation went undetected."""
        return self.status == "survived"

    def is_error(self) -> bool:
        """Return True if pytest could not execute properly."""
        return self.status == "error"

    def __repr__(self) -> str:
        return (
            f"TestResult("
            f"operator={self.mutant_operator!r}, "
            f"occurrence={self.occurrence_index}, "
            f"status={self.status!r}, "
            f"exit_code={self.exit_code}, "
            f"duration={self.duration_seconds:.2f}s"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────── #
# TestRunner                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class TestRunner:
    """
    Runs the project's pytest suite against every mutant in ``mutateList``
    and stores the outcomes in ``results``.

    Parameters
    ----------
    mutateList  : list[Mutant]
        Mutants produced by ``MutationManager.applyMutation()``.
    config      : ConfigLoader
        A fully loaded ``ConfigLoader`` instance used to locate the test
        file, the workspace directory, and the SparkSession.
    max_workers : int
        Maximum number of parallel pytest subprocesses.
        Defaults to ``min(4, os.cpu_count() or 1)``.
    results     : list[TestResult]
        Populated by ``runTest()``.  May be pre-seeded for testing
        purposes but is normally left empty.
    """

    mutateList:  list
    config:      "ConfigLoader"
    max_workers: int = field(default_factory=lambda: min(4, os.cpu_count() or 1))
    results:     list[TestResult] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Post-init validation                                                 #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        self._validate_mutate_list()
        self._validate_config()
        self._validate_max_workers()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def runTest(self) -> list[TestResult]:
        """
        Execute the pytest suite against every mutant in ``mutateList``
        in parallel and store each ``TestResult`` in ``self.results``.

        Steps
        -----
        1. Clear any previous results to avoid stale data.
        2. Submit one task per mutant to a ``ThreadPoolExecutor``.
        3. Each task writes the mutant source to a temp file inside the
           workspace's ``mutants/`` subdirectory, then runs::

               pytest <tests_path> --import-mode=importlib
                      --override-ini="python_files=*.py"
                      -x -q

           with the mutant's directory prepended to ``PYTHONPATH`` so the
           test file imports the mutated module instead of the original.
        4. Collect results as futures complete and append to
           ``self.results``.

        Returns
        -------
        list[TestResult]
            The same list as ``self.results`` (returned for convenience).

        Raises
        ------
        RuntimeError
            If ``mutateList`` is empty (nothing to test).
        """
        if not self.mutateList:
            raise RuntimeError(
                "[TestRunner] mutateList is empty — nothing to test. "
                "Call MutationManager.applyMutation() first."
            )

        self.results.clear()

        print(
            f"[TestRunner] Starting test run: {len(self.mutateList)} mutant(s), "
            f"max_workers={self.max_workers}."
        )

        futures_map = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for mutant in self.mutateList:
                future = executor.submit(self._run_single_mutant, mutant)
                futures_map[future] = mutant

            for future in as_completed(futures_map):
                mutant = futures_map[future]
                try:
                    result = future.result()
                except Exception as exc:           # unexpected worker crash
                    result = TestResult(
                        mutant_operator=mutant.operator_name,
                        occurrence_index=mutant.occurrence_index,
                        status="error",
                        stdout="",
                        stderr=str(exc),
                        exit_code=-1,
                        duration_seconds=0.0,
                    )
                self.results.append(result)

        killed   = sum(1 for r in self.results if r.is_killed())
        survived = sum(1 for r in self.results if r.is_survived())
        errors   = sum(1 for r in self.results if r.is_error())

        print(
            f"[TestRunner] Run complete — "
            f"killed={killed}, survived={survived}, errors={errors}."
        )
        return self.results

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _run_single_mutant(self, mutant: "Mutant") -> TestResult:
        """
        Write one mutant to a temp file and run pytest against it.

        The mutant file is written to a unique subdirectory inside
        ``<workspace>/mutants/`` so concurrent runs never collide.
        The directory is cleaned up automatically after the run.

        Parameters
        ----------
        mutant : Mutant
            The mutant to test.

        Returns
        -------
        TestResult
            The outcome for this mutant.
        """
        mutants_dir = self.config.workspace_path / "mutants"
        mutants_dir.mkdir(parents=True, exist_ok=True)

        # Derive the original program module name from its path
        original_module_name = Path(self.config.programPath).stem

        # Create an isolated temp directory for this mutant
        with tempfile.TemporaryDirectory(dir=mutants_dir) as tmp_dir:
            mutant_file = Path(tmp_dir) / f"{original_module_name}.py"
            mutant_file.write_text(mutant.source_code, encoding="utf-8")

            # Build the environment: inject the mutant's directory first
            # so the test file imports the mutated module, not the original
            env = os.environ.copy()
            python_path_parts = [str(tmp_dir)]
            if existing := env.get("PYTHONPATH"):
                python_path_parts.append(existing)
            env["PYTHONPATH"] = os.pathsep.join(python_path_parts)

            cmd = [
                "pytest",
                str(self.config.testsPath),
                "--import-mode=importlib",
                "-x",     # stop after first failure — faster kill detection
                "-q",     # quiet output
                "--tb=short",
            ]

            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=120,   # safety valve: 2 minutes per mutant
                )
                duration = time.perf_counter() - start
                status   = self._classify(proc.returncode)

                return TestResult(
                    mutant_operator=mutant.operator_name,
                    occurrence_index=mutant.occurrence_index,
                    status=status,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    exit_code=proc.returncode,
                    duration_seconds=duration,
                )

            except subprocess.TimeoutExpired as exc:
                duration = time.perf_counter() - start
                return TestResult(
                    mutant_operator=mutant.operator_name,
                    occurrence_index=mutant.occurrence_index,
                    status="error",
                    stdout=exc.stdout or "",
                    stderr=f"Timeout after {duration:.1f}s",
                    exit_code=-1,
                    duration_seconds=duration,
                )

    @staticmethod
    def _classify(exit_code: int) -> str:
        """
        Map a pytest exit code to a mutant status string.

        Pytest exit codes
        -----------------
        0 — all tests passed            → survived
        1 — some tests failed           → killed
        2 — interrupted (e.g. Ctrl-C)  → error
        3 — internal pytest error       → error
        4 — command-line usage error    → error
        5 — no tests collected          → error
        """
        if exit_code == 0:
            return "survived"
        if exit_code == 1:
            return "killed"
        return "error"

    # ------------------------------------------------------------------ #
    # Validators                                                           #
    # ------------------------------------------------------------------ #

    def _validate_mutate_list(self) -> None:
        if not isinstance(self.mutateList, list):
            raise TypeError(
                f"[TestRunner] mutateList must be a list, "
                f"got: {type(self.mutateList)}"
            )

    def _validate_config(self) -> None:
        required_attrs = (
            "programPath",
            "testsPath",
            "workspace_path",
            "sparkSession",
        )
        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise TypeError(
                    f"[TestRunner] config is missing attribute '{attr}'. "
                    f"Pass a loaded ConfigLoader instance."
                )
        # Trigger _assert_loaded inside ConfigLoader
        try:
            _ = self.config.workspace_path
        except RuntimeError as exc:
            raise RuntimeError(
                "[TestRunner] The provided ConfigLoader has not been loaded "
                "yet. Call config.load() first."
            ) from exc

    def _validate_max_workers(self) -> None:
        if not isinstance(self.max_workers, int) or self.max_workers < 1:
            raise ValueError(
                f"[TestRunner] max_workers must be a positive integer, "
                f"got: {self.max_workers!r}"
            )

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"TestRunner("
            f"mutants={len(self.mutateList)}, "
            f"results={len(self.results)}, "
            f"max_workers={self.max_workers}"
            f")"
        )