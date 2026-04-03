"""
TestRunner
==========
Responsible for executing the project's pytest suite against every mutant
in ``mutant_list`` and collecting one ``TestResult`` per mutant.

Execution strategy
------------------
Mutants are tested in parallel using ``ThreadPoolExecutor``. Each worker
spawns an isolated ``pytest`` subprocess pointed at ``mutant.mutant_path``,
so there is no shared state between concurrent runs.

The degree of parallelism is capped by ``max_workers``
(default: ``min(4, cpu_count)``) to avoid competing with the SparkSession
that may be active in the same environment.

Mutant status
-------------
- **killed**   — at least one test failed (pytest exit code 1).
- **survived** — all tests passed (pytest exit code 0).
- **timeout**  — pytest did not finish within the time limit.
- **error**    — pytest could not run at all (exit codes 2-5 or subprocess
                 crash).

Deliberately out of scope
--------------------------
- Generating mutants        → Operator.build_mutant()
- Aggregating / reporting   → Report (called by MutationManager)
"""

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .test_result import TestResult
from .config_loader import ConfigLoader

if TYPE_CHECKING:
    from .operator import Mutant


@dataclass
class TestRunner:
    """
    Executes the pytest suite against every mutant and returns the results.

    Parameters
    ----------
    mutant_list : list[Mutant]
        Mutants produced by ``Operator.build_mutant()``, received from
        ``MutationManager``.  Each mutant already has its source written
        to ``mutant.mutant_path``.
    config      : ConfigLoader
        Configuration dataclass providing ``tests_path`` (path to the
        pytest file) and ``program_path`` (used to resolve the workspace).
    max_workers : int
        Maximum number of parallel pytest subprocesses.
        Defaults to ``min(4, os.cpu_count() or 1)``.
    results     : list[TestResult]
        Populated by ``run_test()``.  Normally left empty on construction.
    """

    mutant_list: list
    config:      ConfigLoader
    max_workers: int               = field(
        default_factory=lambda: min(4, os.cpu_count() or 1)
    )
    results:     list[TestResult]  = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Post-init validation                                                 #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        self._validate_mutant_list()
        self._validate_config()
        self._validate_max_workers()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run_test(self) -> list[TestResult]:
        """
        Execute the pytest suite against every mutant in ``mutant_list``
        in parallel and return the collected ``TestResult`` instances.

        For each mutant the runner:
          1. Prepends the directory containing ``mutant.mutant_path`` to
             ``PYTHONPATH`` so the test file imports the mutated module.
          2. Runs ``pytest <tests_path> -x -q --tb=short`` in a subprocess.
          3. Wraps the outcome in a ``TestResult`` and appends it to
             ``self.results``.

        Results are also stored in ``self.results`` so callers can inspect
        them after the call.

        Returns
        -------
        list[TestResult]
            One ``TestResult`` per mutant, in completion order.

        Raises
        ------
        RuntimeError
            If ``mutant_list`` is empty.
        """
        if not self.mutant_list:
            raise RuntimeError(
                "[TestRunner] mutant_list is empty — nothing to test. "
                "Ensure apply_mutation() ran before run_tests()."
            )

        self.results.clear()

        print(
            f"[TestRunner] Starting: {len(self.mutant_list)} mutant(s), "
            f"max_workers={self.max_workers}."
        )

        futures_map: dict = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for mutant in self.mutant_list:
                future = executor.submit(self._run_single_mutant, mutant)
                futures_map[future] = mutant

            for future in as_completed(futures_map):
                mutant = futures_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    # Unexpected worker crash — record as error
                    result = TestResult(
                        mutant=mutant.id,
                        status="error",
                        failed_tests=[],
                        execution_time=0.0,
                    )
                    print(
                        f"[TestRunner] Mutant {mutant.id} worker crashed: "
                        f"{exc} — recorded as error."
                    )
                self.results.append(result)

        killed   = sum(1 for r in self.results if r.status == "killed")
        survived = sum(1 for r in self.results if r.status == "survived")
        timeouts = sum(1 for r in self.results if r.status == "timeout")
        errors   = sum(1 for r in self.results if r.status == "error")

        print(
            f"[TestRunner] Done — killed={killed}, survived={survived}, "
            f"timeout={timeouts}, error={errors}."
        )
        return self.results

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _run_single_mutant(self, mutant: "Mutant") -> TestResult:
        """
        Run pytest against a single mutant file and return a ``TestResult``.

        The mutant file already exists at ``mutant.mutant_path`` (written
        by ``Operator.build_mutant``).  Its parent directory is prepended
        to ``PYTHONPATH`` so that ``import <module>`` inside the test file
        resolves to the mutated version.

        Parameters
        ----------
        mutant : Mutant
            The mutant to test.

        Returns
        -------
        TestResult
            Outcome for this mutant.
        """
        mutant_path = Path(mutant.mutant_path)
        mutant_dir  = str(mutant_path.parent)

        # Inject mutant directory at the front of PYTHONPATH
        env = os.environ.copy()
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            os.pathsep.join([mutant_dir, existing_path])
            if existing_path
            else mutant_dir
        )

        cmd = [
            "pytest",
            str(self.config.tests_path),
            "-x",
            "-q",
            "--tb=short",
            "--import-mode=importlib",
        ]

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )

            duration     = time.perf_counter() - start
            status       = self._classify(proc.returncode)
            failed_tests = self._extract_failed_tests(proc.stdout)

            return TestResult(
                mutant=mutant.id,
                status=status,
                failed_tests=failed_tests,
                execution_time=round(duration, 4),
            )

        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start
            return TestResult(
                mutant=mutant.id,
                status="timeout",
                failed_tests=[],
                execution_time=round(duration, 4),
            )

    @staticmethod
    def _classify(exit_code: int) -> str:
        """
        Map a pytest exit code to a mutant status string.

        Codes
        -----
        0 → survived  (all tests passed)
        1 → killed    (at least one test failed)
        2 → error     (interrupted)
        3 → error     (internal pytest error)
        4 → error     (command-line usage error)
        5 → error     (no tests collected)
        """
        if exit_code == 0:
            return "survived"
        if exit_code == 1:
            return "killed"
        return "error"

    @staticmethod
    def _extract_failed_tests(stdout: str) -> list[str]:
        """
        Parse pytest's ``-q`` output and return a list of failed test ids.

        Pytest prints each failure as ``FAILED path::test_name`` on its
        own line — this method collects those names.

        Parameters
        ----------
        stdout : str
            Captured standard output from the pytest subprocess.

        Returns
        -------
        list[str]
            Failed test identifiers, empty if none found.
        """
        failed: list[str] = []
        for line in stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("FAILED "):
                test_id = stripped[len("FAILED "):].strip()
                failed.append(test_id)
        return failed

    # ------------------------------------------------------------------ #
    # Validators                                                           #
    # ------------------------------------------------------------------ #

    def _validate_mutant_list(self) -> None:
        if not isinstance(self.mutant_list, list):
            raise TypeError(
                f"[TestRunner] mutant_list must be a list, "
                f"got: {type(self.mutant_list)}"
            )

    def _validate_config(self) -> None:
        if not isinstance(self.config, ConfigLoader):
            raise TypeError(
                f"[TestRunner] config must be a ConfigLoader instance, "
                f"got: {type(self.config)}"
            )
        if not self.config.tests_path or not self.config.tests_path.strip():
            raise ValueError(
                "[TestRunner] config.tests_path must be a non-empty string."
            )
        if not Path(self.config.tests_path).exists():
            raise FileNotFoundError(
                f"[TestRunner] tests_path not found: {self.config.tests_path}"
            )

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
            f"mutants={len(self.mutant_list)}, "
            f"results={len(self.results)}, "
            f"max_workers={self.max_workers}"
            f")"
        )