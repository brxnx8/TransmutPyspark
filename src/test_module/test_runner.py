"""
TestRunner
==========
Responsible for executing the project's pytest suite against every mutant
in ``mutant_list`` and collecting one ``TestResult`` per mutant.
"""

import os
import shutil
import subprocess
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.model.test_result import TestResult
from src.config.config_loader import ConfigLoader

if TYPE_CHECKING:
    from src.operators.operator import Mutant

logger = logging.getLogger(__name__)


@dataclass
class TestRunner:
    mutant_list: list
    config:      ConfigLoader
    max_workers: int              = field(
        default_factory=lambda: min(4, os.cpu_count() or 1)
    )
    results:     list[TestResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate_mutant_list()
        self._validate_config()
        self._validate_max_workers()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run_test(self) -> list[TestResult]:
        if not self.mutant_list:
            raise RuntimeError(
                "[TestRunner] mutant_list is empty — nothing to test."
            )

        self.results.clear()
        logger.info(
            f"[TestRunner.run_test] Starting: {len(self.mutant_list)} mutant(s), "
            f"max_workers={self.max_workers}."
        )

        # Diretório raiz para todos os sandboxes desta execução
        # Fica dentro do workdir → TransmutPysparkOutput/sandboxes/
        work_dir    = Path(self.config.workspace_dir) / "TransmutPysparkOutput"
        sandbox_root = work_dir / "sandboxes"
        sandbox_root.mkdir(parents=True, exist_ok=True)

        futures_map: dict = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for mutant in self.mutant_list:
                future = executor.submit(
                    self._run_single_mutant, mutant, sandbox_root
                )
                futures_map[future] = mutant

            for future in as_completed(futures_map):
                mutant = futures_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = TestResult(
                        mutant=mutant.id,
                        status="error",
                        failed_tests=[],
                        execution_time=0.0,
                    )
                    logger.error(
                        f"[TestRunner.run_test] Mutant {mutant.id} worker crashed: "
                        f"{exc} — recorded as error."
                    )
                self.results.append(result)

        killed   = sum(1 for r in self.results if r.status == "killed")
        survived = sum(1 for r in self.results if r.status == "survived")
        timeouts = sum(1 for r in self.results if r.status == "timeout")
        errors   = sum(1 for r in self.results if r.status == "error")
        logger.info(
            f"[TestRunner.run_test] Done — killed={killed}, survived={survived}, "
            f"timeout={timeouts}, error={errors}"
        )
        return self.results

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _run_single_mutant(
        self, mutant: "Mutant", sandbox_root: Path
    ) -> TestResult:
        mutant_path   = Path(mutant.mutant_path)
        original_name = Path(mutant.original_path).name   # ex: uts.py

        # Sandbox isolado: TransmutPysparkOutput/sandboxes/sandbox_<id>/
        sandbox_dir = sandbox_root / f"sandbox_{mutant.id}"
        sandbox_dir.mkdir(exist_ok=True)

        target = sandbox_dir / original_name
        target.write_text(
            mutant_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "").strip()
        spark_python_path  = "/opt/bitnami/spark/python"

        path_parts = [str(sandbox_dir)]
        if spark_python_path not in current_pythonpath:
            path_parts.append(spark_python_path)
        if current_pythonpath:
            path_parts.append(current_pythonpath)

        env["PYTHONPATH"] = os.pathsep.join(path_parts)

        cmd = ["pytest", str(self.config.tests_path), "-x", "-q", "--tb=short"]

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

        finally:
            shutil.rmtree(sandbox_dir, ignore_errors=True)

    @staticmethod
    def _classify(exit_code: int) -> str:
        if exit_code == 0:
            return "survived"
        if exit_code == 1:
            return "killed"
        return "error"

    @staticmethod
    def _extract_failed_tests(stdout: str) -> list[str]:
        failed: list[str] = []
        for line in stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("FAILED "):
                failed.append(stripped[len("FAILED "):].strip())
        return failed

    # ------------------------------------------------------------------ #
    # Validators                                                           #
    # ------------------------------------------------------------------ #

    def _validate_mutant_list(self) -> None:
        if not isinstance(self.mutant_list, list):
            raise TypeError(
                f"[TestRunner] mutant_list must be a list, got: {type(self.mutant_list)}"
            )

    def _validate_config(self) -> None:
        if not isinstance(self.config, ConfigLoader):
            raise TypeError(
                f"[TestRunner] config must be a ConfigLoader instance, "
                f"got: {type(self.config)}"
            )
        if not self.config.tests_path or not self.config.tests_path.strip():
            raise ValueError("[TestRunner] config.tests_path must be a non-empty string.")
        if not Path(self.config.tests_path).exists():
            raise FileNotFoundError(
                f"[TestRunner] tests_path not found: {self.config.tests_path}"
            )
        if not self.config.workspace_dir or not self.config.workspace_dir.strip():
            raise ValueError("[TestRunner] config.workspace_dir must be a non-empty string.")

    def _validate_max_workers(self) -> None:
        if not isinstance(self.max_workers, int) or self.max_workers < 1:
            raise ValueError(
                f"[TestRunner] max_workers must be a positive integer, "
                f"got: {self.max_workers!r}"
            )

    def __repr__(self) -> str:
        return (
            f"TestRunner("
            f"mutants={len(self.mutant_list)}, "
            f"results={len(self.results)}, "
            f"max_workers={self.max_workers}"
            f")"
        )