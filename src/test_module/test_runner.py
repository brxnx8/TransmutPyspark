import os
import shutil
import subprocess
import sys
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from src.model.test_result import TestResult

logger = logging.getLogger(__name__)


@dataclass
class TestRunner:
    mutant_list: list
    config:      object
    max_workers: int = field(
        default_factory=lambda: min(4, os.cpu_count() or 1)
    )
    results: list[TestResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.mutant_list, list):
            raise TypeError(f"mutant_list deve ser lista, recebeu: {type(self.mutant_list)}")
        if self.max_workers < 1:
            raise ValueError(f"max_workers deve ser >= 1, recebeu: {self.max_workers}")

    def run_test(self) -> list[TestResult]:
        if not self.mutant_list:
            raise RuntimeError("[TestRunner] mutant_list vazia — nada para testar.")

        self.results.clear()
        logger.info(
            f"[TestRunner] Iniciando: {len(self.mutant_list)} mutante(s), "
            f"workers={self.max_workers}."
        )

        sandbox_root = self.config.workspace_dir / "TransmutPysparkOutput" / "sandboxes"
        sandbox_root.mkdir(parents=True, exist_ok=True)

        worker_sandboxes: dict[int, Path] = {}
        for i in range(self.max_workers):
            sb = sandbox_root / f"worker_{i}_sandbox"
            sb.mkdir(exist_ok=True)
            worker_sandboxes[i] = sb

        worker_index = 0
        lock = threading.Lock()

        futures_map: dict = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for mutant in self.mutant_list:
                with lock:
                    idx = worker_index
                    worker_index = (worker_index + 1) % self.max_workers
                sandbox = worker_sandboxes[idx]
                future = executor.submit(self._run_single_mutant, mutant, sandbox)
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
                    logger.error(f"[TestRunner] Mutant {mutant.id} crash: {exc}")
                self.results.append(result)

        for sb in worker_sandboxes.values():
            shutil.rmtree(sb, ignore_errors=True)

        killed   = sum(1 for r in self.results if r.status == "killed")
        survived = sum(1 for r in self.results if r.status == "survived")
        timeouts = sum(1 for r in self.results if r.status == "timeout")
        errors   = sum(1 for r in self.results if r.status == "error")

        exercised = killed + survived
        mutation_score = (killed / exercised * 100) if exercised > 0 else 0.0

        logger.info(
            f"[TestRunner] Concluído — "
            f"killed={killed}, survived={survived}, "
            f"timeout={timeouts}, error={errors} | "
            f"mutation score={mutation_score:.1f}% "
            f"({killed}/{exercised} mutantes exercitados)"
        )
        return self.results

    def _run_single_mutant(self, mutant, sandbox: Path) -> TestResult:
        mutant_path   = Path(mutant.mutant_path)
        original_name = Path(mutant.original_path).name

        target_file = sandbox / original_name
        target_file.write_text(
            mutant_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

        env = self._build_env(
            sandbox,
            original_source_dir=Path(mutant.original_path).parent,
        )

        if not mutant.test_files:
            logger.debug(f"Mutant {mutant.id}: Nenhum teste mapeado — survived")
            return TestResult(
                mutant=mutant.id,
                status="survived",
                failed_tests=[],
                execution_time=0.0,
            )

        test_paths = [str(tf) for tf in mutant.test_files]

        project_root = self.config.workspace_dir
        cmd = [
            sys.executable, "-m", "pytest",
            *test_paths,
            "-x", "-q", "--tb=short",
            "--import-mode=importlib",
            f"--rootdir={project_root}",
        ]

        if mutant.test_functions:
            k_expr = " or ".join(mutant.test_functions)
            cmd += ["-k", k_expr]

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=120,
            )
            duration = time.perf_counter() - start
            stdout   = proc.stdout or ""
            stderr   = proc.stderr or ""

            logger.debug(
                f"Mutant {mutant.id} | exit={proc.returncode} | "
                f"stdout: {stdout[:500]!r}"
            )
            if proc.returncode not in (0, 1, 5):
                logger.warning(
                    f"Mutant {mutant.id} | stderr: {stderr[:500]!r}"
                )

            status       = self._classify(proc.returncode, stdout)
            failed_tests = self._extract_failed_tests(stdout)

            return TestResult(
                mutant=mutant.id,
                status=status,
                failed_tests=failed_tests,
                execution_time=round(duration, 4),
            )

        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start
            logger.warning(f"Mutant {mutant.id}: timeout após {duration:.1f}s")
            return TestResult(
                mutant=mutant.id,
                status="timeout",
                failed_tests=[],
                execution_time=round(duration, 4),
            )


    @staticmethod
    def _build_env(sandbox_dir: Path, original_source_dir: Path | None = None) -> dict:
        env = os.environ.copy()

        python_bin_dir = str(Path(sys.executable).parent)
        current_path = env.get("PATH", "")
        if python_bin_dir not in current_path:
            env["PATH"] = python_bin_dir + os.pathsep + current_path

        current_pythonpath = env.get("PYTHONPATH", "").strip()
        parts = [str(sandbox_dir)]
        if original_source_dir and str(original_source_dir) not in current_pythonpath:
            parts.append(str(original_source_dir))
        if current_pythonpath:
            parts.append(current_pythonpath)

        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    @staticmethod
    def _classify(exit_code: int, stdout: str = "") -> str:
        if exit_code == 0 or exit_code == 5:
            return "survived"
        if exit_code == 1:
            if "FAILED" in stdout:
                return "killed"
            if "ERROR" in stdout:
                return "error"
            return "error"
        return "error"

    @staticmethod
    def _extract_failed_tests(stdout: str) -> list[str]:
        return [
            line.strip()[len("FAILED "):].strip()
            for line in stdout.splitlines()
            if line.strip().startswith("FAILED ")
        ]

    def __repr__(self) -> str:
        return (
            f"TestRunner("
            f"mutants={len(self.mutant_list)}, "
            f"results={len(self.results)}, "
            f"workers={self.max_workers})"
        )