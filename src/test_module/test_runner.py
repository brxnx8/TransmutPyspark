"""
TestRunner
==========
Executa o pytest contra cada mutante em paralelo.

Mudanças em relação à versão anterior:
  - Aceita ResolvedConfig em vez de ConfigLoader legado
  - Usa mutant.test_files e mutant.test_functions para rodar só os
    testes mapeados (escopo cirúrgico de execução)
  - Fallback para tests_path completo quando não há mapeamento
"""

import os
import shutil
import subprocess
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from src.model.test_result import TestResult

logger = logging.getLogger(__name__)


@dataclass
class TestRunner:
    mutant_list: list
    config:      object      # ResolvedConfig
    max_workers: int = field(
        default_factory=lambda: min(4, os.cpu_count() or 1)
    )
    results: list[TestResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.mutant_list, list):
            raise TypeError(f"mutant_list deve ser lista, recebeu: {type(self.mutant_list)}")
        if self.max_workers < 1:
            raise ValueError(f"max_workers deve ser >= 1, recebeu: {self.max_workers}")

    # ------------------------------------------------------------------ #
    # API pública                                                          #
    # ------------------------------------------------------------------ #

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

        futures_map: dict = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for mutant in self.mutant_list:
                future = executor.submit(self._run_single_mutant, mutant, sandbox_root)
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

        killed   = sum(1 for r in self.results if r.status == "killed")
        survived = sum(1 for r in self.results if r.status == "survived")
        timeouts = sum(1 for r in self.results if r.status == "timeout")
        errors   = sum(1 for r in self.results if r.status == "error")
        logger.info(
            f"[TestRunner] Concluído — "
            f"killed={killed}, survived={survived}, "
            f"timeout={timeouts}, error={errors}"
        )
        return self.results

    # ------------------------------------------------------------------ #
    # Execução de mutante individual                                       #
    # ------------------------------------------------------------------ #

    def _run_single_mutant(self, mutant, sandbox_root: Path) -> TestResult:
        mutant_path   = Path(mutant.mutant_path)
        original_name = Path(mutant.original_path).name   # ex: uts.py

        # Sandbox isolado por mutante
        sandbox_dir = sandbox_root / f"sandbox_{mutant.id}"
        sandbox_dir.mkdir(exist_ok=True)

        # Copia o arquivo mutante com o nome original para que os imports
        # dos testes continuem funcionando
        target_file = sandbox_dir / original_name
        target_file.write_text(
            mutant_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

        env = self._build_env(
            sandbox_dir,
            original_source_dir=Path(mutant.original_path).parent,
        )

        # ── Resolve quais arquivos de teste rodar ──────────────────────
        # Usa os test_files mapeados pelo ast_analyzer; fallback para
        # todos os test_files do config quando não há mapeamento
        if mutant.test_files:
            test_paths = [str(tf) for tf in mutant.test_files]
        else:
            test_paths = [str(tf) for tf in self.config.test_files]

        import sys
        # Usa 'python -m pytest' em vez do binário pytest diretamente.
        # Garante que o pytest correto (do mesmo ambiente Python) é usado,
        # independente de estar num venv, conda ou instalação global.
        cmd = [sys.executable, "-m", "pytest", *test_paths, "-x", "-q", "--tb=short"]

        # Filtro por nome de função de teste quando mapeado (mais rápido)
        if mutant.test_functions:
            k_expr = " or ".join(mutant.test_functions)
            cmd += ["-k", k_expr]

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

            logger.debug(
                f"Mutant {mutant.id}: {status} "
                f"({duration:.2f}s, exit={proc.returncode})"
            )
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

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_env(sandbox_dir: Path, original_source_dir: Path | None = None) -> dict:
        """Monta o PYTHONPATH injetando o sandbox e o diretório fonte no início."""
        import sys
        env = os.environ.copy()

        # Garante que o bin do Python atual (venv) está no PATH do subprocess
        python_bin_dir = str(Path(sys.executable).parent)
        current_path = env.get("PATH", "")
        if python_bin_dir not in current_path:
            env["PATH"] = python_bin_dir + os.pathsep + current_path

        # Monta PYTHONPATH:
        # 1. sandbox_dir      — contém o arquivo mutante (substitui o original)
        # 2. original_source_dir — contém os outros módulos fonte (imports dos testes)
        # 3. paths existentes
        current_pythonpath = env.get("PYTHONPATH", "").strip()

        parts = [str(sandbox_dir)]
        if original_source_dir and str(original_source_dir) not in current_pythonpath:
            parts.append(str(original_source_dir))
        if current_pythonpath:
            parts.append(current_pythonpath)

        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    @staticmethod
    def _classify(exit_code: int) -> str:
        if exit_code == 0:
            return "survived"
        if exit_code == 1:
            return "killed"
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
