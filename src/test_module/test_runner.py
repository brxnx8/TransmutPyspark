import os
import re
import shutil
import subprocess
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from src.model.test_result import TestResult

logger = logging.getLogger(__name__)

_VALID_TEST_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Se True, mutantes cujo pytest retornou "nenhum teste coletado" (exit code 5)
# contam como "survived" no cálculo do mutation score (comportamento antigo).
# Se False, ficam com status próprio ("no_tests_collected") e são excluídos
# do denominador, evitando inflar/deflar o score por um problema de mapeamento
# em vez de uma fraqueza real da suíte.
COUNT_NO_TESTS_COLLECTED_AS_SURVIVED = True


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

        futures_map: dict = {}
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for mutant in self.mutant_list:
                    # Sandbox exclusiva por mutante — sem reciclagem entre threads,
                    # elimina a possibilidade de duas execuções concorrentes
                    # escreverem/lerem o mesmo diretório ao mesmo tempo.
                    sandbox = sandbox_root / f"sandbox_{mutant.id}"
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
        finally:
            # Todas as sandboxes (sandbox_<id>) vivem sob sandbox_root — remover
            # a raiz de uma vez descarta todas ao final da execução, mesmo se
            # alguma tarefa tiver levantado exceção.
            shutil.rmtree(sandbox_root, ignore_errors=True)

        shutil.rmtree(sandbox_root, ignore_errors=True)

        killed    = sum(1 for r in self.results if r.status == "killed")
        survived  = sum(1 for r in self.results if r.status == "survived")
        timeouts  = sum(1 for r in self.results if r.status == "timeout")
        errors    = sum(1 for r in self.results if r.status == "error")
        no_tests  = sum(1 for r in self.results if r.status == "no_tests_collected")

        # NOTA: esta fórmula (killed / (killed + survived)) difere da definida
        # no texto do TCC (Seção 4.3.5), que inclui timeout e error no
        # denominador. Mantida assim por decisão consciente — timeout/error são
        # resultados inconclusivos, não evidência de detecção nem de
        # sobrevivência —, mas o texto do trabalho deve ser atualizado para
        # refletir isso, ou esta fórmula deve ser revista para bater com o TCC.
        exercised = killed + survived
        if COUNT_NO_TESTS_COLLECTED_AS_SURVIVED:
            exercised += no_tests
            survived += no_tests
        mutation_score = (killed / exercised * 100) if exercised > 0 else 0.0

        logger.info(
            f"[TestRunner] Concluído — "
            f"killed={killed}, survived={survived} (no_tests_collected={no_tests}), "
            f"timeout={timeouts}, error={errors}  | "
            f"mutation score={mutation_score:.1f}% "
            f"({killed}/{exercised} mutantes exercitados)"
        )
        return self.results

    def _run_single_mutant(self, mutant, sandbox: Path) -> TestResult:
        sandbox.mkdir(parents=True, exist_ok=True)
        try:
            mutant_path   = Path(mutant.mutant_path)
            original_name = Path(mutant.original_path).name

            if not mutant.test_files:
                logger.debug(f"Mutant {mutant.id}: Nenhum teste mapeado — survived")
                return TestResult(
                    mutant=mutant.id,
                    status="no_tests_collected",
                    failed_tests=[],
                    execution_time=0.0,
                )

            target_file = sandbox / original_name
            target_file.write_text(
                mutant_path.read_text(encoding="utf-8"), encoding="utf-8"
            )

            env = self._build_env(
                sandbox,
                original_source_dir=Path(mutant.original_path).parent,
            )

            test_paths = [str(tf) for tf in mutant.test_files]

            project_root = self.config.workspace_dir
            cmd = [
                sys.executable, "-m", "pytest",
                *test_paths,
                "-x", "-q", "--tb=short",
                "--import-mode=importlib",
                "-p", "no:cacheprovider",
                f"--rootdir={project_root}",
            ]

            safe_functions = self._sanitize_test_functions(mutant.test_functions)
            if safe_functions:
                cmd += ["-k", " or ".join(safe_functions)]

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

                failed_tests = self._extract_failed_tests(stdout)
                status = self._classify(proc.returncode, stdout, failed_tests)

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
        finally:
            # Cada mutante limpa a própria sandbox assim que termina, em vez de
            # acumular todos os diretórios até o fim do run_test — reduz o
            # pico de uso de disco quando há muitos mutantes em paralelo.
            # A remoção de sandbox_root ao final de run_test cobre qualquer
            # resíduo remanescente (ex.: em caso de crash antes deste ponto).
            shutil.rmtree(sandbox, ignore_errors=True)

    @staticmethod
    def _sanitize_test_functions(test_functions) -> list[str]:
        """Filtra apenas identificadores Python válidos, evitando que nomes
        inesperados quebrem a sintaxe da expressão -k do pytest."""
        if not test_functions:
            return []
        safe = [f for f in test_functions if _VALID_TEST_NAME_RE.match(f)]
        dropped = set(test_functions) - set(safe)
        if dropped:
            logger.warning(
                f"[TestRunner] Nomes de teste ignorados na expressão -k "
                f"(não são identificadores válidos): {sorted(dropped)}"
            )
        return safe

    @staticmethod
    def _build_env(sandbox_dir: Path, original_source_dir: Path | None = None) -> dict:
        env = os.environ.copy()

        python_bin_dir = str(Path(sys.executable).parent)
        current_path = env.get("PATH", "")
        path_parts = current_path.split(os.pathsep) if current_path else []
        if python_bin_dir not in path_parts:
            env["PATH"] = python_bin_dir + os.pathsep + current_path

        current_pythonpath = env.get("PYTHONPATH", "").strip()
        pythonpath_parts_existing = (
            current_pythonpath.split(os.pathsep) if current_pythonpath else []
        )

        parts = [str(sandbox_dir)]
        if original_source_dir and str(original_source_dir) not in pythonpath_parts_existing:
            parts.append(str(original_source_dir))
        if current_pythonpath:
            parts.append(current_pythonpath)

        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    @staticmethod
    def _classify(exit_code: int, stdout: str, failed_tests: list[str] | None = None) -> str:
        """Classifica o resultado da execução de um mutante.

        Prioriza a lista de testes efetivamente marcados como FAILED (extraída
        por regex de início de linha) em vez de checar substrings soltas
        ("FAILED"/"ERROR") em todo o stdout, o que reduz falsos positivos caso
        esses termos apareçam por outro motivo na saída (ex.: dentro do nome
        de um teste ou de uma mensagem de asserção).
        """
        if exit_code == 0:
            return "survived"

        if exit_code == 5:
            # "Nenhum teste coletado": pode indicar que o -k não encontrou
            # nenhum teste correspondente (possível falha de mapeamento) ou
            # que os arquivos de teste mapeados estão vazios/mal formados.
            # Tratado como status próprio para não se confundir com um
            # "survived" real (suíte rodou e não detectou a mutação).
            return "no_tests_collected"

        if exit_code == 1:
            if failed_tests:
                return "killed"
            if re.search(r"^ERROR\b", stdout, re.MULTILINE):
                return "error"
            # Exit 1 sem nenhum FAILED/ERROR identificável é inesperado;
            # tratado como erro por segurança (fail-safe), não como killed.
            return "error"

        # Códigos 2, 3, 4 (e qualquer outro não mapeado): erros internos do
        # pytest (uso incorreto, interrupção, erro interno, etc.).
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