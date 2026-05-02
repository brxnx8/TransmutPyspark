"""
MutationManager - Orquestrador do pipeline de mutação
"""

import ast
import importlib
import logging
import shutil
from pathlib import Path

from src.config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# Nome fixo do diretório de saída
_OUTPUT_DIR_NAME = "TransmutPysparkOutput"

# Mapeamento de operadores disponíveis
OPERATOR_REGISTRY = {
    "NFTP": "src.operators.operator_nftp.OperatorNFTP",
    "MTR":  "src.operators.operator_mtr.OperatorMTR",
    "ATR":  "src.operators.operator_atr.OperatorATR",
    "UTS":  "src.operators.operator_uts.OperatorUTS",
}


class MutationManager:
    """Orquestrador do pipeline de mutação teste."""

    def __init__(self, config_path: str) -> None:
        self.config_path  = config_path
        self.config       = None
        self.code_original = ""
        self.code_ast     = None
        self.mutant_list  = []
        self.result_list  = []
        self.work_dir: Path | None = None   # preenchido em load()

    # ------------------------------------------------------------------ #
    # Pipeline steps                                                       #
    # ------------------------------------------------------------------ #

    def load(self) -> "MutationManager":
        """Carrega a configuração, prepara o workdir e lê o código original."""
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config não encontrado: {self.config_path}")

        config_text = Path(self.config_path).read_text(encoding="utf-8")
        config_dict = self._parse_config(config_text)

        required_keys = ["program_path", "tests_path", "operators_list", "workspace_dir"]
        for key in required_keys:
            if key not in config_dict:
                raise ValueError(f"Chave obrigatória faltando no config: {key}")

        operators = [op.strip().upper() for op in config_dict["operators_list"].split(",")]

        self.config = ConfigLoader(
            program_path=config_dict["program_path"].strip(),
            tests_path=config_dict["tests_path"].strip(),
            operators_list=operators,
            workspace_dir=config_dict["workspace_dir"].strip(),
        )

        # ── Prepara o workdir ──────────────────────────────────────────
        self.work_dir = Path(self.config.workspace_dir) / _OUTPUT_DIR_NAME
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
            logger.info(f"Workdir existente removido: {self.work_dir}")
        self.work_dir.mkdir(parents=True)
        logger.info(f"Workdir criado: {self.work_dir}")

        # ── Lê código original ─────────────────────────────────────────
        program_path = Path(self.config.program_path)
        if not program_path.exists():
            raise FileNotFoundError(f"Programa não encontrado: {program_path}")

        self.code_original = program_path.read_text(encoding="utf-8")
        logger.info(f"Config carregado: {self.config.program_path}")
        return self

    def parse_to_ast(self) -> "MutationManager":
        """Converte o código em AST."""
        if not self.code_original:
            raise RuntimeError("Chame load() primeiro")
        try:
            self.code_ast = ast.parse(self.code_original)
            ast.fix_missing_locations(self.code_ast)
            logger.info("AST gerado com sucesso")
        except SyntaxError as e:
            raise ValueError(f"Erro de sintaxe na linha {e.lineno}: {e.msg}")
        return self

    def apply_mutation(self) -> "MutationManager":
        """Aplica mutações usando os operadores configurados."""
        if not self.code_ast:
            raise RuntimeError("Chame parse_to_ast() primeiro")
        if self.work_dir is None:
            raise RuntimeError("Chame load() primeiro")

        # Mutantes ficam dentro do workdir
        mutant_dir = self.work_dir / "mutants"
        mutant_dir.mkdir(parents=True, exist_ok=True)

        global_counter = len(self.mutant_list)

        for op_name in self.config.operators_list:
            try:
                operator = self._load_operator(op_name)
                nodes    = operator.analyse_ast(self.code_ast)

                if not nodes:
                    logger.info(f"Operador '{op_name}': nenhum nó elegível")
                    continue

                mutants = operator.build_mutant(
                    nodes=nodes,
                    original_ast=self.code_ast,
                    original_path=self.config.program_path,
                    mutant_dir=str(mutant_dir),
                )

                # Renumera com IDs globais únicos
                for m in mutants:
                    global_counter += 1
                    m.id = global_counter

                self.mutant_list.extend(mutants)
                logger.info(f"Operador '{op_name}': {len(mutants)} mutante(s) gerado(s)")

            except Exception as e:
                logger.warning(f"Erro no operador '{op_name}': {e}")

        logger.info(f"Total de mutantes: {len(self.mutant_list)}")
        return self

    def run_tests(self) -> "MutationManager":
        """Executa testes para todos os mutantes."""
        if not self.mutant_list:
            raise RuntimeError("Chame apply_mutation() primeiro")

        from src.test_module.test_runner import TestRunner

        runner = TestRunner(mutant_list=self.mutant_list, config=self.config)
        self.result_list = runner.run_test()

        logger.info(f"Testes executados: {len(self.result_list)} resultado(s)")
        return self

    def agregate_results(self) -> "MutationManager":
        """Gera relatório agregado dos resultados."""
        if not self.result_list:
            raise RuntimeError("Chame run_tests() primeiro")
        if self.work_dir is None:
            raise RuntimeError("Chame load() primeiro")

        from src.reporter.reporter import Reporter

        reporter = Reporter(
            result_list=self.result_list,
            code_original=self.code_original,
            mutant_list=self.mutant_list,
            output_dir=self.work_dir,          # ← passa o workdir para o Reporter
        )

        reporter.calculate()
        reporter.make_diff()
        reporter.show_results()

        logger.info("Relatório gerado com sucesso")
        return self

    def run(self) -> "MutationManager":
        """Executa o pipeline completo em uma única chamada."""
        logger.info("Iniciando pipeline de mutação...")
        self.load()
        self.parse_to_ast()
        self.apply_mutation()
        self.run_tests()
        self.agregate_results()
        logger.info(f"Pipeline concluído! Saída em: {self.work_dir}")
        return self

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _parse_config(self, text: str) -> dict:
        config: dict = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                config[key.strip()] = value.strip()
        return config

    def _load_operator(self, op_name: str):
        op_name = op_name.strip().upper()
        if op_name not in OPERATOR_REGISTRY:
            raise KeyError(f"Operador '{op_name}' não registrado")
        dotted_path = OPERATOR_REGISTRY[op_name]
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        operator_class = getattr(module, class_name)
        return operator_class.create()

    # ------------------------------------------------------------------ #
    # Dunder                                                               #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"MutationManager("
            f"config={'loaded' if self.config else 'not_loaded'}, "
            f"work_dir={str(self.work_dir)!r}, "
            f"ast={'yes' if self.code_ast else 'no'}, "
            f"mutants={len(self.mutant_list)}, "
            f"results={len(self.result_list)}"
            f")"
        )