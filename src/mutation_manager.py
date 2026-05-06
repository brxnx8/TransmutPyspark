"""
MutationManager
===============
Orquestrador do pipeline de mutação.

Mudanças em relação à versão anterior:
  - load() usa ConfigLoader novo (suporta 3 modos de entrada)
  - parse_to_ast() constrói um dict source_file → AST (múltiplos arquivos)
  - apply_mutation() itera sobre FunctionTargets em vez da AST inteira,
    garantindo escopo cirúrgico por função
  - run_tests() e agregate_results() inalterados na interface pública
"""

import importlib
import logging
import shutil
from pathlib import Path

from src.config.config_loader import ConfigLoader
from src.config.resolver import ResolvedConfig

logger = logging.getLogger(__name__)

_OUTPUT_DIR_NAME = "TransmutPysparkOutput"

OPERATOR_REGISTRY = {
    "NFTP": "src.operators.operator_nftp.OperatorNFTP",
    "MTR":  "src.operators.operator_mtr.OperatorMTR",
    "ATR":  "src.operators.operator_atr.OperatorATR",
    "UTS":  "src.operators.operator_uts.OperatorUTS",
}


class MutationManager:
    """Orquestrador do pipeline de mutation testing."""

    def __init__(self, config_input: str | dict) -> None:
        """
        Aceita:
          - str  → config.txt, transmut.toml, arquivo .py ou diretório
          - dict → config em memória (gerado pela CLI com --src / --tests)
        """
        self.config_input  = config_input
        self.config: ResolvedConfig | None = None

        # source_file → código fonte (str)
        self.source_codes: dict[Path, str] = {}
        # source_file → AST parseada
        self.source_asts:  dict[Path, object] = {}

        self.mutant_list = []
        self.result_list = []
        self.work_dir: Path | None = None

    # ------------------------------------------------------------------ #
    # Pipeline steps                                                       #
    # ------------------------------------------------------------------ #

    def load(self) -> "MutationManager":
        """Carrega configuração, descobre arquivos e extrai FunctionTargets."""
        self.config = ConfigLoader(self.config_input).load()

        # Prepara workdir
        self.work_dir = self.config.workspace_dir / _OUTPUT_DIR_NAME
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        self.work_dir.mkdir(parents=True)
        logger.info(f"Workdir criado: {self.work_dir}")

        # Lê código de todos os arquivos fonte
        for sf in self.config.source_files:
            self.source_codes[sf.resolve()] = sf.read_text(encoding="utf-8")

        logger.info(
            f"Config carregado — "
            f"{len(self.config.source_files)} fonte(s), "
            f"{len(self.config.test_files)} teste(s), "
            f"{len(self.config.targets)} target(s)."
        )
        return self

    def parse_to_ast(self) -> "MutationManager":
        """Converte cada arquivo fonte em AST."""
        import ast as _ast
        if not self.source_codes:
            raise RuntimeError("Chame load() primeiro.")

        for sf, code in self.source_codes.items():
            try:
                tree = _ast.parse(code, filename=str(sf))
                _ast.fix_missing_locations(tree)
                # Usa sempre o path resolvido (absoluto) como chave
                self.source_asts[sf.resolve()] = tree
            except SyntaxError as e:
                raise ValueError(
                    f"Erro de sintaxe em {sf}, linha {e.lineno}: {e.msg}"
                )

        logger.info(f"ASTs geradas: {len(self.source_asts)} arquivo(s).")
        return self

    def apply_mutation(self) -> "MutationManager":
        """
        Aplica mutações iterando sobre FunctionTargets.
        Cada operador recebe o nó da função (escopo cirúrgico) para
        analyse_ast e a AST completa do arquivo para build_mutant.
        """
        if not self.source_asts:
            raise RuntimeError("Chame parse_to_ast() primeiro.")
        if self.work_dir is None:
            raise RuntimeError("Chame load() primeiro.")

        mutant_dir    = self.work_dir / "mutants"
        mutant_dir.mkdir(parents=True, exist_ok=True)
        global_counter = 0

        for target in self.config.targets:
            file_ast = self.source_asts.get(target.source_file.resolve())
            if file_ast is None:
                logger.warning(f"AST não encontrada para {target.source_file} — pulando.")
                continue

            # Subpasta por arquivo fonte — evita colisão de nomes entre targets
            # ex: mutants/atr/ e mutants/uts/ em vez de tudo em mutants/
            target_mutant_dir = mutant_dir / target.source_file.stem
            target_mutant_dir.mkdir(parents=True, exist_ok=True)

            # Subpasta: mutants/<arquivo_fonte>/<operador>/
            # ex: mutants/atr/ATR/, mutants/atr/MTR/
            for op_name in self.config.operators:
                try:
                    operator = self._load_operator(op_name)
                    op_dir = target_mutant_dir / op_name.upper()
                    op_dir.mkdir(parents=True, exist_ok=True)

                    # analyse_ast recebe só o nó da função — escopo cirúrgico
                    eligible_nodes = operator.analyse_ast(target.node)
                    if not eligible_nodes:
                        continue

                    # Marca quantos mutantes havia antes para pegar só os novos
                    before = len(operator.mutant_list)

                    # build_mutant recebe a AST completa do arquivo para gerar
                    # o arquivo mutante completo e válido
                    operator.build_mutant(
                        nodes=eligible_nodes,
                        original_ast=file_ast,
                        original_path=str(target.source_file),
                        mutant_dir=str(op_dir),
                    )

                    # Pega só os mutantes novos gerados nesta chamada
                    new_mutants = operator.mutant_list[before:]

                    # Propaga mapeamento de testes para cada mutante gerado
                    for m in new_mutants:
                        global_counter += 1
                        m.id             = global_counter
                        m.test_files     = list(target.test_files)
                        m.test_functions = list(target.test_functions)

                    self.mutant_list.extend(new_mutants)
                    logger.info(
                        f"[{op_name}] {target.qualified_name}: "
                        f"{len(new_mutants)} mutante(s)."
                    )

                except Exception as e:
                    logger.warning(
                        f"Erro no operador '{op_name}' "
                        f"para '{target.qualified_name}': {e}"
                    )

        logger.info(f"Total de mutantes gerados: {len(self.mutant_list)}")
        return self

    def run_tests(self) -> "MutationManager":
        """Executa testes para todos os mutantes gerados."""
        if not self.mutant_list:
            logger.warning("Nenhum mutante gerado — nada para testar.")
            return self

        from src.test_module.test_runner import TestRunner

        runner = TestRunner(mutant_list=self.mutant_list, config=self.config)
        self.result_list = runner.run_test()
        logger.info(f"Testes executados: {len(self.result_list)} resultado(s).")
        return self

    def agregate_results(self) -> "MutationManager":
        """Consolida resultados e gera relatório."""
        if not self.result_list:
            raise RuntimeError("Chame run_tests() primeiro.")

        from src.reporter.reporter import Reporter

        # Passa o código do primeiro arquivo fonte para o Reporter
        # (compatibilidade com o Reporter atual que espera uma string)
        first_source = next(iter(self.source_codes.values()), "")

        reporter = Reporter(
            result_list=self.result_list,
            code_original=first_source,
            mutant_list=self.mutant_list,
            output_dir=self.work_dir,
        )
        reporter.calculate()
        reporter.make_diff()
        reporter.show_results()

        logger.info(f"Relatório gerado: {self.work_dir}")
        return self

    def run(self) -> "MutationManager":
        """Executa o pipeline completo em uma única chamada."""
        logger.info("Iniciando pipeline de mutação...")
        self.load()
        self.parse_to_ast()
        self.apply_mutation()
        self.run_tests()
        self.agregate_results()
        logger.info(f"Pipeline concluído. Saída em: {self.work_dir}")
        return self

    # ------------------------------------------------------------------ #
    # Helpers privados                                                     #
    # ------------------------------------------------------------------ #

    def _load_operator(self, op_name: str):
        op_name = op_name.strip().upper()
        if op_name not in OPERATOR_REGISTRY:
            raise KeyError(f"Operador '{op_name}' não registrado. "
                           f"Disponíveis: {list(OPERATOR_REGISTRY)}")
        dotted_path = OPERATOR_REGISTRY[op_name]
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name).create()

    def __repr__(self) -> str:
        return (
            f"MutationManager("
            f"config={'loaded' if self.config else 'not_loaded'}, "
            f"sources={len(self.source_codes)}, "
            f"mutants={len(self.mutant_list)}, "
            f"results={len(self.result_list)})"
        )
