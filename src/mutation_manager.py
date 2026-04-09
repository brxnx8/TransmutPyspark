"""
MutationManager - Orquestrador do pipeline de mutação
"""

import ast
import importlib
import logging
from pathlib import Path

from src.model.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


# Mapeamento de operadores disponíveis
OPERATOR_REGISTRY = {
    "NFTP": "src.operators.operator_nftp.OperatorNFTP",
    "MTR": "src.operators.operator_mtr.OperatorMTR",
}


class MutationManager:
    """Orquestrador do pipeline de mutação teste."""

    def __init__(self, config_path):
        """
        Inicializa o gerenciador de mutação.
        
        Args:
            config_path: Caminho para o arquivo config.txt
        """
        self.config_path = config_path
        self.config = None
        self.code_original = ""
        self.code_ast = None
        self.mutant_list = []
        self.result_list = []

    def load(self):
        """Carrega a configuração e o código original do programa."""
        # Valida arquivo de config
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config não encontrado: {self.config_path}")

        # Lê arquivo de config (formato: key = value)
        config_text = Path(self.config_path).read_text(encoding="utf-8")
        config_dict = self._parse_config(config_text)

        # Valida chaves obrigatórias
        required_keys = ["program_path", "tests_path", "operators_list"]
        for key in required_keys:
            if key not in config_dict:
                raise ValueError(f"Chave obrigatória faltando no config: {key}")

        # Converte operators_list em lista
        operators = [op.strip().upper() for op in config_dict["operators_list"].split(",")]

        # Cria ConfigLoader
        self.config = ConfigLoader(
            program_path=config_dict["program_path"].strip(),
            tests_path=config_dict["tests_path"].strip(),
            operators_list=operators,
        )

        # Lê código original
        program_path = Path(self.config.program_path)
        if not program_path.exists():
            raise FileNotFoundError(f"Programa não encontrado: {program_path}")

        self.code_original = program_path.read_text(encoding="utf-8")
        logger.info(f"Config carregado: {self.config.program_path}")

        return self

    def parse_to_ast(self):
        """Converte o código em AST (árvore sintática)."""
        if not self.code_original:
            raise RuntimeError("Chame load() primeiro")

        try:
            self.code_ast = ast.parse(self.code_original)
            ast.fix_missing_locations(self.code_ast)
            logger.info("AST gerado com sucesso")
        except SyntaxError as e:
            raise ValueError(f"Erro de sintaxe na linha {e.lineno}: {e.msg}")

        return self

    def apply_mutation(self):
        """Aplica mutações usando os operadores configurados."""
        if not self.code_ast:
            raise RuntimeError("Chame parse_to_ast() primeiro")

        # Cria diretório de mutantes
        mutant_dir = Path(self.config.program_path).parent / "mutants"
        mutant_dir.mkdir(parents=True, exist_ok=True)

        # Para cada operador configurado
        for op_name in self.config.operators_list:
            try:
                # Carrega dinamicamente o operador
                operator = self._load_operator(op_name)

                # Encontra nós elegíveis para mutação
                nodes = operator.analyse_ast(self.code_ast)

                if not nodes:
                    logger.info(f"Operador '{op_name}': nenhum nó elegível")
                    continue

                # Gera mutantes
                mutants = operator.build_mutant(
                    nodes=nodes,
                    original_ast=self.code_ast,
                    original_path=self.config.program_path,
                    mutant_dir=str(mutant_dir),
                )
                self.mutant_list.extend(mutants)
                logger.info(f"Operador '{op_name}': {len(mutants)} mutante(s) gerado(s)")

            except Exception as e:
                logger.warning(f"Erro no operador '{op_name}': {e}")

        logger.info(f"Total de mutantes: {len(self.mutant_list)}")
        return self

    def run_tests(self):
        """Executa testes para todos os mutantes."""
        if not self.mutant_list:
            raise RuntimeError("Chame apply_mutation() primeiro")

        from src.test_module.test_runner import TestRunner

        runner = TestRunner(mutant_list=self.mutant_list, config=self.config)
        self.result_list = runner.run_test()

        logger.info(f"Testes executados: {len(self.result_list)} resultado(s)")
        return self

    def agregate_results(self):
        """Gera relatórios agregados dos resultados."""
        if not self.result_list:
            raise RuntimeError("Chame run_tests() primeiro")

        from src.reporter.reporter import Reporter

        reporter = Reporter(
            result_list=self.result_list,
            code_original=self.code_original,
            mutant_list=self.mutant_list,
        )

        reporter.calculate()
        reporter.make_diff()
        reporter.show_results()

        logger.info("Relatório gerado com sucesso")
        return self

    # Métodos privados

    def _parse_config(self, text):
        """Parse de arquivo config no formato: key = value"""
        config = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                config[key.strip()] = value.strip()
        return config

    def _load_operator(self, op_name):
        """Carrega dinamicamente um operador pelo nome."""
        op_name = op_name.strip().upper()

        if op_name not in OPERATOR_REGISTRY:
            raise KeyError(f"Operador '{op_name}' não registrado")

        # Importa dinamicamente
        dotted_path = OPERATOR_REGISTRY[op_name]
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        operator_class = getattr(module, class_name)

        return operator_class()

    def run(self):
        """
        Executa o pipeline completo de mutação teste em uma única chamada.
        
        Returns:
            self — para acesso aos resultados
        """
        logger.info("Iniciando pipeline de mutação...")
        self.load()
        self.parse_to_ast()
        self.apply_mutation()
        self.run_tests()
        self.agregate_results()
        logger.info("Pipeline concluído com sucesso!")
        return self

    def __repr__(self):
        return (
            f"MutationManager("
            f"config={'loaded' if self.config else 'not_loaded'}, "
            f"ast={'yes' if self.code_ast else 'no'}, "
            f"mutants={len(self.mutant_list)}, "
            f"results={len(self.result_list)}"
            f")"
        )