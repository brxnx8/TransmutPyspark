"""
config_loader.py
================
Detecta o modo de entrada (arquivo único, diretório, transmut.toml ou
config.txt legado) e devolve sempre um ResolvedConfig.

O MutationManager nunca vê o modo raw — só enxerga ResolvedConfig.

Ordem de precedência (quando config_input é um diretório ou string vaga):
  1. Argumento é um .toml                   → resolve_from_toml
  2. Argumento é um .py                     → resolve_from_dict (Modo 1)
  3. Argumento é um diretório               → resolve_from_dict (Modo 2)
  4. transmut.toml existe na raiz passada   → resolve_from_toml
  5. config.txt existe na raiz passada      → _parse_txt + resolve_from_dict
"""
from __future__ import annotations

from pathlib import Path

from src.config.resolver import ResolvedConfig, resolve_from_dict, resolve_from_toml
from src.config.ast_analyzer import analyze


class ConfigLoader:
    """
    Detecta o modo de entrada e devolve sempre um ResolvedConfig.
    Também dispara o ast_analyzer para preencher targets.
    """

    def __init__(self, config_input: str | dict) -> None:
        """
        Aceita:
          - str  → caminho para config.txt, transmut.toml, arquivo .py ou diretório
          - dict → config em memória (gerado pela CLI com --src / --tests)
        """
        self._input = config_input

    def load(self) -> ResolvedConfig:
        """Resolve a configuração e retorna ResolvedConfig com targets preenchidos."""
        cfg = self._resolve()
        # Preenche targets via AST analyzer
        cfg.targets = analyze(cfg.source_files, cfg.test_files)
        return cfg

    # ------------------------------------------------------------------ #
    # Resolução de modo                                                    #
    # ------------------------------------------------------------------ #

    def _resolve(self) -> ResolvedConfig:
        # Modo inline: dict passado diretamente pela CLI
        if isinstance(self._input, dict):
            return resolve_from_dict(self._input)

        p = Path(self._input)

        # Modo 3a: arquivo .toml explícito
        if p.suffix == ".toml":
            if not p.exists():
                raise FileNotFoundError(f"Arquivo de config não encontrado: {p}")
            return resolve_from_toml(p)

        # Modo 1: arquivo .py único passado diretamente
        if p.suffix == ".py" and p.is_file():
            return resolve_from_dict({"program_path": str(p),
                                      "tests_path":   "",
                                      "operators_list": "",
                                      "workspace_dir": str(p.parent)})

        # Modo 3b: transmut.toml na raiz (arquivo de config padrão)
        toml_candidate = p.parent / "transmut.toml" if p.is_file() else p / "transmut.toml"
        if toml_candidate.exists():
            return resolve_from_toml(toml_candidate)

        # Modo legado: config.txt
        if p.is_file() and p.suffix in (".txt", ""):
            raw = self._parse_txt(p)
            return resolve_from_dict(raw)

        # Modo 2: diretório passado diretamente (sem config.txt)
        if p.is_dir():
            raise ValueError(
                f"'{p}' é um diretório mas não contém transmut.toml.\n"
                f"Crie um transmut.toml com 'transmut init' ou passe --src e --tests."
            )

        raise FileNotFoundError(f"Config não encontrado: '{self._input}'")

    # ------------------------------------------------------------------ #
    # Parser legado (config.txt)                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_txt(path: Path) -> dict:
        result: dict = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
        return result

    def __repr__(self) -> str:
        return f"ConfigLoader(input={self._input!r})"
