from __future__ import annotations

from pathlib import Path

from src.config.resolver import ResolvedConfig, resolve_from_dict, resolve_from_toml
from src.config.ast_analyzer import analyze


class ConfigLoader:

    def __init__(self, config_input: str | dict) -> None:
        self._input = config_input

    def load(self) -> ResolvedConfig:
        cfg = self._resolve()
        cfg.targets = analyze(cfg.source_files, cfg.test_files)
        return cfg

    def _resolve(self) -> ResolvedConfig:
        if isinstance(self._input, dict):
            return resolve_from_dict(self._input)

        p = Path(self._input)

        if p.suffix == ".toml":
            if not p.exists():
                raise FileNotFoundError(f"Arquivo de config não encontrado: {p}")
            return resolve_from_toml(p)

        if p.suffix == ".py" and p.is_file():
            return resolve_from_dict({"program_path": str(p),
                                      "tests_path":   "",
                                      "operators_list": "",
                                      "workspace_dir": str(p.parent)})

        toml_candidate = p.parent / "transmut.toml" if p.is_file() else p / "transmut.toml"
        if toml_candidate.exists():
            return resolve_from_toml(toml_candidate)

        if p.is_file() and p.suffix in (".txt", ""):
            raw = self._parse_txt(p)
            return resolve_from_dict(raw)

        if p.is_dir():
            raise ValueError(
                f"'{p}' é um diretório mas não contém transmut.toml.\n"
                f"Crie um transmut.toml com 'transmut init' ou passe --src e --tests."
            )

        raise FileNotFoundError(f"Config não encontrado: '{self._input}'")

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
