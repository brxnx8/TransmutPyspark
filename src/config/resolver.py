from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


_IGNORE_FILES: frozenset[str] = frozenset({
    "__init__.py", "conftest.py", "setup.py",
    "config.py", "settings.py",
})

_IGNORE_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".venv", "venv", ".env",
    "node_modules", ".git", ".tox", "dist", "build",
})


@dataclass
class ResolvedConfig:
    source_files:  list[Path]
    test_files:    list[Path]
    operators:     list[str]
    workspace_dir: Path
    targets:       list = field(default_factory=list)

    def validate(self) -> None:
        if not self.source_files:
            raise ValueError("Nenhum arquivo fonte encontrado.")
        if not self.test_files:
            raise ValueError("Nenhum arquivo de teste encontrado.")
        missing = [f for f in self.source_files + self.test_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                "Arquivos não encontrados:\n" + "\n".join(f"  {f}" for f in missing)
            )

    def __repr__(self) -> str:
        return (
            f"ResolvedConfig("
            f"sources={len(self.source_files)}, "
            f"tests={len(self.test_files)}, "
            f"operators={self.operators}, "
            f"workspace={self.workspace_dir})"
        )



def resolve_from_dict(raw: dict) -> ResolvedConfig:
    operators = [
        op.strip().upper()
        for op in raw.get("operators_list", "").split(",")
        if op.strip()
    ]
    if not operators:
        operators = raw.get("operators", [])

    workspace = Path(raw.get("workspace_dir", ".").strip())

    src_entry  = raw.get("program_path") or raw.get("source_dirs")
    test_entry = raw.get("tests_path")   or raw.get("tests_dirs")

    source_files = _resolve_entry(src_entry)
    test_files   = _resolve_entry(test_entry)

    cfg = ResolvedConfig(
        source_files=source_files,
        test_files=test_files,
        operators=operators,
        workspace_dir=workspace,
    )
    cfg.validate()
    return cfg


def resolve_from_toml(toml_path: Path) -> ResolvedConfig:
    if tomllib is None:
        raise ImportError(
            "Python < 3.11 requer 'tomli': pip install tomli"
        )
    with open(toml_path, "rb") as f:
        data = tomllib.load(f).get("transmut", {})

    source_files = _resolve_entry(data.get("source_dirs", []))
    test_files   = _resolve_entry(data.get("tests_dirs",  []))

    cfg = ResolvedConfig(
        source_files=source_files,
        test_files=test_files,
        operators=[op.upper() for op in data.get("operators", [])],
        workspace_dir=Path(data.get("workspace_dir", ".")),
    )
    cfg.validate()
    return cfg


def _resolve_entry(entry: str | list | None) -> list[Path]:
    if not entry:
        return []

    entries: list[str] = [entry] if isinstance(entry, str) else list(entry)
    result: list[Path] = []

    for e in entries:
        p = Path(str(e).strip())
        if p.is_file() and p.suffix == ".py":
            result.append(p)                  # Modo 1
        elif p.is_dir():
            result.extend(_discover_py(p))    # Modo 2
        else:
            raise ValueError(
                f"Caminho inválido ou não encontrado: '{p}'\n"
                f"Verifique se o arquivo/diretório existe."
            )

    return result


def _discover_py(directory: Path) -> list[Path]:
    found: list[Path] = []
    for f in sorted(directory.rglob("*.py")):
        if f.name in _IGNORE_FILES:
            continue
        if any(part in _IGNORE_DIRS for part in f.parts):
            continue
        if any(part.startswith(".") for part in f.parts):
            continue
        found.append(f)
    return found
