from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Importações do módulo sob teste
# ---------------------------------------------------------------------------
from src.config.resolver import (
    ResolvedConfig,
    _discover_py,
    _resolve_entry,
    resolve_from_dict,
    resolve_from_toml,
)


# ===========================================================================
# Fixtures compartilhadas
# ===========================================================================

@pytest.fixture()
def tmp_py_file(tmp_path: Path) -> Path:
    """Cria um único arquivo .py temporário."""
    f = tmp_path / "sample.py"
    f.write_text("x = 1\n")
    return f


@pytest.fixture()
def tmp_src_dir(tmp_path: Path) -> Path:
    """Cria uma hierarquia de arquivos .py em tmp_path/src."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "module_a.py").write_text("a = 1\n")
    (src / "module_b.py").write_text("b = 2\n")
    sub = src / "sub"
    sub.mkdir()
    (sub / "deep.py").write_text("d = 3\n")
    return src


@pytest.fixture()
def tmp_test_dir(tmp_path: Path) -> Path:
    """Cria uma hierarquia de arquivos de teste em tmp_path/tests."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_a.py").write_text("def test_a(): pass\n")
    (tests / "test_b.py").write_text("def test_b(): pass\n")
    return tests


@pytest.fixture()
def valid_resolved_config(tmp_src_dir: Path, tmp_test_dir: Path) -> ResolvedConfig:
    """ResolvedConfig válido com arquivos existentes."""
    src_files = list(tmp_src_dir.rglob("*.py"))
    test_files = list(tmp_test_dir.rglob("*.py"))
    return ResolvedConfig(
        source_files=src_files,
        test_files=test_files,
        operators=["AOR", "ROR"],
        workspace_dir=tmp_src_dir.parent,
    )


# ===========================================================================
# ResolvedConfig — testes de validação
# ===========================================================================

class TestResolvedConfigValidate:

    def test_should_pass_when_all_files_exist_and_lists_are_non_empty(
        self, valid_resolved_config: ResolvedConfig
    ):
        # Não deve lançar nenhuma exceção
        valid_resolved_config.validate()

    def test_should_raise_value_error_when_source_files_list_is_empty(
        self, tmp_test_dir: Path
    ):
        cfg = ResolvedConfig(
            source_files=[],
            test_files=[tmp_test_dir / "test_a.py"],
            operators=["AOR"],
            workspace_dir=Path("."),
        )
        with pytest.raises(ValueError, match="Nenhum arquivo fonte encontrado"):
            cfg.validate()

    def test_should_raise_value_error_when_test_files_list_is_empty(
        self, tmp_py_file: Path
    ):
        cfg = ResolvedConfig(
            source_files=[tmp_py_file],
            test_files=[],
            operators=["AOR"],
            workspace_dir=Path("."),
        )
        with pytest.raises(ValueError, match="Nenhum arquivo de teste encontrado"):
            cfg.validate()

    def test_should_raise_file_not_found_when_source_file_does_not_exist(
        self, tmp_path: Path, tmp_test_dir: Path
    ):
        ghost = tmp_path / "ghost.py"
        test_file = tmp_test_dir / "test_a.py"
        cfg = ResolvedConfig(
            source_files=[ghost],
            test_files=[test_file],
            operators=["AOR"],
            workspace_dir=tmp_path,
        )
        with pytest.raises(FileNotFoundError, match="ghost.py"):
            cfg.validate()

    def test_should_raise_file_not_found_when_test_file_does_not_exist(
        self, tmp_py_file: Path, tmp_path: Path
    ):
        ghost_test = tmp_path / "ghost_test.py"
        cfg = ResolvedConfig(
            source_files=[tmp_py_file],
            test_files=[ghost_test],
            operators=["AOR"],
            workspace_dir=tmp_path,
        )
        with pytest.raises(FileNotFoundError, match="ghost_test.py"):
            cfg.validate()

    def test_should_report_all_missing_files_in_error_message(
        self, tmp_path: Path
    ):
        cfg = ResolvedConfig(
            source_files=[tmp_path / "a.py", tmp_path / "b.py"],
            test_files=[tmp_path / "t.py"],
            operators=[],
            workspace_dir=tmp_path,
        )
        with pytest.raises(FileNotFoundError) as exc_info:
            cfg.validate()
        msg = str(exc_info.value)
        assert "a.py" in msg
        assert "b.py" in msg
        assert "t.py" in msg


class TestResolvedConfigRepr:

    def test_should_include_counts_operators_and_workspace_in_repr(
        self, valid_resolved_config: ResolvedConfig
    ):
        r = repr(valid_resolved_config)
        assert "ResolvedConfig(" in r
        assert "sources=" in r
        assert "tests=" in r
        assert "operators=" in r
        assert "workspace=" in r

    def test_should_show_correct_file_counts_in_repr(
        self, valid_resolved_config: ResolvedConfig
    ):
        r = repr(valid_resolved_config)
        n_src = len(valid_resolved_config.source_files)
        n_tst = len(valid_resolved_config.test_files)
        assert f"sources={n_src}" in r
        assert f"tests={n_tst}" in r


class TestResolvedConfigDefaults:

    def test_should_initialize_targets_as_empty_list_by_default(
        self, tmp_py_file: Path
    ):
        cfg = ResolvedConfig(
            source_files=[tmp_py_file],
            test_files=[tmp_py_file],
            operators=[],
            workspace_dir=Path("."),
        )
        assert cfg.targets == []

    def test_should_accept_explicit_targets_list(self, tmp_py_file: Path):
        cfg = ResolvedConfig(
            source_files=[tmp_py_file],
            test_files=[tmp_py_file],
            operators=[],
            workspace_dir=Path("."),
            targets=["func_a", "func_b"],
        )
        assert cfg.targets == ["func_a", "func_b"]


# ===========================================================================
# resolve_from_dict — testes
# ===========================================================================

class TestResolveFromDict:

    def test_should_resolve_using_legacy_program_path_key(
        self, tmp_py_file: Path, tmp_path: Path
    ):
        test_file = tmp_path / "test_x.py"
        test_file.write_text("def test_x(): pass\n")
        raw = {
            "program_path": str(tmp_py_file),
            "tests_path": str(test_file),
            "operators_list": "AOR,ROR",
            "workspace_dir": str(tmp_path),
        }
        cfg = resolve_from_dict(raw)
        assert tmp_py_file in cfg.source_files
        assert test_file in cfg.test_files
        assert cfg.operators == ["AOR", "ROR"]

    def test_should_resolve_using_new_source_dirs_key(
        self, tmp_src_dir: Path, tmp_test_dir: Path
    ):
        raw = {
            "source_dirs": str(tmp_src_dir),
            "tests_dirs": str(tmp_test_dir),
            "operators_list": "LCR",
            "workspace_dir": str(tmp_src_dir.parent),
        }
        cfg = resolve_from_dict(raw)
        assert len(cfg.source_files) > 0
        assert len(cfg.test_files) > 0

    def test_should_parse_operators_from_comma_separated_string(
        self, tmp_py_file: Path, tmp_path: Path
    ):
        test_file = tmp_path / "test_ops.py"
        test_file.write_text("")
        raw = {
            "program_path": str(tmp_py_file),
            "tests_path": str(test_file),
            "operators_list": " aor , ror , lcr ",
        }
        cfg = resolve_from_dict(raw)
        assert cfg.operators == ["AOR", "ROR", "LCR"]

    def test_should_use_operators_list_from_raw_dict_when_operators_list_key_is_empty(
        self, tmp_py_file: Path, tmp_path: Path
    ):
        test_file = tmp_path / "test_ops.py"
        test_file.write_text("")
        raw = {
            "program_path": str(tmp_py_file),
            "tests_path": str(test_file),
            "operators_list": "",
            "operators": ["BCR", "SVR"],
        }
        cfg = resolve_from_dict(raw)
        assert cfg.operators == ["BCR", "SVR"]

    def test_should_default_workspace_to_current_dir_when_key_is_absent(
        self, tmp_py_file: Path, tmp_path: Path
    ):
        test_file = tmp_path / "test_ws.py"
        test_file.write_text("")
        raw = {
            "program_path": str(tmp_py_file),
            "tests_path": str(test_file),
        }
        cfg = resolve_from_dict(raw)
        assert cfg.workspace_dir == Path(".")

    def test_should_raise_value_error_when_program_path_does_not_exist(
        self, tmp_path: Path
    ):
        raw = {
            "program_path": str(tmp_path / "nonexistent.py"),
            "tests_path": str(tmp_path / "test_x.py"),
        }
        with pytest.raises(ValueError, match="Caminho inválido"):
            resolve_from_dict(raw)

    def test_should_raise_value_error_when_tests_path_does_not_exist(
        self, tmp_py_file: Path, tmp_path: Path
    ):
        raw = {
            "program_path": str(tmp_py_file),
            "tests_path": str(tmp_path / "nonexistent_test.py"),
        }
        with pytest.raises(ValueError, match="Caminho inválido"):
            resolve_from_dict(raw)

    def test_should_strip_whitespace_from_operators(
        self, tmp_py_file: Path, tmp_path: Path
    ):
        test_file = tmp_path / "test_strip.py"
        test_file.write_text("")
        raw = {
            "program_path": str(tmp_py_file),
            "tests_path": str(test_file),
            "operators_list": "  AOR  ,  ROR  ",
        }
        cfg = resolve_from_dict(raw)
        assert cfg.operators == ["AOR", "ROR"]

    def test_should_produce_empty_operators_list_when_both_keys_absent(
        self, tmp_py_file: Path, tmp_path: Path
    ):
        test_file = tmp_path / "test_no_ops.py"
        test_file.write_text("")
        raw = {
            "program_path": str(tmp_py_file),
            "tests_path": str(test_file),
        }
        cfg = resolve_from_dict(raw)
        assert cfg.operators == []

    def test_should_prefer_program_path_over_source_dirs_when_both_present(
        self, tmp_py_file: Path, tmp_src_dir: Path, tmp_test_dir: Path
    ):
        test_file = next(tmp_test_dir.glob("*.py"))
        raw = {
            "program_path": str(tmp_py_file),
            "source_dirs": str(tmp_src_dir),
            "tests_path": str(test_file),
        }
        cfg = resolve_from_dict(raw)
        # program_path tem prioridade (raw.get usa 'or', então program_path vence se truthy)
        assert cfg.source_files == [tmp_py_file]


# ===========================================================================
# resolve_from_toml — testes
# ===========================================================================

class TestResolveFromToml:

    def _write_toml(self, path: Path, content: str) -> Path:
        toml_file = path / "transmut.toml"
        toml_file.write_bytes(content.encode())
        return toml_file

    def test_should_resolve_config_from_valid_toml_file(
        self, tmp_src_dir: Path, tmp_test_dir: Path, tmp_path: Path
    ):
        toml_content = (
            "[transmut]\n"
            f'source_dirs = ["{tmp_src_dir}"]\n'
            f'tests_dirs  = ["{tmp_test_dir}"]\n'
            'operators   = ["aor", "ror"]\n'
            f'workspace_dir = "{tmp_path}"\n'
        )
        toml_file = self._write_toml(tmp_path, toml_content)
        cfg = resolve_from_toml(toml_file)
        assert len(cfg.source_files) > 0
        assert len(cfg.test_files) > 0
        assert cfg.operators == ["AOR", "ROR"]
        assert cfg.workspace_dir == tmp_path

    def test_should_uppercase_operators_loaded_from_toml(
        self, tmp_src_dir: Path, tmp_test_dir: Path, tmp_path: Path
    ):
        toml_content = (
            "[transmut]\n"
            f'source_dirs = ["{tmp_src_dir}"]\n'
            f'tests_dirs  = ["{tmp_test_dir}"]\n'
            'operators   = ["lcr", "bcr"]\n'
        )
        toml_file = self._write_toml(tmp_path, toml_content)
        cfg = resolve_from_toml(toml_file)
        assert cfg.operators == ["LCR", "BCR"]

    def test_should_default_workspace_to_dot_when_key_absent_in_toml(
        self, tmp_src_dir: Path, tmp_test_dir: Path, tmp_path: Path
    ):
        toml_content = (
            "[transmut]\n"
            f'source_dirs = ["{tmp_src_dir}"]\n'
            f'tests_dirs  = ["{tmp_test_dir}"]\n'
        )
        toml_file = self._write_toml(tmp_path, toml_content)
        cfg = resolve_from_toml(toml_file)
        assert cfg.workspace_dir == Path(".")

    def test_should_raise_import_error_when_tomllib_is_none(
        self, tmp_path: Path
    ):
        toml_file = tmp_path / "transmut.toml"
        toml_file.write_bytes(b"[transmut]\n")
        with patch("resolver.tomllib", None):
            with pytest.raises(ImportError, match="tomli"):
                resolve_from_toml(toml_file)

    def test_should_raise_value_error_when_source_dirs_path_is_invalid_in_toml(
        self, tmp_test_dir: Path, tmp_path: Path
    ):
        toml_content = (
            "[transmut]\n"
            'source_dirs = ["/this/does/not/exist"]\n'
            f'tests_dirs  = ["{tmp_test_dir}"]\n'
        )
        toml_file = self._write_toml(tmp_path, toml_content)
        with pytest.raises(ValueError, match="Caminho inválido"):
            resolve_from_toml(toml_file)

    def test_should_raise_value_error_when_source_dirs_is_empty_in_toml(
        self, tmp_test_dir: Path, tmp_path: Path
    ):
        toml_content = (
            "[transmut]\n"
            'source_dirs = []\n'
            f'tests_dirs  = ["{tmp_test_dir}"]\n'
        )
        toml_file = self._write_toml(tmp_path, toml_content)
        with pytest.raises(ValueError, match="Nenhum arquivo fonte"):
            resolve_from_toml(toml_file)

    def test_should_resolve_transmut_section_only_ignoring_other_sections(
        self, tmp_src_dir: Path, tmp_test_dir: Path, tmp_path: Path
    ):
        toml_content = (
            "[other_section]\n"
            'key = "value"\n'
            "[transmut]\n"
            f'source_dirs = ["{tmp_src_dir}"]\n'
            f'tests_dirs  = ["{tmp_test_dir}"]\n'
        )
        toml_file = self._write_toml(tmp_path, toml_content)
        cfg = resolve_from_toml(toml_file)
        assert len(cfg.source_files) > 0


# ===========================================================================
# _resolve_entry — testes
# ===========================================================================

class TestResolveEntry:

    def test_should_return_empty_list_when_entry_is_none(self):
        assert _resolve_entry(None) == []

    def test_should_return_empty_list_when_entry_is_empty_string(self):
        assert _resolve_entry("") == []

    def test_should_return_empty_list_when_entry_is_empty_list(self):
        assert _resolve_entry([]) == []

    def test_should_return_single_file_when_entry_is_path_to_py_file(
        self, tmp_py_file: Path
    ):
        result = _resolve_entry(str(tmp_py_file))
        assert result == [tmp_py_file]

    def test_should_discover_all_py_files_when_entry_is_directory_path(
        self, tmp_src_dir: Path
    ):
        result = _resolve_entry(str(tmp_src_dir))
        assert len(result) >= 3  # module_a, module_b, deep
        assert all(p.suffix == ".py" for p in result)

    def test_should_handle_list_of_single_file_entries(self, tmp_py_file: Path):
        result = _resolve_entry([str(tmp_py_file)])
        assert result == [tmp_py_file]

    def test_should_handle_list_of_directory_entries(
        self, tmp_src_dir: Path, tmp_test_dir: Path
    ):
        result = _resolve_entry([str(tmp_src_dir), str(tmp_test_dir)])
        assert len(result) >= 5  # 3 src + 2 test

    def test_should_handle_list_mixing_files_and_directories(
        self, tmp_py_file: Path, tmp_src_dir: Path
    ):
        result = _resolve_entry([str(tmp_py_file), str(tmp_src_dir)])
        assert tmp_py_file in result
        assert len(result) > 1

    def test_should_raise_value_error_when_path_does_not_exist(self, tmp_path: Path):
        ghost = tmp_path / "ghost_dir"
        with pytest.raises(ValueError, match="Caminho inválido"):
            _resolve_entry(str(ghost))

    def test_should_raise_value_error_when_file_has_non_py_extension(
        self, tmp_path: Path
    ):
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="Caminho inválido"):
            _resolve_entry(str(txt_file))

    def test_should_strip_whitespace_from_entry_string(self, tmp_py_file: Path):
        result = _resolve_entry(f"  {tmp_py_file}  ")
        assert result == [tmp_py_file]


# ===========================================================================
# _discover_py — testes
# ===========================================================================

class TestDiscoverPy:

    def test_should_find_all_py_files_in_flat_directory(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        result = _discover_py(tmp_path)
        names = {f.name for f in result}
        assert names == {"a.py", "b.py"}

    def test_should_find_py_files_recursively_in_subdirectories(
        self, tmp_path: Path
    ):
        (tmp_path / "top.py").write_text("")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("")
        result = _discover_py(tmp_path)
        names = {f.name for f in result}
        assert "top.py" in names
        assert "nested.py" in names

    def test_should_ignore_init_py_file(self, tmp_path: Path):
        (tmp_path / "__init__.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any(f.name == "__init__.py" for f in result)

    def test_should_ignore_conftest_py_file(self, tmp_path: Path):
        (tmp_path / "conftest.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any(f.name == "conftest.py" for f in result)

    def test_should_ignore_setup_py_file(self, tmp_path: Path):
        (tmp_path / "setup.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any(f.name == "setup.py" for f in result)

    def test_should_ignore_config_py_file(self, tmp_path: Path):
        (tmp_path / "config.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any(f.name == "config.py" for f in result)

    def test_should_ignore_settings_py_file(self, tmp_path: Path):
        (tmp_path / "settings.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any(f.name == "settings.py" for f in result)

    def test_should_ignore_files_inside_pycache_directory(self, tmp_path: Path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "cached.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any("__pycache__" in str(f) for f in result)

    def test_should_ignore_files_inside_venv_directory(self, tmp_path: Path):
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        (venv_dir / "venv_module.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any("venv" in str(f) for f in result)

    def test_should_ignore_files_inside_dot_venv_directory(self, tmp_path: Path):
        dot_venv = tmp_path / ".venv"
        dot_venv.mkdir()
        (dot_venv / "venv_mod.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any(".venv" in str(f) for f in result)

    def test_should_ignore_files_inside_node_modules_directory(self, tmp_path: Path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "script.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any("node_modules" in str(f) for f in result)

    def test_should_ignore_files_inside_git_directory(self, tmp_path: Path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "hook.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any(".git" in str(f) for f in result)

    def test_should_ignore_files_inside_dist_directory(self, tmp_path: Path):
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "pkg.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any("dist" in str(f) for f in result)

    def test_should_ignore_files_inside_build_directory(self, tmp_path: Path):
        build = tmp_path / "build"
        build.mkdir()
        (build / "artifact.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any("build" in str(f) for f in result)

    def test_should_ignore_hidden_files_with_dot_prefix(self, tmp_path: Path):
        (tmp_path / ".hidden.py").write_text("")
        (tmp_path / "visible.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any(f.name.startswith(".") for f in result)

    def test_should_return_empty_list_when_directory_has_no_eligible_py_files(
        self, tmp_path: Path
    ):
        (tmp_path / "__init__.py").write_text("")
        (tmp_path / "conftest.py").write_text("")
        result = _discover_py(tmp_path)
        assert result == []

    def test_should_return_sorted_results(self, tmp_path: Path):
        (tmp_path / "z_mod.py").write_text("")
        (tmp_path / "a_mod.py").write_text("")
        (tmp_path / "m_mod.py").write_text("")
        result = _discover_py(tmp_path)
        names = [f.name for f in result]
        assert names == sorted(names)

    def test_should_ignore_tox_directory(self, tmp_path: Path):
        tox = tmp_path / ".tox"
        tox.mkdir()
        (tox / "tox_run.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any(".tox" in str(f) for f in result)

    def test_should_not_include_non_py_files(self, tmp_path: Path):
        (tmp_path / "readme.md").write_text("")
        (tmp_path / "data.json").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert all(f.suffix == ".py" for f in result)
        assert len(result) == 1

    def test_should_ignore_env_directory(self, tmp_path: Path):
        env_dir = tmp_path / ".env"
        env_dir.mkdir()
        (env_dir / "env_mod.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = _discover_py(tmp_path)
        assert not any(".env" in str(f) for f in result)


# ===========================================================================
# Integração: resolve_from_dict → validate completo
# ===========================================================================

class TestResolveFromDictIntegration:

    def test_should_return_valid_resolved_config_with_all_fields_populated(
        self, tmp_src_dir: Path, tmp_test_dir: Path
    ):
        raw = {
            "source_dirs": str(tmp_src_dir),
            "tests_dirs": str(tmp_test_dir),
            "operators_list": "AOR,ROR,LCR",
            "workspace_dir": str(tmp_src_dir.parent),
        }
        cfg = resolve_from_dict(raw)
        assert isinstance(cfg, ResolvedConfig)
        assert all(isinstance(f, Path) for f in cfg.source_files)
        assert all(isinstance(f, Path) for f in cfg.test_files)
        assert cfg.operators == ["AOR", "ROR", "LCR"]
        assert isinstance(cfg.workspace_dir, Path)

    def test_should_return_config_that_passes_validation(
        self, tmp_src_dir: Path, tmp_test_dir: Path
    ):
        raw = {
            "source_dirs": str(tmp_src_dir),
            "tests_dirs": str(tmp_test_dir),
            "operators": ["BCR"],
        }
        cfg = resolve_from_dict(raw)
        # validate() não deve lançar exceção
        cfg.validate()

    def test_should_resolve_multiple_source_dirs_as_list(
        self, tmp_path: Path
    ):
        src1 = tmp_path / "src1"
        src1.mkdir()
        (src1 / "mod1.py").write_text("")
        src2 = tmp_path / "src2"
        src2.mkdir()
        (src2 / "mod2.py").write_text("")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_all.py").write_text("")

        raw = {
            "source_dirs": [str(src1), str(src2)],
            "tests_dirs": str(tests),
            "operators": ["AOR"],
        }
        cfg = resolve_from_dict(raw)
        names = {f.name for f in cfg.source_files}
        assert "mod1.py" in names
        assert "mod2.py" in names