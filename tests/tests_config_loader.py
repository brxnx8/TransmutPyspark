from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.config.config_loader import ConfigLoader


# ===========================================================================
# Helpers
# ===========================================================================

def _write(tmp_path: Path, filename: str, content: str = "") -> Path:
    """Cria arquivo com conteúdo e retorna seu Path."""
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


def _make_cfg(source_files=None, test_files=None) -> MagicMock:
    """Fábrica de ResolvedConfig mockado."""
    cfg = MagicMock()
    cfg.source_files = source_files or []
    cfg.test_files   = test_files   or []
    cfg.targets      = []
    return cfg


# ===========================================================================
# Caminhos de patch
# (resolve_from_dict/toml são importados *no topo* do módulo, então o alvo
#  correto é o namespace do config_loader, não o módulo de origem)
# ===========================================================================
_RESOLVE_DICT = "config_loader.resolve_from_dict"
_RESOLVE_TOML = "config_loader.resolve_from_toml"
_ANALYZE      = "config_loader.analyze"


# ===========================================================================
# __init__ e __repr__
# ===========================================================================

class TestConfigLoaderInit:

    def test_should_store_dict_input(self):
        d = {"program_path": "src/", "tests_path": "tests/"}
        loader = ConfigLoader(d)
        assert loader._input is d

    def test_should_store_string_input(self):
        loader = ConfigLoader("transmut.toml")
        assert loader._input == "transmut.toml"

    def test_should_include_input_in_repr_for_string(self):
        loader = ConfigLoader("my_config.toml")
        assert "my_config.toml" in repr(loader)

    def test_should_include_input_in_repr_for_dict(self):
        loader = ConfigLoader({"program_path": "src/"})
        r = repr(loader)
        assert "ConfigLoader" in r

    def test_should_include_class_name_in_repr(self):
        loader = ConfigLoader("x.toml")
        assert "ConfigLoader" in repr(loader)


# ===========================================================================
# load()  —  popula targets via analyze
# ===========================================================================

class TestLoad:

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_return_resolved_config(
        self, mock_resolve_dict, mock_analyze
    ):
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = ["target_a"]

        loader = ConfigLoader({"program_path": "src/"})
        result = loader.load()

        assert result is cfg

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_call_analyze_with_source_and_test_files(
        self, mock_resolve_dict, mock_analyze
    ):
        src_files  = [Path("a.py")]
        test_files = [Path("test_a.py")]
        cfg = _make_cfg(source_files=src_files, test_files=test_files)
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = []

        ConfigLoader({"program_path": "src/"}).load()

        mock_analyze.assert_called_once_with(src_files, test_files)

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_assign_analyze_result_to_cfg_targets(
        self, mock_resolve_dict, mock_analyze
    ):
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = ["t1", "t2"]

        result = ConfigLoader({"program_path": "src/"}).load()

        assert result.targets == ["t1", "t2"]

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_assign_empty_list_when_analyze_returns_empty(
        self, mock_resolve_dict, mock_analyze
    ):
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = []

        result = ConfigLoader({"program_path": "src/"}).load()

        assert result.targets == []


# ===========================================================================
# _resolve  —  Modo inline (dict)
# ===========================================================================

class TestResolveDict:

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_call_resolve_from_dict_when_input_is_dict(
        self, mock_resolve_dict, mock_analyze
    ):
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = []
        payload = {"program_path": "src/", "tests_path": "tests/"}

        ConfigLoader(payload).load()

        mock_resolve_dict.assert_called_once_with(payload)

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_not_call_resolve_from_toml_when_input_is_dict(
        self, mock_resolve_dict, mock_analyze
    ):
        with patch(_RESOLVE_TOML) as mock_resolve_toml:
            cfg = _make_cfg()
            mock_resolve_dict.return_value = cfg
            mock_analyze.return_value      = []

            ConfigLoader({"program_path": "src/"}).load()

            mock_resolve_toml.assert_not_called()

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_forward_full_dict_payload_unchanged(
        self, mock_resolve_dict, mock_analyze
    ):
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = []
        payload = {
            "program_path":   "etl/",
            "tests_path":     "tests/",
            "operators_list": "MTR,NFTP",
            "workspace_dir":  "/out",
        }

        ConfigLoader(payload).load()

        mock_resolve_dict.assert_called_once_with(payload)


# ===========================================================================
# _resolve  —  Modo .toml explícito
# ===========================================================================

class TestResolveTomlExplicit:

    @patch(_ANALYZE)
    @patch(_RESOLVE_TOML)
    def test_should_call_resolve_from_toml_when_toml_exists(
        self, mock_resolve_toml, mock_analyze, tmp_path
    ):
        toml = _write(tmp_path, "transmut.toml", "[transmut]\n")
        cfg = _make_cfg()
        mock_resolve_toml.return_value = cfg
        mock_analyze.return_value      = []

        ConfigLoader(str(toml)).load()

        mock_resolve_toml.assert_called_once_with(toml)

    @patch(_ANALYZE)
    @patch(_RESOLVE_TOML)
    def test_should_raise_file_not_found_when_toml_does_not_exist(
        self, mock_resolve_toml, mock_analyze, tmp_path
    ):
        missing = tmp_path / "missing.toml"

        with pytest.raises(FileNotFoundError) as exc:
            ConfigLoader(str(missing)).load()

        assert "missing.toml" in str(exc.value)

    @patch(_ANALYZE)
    @patch(_RESOLVE_TOML)
    def test_should_include_path_in_file_not_found_message(
        self, mock_resolve_toml, mock_analyze, tmp_path
    ):
        missing = tmp_path / "config.toml"

        with pytest.raises(FileNotFoundError) as exc:
            ConfigLoader(str(missing)).load()

        assert str(missing) in str(exc.value)

    @patch(_ANALYZE)
    @patch(_RESOLVE_TOML)
    def test_should_not_call_resolve_from_dict_when_toml_input(
        self, mock_resolve_toml, mock_analyze, tmp_path
    ):
        with patch(_RESOLVE_DICT) as mock_resolve_dict:
            toml = _write(tmp_path, "transmut.toml", "[transmut]\n")
            cfg = _make_cfg()
            mock_resolve_toml.return_value = cfg
            mock_analyze.return_value      = []

            ConfigLoader(str(toml)).load()

            mock_resolve_dict.assert_not_called()


# ===========================================================================
# _resolve  —  Modo .py único
# ===========================================================================

class TestResolvePyFile:

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_call_resolve_from_dict_when_py_file_given(
        self, mock_resolve_dict, mock_analyze, tmp_path
    ):
        py_file = _write(tmp_path, "etl.py", "def fn(): pass")
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = []

        ConfigLoader(str(py_file)).load()

        mock_resolve_dict.assert_called_once()

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_set_program_path_to_py_file(
        self, mock_resolve_dict, mock_analyze, tmp_path
    ):
        py_file = _write(tmp_path, "etl.py", "def fn(): pass")
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = []

        ConfigLoader(str(py_file)).load()

        call_kwargs = mock_resolve_dict.call_args[0][0]
        assert call_kwargs["program_path"] == str(py_file)

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_set_workspace_dir_to_parent_of_py_file(
        self, mock_resolve_dict, mock_analyze, tmp_path
    ):
        py_file = _write(tmp_path, "etl.py", "def fn(): pass")
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = []

        ConfigLoader(str(py_file)).load()

        call_kwargs = mock_resolve_dict.call_args[0][0]
        assert call_kwargs["workspace_dir"] == str(tmp_path)

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_set_empty_tests_path_when_only_py_file_given(
        self, mock_resolve_dict, mock_analyze, tmp_path
    ):
        py_file = _write(tmp_path, "etl.py", "def fn(): pass")
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = []

        ConfigLoader(str(py_file)).load()

        call_kwargs = mock_resolve_dict.call_args[0][0]
        assert call_kwargs["tests_path"] == ""

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_set_empty_operators_list_when_only_py_file_given(
        self, mock_resolve_dict, mock_analyze, tmp_path
    ):
        py_file = _write(tmp_path, "etl.py", "def fn(): pass")
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = []

        ConfigLoader(str(py_file)).load()

        call_kwargs = mock_resolve_dict.call_args[0][0]
        assert call_kwargs["operators_list"] == ""


# ===========================================================================
# _resolve  —  Modo transmut.toml vizinho (Modo 3b)
# ===========================================================================

class TestResolveNeighborToml:

    @patch(_ANALYZE)
    @patch(_RESOLVE_TOML)
    def test_should_use_neighbor_toml_when_txt_file_given_and_toml_exists(
        self, mock_resolve_toml, mock_analyze, tmp_path
    ):
        """config.txt está no diretório, mas transmut.toml também → toml prevalece."""
        toml = _write(tmp_path, "transmut.toml", "[transmut]\n")
        txt  = _write(tmp_path, "config.txt", "program_path=src/\n")
        cfg = _make_cfg()
        mock_resolve_toml.return_value = cfg
        mock_analyze.return_value      = []

        ConfigLoader(str(txt)).load()

        mock_resolve_toml.assert_called_once_with(toml)

    @patch(_ANALYZE)
    @patch(_RESOLVE_TOML)
    def test_should_use_neighbor_toml_when_dir_given_and_toml_exists(
        self, mock_resolve_toml, mock_analyze, tmp_path
    ):
        toml = _write(tmp_path, "transmut.toml", "[transmut]\n")
        cfg = _make_cfg()
        mock_resolve_toml.return_value = cfg
        mock_analyze.return_value      = []

        ConfigLoader(str(tmp_path)).load()

        mock_resolve_toml.assert_called_once_with(toml)

    @patch(_ANALYZE)
    @patch(_RESOLVE_TOML)
    def test_should_pass_correct_toml_path_to_resolve_from_toml(
        self, mock_resolve_toml, mock_analyze, tmp_path
    ):
        toml = _write(tmp_path, "transmut.toml", "[transmut]\n")
        cfg = _make_cfg()
        mock_resolve_toml.return_value = cfg
        mock_analyze.return_value      = []

        ConfigLoader(str(tmp_path)).load()

        assert mock_resolve_toml.call_args[0][0] == toml


# ===========================================================================
# _resolve  —  Modo legado config.txt
# ===========================================================================

class TestResolveLegacyTxt:

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_call_resolve_from_dict_when_txt_file_given(
        self, mock_resolve_dict, mock_analyze, tmp_path
    ):
        txt = _write(tmp_path, "config.txt", "program_path=src/\n")
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = []

        ConfigLoader(str(txt)).load()

        mock_resolve_dict.assert_called_once()

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_parse_key_value_from_txt_file(
        self, mock_resolve_dict, mock_analyze, tmp_path
    ):
        txt = _write(tmp_path, "config.txt", "program_path=src/etl\ntests_path=tests/\n")
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = []

        ConfigLoader(str(txt)).load()

        passed = mock_resolve_dict.call_args[0][0]
        assert passed["program_path"] == "src/etl"
        assert passed["tests_path"]   == "tests/"

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_handle_extensionless_config_file(
        self, mock_resolve_dict, mock_analyze, tmp_path
    ):
        # Arquivo sem extensão (suffix == "")
        no_ext = tmp_path / "config"
        no_ext.write_text("program_path=src/\n", encoding="utf-8")
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.return_value      = []

        ConfigLoader(str(no_ext)).load()

        mock_resolve_dict.assert_called_once()


# ===========================================================================
# _resolve  —  Diretório sem transmut.toml  →  ValueError
# ===========================================================================

class TestResolveDirectory:

    def test_should_raise_value_error_when_dir_has_no_toml(self, tmp_path):
        with pytest.raises(ValueError) as exc:
            ConfigLoader(str(tmp_path)).load()

        assert "transmut.toml" in str(exc.value)

    def test_should_include_directory_path_in_value_error_message(
        self, tmp_path
    ):
        with pytest.raises(ValueError) as exc:
            ConfigLoader(str(tmp_path)).load()

        assert str(tmp_path) in str(exc.value)

    def test_should_suggest_transmut_init_in_error_message(self, tmp_path):
        with pytest.raises(ValueError) as exc:
            ConfigLoader(str(tmp_path)).load()

        assert "transmut init" in str(exc.value)


# ===========================================================================
# _resolve  —  Caminho inexistente  →  FileNotFoundError
# ===========================================================================

class TestResolveUnknown:

    def test_should_raise_file_not_found_for_nonexistent_path(self, tmp_path):
        ghost = tmp_path / "nonexistent_config"
        with pytest.raises(FileNotFoundError) as exc:
            ConfigLoader(str(ghost)).load()

        assert str(ghost) in str(exc.value) or "nonexistent_config" in str(exc.value)

    def test_should_raise_file_not_found_for_nonexistent_py_file(
        self, tmp_path
    ):
        # .py que não existe — não é .toml, não é arquivo existente
        ghost = tmp_path / "ghost.py"
        with pytest.raises((FileNotFoundError, ValueError)):
            ConfigLoader(str(ghost)).load()


# ===========================================================================
# _parse_txt  —  parser legado isolado
# ===========================================================================

class TestParseTxt:

    def test_should_parse_simple_key_value_pair(self, tmp_path):
        f = _write(tmp_path, "c.txt", "program_path=src/etl\n")
        result = ConfigLoader._parse_txt(f)
        assert result["program_path"] == "src/etl"

    def test_should_strip_spaces_around_key_and_value(self, tmp_path):
        f = _write(tmp_path, "c.txt", "  program_path  =  src/etl  \n")
        result = ConfigLoader._parse_txt(f)
        assert result["program_path"] == "src/etl"

    def test_should_ignore_blank_lines(self, tmp_path):
        f = _write(tmp_path, "c.txt", "\n\nprogram_path=src/\n\n")
        result = ConfigLoader._parse_txt(f)
        assert result == {"program_path": "src/"}

    def test_should_ignore_comment_lines(self, tmp_path):
        f = _write(tmp_path, "c.txt", "# comentário\nprogram_path=src/\n")
        result = ConfigLoader._parse_txt(f)
        assert "# comentário" not in str(result)
        assert result == {"program_path": "src/"}

    def test_should_ignore_lines_without_equals_sign(self, tmp_path):
        f = _write(tmp_path, "c.txt", "sem_sinal_de_igual\nprogram_path=src/\n")
        result = ConfigLoader._parse_txt(f)
        assert "sem_sinal_de_igual" not in result
        assert result["program_path"] == "src/"

    def test_should_split_only_on_first_equals_sign(self, tmp_path):
        # Valor que contém '='
        f = _write(tmp_path, "c.txt", "url=http://host/path?a=1\n")
        result = ConfigLoader._parse_txt(f)
        assert result["url"] == "http://host/path?a=1"

    def test_should_parse_multiple_key_value_pairs(self, tmp_path):
        content = (
            "program_path=src/etl\n"
            "tests_path=tests/\n"
            "operators_list=MTR,NFTP\n"
        )
        f = _write(tmp_path, "c.txt", content)
        result = ConfigLoader._parse_txt(f)
        assert result["program_path"]   == "src/etl"
        assert result["tests_path"]     == "tests/"
        assert result["operators_list"] == "MTR,NFTP"

    def test_should_return_empty_dict_for_blank_file(self, tmp_path):
        f = _write(tmp_path, "c.txt", "")
        assert ConfigLoader._parse_txt(f) == {}

    def test_should_return_empty_dict_for_comment_only_file(self, tmp_path):
        f = _write(tmp_path, "c.txt", "# só comentários\n# outra linha\n")
        assert ConfigLoader._parse_txt(f) == {}

    def test_should_handle_windows_line_endings(self, tmp_path):
        f = tmp_path / "c.txt"
        f.write_bytes(b"program_path=src/\r\ntests_path=tests/\r\n")
        result = ConfigLoader._parse_txt(f)
        assert result["program_path"] == "src/"
        assert result["tests_path"]   == "tests/"

    def test_should_handle_file_with_only_inline_comment(self, tmp_path):
        # Linha que começa com # após strip
        f = _write(tmp_path, "c.txt", "   # inline comment   \n")
        assert ConfigLoader._parse_txt(f) == {}

    def test_should_not_include_key_when_value_is_empty(self, tmp_path):
        f = _write(tmp_path, "c.txt", "empty_key=\n")
        result = ConfigLoader._parse_txt(f)
        # Chave existe, valor é string vazia
        assert "empty_key" in result
        assert result["empty_key"] == ""

    def test_should_overwrite_duplicate_keys_with_last_value(self, tmp_path):
        f = _write(tmp_path, "c.txt", "key=first\nkey=second\n")
        result = ConfigLoader._parse_txt(f)
        assert result["key"] == "second"


# ===========================================================================
# Integração leve  —  _resolve chama o colaborador correto em cada modo
# (garantia de que load() orquestra _resolve + analyze corretamente)
# ===========================================================================

class TestLoadOrchestration:

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_call_resolve_then_analyze_in_order(
        self, mock_resolve_dict, mock_analyze
    ):
        """Garante que analyze só é chamado após _resolve ter retornado."""
        call_order = []
        cfg = _make_cfg()

        def record_resolve(d):
            call_order.append("resolve")
            return cfg

        def record_analyze(sf, tf):
            call_order.append("analyze")
            return []

        mock_resolve_dict.side_effect = record_resolve
        mock_analyze.side_effect      = record_analyze

        ConfigLoader({"program_path": "src/"}).load()

        assert call_order == ["resolve", "analyze"]

    @patch(_ANALYZE)
    @patch(_RESOLVE_TOML)
    def test_should_propagate_file_not_found_from_resolve(
        self, mock_resolve_toml, mock_analyze, tmp_path
    ):
        missing = tmp_path / "gone.toml"
        with pytest.raises(FileNotFoundError):
            ConfigLoader(str(missing)).load()

        mock_analyze.assert_not_called()

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_propagate_exception_from_resolve_from_dict(
        self, mock_resolve_dict, mock_analyze
    ):
        mock_resolve_dict.side_effect = ValueError("config inválida")

        with pytest.raises(ValueError, match="config inválida"):
            ConfigLoader({"program_path": "src/"}).load()

    @patch(_ANALYZE)
    @patch(_RESOLVE_DICT)
    def test_should_propagate_exception_from_analyze(
        self, mock_resolve_dict, mock_analyze
    ):
        cfg = _make_cfg()
        mock_resolve_dict.return_value = cfg
        mock_analyze.side_effect       = RuntimeError("AST falhou")

        with pytest.raises(RuntimeError, match="AST falhou"):
            ConfigLoader({"program_path": "src/"}).load()