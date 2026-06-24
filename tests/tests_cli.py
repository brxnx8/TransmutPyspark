from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Importações do módulo sob teste
# ---------------------------------------------------------------------------
import src.cli.cli
from src.cli.cli import (
    _cmd_init,
    _cmd_run,
    _cmd_show,
    _die,
    _print_banner,
    _resolve_config_input,
    _setup_logging,
    build_parser,
    main,
)


# ===========================================================================
# Fixtures compartilhadas
# ===========================================================================

@pytest.fixture()
def parser() -> argparse.ArgumentParser:
    """Retorna uma instância fresca do parser CLI."""
    return build_parser()


@pytest.fixture()
def run_args_src_tests() -> argparse.Namespace:
    """Namespace simulando `transmut run --src src/ --tests tests/`."""
    return argparse.Namespace(
        command="run",
        src="src/",
        tests="tests/",
        config=None,
        operators=["MTR", "NFTP", "ATR", "UTS"],
        output=".",
        workers=4,
        verbose=False,
    )


@pytest.fixture()
def run_args_config() -> argparse.Namespace:
    """Namespace simulando `transmut run --config transmut.toml`."""
    return argparse.Namespace(
        command="run",
        src=None,
        tests=None,
        config="transmut.toml",
        operators=["MTR", "NFTP", "ATR", "UTS"],
        output=".",
        workers=4,
        verbose=False,
    )


@pytest.fixture()
def run_args_empty() -> argparse.Namespace:
    """Namespace sem nenhuma fonte de configuração."""
    return argparse.Namespace(
        command="run",
        src=None,
        tests=None,
        config=None,
        operators=["MTR", "NFTP", "ATR", "UTS"],
        output=".",
        workers=4,
        verbose=False,
    )


@pytest.fixture()
def init_args() -> argparse.Namespace:
    """Namespace simulando `transmut init`."""
    return argparse.Namespace(
        command="init",
        src="src/",
        tests="tests/",
        output=".",
    )


@pytest.fixture()
def mock_manager() -> MagicMock:
    """MutationManager mockado com 5 mutantes (3 killed, 2 survived)."""
    manager = MagicMock()
    manager.work_dir = "/tmp/TransmutPysparkOutput"

    killed_result   = Mock(status="killed")
    survived_result = Mock(status="survived")
    manager.mutant_list  = [object()] * 5
    manager.result_list  = [killed_result] * 3 + [survived_result] * 2
    return manager


# ===========================================================================
# build_parser
# ===========================================================================

class TestBuildParser:

    def test_should_return_argparse_instance_when_called(self, parser):
        assert isinstance(parser, argparse.ArgumentParser)

    def test_should_set_prog_name_to_transmut(self, parser):
        assert parser.prog == "transmut"

    # ── transmut run ──────────────────────────────────────────────────

    def test_should_parse_run_with_src_and_tests_flags(self, parser):
        args = parser.parse_args(["run", "--src", "etl/", "--tests", "tests/"])
        assert args.command == "run"
        assert args.src     == "etl/"
        assert args.tests   == "tests/"

    def test_should_use_default_operators_when_not_specified(self, parser):
        args = parser.parse_args(["run", "--src", "s/", "--tests", "t/"])
        assert args.operators == ["MTR", "NFTP", "ATR", "UTS"]

    def test_should_parse_custom_operators_when_provided(self, parser):
        args = parser.parse_args(
            ["run", "--src", "s/", "--tests", "t/", "--operators", "MTR", "ATR"]
        )
        assert args.operators == ["MTR", "ATR"]

    def test_should_use_default_output_dot_when_not_specified(self, parser):
        args = parser.parse_args(["run", "--src", "s/", "--tests", "t/"])
        assert args.output == "."

    def test_should_parse_custom_output_dir(self, parser):
        args = parser.parse_args(
            ["run", "--src", "s/", "--tests", "t/", "--output", "/tmp/out"]
        )
        assert args.output == "/tmp/out"

    def test_should_use_default_workers_4_when_not_specified(self, parser):
        args = parser.parse_args(["run", "--src", "s/", "--tests", "t/"])
        assert args.workers == 4

    def test_should_parse_custom_workers(self, parser):
        args = parser.parse_args(
            ["run", "--src", "s/", "--tests", "t/", "--workers", "8"]
        )
        assert args.workers == 8

    def test_should_set_verbose_false_by_default(self, parser):
        args = parser.parse_args(["run", "--src", "s/", "--tests", "t/"])
        assert args.verbose is False

    def test_should_set_verbose_true_when_flag_present(self, parser):
        args = parser.parse_args(["run", "--src", "s/", "--tests", "t/", "--verbose"])
        assert args.verbose is True

    def test_should_accept_short_verbose_flag(self, parser):
        args = parser.parse_args(["run", "--src", "s/", "--tests", "t/", "-v"])
        assert args.verbose is True

    def test_should_parse_run_with_config_flag(self, parser):
        args = parser.parse_args(["run", "--config", "my_config.toml"])
        assert args.config == "my_config.toml"
        assert args.src    is None
        assert args.tests  is None

    # ── transmut init ─────────────────────────────────────────────────

    def test_should_parse_init_command(self, parser):
        args = parser.parse_args(["init"])
        assert args.command == "init"

    def test_should_use_default_src_in_init(self, parser):
        args = parser.parse_args(["init"])
        assert args.src == "src/"

    def test_should_use_default_tests_in_init(self, parser):
        args = parser.parse_args(["init"])
        assert args.tests == "tests/"

    def test_should_use_default_output_in_init(self, parser):
        args = parser.parse_args(["init"])
        assert args.output == "."

    def test_should_parse_custom_src_in_init(self, parser):
        args = parser.parse_args(["init", "--src", "etl/"])
        assert args.src == "etl/"

    # ── transmut show ─────────────────────────────────────────────────

    def test_should_parse_show_command(self, parser):
        args = parser.parse_args(["show"])
        assert args.command == "show"

    # ── erros ─────────────────────────────────────────────────────────

    def test_should_exit_when_no_command_provided(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_should_exit_when_unknown_command_provided(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["unknown"])


# ===========================================================================
# _resolve_config_input
# ===========================================================================

class TestResolveConfigInput:

    def test_should_return_dict_when_src_and_tests_provided(
        self, run_args_src_tests
    ):
        result = _resolve_config_input(run_args_src_tests)
        assert isinstance(result, dict)
        assert result["program_path"]   == "src/"
        assert result["tests_path"]     == "tests/"
        assert result["workspace_dir"]  == "."

    def test_should_join_operators_as_csv_when_src_and_tests_provided(
        self, run_args_src_tests
    ):
        result = _resolve_config_input(run_args_src_tests)
        assert result["operators_list"] == "MTR,NFTP,ATR,UTS"

    def test_should_include_custom_operators_in_dict(self):
        args = argparse.Namespace(
            src="s/", tests="t/",
            config=None, operators=["MTR", "ATR"],
            output="/out", workers=4, verbose=False,
        )
        result = _resolve_config_input(args)
        assert result["operators_list"] == "MTR,ATR"

    def test_should_return_config_path_string_when_explicit_config_exists(
        self, run_args_config, tmp_path
    ):
        config_file = tmp_path / "transmut.toml"
        config_file.write_text("[transmut]\n")
        run_args_config.config = str(config_file)

        result = _resolve_config_input(run_args_config)
        assert result == str(config_file)

    def test_should_die_when_explicit_config_file_does_not_exist(
        self, run_args_config
    ):
        run_args_config.config = "/nonexistent/path/config.toml"
        with pytest.raises(SystemExit):
            _resolve_config_input(run_args_config)

    def test_should_return_transmut_toml_when_present_in_cwd(
        self, run_args_empty, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "transmut.toml").write_text("[transmut]\n")

        result = _resolve_config_input(run_args_empty)
        assert result == "transmut.toml"

    def test_should_return_config_txt_when_only_legacy_file_present(
        self, run_args_empty, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.txt").write_text("source_dirs=src/\n")

        result = _resolve_config_input(run_args_empty)
        assert result == "config.txt"

    def test_should_prefer_transmut_toml_over_config_txt(
        self, run_args_empty, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "transmut.toml").write_text("[transmut]\n")
        (tmp_path / "config.txt").write_text("source_dirs=src/\n")

        result = _resolve_config_input(run_args_empty)
        assert result == "transmut.toml"

    def test_should_die_when_no_configuration_source_found(
        self, run_args_empty, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)  # diretório vazio — sem configs
        with pytest.raises(SystemExit):
            _resolve_config_input(run_args_empty)

    def test_should_prefer_src_tests_flags_over_config_file(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "transmut.toml").write_text("[transmut]\n")
        args = argparse.Namespace(
            src="my_src/", tests="my_tests/",
            config="transmut.toml",
            operators=["MTR"],
            output=".", workers=4, verbose=False,
        )
        result = _resolve_config_input(args)
        assert isinstance(result, dict)
        assert result["program_path"] == "my_src/"


# ===========================================================================
# _cmd_init
# ===========================================================================

class TestCmdInit:

    def test_should_create_transmut_toml_when_not_exists(
        self, init_args, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        _cmd_init(init_args)
        assert (tmp_path / "transmut.toml").exists()

    def test_should_write_correct_source_dir_in_toml(
        self, init_args, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        _cmd_init(init_args)
        content = (tmp_path / "transmut.toml").read_text()
        assert 'source_dirs   = ["src/"]' in content

    def test_should_write_correct_tests_dir_in_toml(
        self, init_args, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        _cmd_init(init_args)
        content = (tmp_path / "transmut.toml").read_text()
        assert 'tests_dirs    = ["tests/"]' in content

    def test_should_write_default_operators_in_toml(
        self, init_args, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        _cmd_init(init_args)
        content = (tmp_path / "transmut.toml").read_text()
        assert 'operators     = ["MTR", "NFTP", "ATR", "UTS"]' in content

    def test_should_write_workspace_dir_in_toml(
        self, init_args, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        _cmd_init(init_args)
        content = (tmp_path / "transmut.toml").read_text()
        assert 'workspace_dir = "."' in content

    def test_should_print_success_message_when_toml_created(
        self, init_args, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        _cmd_init(init_args)
        out = capsys.readouterr().out
        assert "transmut.toml criado" in out

    def test_should_exit_when_transmut_toml_already_exists(
        self, init_args, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "transmut.toml").write_text("[transmut]\n")
        with pytest.raises(SystemExit) as exc:
            _cmd_init(init_args)
        assert exc.value.code == 1

    def test_should_print_warning_when_toml_already_exists(
        self, init_args, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "transmut.toml").write_text("[transmut]\n")
        with pytest.raises(SystemExit):
            _cmd_init(init_args)
        out = capsys.readouterr().out
        assert "já existe" in out

    def test_should_include_optional_exclude_comment_in_toml(
        self, init_args, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        _cmd_init(init_args)
        content = (tmp_path / "transmut.toml").read_text()
        assert "# exclude" in content

    def test_should_respect_custom_src_path_in_toml(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        args = argparse.Namespace(
            command="init", src="etl/transforms/", tests="tests/", output="/out"
        )
        _cmd_init(args)
        content = (tmp_path / "transmut.toml").read_text()
        assert 'source_dirs   = ["etl/transforms/"]' in content


# ===========================================================================
# _cmd_show
# ===========================================================================

class TestCmdShow:

    def _make_report(self, base: Path, name: str = "report_2024.html") -> Path:
        """Cria um relatório dentro da estrutura TransmutPysparkOutput."""
        report_dir = base / "TransmutPysparkOutput"
        report_dir.mkdir(parents=True, exist_ok=True)
        report = report_dir / name
        report.write_text("<html></html>")
        return report

    def test_should_open_browser_when_report_exists(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        report = self._make_report(tmp_path)

        with patch("cli.webbrowser.open") as mock_open:
            _cmd_show()
            mock_open.assert_called_once()

    def test_should_open_correct_report_uri(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        report = self._make_report(tmp_path, "report_20240101_120000.html")

        with patch("cli.webbrowser.open") as mock_open:
            _cmd_show()
            called_uri = mock_open.call_args[0][0]
            assert called_uri.startswith("file://")
            assert "report_20240101_120000.html" in called_uri

    def test_should_open_most_recent_report_when_multiple_exist(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        report_dir = tmp_path / "TransmutPysparkOutput"
        report_dir.mkdir()
        (report_dir / "report_20240101.html").write_text("<html>old</html>")
        (report_dir / "report_20240202.html").write_text("<html>new</html>")

        with patch("cli.webbrowser.open") as mock_open:
            _cmd_show()
            called_uri = mock_open.call_args[0][0]
            assert "report_20240202.html" in called_uri

    def test_should_die_when_no_report_exists(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            _cmd_show()

    def test_should_not_open_html_outside_transmut_output_dir(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        # HTML fora do diretório esperado
        (tmp_path / "report_spurious.html").write_text("<html></html>")

        with pytest.raises(SystemExit):
            _cmd_show()

    def test_should_not_open_non_report_html_inside_transmut_dir(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        report_dir = tmp_path / "TransmutPysparkOutput"
        report_dir.mkdir()
        (report_dir / "index.html").write_text("<html></html>")  # não começa com "report"

        with pytest.raises(SystemExit):
            _cmd_show()

    def test_should_print_path_before_opening_browser(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        self._make_report(tmp_path, "report_x.html")

        with patch("cli.webbrowser.open"):
            _cmd_show()

        out = capsys.readouterr().out
        assert "report_x.html" in out


# ===========================================================================
# _cmd_run
# ===========================================================================
#
# NOTA SOBRE O PATCH:
# _cmd_run usa um import *local/lazy*:
#     from src.mutation_manager import MutationManager
# Por isso o alvo do patch é "src.mutation_manager.MutationManager"
# (onde o objeto vive), e não "cli.MutationManager"
# (que nunca existirá no namespace do módulo cli).
# ===========================================================================

_MANAGER_PATH = "src.mutation_manager.MutationManager"


class TestCmdRun:

    @patch(_MANAGER_PATH, autospec=True)
    def test_should_create_mutation_manager_with_correct_config(
        self, MockManager, run_args_src_tests
    ):
        instance = MockManager.return_value
        instance.mutant_list = []
        instance.result_list = []
        instance.work_dir    = "/tmp/out"

        _cmd_run(run_args_src_tests)

        MockManager.assert_called_once()
        config_arg = MockManager.call_args[0][0]
        assert isinstance(config_arg, dict)
        assert config_arg["program_path"] == "src/"

    @patch(_MANAGER_PATH, autospec=True)
    def test_should_call_manager_run(
        self, MockManager, run_args_src_tests
    ):
        instance = MockManager.return_value
        instance.mutant_list = [object()] * 3
        instance.result_list = [Mock(status="killed")] * 3
        instance.work_dir    = "/tmp/out"

        _cmd_run(run_args_src_tests)

        instance.run.assert_called_once()

    @patch(_MANAGER_PATH, autospec=True)
    def test_should_print_mutation_score_after_run(
        self, MockManager, run_args_src_tests, capsys
    ):
        instance = MockManager.return_value
        instance.work_dir    = "/tmp/out"
        instance.mutant_list = [object()] * 4
        instance.result_list = (
            [Mock(status="killed")] * 3 + [Mock(status="survived")]
        )

        _cmd_run(run_args_src_tests)

        out = capsys.readouterr().out
        assert "75.0%" in out

    @patch(_MANAGER_PATH, autospec=True)
    def test_should_print_zero_score_when_no_mutants(
        self, MockManager, run_args_src_tests, capsys
    ):
        instance = MockManager.return_value
        instance.work_dir    = "/tmp/out"
        instance.mutant_list = []
        instance.result_list = []

        _cmd_run(run_args_src_tests)

        out = capsys.readouterr().out
        assert "0%" in out

    @patch(_MANAGER_PATH, autospec=True)
    def test_should_print_low_score_warning_when_score_below_60(
        self, MockManager, run_args_src_tests, capsys
    ):
        instance = MockManager.return_value
        instance.work_dir    = "/tmp/out"
        instance.mutant_list = [object()] * 10
        instance.result_list = [Mock(status="killed")] * 5 + [Mock(status="survived")] * 5

        _cmd_run(run_args_src_tests)

        out = capsys.readouterr().out
        assert "Score baixo" in out or "⚠" in out

    @patch(_MANAGER_PATH, autospec=True)
    def test_should_not_print_warning_when_score_at_60(
        self, MockManager, run_args_src_tests, capsys
    ):
        instance = MockManager.return_value
        instance.work_dir    = "/tmp/out"
        instance.mutant_list = [object()] * 10
        instance.result_list = [Mock(status="killed")] * 6 + [Mock(status="survived")] * 4

        _cmd_run(run_args_src_tests)

        out = capsys.readouterr().out
        assert "Score baixo" not in out

    @patch(_MANAGER_PATH, autospec=True)
    def test_should_exit_with_error_when_file_not_found(
        self, MockManager, run_args_src_tests
    ):
        MockManager.return_value.run.side_effect = FileNotFoundError("src/etl.py")

        with pytest.raises(SystemExit) as exc:
            _cmd_run(run_args_src_tests)
        assert exc.value.code == 1

    @patch(_MANAGER_PATH, autospec=True)
    def test_should_print_file_not_found_message_to_stderr(
        self, MockManager, run_args_src_tests, capsys
    ):
        MockManager.return_value.run.side_effect = FileNotFoundError("src/etl.py")

        with pytest.raises(SystemExit):
            _cmd_run(run_args_src_tests)

        err = capsys.readouterr().err
        assert "não encontrado" in err or "Arquivo" in err

    @patch(_MANAGER_PATH, autospec=True)
    def test_should_exit_with_error_when_value_error_raised(
        self, MockManager, run_args_src_tests
    ):
        MockManager.return_value.run.side_effect = ValueError("Operator inválido")

        with pytest.raises(SystemExit) as exc:
            _cmd_run(run_args_src_tests)
        assert exc.value.code == 1

    @patch(_MANAGER_PATH, autospec=True)
    def test_should_print_config_error_message_when_value_error(
        self, MockManager, run_args_src_tests, capsys
    ):
        MockManager.return_value.run.side_effect = ValueError("bad config")

        with pytest.raises(SystemExit):
            _cmd_run(run_args_src_tests)

        err = capsys.readouterr().err
        assert "Configuração inválida" in err or "inválid" in err

    @patch(_MANAGER_PATH, autospec=True)
    def test_should_exit_cleanly_when_keyboard_interrupt(
        self, MockManager, run_args_src_tests
    ):
        MockManager.return_value.run.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit) as exc:
            _cmd_run(run_args_src_tests)
        assert exc.value.code == 1

    @patch(_MANAGER_PATH, autospec=True)
    def test_should_print_report_path_in_summary(
        self, MockManager, run_args_src_tests, capsys
    ):
        instance = MockManager.return_value
        instance.work_dir    = "/expected/path"
        instance.mutant_list = [object()]
        instance.result_list = [Mock(status="killed")]

        _cmd_run(run_args_src_tests)

        out = capsys.readouterr().out
        assert "/expected/path/report.html" in out

    @patch(_MANAGER_PATH, autospec=True)
    def test_should_report_correct_survived_count(
        self, MockManager, run_args_src_tests, capsys
    ):
        instance = MockManager.return_value
        instance.work_dir    = "/tmp"
        instance.mutant_list = [object()] * 5
        instance.result_list = [Mock(status="killed")] * 2 + [Mock(status="survived")] * 3

        _cmd_run(run_args_src_tests)

        out = capsys.readouterr().out
        assert "3" in out  # 3 sobreviventes


# ===========================================================================
# _setup_logging
# ===========================================================================

class TestSetupLogging:

    def test_should_set_root_level_to_warning_by_default(self):
        _setup_logging(verbose=False)
        assert logging.getLogger().level == logging.WARNING

    def test_should_set_root_level_to_debug_when_verbose(self):
        # basicConfig é idempotente se já existem handlers; limpa antes para
        # garantir que a chamada seja processada em ambiente de teste.
        root = logging.getLogger()
        original_level    = root.level
        original_handlers = root.handlers[:]
        for h in original_handlers:
            root.removeHandler(h)

        try:
            _setup_logging(verbose=True)
            assert root.level == logging.DEBUG
        finally:
            root.setLevel(original_level)
            for h in original_handlers:
                root.addHandler(h)

    def test_should_set_mutation_manager_logger_to_info(self):
        _setup_logging(verbose=False)
        logger = logging.getLogger("src.mutation_manager")
        assert logger.level == logging.INFO

    def test_should_set_ast_analyzer_logger_to_info(self):
        _setup_logging(verbose=False)
        logger = logging.getLogger("src.config.ast_analyzer")
        assert logger.level == logging.INFO

    def test_should_set_test_runner_logger_to_info(self):
        _setup_logging(verbose=False)
        logger = logging.getLogger("src.test_module.test_runner")
        assert logger.level == logging.INFO


# ===========================================================================
# _die
# ===========================================================================

class TestDie:

    def test_should_exit_with_code_1(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _die("algum erro")
        assert exc.value.code == 1

    def test_should_write_message_to_stderr(self, capsys):
        with pytest.raises(SystemExit):
            _die("mensagem de erro")
        err = capsys.readouterr().err
        assert "mensagem de erro" in err

    def test_should_include_erro_prefix_in_stderr(self, capsys):
        with pytest.raises(SystemExit):
            _die("detalhe")
        err = capsys.readouterr().err
        assert "Erro:" in err or "Erro" in err

    def test_should_handle_multiline_message(self, capsys):
        msg = "linha 1\n  linha 2\n  linha 3"
        with pytest.raises(SystemExit):
            _die(msg)
        err = capsys.readouterr().err
        assert "linha 1" in err
        assert "linha 2" in err


# ===========================================================================
# _print_banner
# ===========================================================================

class TestPrintBanner:

    def test_should_print_tool_name_in_banner(self, capsys):
        _print_banner()
        out = capsys.readouterr().out
        assert "TransmutPySpark" in out

    def test_should_print_separator_line_in_banner(self, capsys):
        _print_banner()
        out = capsys.readouterr().out
        assert "─" in out


# ===========================================================================
# main (integração leve — valida roteamento de comandos)
# ===========================================================================

class TestMain:

    @patch("cli._cmd_init")
    @patch("cli.build_parser")
    def test_should_call_cmd_init_when_command_is_init(
        self, mock_parser_factory, mock_cmd_init
    ):
        mock_args = argparse.Namespace(command="init", src="src/", tests="tests/",
                                       output=".", verbose=False)
        mock_parser_factory.return_value.parse_args.return_value = mock_args
        main()
        mock_cmd_init.assert_called_once_with(mock_args)

    @patch("cli._cmd_run")
    @patch("cli.build_parser")
    def test_should_call_cmd_run_when_command_is_run(
        self, mock_parser_factory, mock_cmd_run
    ):
        mock_args = argparse.Namespace(command="run", src="s/", tests="t/",
                                       config=None, operators=["MTR"],
                                       output=".", workers=4, verbose=False)
        mock_parser_factory.return_value.parse_args.return_value = mock_args
        main()
        mock_cmd_run.assert_called_once_with(mock_args)

    @patch("cli._cmd_show")
    @patch("cli.build_parser")
    def test_should_call_cmd_show_when_command_is_show(
        self, mock_parser_factory, mock_cmd_show
    ):
        mock_args = argparse.Namespace(command="show", verbose=False)
        mock_parser_factory.return_value.parse_args.return_value = mock_args
        main()
        mock_cmd_show.assert_called_once()

    @patch("cli._cmd_run")
    @patch("cli.build_parser")
    def test_should_call_setup_logging_with_verbose_flag(
        self, mock_parser_factory, mock_cmd_run
    ):
        mock_args = argparse.Namespace(command="run", src="s/", tests="t/",
                                       config=None, operators=["MTR"],
                                       output=".", workers=4, verbose=True)
        mock_parser_factory.return_value.parse_args.return_value = mock_args

        with patch("cli._setup_logging") as mock_logging:
            main()
            mock_logging.assert_called_once_with(verbose=True)

    @patch("cli._cmd_init")
    @patch("cli.build_parser")
    def test_should_pass_verbose_false_when_command_has_no_verbose_attr(
        self, mock_parser_factory, mock_cmd_init
    ):
        # 'show' e 'init' não têm --verbose; getattr usa False como default
        mock_args = argparse.Namespace(command="init", src="src/", tests="tests/",
                                       output=".")
        mock_parser_factory.return_value.parse_args.return_value = mock_args

        with patch("cli._setup_logging") as mock_logging:
            main()
            mock_logging.assert_called_once_with(verbose=False)