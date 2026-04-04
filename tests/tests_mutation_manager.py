"""
Unit tests for MutationManager (orchestrator)
===============================================
Coverage targets
----------------
- __post_init__            : valid path, empty string, non-string, whitespace
- load()                   : FileNotFoundError (config missing, program missing),
                             ValueError (missing keys, empty program file),
                             success (config populated, code_original read),
                             method chaining, blank/comment lines in config,
                             operators normalised to uppercase
- parse_to_ast()           : RuntimeError before load, ValueError on syntax
                             error, success (AST stored, fix_missing_locations),
                             method chaining
- apply_mutation()         : RuntimeError before parse_to_ast,
                             unregistered operator skipped (KeyError),
                             ImportError skipped, AttributeError skipped,
                             operator with no eligible nodes skipped,
                             operator exception skipped,
                             mutants accumulated across operators,
                             method chaining, mutant_dir created
- run_tests()              : RuntimeError before apply_mutation,
                             delegates to TestRunner, result_list populated,
                             method chaining
- agregate_results()       : RuntimeError before run_tests, delegates to
                             Reporter (calculate, make_diff, show_results
                             called in order), method chaining
- _resolve_operator()      : KeyError on unknown, ImportError propagated,
                             AttributeError propagated, success,
                             name normalised to uppercase
- _ensure_mutant_dir()     : creates directory, returns correct Path, idempotent
- _parse_config_file()     : key=value, blank lines, comment lines,
                             lines without '=', values with '=' in them,
                             whitespace stripped, empty input, only comments
- _validate_config_path()  : FileNotFoundError, ValueError (dir not file)
- _assert_loaded()         : RuntimeError when config None, when source empty,
                             no error when ready
- _assert_ast_ready()      : RuntimeError when None, no error when set
- _assert_mutants_ready()  : RuntimeError when empty, no error when populated
- _assert_results_ready()  : RuntimeError when empty, no error when populated
- __repr__                 : before and after each pipeline step

Run with:
    pytest test_mutation_manager.py -v \
        --cov=src.mutation_manager \
        --cov-report=term-missing
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.mutation_manager import MutationManager, _OPERATOR_REGISTRY
from src.config_loader import ConfigLoader


# ═══════════════════════════════════════════════════════════════════════════ #
# Helpers / factories                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

VALID_PROGRAM        = "x = 1 + 2\ny = x * 3\n"
SYNTAX_ERROR_PROGRAM = "def foo(:\n    pass\n"


def _write_program(tmp_path: Path,
                   content: str = VALID_PROGRAM) -> Path:
    f = tmp_path / "spark_job.py"
    f.write_text(content, encoding="utf-8")
    return f


def _write_tests(tmp_path: Path) -> Path:
    f = tmp_path / "test_suite.py"
    f.write_text("def test_placeholder(): pass\n", encoding="utf-8")
    return f


def _write_config(tmp_path: Path,
                  program_path: str | None = None,
                  tests_path:   str | None = None,
                  operators:    str = "NFTP",
                  extra_lines:  str = "") -> Path:
    program_path = program_path or str(tmp_path / "spark_job.py")
    tests_path   = tests_path   or str(tmp_path / "test_suite.py")
    content = (
        f"program_path   = {program_path}\n"
        f"tests_path     = {tests_path}\n"
        f"operators_list = {operators}\n"
        f"{extra_lines}"
    )
    cfg = tmp_path / "config.txt"
    cfg.write_text(content, encoding="utf-8")
    return cfg


def _make_manager(tmp_path: Path,
                  program_content: str = VALID_PROGRAM,
                  operators: str = "NFTP") -> MutationManager:
    _write_program(tmp_path, program_content)
    _write_tests(tmp_path)
    cfg = _write_config(tmp_path, operators=operators)
    return MutationManager(str(cfg))


def _loaded(tmp_path: Path, **kwargs) -> MutationManager:
    return _make_manager(tmp_path, **kwargs).load()


def _parsed(tmp_path: Path, **kwargs) -> MutationManager:
    return _loaded(tmp_path, **kwargs).parse_to_ast()


def _mock_mutant(mutant_id: int = 1) -> MagicMock:
    m = MagicMock()
    m.id = mutant_id
    return m


def _mock_result(mutant_id: int = 1) -> MagicMock:
    r = MagicMock()
    r.mutant = mutant_id
    return r


# ═══════════════════════════════════════════════════════════════════════════ #
# __post_init__                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestPostInit:

    def test_valid_path_string_accepted(self, tmp_path):
        cfg = _write_config(tmp_path)
        _write_program(tmp_path)
        _write_tests(tmp_path)
        m = MutationManager(str(cfg))
        assert m.config_path == str(cfg)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="config_path must be a non-empty string"):
            MutationManager("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="config_path must be a non-empty string"):
            MutationManager("   ")

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="config_path must be a non-empty string"):
            MutationManager(42)

    def test_initial_state_is_clean(self, tmp_path):
        cfg = _write_config(tmp_path)
        _write_program(tmp_path)
        _write_tests(tmp_path)
        m = MutationManager(str(cfg))
        assert m.config        is None
        assert m.code_original == ""
        assert m.code_ast      is None
        assert m.mutant_list   == []
        assert m.result_list   == []


# ═══════════════════════════════════════════════════════════════════════════ #
# load()                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestLoad:

    def test_config_file_not_found_raises(self, tmp_path):
        m = MutationManager(str(tmp_path / "ghost.txt"))
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            m.load()

    def test_config_path_is_directory_raises(self, tmp_path):
        m = MutationManager(str(tmp_path))
        with pytest.raises(ValueError, match="config_path must point to a file"):
            m.load()

    def test_missing_program_path_key_raises(self, tmp_path):
        _write_tests(tmp_path)
        cfg = tmp_path / "config.txt"
        cfg.write_text(
            f"tests_path     = {tmp_path / 'test_suite.py'}\n"
            f"operators_list = NFTP\n"
        )
        with pytest.raises(ValueError, match="Missing required config keys"):
            MutationManager(str(cfg)).load()

    def test_missing_tests_path_key_raises(self, tmp_path):
        _write_program(tmp_path)
        cfg = tmp_path / "config.txt"
        cfg.write_text(
            f"program_path   = {tmp_path / 'spark_job.py'}\n"
            f"operators_list = NFTP\n"
        )
        with pytest.raises(ValueError, match="Missing required config keys"):
            MutationManager(str(cfg)).load()

    def test_missing_operators_list_key_raises(self, tmp_path):
        _write_program(tmp_path)
        _write_tests(tmp_path)
        cfg = tmp_path / "config.txt"
        cfg.write_text(
            f"program_path = {tmp_path / 'spark_job.py'}\n"
            f"tests_path   = {tmp_path / 'test_suite.py'}\n"
        )
        with pytest.raises(ValueError, match="Missing required config keys"):
            MutationManager(str(cfg)).load()

    def test_program_file_not_found_raises(self, tmp_path):
        _write_tests(tmp_path)
        cfg = _write_config(tmp_path,
                            program_path=str(tmp_path / "missing.py"))
        with pytest.raises(FileNotFoundError, match="PySpark program not found"):
            MutationManager(str(cfg)).load()

    def test_empty_program_file_raises(self, tmp_path):
        (tmp_path / "spark_job.py").write_text("   \n", encoding="utf-8")
        _write_tests(tmp_path)
        cfg = _write_config(tmp_path)
        with pytest.raises(ValueError, match="Program file is empty"):
            MutationManager(str(cfg)).load()

    def test_success_populates_config(self, tmp_path):
        m = _loaded(tmp_path)
        assert isinstance(m.config, ConfigLoader)

    def test_success_populates_code_original(self, tmp_path):
        m = _loaded(tmp_path)
        assert m.code_original == VALID_PROGRAM

    def test_operators_normalised_to_uppercase(self, tmp_path):
        m = _loaded(tmp_path, operators="nftp")
        assert m.config.operators_list == ["NFTP"]

    def test_multiple_operators_parsed(self, tmp_path):
        _write_program(tmp_path)
        _write_tests(tmp_path)
        cfg = _write_config(tmp_path, operators="NFTP, AOR, ROR")
        m   = MutationManager(str(cfg)).load()
        assert m.config.operators_list == ["NFTP", "AOR", "ROR"]

    def test_returns_self_for_chaining(self, tmp_path):
        m = _make_manager(tmp_path)
        assert m.load() is m

    def test_blank_lines_in_config_ignored(self, tmp_path):
        _write_program(tmp_path)
        _write_tests(tmp_path)
        cfg = _write_config(tmp_path, extra_lines="\n\n")
        assert MutationManager(str(cfg)).load().config is not None

    def test_comment_lines_in_config_ignored(self, tmp_path):
        _write_program(tmp_path)
        _write_tests(tmp_path)
        cfg = _write_config(tmp_path, extra_lines="# comment\n")
        assert MutationManager(str(cfg)).load().config is not None

    def test_value_with_equals_sign_parsed(self, tmp_path):
        _write_program(tmp_path)
        _write_tests(tmp_path)
        cfg = tmp_path / "config.txt"
        cfg.write_text(
            f"program_path   = {tmp_path / 'spark_job.py'}\n"
            f"tests_path     = {tmp_path / 'test_suite.py'}\n"
            f"operators_list = NFTP\n"
            f"extra_key      = a=b=c\n"
        )
        m = MutationManager(str(cfg)).load()
        assert m.config is not None


# ═══════════════════════════════════════════════════════════════════════════ #
# parse_to_ast()                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestParseToAst:

    def test_raises_before_load(self, tmp_path):
        m = _make_manager(tmp_path)
        with pytest.raises(RuntimeError, match="Call load\\(\\) first"):
            m.parse_to_ast()

    def test_syntax_error_raises_value_error(self, tmp_path):
        m = _loaded(tmp_path, program_content=SYNTAX_ERROR_PROGRAM)
        with pytest.raises(ValueError, match="Syntax error at line"):
            m.parse_to_ast()

    def test_success_stores_ast(self, tmp_path):
        m = _parsed(tmp_path)
        assert isinstance(m.code_ast, ast.AST)

    def test_fix_missing_locations_applied(self, tmp_path):
        m = _parsed(tmp_path)
        for node in ast.walk(m.code_ast):
            if hasattr(node, "lineno"):
                assert node.lineno is not None

    def test_returns_self_for_chaining(self, tmp_path):
        m = _loaded(tmp_path)
        assert m.parse_to_ast() is m

    def test_ast_unparseable_reflects_source(self, tmp_path):
        m = _parsed(tmp_path)
        assert ast.unparse(m.code_ast) != ""


# ═══════════════════════════════════════════════════════════════════════════ #
# apply_mutation()                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestApplyMutation:

    def test_raises_before_parse_to_ast(self, tmp_path):
        m = _loaded(tmp_path)
        with pytest.raises(RuntimeError, match="Call parse_to_ast\\(\\) first"):
            m.apply_mutation()

    def test_unregistered_operator_skipped(self, tmp_path):
        _write_program(tmp_path)
        _write_tests(tmp_path)
        cfg = _write_config(tmp_path, operators="UNKNOWN_OP")
        m   = MutationManager(str(cfg)).load().parse_to_ast()
        m.apply_mutation()
        assert m.mutant_list == []

    def test_import_error_skipped(self, tmp_path):
        m = _parsed(tmp_path)
        with patch.dict(_OPERATOR_REGISTRY, {"NFTP": "bad.module.Op"}):
            with patch("src.mutation_manager.importlib.import_module",
                       side_effect=ImportError("no module")):
                m.apply_mutation()
        assert m.mutant_list == []

    def test_attribute_error_skipped(self, tmp_path):
        m           = _parsed(tmp_path)
        fake_module = MagicMock(spec=[])           # has no attributes
        with patch.dict(_OPERATOR_REGISTRY, {"NFTP": "src.op.OpClass"}):
            with patch("src.mutation_manager.importlib.import_module",
                       return_value=fake_module):
                m.apply_mutation()
        assert m.mutant_list == []

    def test_operator_with_no_nodes_skipped(self, tmp_path):
        m       = _parsed(tmp_path)
        mock_op = MagicMock()
        mock_op.analyse_ast.return_value = []

        with patch.object(m, "_resolve_operator", return_value=mock_op):
            m.apply_mutation()

        mock_op.build_mutant.assert_not_called()
        assert m.mutant_list == []

    def test_operator_exception_skipped(self, tmp_path):
        m       = _parsed(tmp_path)
        mock_op = MagicMock()
        mock_op.analyse_ast.side_effect = RuntimeError("unexpected")

        with patch.object(m, "_resolve_operator", return_value=mock_op):
            m.apply_mutation()

        assert m.mutant_list == []

    def test_mutants_accumulated_from_one_operator(self, tmp_path):
        m       = _parsed(tmp_path)
        mut_a   = _mock_mutant(1)
        mut_b   = _mock_mutant(2)
        mock_op = MagicMock()
        mock_op.analyse_ast.return_value  = [MagicMock()]
        mock_op.build_mutant.return_value = [mut_a, mut_b]

        with patch.object(m, "_resolve_operator", return_value=mock_op):
            m.apply_mutation()

        assert len(m.mutant_list) == 2
        assert mut_a in m.mutant_list
        assert mut_b in m.mutant_list

    def test_mutants_accumulated_across_two_operators(self, tmp_path):
        _write_program(tmp_path)
        _write_tests(tmp_path)
        cfg = _write_config(tmp_path, operators="NFTP, AOR")
        m   = MutationManager(str(cfg)).load().parse_to_ast()

        ops = []
        for i in range(2):
            op = MagicMock()
            op.analyse_ast.return_value  = [MagicMock()]
            op.build_mutant.return_value = [_mock_mutant(i + 1)]
            ops.append(op)

        with patch.object(m, "_resolve_operator", side_effect=ops):
            m.apply_mutation()

        assert len(m.mutant_list) == 2

    def test_returns_self_for_chaining(self, tmp_path):
        m       = _parsed(tmp_path)
        mock_op = MagicMock()
        mock_op.analyse_ast.return_value = []

        with patch.object(m, "_resolve_operator", return_value=mock_op):
            result = m.apply_mutation()

        assert result is m

    def test_mutant_dir_created(self, tmp_path):
        m       = _parsed(tmp_path)
        mock_op = MagicMock()
        mock_op.analyse_ast.return_value = []

        with patch.object(m, "_resolve_operator", return_value=mock_op):
            m.apply_mutation()

        assert (tmp_path / "mutants").is_dir()

    def test_build_mutant_receives_correct_args(self, tmp_path):
        m       = _parsed(tmp_path)
        node    = MagicMock()
        mock_op = MagicMock()
        mock_op.analyse_ast.return_value  = [node]
        mock_op.build_mutant.return_value = []

        with patch.object(m, "_resolve_operator", return_value=mock_op):
            m.apply_mutation()

        call_kwargs = mock_op.build_mutant.call_args[1]
        assert call_kwargs["nodes"]         == [node]
        assert call_kwargs["original_ast"]  is m.code_ast
        assert call_kwargs["original_path"] == m.config.program_path
        assert "mutants" in call_kwargs["mutant_dir"]


# ═══════════════════════════════════════════════════════════════════════════ #
# run_tests()                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestRunTests:

    def _with_mutants(self, tmp_path: Path) -> MutationManager:
        m = _parsed(tmp_path)
        m.mutant_list = [_mock_mutant(1)]
        return m

    def test_raises_before_apply_mutation(self, tmp_path):
        m = _parsed(tmp_path)
        with pytest.raises(RuntimeError, match="Call apply_mutation\\(\\) first"):
            m.run_tests()

    def test_test_runner_instantiated_with_correct_args(self, tmp_path):
        m           = self._with_mutants(tmp_path)
        mock_runner = MagicMock()
        mock_runner.run_test.return_value = [_mock_result()]

        with patch("src.mutation_manager.TestRunner",
                   return_value=mock_runner) as mock_cls:
            m.run_tests()

        mock_cls.assert_called_once_with(
            mutant_list=m.mutant_list,
            config=m.config,
        )

    def test_run_test_method_called(self, tmp_path):
        m           = self._with_mutants(tmp_path)
        mock_runner = MagicMock()
        mock_runner.run_test.return_value = [_mock_result()]

        with patch("src.mutation_manager.TestRunner", return_value=mock_runner):
            m.run_tests()

        mock_runner.run_test.assert_called_once()

    def test_result_list_populated(self, tmp_path):
        m           = self._with_mutants(tmp_path)
        results     = [_mock_result(1), _mock_result(2)]
        mock_runner = MagicMock()
        mock_runner.run_test.return_value = results

        with patch("src.mutation_manager.TestRunner", return_value=mock_runner):
            m.run_tests()

        assert m.result_list == results

    def test_returns_self_for_chaining(self, tmp_path):
        m           = self._with_mutants(tmp_path)
        mock_runner = MagicMock()
        mock_runner.run_test.return_value = [_mock_result()]

        with patch("src.mutation_manager.TestRunner", return_value=mock_runner):
            result = m.run_tests()

        assert result is m


# ═══════════════════════════════════════════════════════════════════════════ #
# agregate_results()                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestAgregateResults:

    def _with_results(self, tmp_path: Path) -> MutationManager:
        m = _parsed(tmp_path)
        m.mutant_list = [_mock_mutant(1)]
        m.result_list = [_mock_result(1)]
        return m

    def test_raises_before_run_tests(self, tmp_path):
        m = _parsed(tmp_path)
        m.mutant_list = [_mock_mutant()]
        with pytest.raises(RuntimeError, match="Call run_tests\\(\\) first"):
            m.agregate_results()

    def test_reporter_instantiated_with_correct_args(self, tmp_path):
        m             = self._with_results(tmp_path)
        mock_reporter = MagicMock()

        with patch("src.mutation_manager.Reporter",
                   return_value=mock_reporter) as mock_cls:
            m.agregate_results()

        mock_cls.assert_called_once_with(
            result_list=m.result_list,
            code_original=m.code_original,
            mutant_list=m.mutant_list,
        )

    def test_reporter_methods_called_in_correct_order(self, tmp_path):
        m             = self._with_results(tmp_path)
        mock_reporter = MagicMock()
        order         = []

        mock_reporter.calculate.side_effect    = lambda: order.append("calculate")
        mock_reporter.make_diff.side_effect    = lambda: order.append("make_diff")
        mock_reporter.show_results.side_effect = lambda: order.append("show_results")

        with patch("src.mutation_manager.Reporter", return_value=mock_reporter):
            m.agregate_results()

        assert order == ["calculate", "make_diff", "show_results"]

    def test_returns_self_for_chaining(self, tmp_path):
        m             = self._with_results(tmp_path)
        mock_reporter = MagicMock()

        with patch("src.mutation_manager.Reporter", return_value=mock_reporter):
            result = m.agregate_results()

        assert result is m


# ═══════════════════════════════════════════════════════════════════════════ #
# _resolve_operator()                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestResolveOperator:

    def test_unknown_operator_raises_key_error(self, tmp_path):
        m = _parsed(tmp_path)
        with pytest.raises(KeyError, match="is not registered"):
            m._resolve_operator("GHOST")

    def test_import_error_propagated(self, tmp_path):
        m = _parsed(tmp_path)
        with patch.dict(_OPERATOR_REGISTRY, {"NFTP": "bad.module.Op"}):
            with patch("src.mutation_manager.importlib.import_module",
                       side_effect=ImportError("boom")):
                with pytest.raises(ImportError):
                    m._resolve_operator("NFTP")

    def test_attribute_error_propagated(self, tmp_path):
        m           = _parsed(tmp_path)
        fake_module = MagicMock(spec=[])    # no attributes at all
        with patch.dict(_OPERATOR_REGISTRY, {"NFTP": "src.op.OpClass"}):
            with patch("src.mutation_manager.importlib.import_module",
                       return_value=fake_module):
                with pytest.raises(AttributeError):
                    m._resolve_operator("NFTP")

    def test_success_returns_operator_instance(self, tmp_path):
        m           = _parsed(tmp_path)
        mock_op     = MagicMock()
        mock_cls    = MagicMock(return_value=mock_op)
        fake_module = MagicMock()
        fake_module.OpClass = mock_cls

        with patch.dict(_OPERATOR_REGISTRY, {"NFTP": "src.op.OpClass"}):
            with patch("src.mutation_manager.importlib.import_module",
                       return_value=fake_module):
                result = m._resolve_operator("NFTP")

        assert result is mock_op

    def test_lowercase_name_normalised(self, tmp_path):
        m           = _parsed(tmp_path)
        mock_op     = MagicMock()
        mock_cls    = MagicMock(return_value=mock_op)
        fake_module = MagicMock()
        fake_module.OpClass = mock_cls

        with patch.dict(_OPERATOR_REGISTRY, {"NFTP": "src.op.OpClass"}):
            with patch("src.mutation_manager.importlib.import_module",
                       return_value=fake_module):
                result = m._resolve_operator("nftp")

        assert result is mock_op


# ═══════════════════════════════════════════════════════════════════════════ #
# _ensure_mutant_dir()                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestEnsureMutantDir:

    def test_creates_directory(self, tmp_path):
        m = _parsed(tmp_path)
        d = m._ensure_mutant_dir()
        assert d.is_dir()

    def test_returns_correct_path(self, tmp_path):
        m = _parsed(tmp_path)
        assert m._ensure_mutant_dir() == tmp_path / "mutants"

    def test_idempotent_on_second_call(self, tmp_path):
        m = _parsed(tmp_path)
        m._ensure_mutant_dir()
        m._ensure_mutant_dir()   # must not raise
        assert (tmp_path / "mutants").is_dir()


# ═══════════════════════════════════════════════════════════════════════════ #
# _parse_config_file()                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestParseConfigFile:

    def test_basic_key_value(self):
        result = MutationManager._parse_config_file("foo = bar\nbaz = qux\n")
        assert result == {"foo": "bar", "baz": "qux"}

    def test_blank_lines_ignored(self):
        result = MutationManager._parse_config_file("\nfoo = bar\n\n")
        assert result == {"foo": "bar"}

    def test_comment_lines_ignored(self):
        result = MutationManager._parse_config_file("# comment\nfoo = bar\n")
        assert result == {"foo": "bar"}

    def test_lines_without_equals_ignored(self):
        result = MutationManager._parse_config_file("no_equals\nfoo = bar\n")
        assert result == {"foo": "bar"}

    def test_value_with_equals_preserved(self):
        result = MutationManager._parse_config_file("key = a=b=c\n")
        assert result == {"key": "a=b=c"}

    def test_surrounding_whitespace_stripped(self):
        result = MutationManager._parse_config_file("  key  =  value  \n")
        assert result == {"key": "value"}

    def test_empty_string_returns_empty_dict(self):
        assert MutationManager._parse_config_file("") == {}

    def test_only_comments_returns_empty_dict(self):
        assert MutationManager._parse_config_file("# a\n# b\n") == {}


# ═══════════════════════════════════════════════════════════════════════════ #
# Guards                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestGuards:

    # _assert_loaded ----------------------------------------------------------

    def test_loaded_raises_when_config_none(self, tmp_path):
        m = _make_manager(tmp_path)
        with pytest.raises(RuntimeError, match="Call load\\(\\) first"):
            m._assert_loaded()

    def test_loaded_raises_when_source_empty(self, tmp_path):
        m               = _make_manager(tmp_path)
        m.config        = ConfigLoader(str(tmp_path / "spark_job.py"),
                                       str(tmp_path / "test_suite.py"),
                                       ["NFTP"])
        m.code_original = ""
        with pytest.raises(RuntimeError, match="Call load\\(\\) first"):
            m._assert_loaded()

    def test_loaded_passes_when_ready(self, tmp_path):
        _loaded(tmp_path)._assert_loaded()

    # _assert_ast_ready -------------------------------------------------------

    def test_ast_ready_raises_when_none(self, tmp_path):
        m = _loaded(tmp_path)
        with pytest.raises(RuntimeError, match="Call parse_to_ast\\(\\) first"):
            m._assert_ast_ready()

    def test_ast_ready_passes_when_set(self, tmp_path):
        _parsed(tmp_path)._assert_ast_ready()

    # _assert_mutants_ready ---------------------------------------------------

    def test_mutants_ready_raises_when_empty(self, tmp_path):
        m = _parsed(tmp_path)
        with pytest.raises(RuntimeError, match="Call apply_mutation\\(\\) first"):
            m._assert_mutants_ready()

    def test_mutants_ready_passes_when_populated(self, tmp_path):
        m             = _parsed(tmp_path)
        m.mutant_list = [_mock_mutant()]
        m._assert_mutants_ready()

    # _assert_results_ready ---------------------------------------------------

    def test_results_ready_raises_when_empty(self, tmp_path):
        m = _parsed(tmp_path)
        with pytest.raises(RuntimeError, match="Call run_tests\\(\\) first"):
            m._assert_results_ready()

    def test_results_ready_passes_when_populated(self, tmp_path):
        m             = _parsed(tmp_path)
        m.result_list = [_mock_result()]
        m._assert_results_ready()


# ═══════════════════════════════════════════════════════════════════════════ #
# __repr__                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestRepr:

    def test_before_load(self, tmp_path):
        m = _make_manager(tmp_path)
        r = repr(m)
        assert "config=not set" in r
        assert "ast=not set"    in r
        assert "mutants=0"      in r
        assert "results=0"      in r

    def test_after_load(self, tmp_path):
        assert "config=set" in repr(_loaded(tmp_path))

    def test_after_parse_to_ast(self, tmp_path):
        assert "ast=set" in repr(_parsed(tmp_path))

    def test_mutant_count(self, tmp_path):
        m             = _parsed(tmp_path)
        m.mutant_list = [_mock_mutant(i) for i in range(4)]
        assert "mutants=4" in repr(m)

    def test_result_count(self, tmp_path):
        m             = _parsed(tmp_path)
        m.result_list = [_mock_result(i) for i in range(3)]
        assert "results=3" in repr(m)


# ═══════════════════════════════════════════════════════════════════════════ #
# Integration                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestIntegration:

    def test_load_then_parse(self, tmp_path):
        m = _parsed(tmp_path)
        assert m.config        is not None
        assert m.code_original == VALID_PROGRAM
        assert isinstance(m.code_ast, ast.AST)

    def test_full_pipeline_with_mocks(self, tmp_path):
        m = _parsed(tmp_path)

        mut_a   = _mock_mutant(1)
        mut_b   = _mock_mutant(2)
        mock_op = MagicMock()
        mock_op.analyse_ast.return_value  = [MagicMock()]
        mock_op.build_mutant.return_value = [mut_a, mut_b]

        results      = [_mock_result(1), _mock_result(2)]
        mock_runner  = MagicMock()
        mock_runner.run_test.return_value = results
        mock_reporter = MagicMock()

        with patch.object(m, "_resolve_operator", return_value=mock_op), \
             patch("src.mutation_manager.TestRunner", return_value=mock_runner), \
             patch("src.mutation_manager.Reporter", return_value=mock_reporter):

            result = (
                m.apply_mutation()
                 .run_tests()
                 .agregate_results()
            )

        assert result is m
        assert len(m.mutant_list) == 2
        assert m.result_list      == results
        mock_reporter.calculate.assert_called_once()
        mock_reporter.make_diff.assert_called_once()
        mock_reporter.show_results.assert_called_once()

    def test_full_chain_returns_same_instance(self, tmp_path):
        m       = _parsed(tmp_path)
        mock_op = MagicMock()
        mock_op.analyse_ast.return_value = []

        with patch.object(m, "_resolve_operator", return_value=mock_op):
            result = m.apply_mutation()

        assert result is m