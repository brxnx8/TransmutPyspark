"""
Unit tests for ConfigLoader
============================
Coverage targets
----------------
- load()                  : happy path + method chaining
- _validate_program_path  : all TypeError / FileNotFoundError / ValueError branches + success
- _validate_tests_path    : all TypeError / FileNotFoundError / ValueError branches + success
- _setup_workspace        : TypeError, conflict with existing file, creation + subdirs
- _validate_operators_list: TypeError, empty list, invalid items, normalisation
- _validate_spark_session : TypeError, stopped session, active session
- _assert_loaded          : RuntimeError before load(); no error after load()
- properties              : program_source, tests_source, workspace_path (before + after load)
- __repr__                : before and after load()

Run with:
    pytest test_config_loader.py -v --cov=config_loader --cov-report=term-missing
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

from code.config_loader import ConfigLoader


# ═══════════════════════════════════════════════════════════════════════════ #
# Fixtures                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

PROGRAM_CONTENT = """\
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("app").getOrCreate()
df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])
df.show()
"""

TESTS_CONTENT = """\
def test_dataframe_count(spark):
    df = spark.createDataFrame([(1,)], ["id"])
    assert df.count() == 1
"""

TESTS_CONTENT_NO_FUNCTIONS = """\
# This file has no test functions at all
def helper():
    pass
"""


@pytest.fixture
def program_file(tmp_path: Path) -> Path:
    """A valid .py PySpark program file."""
    f = tmp_path / "program.py"
    f.write_text(PROGRAM_CONTENT, encoding="utf-8")
    return f


@pytest.fixture
def tests_file(tmp_path: Path) -> Path:
    """A valid .py pytest file."""
    f = tmp_path / "test_suite.py"
    f.write_text(TESTS_CONTENT, encoding="utf-8")
    return f


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A clean workspace directory path (not yet created)."""
    return tmp_path / "workspace"


@pytest.fixture
def mock_spark() -> MagicMock:
    """A fully mocked active SparkSession."""
    from pyspark.sql import SparkSession

    spark = MagicMock(spec=SparkSession)
    spark.sparkContext._jsc = MagicMock()          # non-None → session is active
    spark.sparkContext.appName = "test-app"
    return spark


@pytest.fixture
def valid_loader(program_file, tests_file, workspace, mock_spark) -> ConfigLoader:
    """A ConfigLoader with all valid parameters, not yet loaded."""
    return ConfigLoader(
        programPath=str(program_file),
        testsPath=str(tests_file),
        workspaceDir=str(workspace),
        sparkSession=mock_spark,
        operatorsList=["aor", "ror"],
    )


@pytest.fixture
def loaded_loader(valid_loader) -> ConfigLoader:
    """A ConfigLoader that has already had .load() called."""
    return valid_loader.load()


# ═══════════════════════════════════════════════════════════════════════════ #
# load()                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestLoad:
    def test_load_returns_self(self, valid_loader):
        result = valid_loader.load()
        assert result is valid_loader

    def test_load_sets_loaded_flag(self, valid_loader):
        assert valid_loader._loaded is False
        valid_loader.load()
        assert valid_loader._loaded is True

    def test_load_method_chaining(self, program_file, tests_file, workspace, mock_spark):
        cfg = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        ).load()
        assert cfg._loaded is True

    def test_load_calls_all_validators(self, valid_loader):
        """Ensure every private validator is invoked during load()."""
        with (
            patch.object(valid_loader, "_validate_program_path") as mock_prog,
            patch.object(valid_loader, "_validate_tests_path") as mock_tests,
            patch.object(valid_loader, "_setup_workspace") as mock_ws,
            patch.object(valid_loader, "_validate_operators_list") as mock_ops,
            patch.object(valid_loader, "_validate_spark_session") as mock_spark,
        ):
            valid_loader.load()

        mock_prog.assert_called_once()
        mock_tests.assert_called_once()
        mock_ws.assert_called_once()
        mock_ops.assert_called_once()
        mock_spark.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════ #
# _validate_program_path                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestValidateProgramPath:

    # --- TypeError -----------------------------------------------------------

    def test_program_path_not_a_string(self, tests_file, workspace, mock_spark):
        loader = ConfigLoader(
            programPath=123,
            testsPath=str(tests_file),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(TypeError, match="programPath must be a non-empty string"):
            loader._validate_program_path()

    def test_program_path_empty_string(self, tests_file, workspace, mock_spark):
        loader = ConfigLoader(
            programPath="   ",
            testsPath=str(tests_file),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(TypeError, match="programPath must be a non-empty string"):
            loader._validate_program_path()

    # --- FileNotFoundError ---------------------------------------------------

    def test_program_path_does_not_exist(self, tests_file, workspace, mock_spark):
        loader = ConfigLoader(
            programPath="/nonexistent/path/program.py",
            testsPath=str(tests_file),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(FileNotFoundError, match="PySpark program not found"):
            loader._validate_program_path()

    # --- ValueError (directory) ----------------------------------------------

    def test_program_path_is_a_directory(self, tmp_path, tests_file, workspace, mock_spark):
        loader = ConfigLoader(
            programPath=str(tmp_path),   # tmp_path is a directory, not a file
            testsPath=str(tests_file),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(ValueError, match="programPath must point to a file"):
            loader._validate_program_path()

    # --- ValueError (wrong extension) ----------------------------------------

    def test_program_path_wrong_extension(self, tmp_path, tests_file, workspace, mock_spark):
        bad_file = tmp_path / "program.txt"
        bad_file.write_text("content", encoding="utf-8")
        loader = ConfigLoader(
            programPath=str(bad_file),
            testsPath=str(tests_file),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(ValueError, match="programPath must point to a .py file"):
            loader._validate_program_path()

    # --- Success -------------------------------------------------------------

    def test_program_source_loaded_correctly(self, program_file, tests_file, workspace, mock_spark):
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        loader._validate_program_path()
        assert loader._program_source == PROGRAM_CONTENT


# ═══════════════════════════════════════════════════════════════════════════ #
# _validate_tests_path                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestValidateTestsPath:

    # --- TypeError -----------------------------------------------------------

    def test_tests_path_not_a_string(self, program_file, workspace, mock_spark):
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=None,
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(TypeError, match="testsPath must be a non-empty string"):
            loader._validate_tests_path()

    def test_tests_path_empty_string(self, program_file, workspace, mock_spark):
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath="",
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(TypeError, match="testsPath must be a non-empty string"):
            loader._validate_tests_path()

    def test_tests_path_whitespace_only(self, program_file, workspace, mock_spark):
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath="   \t",
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(TypeError, match="testsPath must be a non-empty string"):
            loader._validate_tests_path()

    # --- FileNotFoundError ---------------------------------------------------

    def test_tests_path_does_not_exist(self, program_file, workspace, mock_spark):
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath="/nonexistent/test_file.py",
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(FileNotFoundError, match="Tests file not found"):
            loader._validate_tests_path()

    # --- ValueError (directory) ----------------------------------------------

    def test_tests_path_is_a_directory(self, tmp_path, program_file, workspace, mock_spark):
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tmp_path),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(ValueError, match="testsPath must point to a file"):
            loader._validate_tests_path()

    # --- ValueError (wrong extension) ----------------------------------------

    def test_tests_path_wrong_extension(self, tmp_path, program_file, workspace, mock_spark):
        bad_file = tmp_path / "tests.txt"
        bad_file.write_text(TESTS_CONTENT, encoding="utf-8")
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(bad_file),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(ValueError, match="testsPath must point to a .py file"):
            loader._validate_tests_path()

    # --- ValueError (no test functions) --------------------------------------

    def test_tests_path_no_test_functions(self, tmp_path, program_file, workspace, mock_spark):
        bad_tests = tmp_path / "no_tests.py"
        bad_tests.write_text(TESTS_CONTENT_NO_FUNCTIONS, encoding="utf-8")
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(bad_tests),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(ValueError, match="No pytest test functions"):
            loader._validate_tests_path()

    def test_tests_path_empty_file(self, tmp_path, program_file, workspace, mock_spark):
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("", encoding="utf-8")
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(empty_file),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(ValueError, match="No pytest test functions"):
            loader._validate_tests_path()

    # --- Success -------------------------------------------------------------

    def test_tests_source_loaded_correctly(self, program_file, tests_file, workspace, mock_spark):
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        loader._validate_tests_path()
        assert loader._tests_source == TESTS_CONTENT

    def test_tests_indented_def_test_is_accepted(self, tmp_path, program_file, workspace, mock_spark):
        """test_ functions inside a class are also valid pytest tests."""
        indented = tmp_path / "test_class.py"
        indented.write_text(
            "class TestSomething:\n    def test_case(self):\n        pass\n",
            encoding="utf-8",
        )
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(indented),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        loader._validate_tests_path()   # must not raise
        assert "def test_case" in loader._tests_source


# ═══════════════════════════════════════════════════════════════════════════ #
# _setup_workspace                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestSetupWorkspace:

    # --- TypeError -----------------------------------------------------------

    def test_workspace_not_a_string(self, program_file, tests_file, mock_spark):
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=42,
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(TypeError, match="workspaceDir must be a non-empty string"):
            loader._setup_workspace()

    def test_workspace_empty_string(self, program_file, tests_file, mock_spark):
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir="  ",
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(TypeError, match="workspaceDir must be a non-empty string"):
            loader._setup_workspace()

    # --- ValueError (path exists as a file, not directory) -------------------

    def test_workspace_path_is_a_file(self, tmp_path, program_file, tests_file, mock_spark):
        conflicting_file = tmp_path / "workspace_conflict"
        conflicting_file.write_text("I am a file", encoding="utf-8")
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(conflicting_file),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(ValueError, match="workspaceDir path exists but is not a directory"):
            loader._setup_workspace()

    # --- Success: directory created from scratch -----------------------------

    def test_workspace_created_when_absent(self, tmp_path, program_file, tests_file, mock_spark):
        new_workspace = tmp_path / "brand_new_workspace"
        assert not new_workspace.exists()
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(new_workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        loader._setup_workspace()
        assert new_workspace.is_dir()

    def test_workspace_subdirectories_created(self, tmp_path, program_file, tests_file, mock_spark):
        ws = tmp_path / "ws"
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(ws),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        loader._setup_workspace()
        assert (ws / "mutants").is_dir()
        assert (ws / "results").is_dir()
        assert (ws / "logs").is_dir()

    def test_workspace_path_attribute_set(self, tmp_path, program_file, tests_file, mock_spark):
        ws = tmp_path / "ws"
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(ws),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        loader._setup_workspace()
        assert loader._workspace_path == ws.resolve()

    def test_workspace_already_exists_is_ok(self, tmp_path, program_file, tests_file, mock_spark):
        """Calling _setup_workspace on an existing directory must not raise."""
        existing_ws = tmp_path / "existing"
        existing_ws.mkdir()
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(existing_ws),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        loader._setup_workspace()   # must not raise
        assert existing_ws.is_dir()

    def test_workspace_nested_path_created(self, tmp_path, program_file, tests_file, mock_spark):
        """parents=True must allow deeply nested paths."""
        deep_ws = tmp_path / "a" / "b" / "c" / "workspace"
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(deep_ws),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        loader._setup_workspace()
        assert deep_ws.is_dir()


# ═══════════════════════════════════════════════════════════════════════════ #
# _validate_operators_list                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestValidateOperatorsList:

    def _make_loader(self, operators, program_file, tests_file, workspace, mock_spark):
        return ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=operators,
        )

    # --- TypeError -----------------------------------------------------------

    def test_operators_not_a_list(self, program_file, tests_file, workspace, mock_spark):
        loader = self._make_loader("AOR", program_file, tests_file, workspace, mock_spark)
        with pytest.raises(TypeError, match="operatorsList must be a list"):
            loader._validate_operators_list()

    def test_operators_is_none(self, program_file, tests_file, workspace, mock_spark):
        loader = self._make_loader(None, program_file, tests_file, workspace, mock_spark)
        with pytest.raises(TypeError, match="operatorsList must be a list"):
            loader._validate_operators_list()

    def test_operators_is_tuple(self, program_file, tests_file, workspace, mock_spark):
        loader = self._make_loader(("AOR",), program_file, tests_file, workspace, mock_spark)
        with pytest.raises(TypeError, match="operatorsList must be a list"):
            loader._validate_operators_list()

    # --- ValueError (empty list) ---------------------------------------------

    def test_operators_empty_list(self, program_file, tests_file, workspace, mock_spark):
        loader = self._make_loader([], program_file, tests_file, workspace, mock_spark)
        with pytest.raises(ValueError, match="operatorsList must contain at least one"):
            loader._validate_operators_list()

    # --- ValueError (invalid items) ------------------------------------------

    def test_operators_contains_non_string(self, program_file, tests_file, workspace, mock_spark):
        loader = self._make_loader(["AOR", 99], program_file, tests_file, workspace, mock_spark)
        with pytest.raises(ValueError, match="All items in operatorsList must be non-empty strings"):
            loader._validate_operators_list()

    def test_operators_contains_empty_string(self, program_file, tests_file, workspace, mock_spark):
        loader = self._make_loader(["AOR", ""], program_file, tests_file, workspace, mock_spark)
        with pytest.raises(ValueError, match="All items in operatorsList must be non-empty strings"):
            loader._validate_operators_list()

    def test_operators_contains_whitespace_only(self, program_file, tests_file, workspace, mock_spark):
        loader = self._make_loader(["AOR", "  "], program_file, tests_file, workspace, mock_spark)
        with pytest.raises(ValueError, match="All items in operatorsList must be non-empty strings"):
            loader._validate_operators_list()

    # --- Success: normalisation ----------------------------------------------

    def test_operators_normalised_to_uppercase(self, program_file, tests_file, workspace, mock_spark):
        loader = self._make_loader(["aor", "ror", "lcr"], program_file, tests_file, workspace, mock_spark)
        loader._validate_operators_list()
        assert loader.operatorsList == ["AOR", "ROR", "LCR"]

    def test_operators_strips_surrounding_whitespace(self, program_file, tests_file, workspace, mock_spark):
        loader = self._make_loader(["  aor  ", " ror"], program_file, tests_file, workspace, mock_spark)
        loader._validate_operators_list()
        assert loader.operatorsList == ["AOR", "ROR"]

    def test_single_operator_accepted(self, program_file, tests_file, workspace, mock_spark):
        loader = self._make_loader(["AOR"], program_file, tests_file, workspace, mock_spark)
        loader._validate_operators_list()
        assert loader.operatorsList == ["AOR"]


# ═══════════════════════════════════════════════════════════════════════════ #
# _validate_spark_session                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestValidateSparkSession:

    # --- TypeError -----------------------------------------------------------

    def test_spark_not_a_spark_session(self, program_file, tests_file, workspace):
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(workspace),
            sparkSession="not-a-spark-session",
            operatorsList=["AOR"],
        )
        with pytest.raises(TypeError, match="sparkSession must be a pyspark.sql.SparkSession"):
            loader._validate_spark_session()

    def test_spark_is_none(self, program_file, tests_file, workspace):
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(workspace),
            sparkSession=None,
            operatorsList=["AOR"],
        )
        with pytest.raises(TypeError, match="sparkSession must be a pyspark.sql.SparkSession"):
            loader._validate_spark_session()

    # --- ValueError (stopped session) ----------------------------------------

    def test_spark_session_stopped(self, program_file, tests_file, workspace):
        from pyspark.sql import SparkSession

        stopped_spark = MagicMock(spec=SparkSession)
        stopped_spark.sparkContext._jsc = None   # signals a stopped session

        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(workspace),
            sparkSession=stopped_spark,
            operatorsList=["AOR"],
        )
        with pytest.raises(ValueError, match="SparkSession appears to have been stopped"):
            loader._validate_spark_session()

    # --- Success -------------------------------------------------------------

    def test_active_spark_session_passes(self, program_file, tests_file, workspace, mock_spark):
        loader = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(workspace),
            sparkSession=mock_spark,
            operatorsList=["AOR"],
        )
        loader._validate_spark_session()   # must not raise


# ═══════════════════════════════════════════════════════════════════════════ #
# _assert_loaded                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestAssertLoaded:

    def test_raises_runtime_error_before_load(self, valid_loader):
        with pytest.raises(RuntimeError, match="Call .load\\(\\) first"):
            valid_loader._assert_loaded()

    def test_does_not_raise_after_load(self, loaded_loader):
        loaded_loader._assert_loaded()   # must not raise


# ═══════════════════════════════════════════════════════════════════════════ #
# Properties                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestProperties:

    def test_program_source_raises_before_load(self, valid_loader):
        with pytest.raises(RuntimeError):
            _ = valid_loader.program_source

    def test_tests_source_raises_before_load(self, valid_loader):
        with pytest.raises(RuntimeError):
            _ = valid_loader.tests_source

    def test_workspace_path_raises_before_load(self, valid_loader):
        with pytest.raises(RuntimeError):
            _ = valid_loader.workspace_path

    def test_program_source_after_load(self, loaded_loader):
        assert loaded_loader.program_source == PROGRAM_CONTENT

    def test_tests_source_after_load(self, loaded_loader):
        assert loaded_loader.tests_source == TESTS_CONTENT

    def test_workspace_path_after_load(self, loaded_loader, workspace):
        assert loaded_loader.workspace_path == workspace.resolve()

    def test_workspace_path_is_path_instance(self, loaded_loader):
        assert isinstance(loaded_loader.workspace_path, Path)


# ═══════════════════════════════════════════════════════════════════════════ #
# __repr__                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestRepr:

    def test_repr_before_load_contains_loaded_false(self, valid_loader):
        r = repr(valid_loader)
        assert "loaded=False" in r

    def test_repr_after_load_contains_loaded_true(self, loaded_loader):
        r = repr(loaded_loader)
        assert "loaded=True" in r

    def test_repr_contains_program_path(self, valid_loader, program_file):
        assert str(program_file) in repr(valid_loader)

    def test_repr_contains_tests_path(self, valid_loader, tests_file):
        assert str(tests_file) in repr(valid_loader)

    def test_repr_contains_workspace_dir(self, valid_loader, workspace):
        assert str(workspace) in repr(valid_loader)

    def test_repr_contains_operators(self, loaded_loader):
        r = repr(loaded_loader)
        assert "AOR" in r
        assert "ROR" in r


# ═══════════════════════════════════════════════════════════════════════════ #
# Integration: full happy path                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestIntegration:

    def test_full_happy_path(self, program_file, tests_file, tmp_path, mock_spark):
        ws = tmp_path / "workspace"
        cfg = ConfigLoader(
            programPath=str(program_file),
            testsPath=str(tests_file),
            workspaceDir=str(ws),
            sparkSession=mock_spark,
            operatorsList=["aor", "ror", "lcr"],
        ).load()

        # loaded flag
        assert cfg._loaded is True

        # source files read correctly
        assert cfg.program_source == PROGRAM_CONTENT
        assert cfg.tests_source == TESTS_CONTENT

        # workspace + subdirs created
        assert cfg.workspace_path.is_dir()
        assert (cfg.workspace_path / "mutants").is_dir()
        assert (cfg.workspace_path / "results").is_dir()
        assert (cfg.workspace_path / "logs").is_dir()

        # operators normalised
        assert cfg.operatorsList == ["AOR", "ROR", "LCR"]

    def test_load_is_idempotent_second_call(self, valid_loader):
        """Calling load() twice must not raise and must keep loaded=True."""
        valid_loader.load()
        valid_loader.load()
        assert valid_loader._loaded is True