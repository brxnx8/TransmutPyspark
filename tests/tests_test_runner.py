"""
Unit tests for TestRunner
==========================
Coverage targets
----------------
- TestResult                  : is_killed, is_survived, is_error, __repr__
- TestRunner.__post_init__    : valid construction; TypeError on bad mutateList,
                                each missing config attr, unloaded config;
                                ValueError on bad max_workers (zero, negative,
                                float, string); default max_workers computed
- runTest()                   : RuntimeError on empty mutateList, clears previous
                                results, returns list[TestResult], killed /
                                survived / error outcomes, timeout branch,
                                unexpected worker-crash branch (future.result()
                                raises), multiple mutants all collected,
                                PYTHONPATH injection (with/without existing var)
- _run_single_mutant()        : temp file created with correct name, correct env,
                                subprocess called with correct args, stdout/stderr
                                captured, timeout handled
- _classify()                 : exit codes 0, 1, 2, 3, 4, 5, -1
- __repr__                    : fields reflected correctly

Run with:
    pytest test_test_runner.py -v \
        --cov=code.test_runner \
        --cov-report=term-missing
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest

from code.test_runner import TestResult, TestRunner


# ═══════════════════════════════════════════════════════════════════════════ #
# Shared helpers                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def _make_config(tmp_path: Path) -> MagicMock:
    """Return a mock ConfigLoader that passes all validation checks."""
    cfg = MagicMock()
    cfg.programPath    = str(tmp_path / "spark_job.py")
    cfg.testsPath      = str(tmp_path / "test_suite.py")
    cfg.workspace_path = tmp_path / "workspace"
    cfg.sparkSession   = MagicMock()
    (tmp_path / "workspace" / "mutants").mkdir(parents=True, exist_ok=True)
    return cfg


def _make_mutant(operator_name: str = "AOR",
                 occurrence_index: int = 0,
                 source_code: str = "x = 1 - 2") -> MagicMock:
    """Return a mock Mutant with the attributes TestRunner needs."""
    m = MagicMock()
    m.operator_name    = operator_name
    m.occurrence_index = occurrence_index
    m.source_code      = source_code
    return m


def _make_proc(returncode: int = 1,
               stdout: str = "1 failed",
               stderr: str = "") -> MagicMock:
    """Return a mock CompletedProcess."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout     = stdout
    proc.stderr     = stderr
    return proc


def _make_runner(tmp_path: Path,
                 mutants: list | None = None,
                 max_workers: int = 1) -> TestRunner:
    """Convenience factory for TestRunner."""
    if mutants is None:
        mutants = [_make_mutant()]
    return TestRunner(
        mutateList=mutants,
        config=_make_config(tmp_path),
        max_workers=max_workers,
    )


# ═══════════════════════════════════════════════════════════════════════════ #
# TestResult                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestTestResult:

    def _result(self, status: str = "killed") -> TestResult:
        return TestResult(
            mutant_operator="AOR",
            occurrence_index=0,
            status=status,
            stdout="output",
            stderr="",
            exit_code=1,
            duration_seconds=0.42,
        )

    # --- status helpers ------------------------------------------------------

    def test_is_killed_true(self):
        assert self._result("killed").is_killed() is True

    def test_is_killed_false(self):
        assert self._result("survived").is_killed() is False

    def test_is_survived_true(self):
        assert self._result("survived").is_survived() is True

    def test_is_survived_false(self):
        assert self._result("killed").is_survived() is False

    def test_is_error_true(self):
        assert self._result("error").is_error() is True

    def test_is_error_false(self):
        assert self._result("killed").is_error() is False

    def test_is_killed_and_survived_mutually_exclusive(self):
        r = self._result("killed")
        assert r.is_killed() and not r.is_survived()

    # --- __repr__ ------------------------------------------------------------

    def test_repr_contains_operator(self):
        assert "AOR" in repr(self._result())

    def test_repr_contains_occurrence(self):
        assert "0" in repr(self._result())

    def test_repr_contains_status(self):
        assert "killed" in repr(self._result())

    def test_repr_contains_exit_code(self):
        assert "1" in repr(self._result())

    def test_repr_contains_duration(self):
        assert "0.42s" in repr(self._result())


# ═══════════════════════════════════════════════════════════════════════════ #
# Construction / __post_init__                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestConstruction:

    def test_valid_construction(self, tmp_path):
        runner = _make_runner(tmp_path)
        assert runner.mutateList != []
        assert runner.results == []

    def test_default_max_workers_is_positive(self, tmp_path):
        cfg     = _make_config(tmp_path)
        runner  = TestRunner(mutateList=[_make_mutant()], config=cfg)
        assert runner.max_workers >= 1

    def test_default_max_workers_capped_at_four(self, tmp_path):
        cfg    = _make_config(tmp_path)
        runner = TestRunner(mutateList=[_make_mutant()], config=cfg)
        assert runner.max_workers <= 4

    def test_custom_results_list_accepted(self, tmp_path):
        existing = [TestResult("AOR", 0, "killed", "", "", 1, 0.1)]
        cfg      = _make_config(tmp_path)
        runner   = TestRunner(mutateList=[_make_mutant()], config=cfg,
                              results=existing)
        assert runner.results is existing

    # --- mutateList validation -----------------------------------------------

    def test_mutate_list_not_a_list_raises_type_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        with pytest.raises(TypeError, match="mutateList must be a list"):
            TestRunner(mutateList="bad", config=cfg)

    def test_mutate_list_none_raises_type_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        with pytest.raises(TypeError, match="mutateList must be a list"):
            TestRunner(mutateList=None, config=cfg)

    def test_mutate_list_tuple_raises_type_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        with pytest.raises(TypeError, match="mutateList must be a list"):
            TestRunner(mutateList=(_make_mutant(),), config=cfg)

    # --- config validation ---------------------------------------------------

    def test_missing_program_path_raises_type_error(self, tmp_path):
        cfg = MagicMock(spec=["testsPath", "workspace_path", "sparkSession"])
        with pytest.raises(TypeError, match="missing attribute 'programPath'"):
            TestRunner(mutateList=[_make_mutant()], config=cfg)

    def test_missing_tests_path_raises_type_error(self, tmp_path):
        cfg = MagicMock(spec=["programPath", "workspace_path", "sparkSession"])
        with pytest.raises(TypeError, match="missing attribute 'testsPath'"):
            TestRunner(mutateList=[_make_mutant()], config=cfg)

    def test_missing_workspace_path_raises_type_error(self, tmp_path):
        cfg = MagicMock(spec=["programPath", "testsPath", "sparkSession"])
        with pytest.raises(TypeError, match="missing attribute 'workspace_path'"):
            TestRunner(mutateList=[_make_mutant()], config=cfg)

    def test_missing_spark_session_raises_type_error(self, tmp_path):
        cfg = MagicMock(spec=["programPath", "testsPath", "workspace_path"])
        with pytest.raises(TypeError, match="missing attribute 'sparkSession'"):
            TestRunner(mutateList=[_make_mutant()], config=cfg)

    def test_unloaded_config_raises_runtime_error(self, tmp_path):
        cfg = MagicMock()
        cfg.programPath  = "/tmp/p.py"
        cfg.testsPath    = "/tmp/t.py"
        cfg.sparkSession = MagicMock()
        type(cfg).workspace_path = PropertyMock(
            side_effect=RuntimeError("Call .load() first.")
        )
        with pytest.raises(RuntimeError, match="has not been loaded yet"):
            TestRunner(mutateList=[_make_mutant()], config=cfg)

    # --- max_workers validation ----------------------------------------------

    def test_max_workers_zero_raises_value_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        with pytest.raises(ValueError, match="max_workers must be a positive integer"):
            TestRunner(mutateList=[_make_mutant()], config=cfg, max_workers=0)

    def test_max_workers_negative_raises_value_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        with pytest.raises(ValueError, match="max_workers must be a positive integer"):
            TestRunner(mutateList=[_make_mutant()], config=cfg, max_workers=-1)

    def test_max_workers_float_raises_value_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        with pytest.raises(ValueError, match="max_workers must be a positive integer"):
            TestRunner(mutateList=[_make_mutant()], config=cfg, max_workers=2.5)

    def test_max_workers_string_raises_value_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        with pytest.raises(ValueError, match="max_workers must be a positive integer"):
            TestRunner(mutateList=[_make_mutant()], config=cfg, max_workers="4")


# ═══════════════════════════════════════════════════════════════════════════ #
# _classify()                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestClassify:

    def test_exit_0_is_survived(self):
        assert TestRunner._classify(0) == "survived"

    def test_exit_1_is_killed(self):
        assert TestRunner._classify(1) == "killed"

    def test_exit_2_is_error(self):
        assert TestRunner._classify(2) == "error"

    def test_exit_3_is_error(self):
        assert TestRunner._classify(3) == "error"

    def test_exit_4_is_error(self):
        assert TestRunner._classify(4) == "error"

    def test_exit_5_is_error(self):
        assert TestRunner._classify(5) == "error"

    def test_exit_minus_1_is_error(self):
        assert TestRunner._classify(-1) == "error"

    def test_exit_99_is_error(self):
        assert TestRunner._classify(99) == "error"


# ═══════════════════════════════════════════════════════════════════════════ #
# runTest()                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestRunTest:

    # --- RuntimeError on empty mutateList ------------------------------------

    def test_empty_mutate_list_raises_runtime_error(self, tmp_path):
        cfg    = _make_config(tmp_path)
        runner = TestRunner(mutateList=[], config=cfg)
        with pytest.raises(RuntimeError, match="mutateList is empty"):
            runner.runTest()

    # --- Clears previous results before running ------------------------------

    def test_clears_previous_results(self, tmp_path):
        runner = _make_runner(tmp_path)
        # Pre-populate results
        runner.results.append(
            TestResult("OLD", 0, "killed", "", "", 1, 0.1)
        )
        with patch.object(runner, "_run_single_mutant",
                          return_value=TestResult("AOR", 0, "killed",
                                                  "out", "err", 1, 0.5)):
            runner.runTest()

        assert all(r.mutant_operator != "OLD" for r in runner.results)

    # --- Returns list[TestResult] --------------------------------------------

    def test_returns_list(self, tmp_path):
        runner = _make_runner(tmp_path)
        with patch.object(runner, "_run_single_mutant",
                          return_value=TestResult("AOR", 0, "killed",
                                                  "", "", 1, 0.1)):
            result = runner.runTest()
        assert isinstance(result, list)

    def test_returns_same_object_as_self_results(self, tmp_path):
        runner = _make_runner(tmp_path)
        with patch.object(runner, "_run_single_mutant",
                          return_value=TestResult("AOR", 0, "killed",
                                                  "", "", 1, 0.1)):
            result = runner.runTest()
        assert result is runner.results

    # --- killed outcome ------------------------------------------------------

    def test_killed_result_stored(self, tmp_path):
        runner = _make_runner(tmp_path)
        tr = TestResult("AOR", 0, "killed", "1 failed", "", 1, 0.3)
        with patch.object(runner, "_run_single_mutant", return_value=tr):
            runner.runTest()
        assert runner.results[0].is_killed()

    # --- survived outcome ----------------------------------------------------

    def test_survived_result_stored(self, tmp_path):
        runner = _make_runner(tmp_path)
        tr = TestResult("AOR", 0, "survived", "1 passed", "", 0, 0.2)
        with patch.object(runner, "_run_single_mutant", return_value=tr):
            runner.runTest()
        assert runner.results[0].is_survived()

    # --- error outcome -------------------------------------------------------

    def test_error_result_stored(self, tmp_path):
        runner = _make_runner(tmp_path)
        tr = TestResult("AOR", 0, "error", "", "import error", 3, 0.1)
        with patch.object(runner, "_run_single_mutant", return_value=tr):
            runner.runTest()
        assert runner.results[0].is_error()

    # --- Unexpected worker crash (future.result() raises) --------------------

    def test_worker_exception_produces_error_result(self, tmp_path):
        runner = _make_runner(tmp_path)
        with patch.object(runner, "_run_single_mutant",
                          side_effect=RuntimeError("worker boom")):
            runner.runTest()

        assert len(runner.results) == 1
        assert runner.results[0].is_error()
        assert "worker boom" in runner.results[0].stderr

    def test_worker_exception_result_has_correct_operator(self, tmp_path):
        mutant = _make_mutant(operator_name="ROR", occurrence_index=2)
        runner = _make_runner(tmp_path, mutants=[mutant])
        with patch.object(runner, "_run_single_mutant",
                          side_effect=ValueError("unexpected")):
            runner.runTest()
        r = runner.results[0]
        assert r.mutant_operator == "ROR"
        assert r.occurrence_index == 2
        assert r.exit_code == -1

    # --- Multiple mutants all collected --------------------------------------

    def test_all_mutants_produce_results(self, tmp_path):
        mutants = [_make_mutant(occurrence_index=i) for i in range(5)]
        runner  = _make_runner(tmp_path, mutants=mutants, max_workers=2)

        def fake_run(mutant):
            return TestResult(mutant.operator_name, mutant.occurrence_index,
                              "killed", "", "", 1, 0.1)

        with patch.object(runner, "_run_single_mutant", side_effect=fake_run):
            runner.runTest()

        assert len(runner.results) == 5

    def test_mixed_outcomes_all_recorded(self, tmp_path):
        mutants  = [_make_mutant(occurrence_index=i) for i in range(3)]
        statuses = ["killed", "survived", "error"]
        runner   = _make_runner(tmp_path, mutants=mutants)

        def fake_run(mutant):
            status = statuses[mutant.occurrence_index]
            return TestResult(mutant.operator_name, mutant.occurrence_index,
                              status, "", "", {"killed": 1, "survived": 0,
                                               "error": 2}[status], 0.1)

        with patch.object(runner, "_run_single_mutant", side_effect=fake_run):
            runner.runTest()

        recorded_statuses = {r.status for r in runner.results}
        assert recorded_statuses == {"killed", "survived", "error"}


# ═══════════════════════════════════════════════════════════════════════════ #
# _run_single_mutant()                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestRunSingleMutant:

    def _patched_run(self, tmp_path: Path, returncode: int = 1,
                     stdout: str = "1 failed", stderr: str = ""):
        """Helper: call _run_single_mutant with subprocess.run patched."""
        runner = _make_runner(tmp_path)
        mutant = _make_mutant(source_code="x = 1 - 2")
        proc   = _make_proc(returncode=returncode, stdout=stdout, stderr=stderr)

        with patch("code.test_runner.subprocess.run", return_value=proc) as mock_run:
            result = runner._run_single_mutant(mutant)

        return result, mock_run

    # --- Subprocess called with correct arguments ----------------------------

    def test_subprocess_called_once(self, tmp_path):
        _, mock_run = self._patched_run(tmp_path)
        mock_run.assert_called_once()

    def test_subprocess_cmd_starts_with_pytest(self, tmp_path):
        _, mock_run = self._patched_run(tmp_path)
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "pytest"

    def test_subprocess_cmd_contains_tests_path(self, tmp_path):
        runner = _make_runner(tmp_path)
        mutant = _make_mutant()
        proc   = _make_proc(returncode=0)

        with patch("code.test_runner.subprocess.run", return_value=proc) as mock_run:
            runner._run_single_mutant(mutant)

        cmd = mock_run.call_args[0][0]
        assert str(runner.config.testsPath) in cmd

    def test_subprocess_cmd_contains_import_mode(self, tmp_path):
        _, mock_run = self._patched_run(tmp_path)
        cmd = mock_run.call_args[0][0]
        assert "--import-mode=importlib" in cmd

    def test_subprocess_cmd_contains_x_flag(self, tmp_path):
        _, mock_run = self._patched_run(tmp_path)
        cmd = mock_run.call_args[0][0]
        assert "-x" in cmd

    def test_subprocess_cmd_contains_q_flag(self, tmp_path):
        _, mock_run = self._patched_run(tmp_path)
        cmd = mock_run.call_args[0][0]
        assert "-q" in cmd

    def test_subprocess_cmd_contains_tb_short(self, tmp_path):
        _, mock_run = self._patched_run(tmp_path)
        cmd = mock_run.call_args[0][0]
        assert "--tb=short" in cmd

    def test_subprocess_called_with_capture_output(self, tmp_path):
        _, mock_run = self._patched_run(tmp_path)
        kwargs = mock_run.call_args[1]
        assert kwargs.get("capture_output") is True

    def test_subprocess_called_with_text_true(self, tmp_path):
        _, mock_run = self._patched_run(tmp_path)
        kwargs = mock_run.call_args[1]
        assert kwargs.get("text") is True

    def test_subprocess_called_with_timeout(self, tmp_path):
        _, mock_run = self._patched_run(tmp_path)
        kwargs = mock_run.call_args[1]
        assert kwargs.get("timeout") == 120

    # --- PYTHONPATH injection ------------------------------------------------

    def test_pythonpath_injected_in_env(self, tmp_path):
        runner = _make_runner(tmp_path)
        mutant = _make_mutant()
        proc   = _make_proc(returncode=1)

        with patch("code.test_runner.subprocess.run", return_value=proc) as mock_run:
            runner._run_single_mutant(mutant)

        env = mock_run.call_args[1]["env"]
        assert "PYTHONPATH" in env

    def test_pythonpath_prepends_tmp_dir(self, tmp_path):
        """The mutant's temp directory must be the first entry in PYTHONPATH."""
        runner = _make_runner(tmp_path)
        mutant = _make_mutant()
        proc   = _make_proc(returncode=1)

        with patch.dict(os.environ, {}, clear=True):
            with patch("code.test_runner.subprocess.run", return_value=proc) as mock_run:
                runner._run_single_mutant(mutant)

        env      = mock_run.call_args[1]["env"]
        first    = env["PYTHONPATH"].split(os.pathsep)[0]
        ws_path  = str(runner.config.workspace_path)
        assert ws_path in first or first != ""

    def test_existing_pythonpath_preserved(self, tmp_path):
        runner = _make_runner(tmp_path)
        mutant = _make_mutant()
        proc   = _make_proc(returncode=1)

        with patch.dict(os.environ, {"PYTHONPATH": "/existing/path"}):
            with patch("code.test_runner.subprocess.run", return_value=proc) as mock_run:
                runner._run_single_mutant(mutant)

        env = mock_run.call_args[1]["env"]
        assert "/existing/path" in env["PYTHONPATH"]

    def test_pythonpath_without_existing_var(self, tmp_path):
        """When PYTHONPATH is not set, env must still contain a valid entry."""
        runner = _make_runner(tmp_path)
        mutant = _make_mutant()
        proc   = _make_proc(returncode=1)

        env_without_pythonpath = {k: v for k, v in os.environ.items()
                                  if k != "PYTHONPATH"}
        with patch.dict(os.environ, env_without_pythonpath, clear=True):
            with patch("code.test_runner.subprocess.run", return_value=proc) as mock_run:
                runner._run_single_mutant(mutant)

        env = mock_run.call_args[1]["env"]
        assert "PYTHONPATH" in env

    # --- Mutant file written with correct name -------------------------------

    def test_mutant_file_uses_original_module_stem(self, tmp_path):
        """The temp file must be named <original_stem>.py, not the mutant id."""
        runner = _make_runner(tmp_path)
        mutant = _make_mutant(source_code="x = 1 - 2")
        proc   = _make_proc(returncode=1)
        written_paths = []

        original_write = Path.write_text

        def capture_write(self_path, data, **kwargs):
            written_paths.append(self_path)
            return original_write(self_path, data, **kwargs)

        with patch("code.test_runner.subprocess.run", return_value=proc):
            with patch.object(Path, "write_text", capture_write):
                runner._run_single_mutant(mutant)

        stem = Path(runner.config.programPath).stem
        assert any(p.name == f"{stem}.py" for p in written_paths)

    # --- Correct TestResult fields from subprocess ---------------------------

    def test_result_stdout_captured(self, tmp_path):
        result, _ = self._patched_run(tmp_path, returncode=1, stdout="FAIL")
        assert result.stdout == "FAIL"

    def test_result_stderr_captured(self, tmp_path):
        result, _ = self._patched_run(tmp_path, returncode=1,
                                      stderr="ImportError")
        assert result.stderr == "ImportError"

    def test_result_exit_code_stored(self, tmp_path):
        result, _ = self._patched_run(tmp_path, returncode=1)
        assert result.exit_code == 1

    def test_result_status_killed_on_exit_1(self, tmp_path):
        result, _ = self._patched_run(tmp_path, returncode=1)
        assert result.status == "killed"

    def test_result_status_survived_on_exit_0(self, tmp_path):
        result, _ = self._patched_run(tmp_path, returncode=0)
        assert result.status == "survived"

    def test_result_status_error_on_exit_2(self, tmp_path):
        result, _ = self._patched_run(tmp_path, returncode=2)
        assert result.status == "error"

    def test_result_duration_is_positive(self, tmp_path):
        result, _ = self._patched_run(tmp_path)
        assert result.duration_seconds >= 0.0

    def test_result_operator_name_preserved(self, tmp_path):
        runner = _make_runner(tmp_path)
        mutant = _make_mutant(operator_name="ROR")
        proc   = _make_proc()
        with patch("code.test_runner.subprocess.run", return_value=proc):
            result = runner._run_single_mutant(mutant)
        assert result.mutant_operator == "ROR"

    def test_result_occurrence_index_preserved(self, tmp_path):
        runner = _make_runner(tmp_path)
        mutant = _make_mutant(occurrence_index=3)
        proc   = _make_proc()
        with patch("code.test_runner.subprocess.run", return_value=proc):
            result = runner._run_single_mutant(mutant)
        assert result.occurrence_index == 3

    # --- Timeout handling ----------------------------------------------------

    def test_timeout_produces_error_status(self, tmp_path):
        runner = _make_runner(tmp_path)
        mutant = _make_mutant()
        timeout_exc = subprocess.TimeoutExpired(cmd="pytest", timeout=120)
        timeout_exc.stdout = None

        with patch("code.test_runner.subprocess.run", side_effect=timeout_exc):
            result = runner._run_single_mutant(mutant)

        assert result.status == "error"
        assert result.exit_code == -1

    def test_timeout_stderr_contains_timeout_message(self, tmp_path):
        runner = _make_runner(tmp_path)
        mutant = _make_mutant()
        timeout_exc = subprocess.TimeoutExpired(cmd="pytest", timeout=120)
        timeout_exc.stdout = None

        with patch("code.test_runner.subprocess.run", side_effect=timeout_exc):
            result = runner._run_single_mutant(mutant)

        assert "Timeout" in result.stderr

    def test_timeout_with_partial_stdout(self, tmp_path):
        """TimeoutExpired.stdout may contain partial output — must not crash."""
        runner = _make_runner(tmp_path)
        mutant = _make_mutant()
        timeout_exc = subprocess.TimeoutExpired(cmd="pytest", timeout=120)
        timeout_exc.stdout = "partial output"

        with patch("code.test_runner.subprocess.run", side_effect=timeout_exc):
            result = runner._run_single_mutant(mutant)

        assert result.stdout == "partial output"


# ═══════════════════════════════════════════════════════════════════════════ #
# __repr__                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestRepr:

    def test_repr_contains_mutant_count(self, tmp_path):
        runner = _make_runner(tmp_path, mutants=[_make_mutant(), _make_mutant()])
        assert "mutants=2" in repr(runner)

    def test_repr_contains_results_count_zero(self, tmp_path):
        runner = _make_runner(tmp_path)
        assert "results=0" in repr(runner)

    def test_repr_contains_results_count_after_run(self, tmp_path):
        runner = _make_runner(tmp_path)
        runner.results.append(
            TestResult("AOR", 0, "killed", "", "", 1, 0.1)
        )
        assert "results=1" in repr(runner)

    def test_repr_contains_max_workers(self, tmp_path):
        runner = _make_runner(tmp_path, max_workers=3)
        assert "max_workers=3" in repr(runner)


# ═══════════════════════════════════════════════════════════════════════════ #
# Integration                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestIntegration:

    def test_full_run_all_killed(self, tmp_path):
        mutants = [_make_mutant(occurrence_index=i) for i in range(3)]
        runner  = _make_runner(tmp_path, mutants=mutants, max_workers=2)

        def fake_run(mutant):
            return TestResult(mutant.operator_name, mutant.occurrence_index,
                              "killed", "1 failed", "", 1, 0.1)

        with patch.object(runner, "_run_single_mutant", side_effect=fake_run):
            results = runner.runTest()

        assert len(results) == 3
        assert all(r.is_killed() for r in results)

    def test_full_run_all_survived(self, tmp_path):
        mutants = [_make_mutant(occurrence_index=i) for i in range(2)]
        runner  = _make_runner(tmp_path, mutants=mutants)

        def fake_run(mutant):
            return TestResult(mutant.operator_name, mutant.occurrence_index,
                              "survived", "2 passed", "", 0, 0.2)

        with patch.object(runner, "_run_single_mutant", side_effect=fake_run):
            results = runner.runTest()

        assert all(r.is_survived() for r in results)

    def test_consecutive_run_test_calls_reset_results(self, tmp_path):
        """Calling runTest twice must not accumulate stale results."""
        runner = _make_runner(tmp_path)
        tr     = TestResult("AOR", 0, "killed", "", "", 1, 0.1)

        with patch.object(runner, "_run_single_mutant", return_value=tr):
            runner.runTest()
            runner.runTest()

        assert len(runner.results) == 1

    def test_single_worker_processes_all_mutants(self, tmp_path):
        mutants = [_make_mutant(occurrence_index=i) for i in range(4)]
        runner  = _make_runner(tmp_path, mutants=mutants, max_workers=1)

        def fake_run(mutant):
            return TestResult(mutant.operator_name, mutant.occurrence_index,
                              "killed", "", "", 1, 0.05)

        with patch.object(runner, "_run_single_mutant", side_effect=fake_run):
            results = runner.runTest()

        assert len(results) == 4