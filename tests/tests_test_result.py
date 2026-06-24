from __future__ import annotations

import pytest

from src.model.test_result import TestResult


# ===========================================================================
# Fixtures compartilhadas
# ===========================================================================

@pytest.fixture()
def killed_result() -> TestResult:
    """TestResult representando um mutante morto."""
    return TestResult(
        mutant=1,
        status="Killed",
        failed_tests=["test_add", "test_subtract"],
        execution_time=0.42,
    )


@pytest.fixture()
def survived_result() -> TestResult:
    """TestResult representando um mutante sobrevivente."""
    return TestResult(
        mutant=2,
        status="Survived",
        failed_tests=[],
        execution_time=1.05,
    )


@pytest.fixture()
def timeout_result() -> TestResult:
    """TestResult representando um mutante que excedeu o tempo."""
    return TestResult(
        mutant=3,
        status="Timeout",
        failed_tests=[],
        execution_time=30.0,
    )


# ===========================================================================
# Instanciação e campos obrigatórios
# ===========================================================================

class TestTestResultInstantiation:

    def test_should_create_instance_with_all_required_fields(
        self, killed_result: TestResult
    ):
        assert killed_result.mutant == 1
        assert killed_result.status == "Killed"
        assert killed_result.failed_tests == ["test_add", "test_subtract"]
        assert killed_result.execution_time == 0.42

    def test_should_store_mutant_as_integer(self, killed_result: TestResult):
        assert isinstance(killed_result.mutant, int)

    def test_should_store_status_as_string(self, killed_result: TestResult):
        assert isinstance(killed_result.status, str)

    def test_should_store_failed_tests_as_list(self, killed_result: TestResult):
        assert isinstance(killed_result.failed_tests, list)

    def test_should_store_execution_time_as_float(self, killed_result: TestResult):
        assert isinstance(killed_result.execution_time, float)

    def test_should_accept_zero_as_mutant_id(self):
        r = TestResult(mutant=0, status="Killed", failed_tests=[], execution_time=0.1)
        assert r.mutant == 0

    def test_should_accept_negative_mutant_id(self):
        r = TestResult(mutant=-1, status="Survived", failed_tests=[], execution_time=0.0)
        assert r.mutant == -1

    def test_should_accept_large_mutant_id(self):
        r = TestResult(mutant=999_999, status="Killed", failed_tests=[], execution_time=0.5)
        assert r.mutant == 999_999

    def test_should_accept_empty_string_for_status(self):
        r = TestResult(mutant=1, status="", failed_tests=[], execution_time=0.0)
        assert r.status == ""

    def test_should_accept_arbitrary_status_string(self):
        r = TestResult(mutant=1, status="CompileError", failed_tests=[], execution_time=0.0)
        assert r.status == "CompileError"

    def test_should_accept_empty_list_for_failed_tests(
        self, survived_result: TestResult
    ):
        assert survived_result.failed_tests == []

    def test_should_accept_single_item_in_failed_tests(self):
        r = TestResult(mutant=1, status="Killed", failed_tests=["test_x"], execution_time=0.1)
        assert r.failed_tests == ["test_x"]

    def test_should_accept_many_items_in_failed_tests(self):
        tests = [f"test_{i}" for i in range(50)]
        r = TestResult(mutant=1, status="Killed", failed_tests=tests, execution_time=0.9)
        assert r.failed_tests == tests

    def test_should_accept_zero_execution_time(self):
        r = TestResult(mutant=1, status="Killed", failed_tests=[], execution_time=0.0)
        assert r.execution_time == 0.0

    def test_should_accept_very_small_execution_time(self):
        r = TestResult(mutant=1, status="Killed", failed_tests=[], execution_time=1e-9)
        assert r.execution_time == pytest.approx(1e-9)

    def test_should_accept_large_execution_time_for_timeout(
        self, timeout_result: TestResult
    ):
        assert timeout_result.execution_time == 30.0

    def test_should_accept_integer_as_execution_time(self):
        # Python aceita int onde float é esperado em dataclasses sem validação
        r = TestResult(mutant=1, status="Killed", failed_tests=[], execution_time=1)
        assert r.execution_time == 1


# ===========================================================================
# Igualdade (dataclass __eq__ gerado automaticamente)
# ===========================================================================

class TestTestResultEquality:

    def test_should_be_equal_when_all_fields_are_identical(self):
        r1 = TestResult(mutant=1, status="Killed", failed_tests=["t"], execution_time=0.5)
        r2 = TestResult(mutant=1, status="Killed", failed_tests=["t"], execution_time=0.5)
        assert r1 == r2

    def test_should_not_be_equal_when_mutant_ids_differ(self):
        r1 = TestResult(mutant=1, status="Killed", failed_tests=[], execution_time=0.5)
        r2 = TestResult(mutant=2, status="Killed", failed_tests=[], execution_time=0.5)
        assert r1 != r2

    def test_should_not_be_equal_when_statuses_differ(self):
        r1 = TestResult(mutant=1, status="Killed", failed_tests=[], execution_time=0.5)
        r2 = TestResult(mutant=1, status="Survived", failed_tests=[], execution_time=0.5)
        assert r1 != r2

    def test_should_not_be_equal_when_failed_tests_differ(self):
        r1 = TestResult(mutant=1, status="Killed", failed_tests=["test_a"], execution_time=0.5)
        r2 = TestResult(mutant=1, status="Killed", failed_tests=["test_b"], execution_time=0.5)
        assert r1 != r2

    def test_should_not_be_equal_when_execution_times_differ(self):
        r1 = TestResult(mutant=1, status="Killed", failed_tests=[], execution_time=0.1)
        r2 = TestResult(mutant=1, status="Killed", failed_tests=[], execution_time=0.9)
        assert r1 != r2

    def test_should_not_be_same_object_when_created_separately(self):
        r1 = TestResult(mutant=1, status="Killed", failed_tests=[], execution_time=0.5)
        r2 = TestResult(mutant=1, status="Killed", failed_tests=[], execution_time=0.5)
        assert r1 is not r2


# ===========================================================================
# Mutabilidade dos campos
# ===========================================================================

class TestTestResultMutability:

    def test_should_allow_updating_mutant_id_after_creation(
        self, killed_result: TestResult
    ):
        killed_result.mutant = 99
        assert killed_result.mutant == 99

    def test_should_allow_updating_status_after_creation(
        self, killed_result: TestResult
    ):
        killed_result.status = "Survived"
        assert killed_result.status == "Survived"

    def test_should_allow_replacing_failed_tests_list_after_creation(
        self, killed_result: TestResult
    ):
        killed_result.failed_tests = ["test_new"]
        assert killed_result.failed_tests == ["test_new"]

    def test_should_allow_appending_to_failed_tests_after_creation(
        self, killed_result: TestResult
    ):
        original_len = len(killed_result.failed_tests)
        killed_result.failed_tests.append("test_extra")
        assert len(killed_result.failed_tests) == original_len + 1

    def test_should_allow_clearing_failed_tests_list(
        self, killed_result: TestResult
    ):
        killed_result.failed_tests.clear()
        assert killed_result.failed_tests == []

    def test_should_allow_updating_execution_time_after_creation(
        self, killed_result: TestResult
    ):
        killed_result.execution_time = 9.99
        assert killed_result.execution_time == pytest.approx(9.99)


# ===========================================================================
# __repr__
# ===========================================================================

class TestTestResultRepr:

    # [BUG] O __repr__ retorna "ConfigLoader(" em vez de "TestResult(".
    # Os testes abaixo documentam o comportamento ATUAL do código.
    # Quando o bug for corrigido, substituir "ConfigLoader(" por "TestResult(".

    def test_should_start_with_config_loader_due_to_bug_in_repr(
        self, killed_result: TestResult
    ):
        # [BUG] deveria ser "TestResult("
        assert repr(killed_result).startswith("ConfigLoader(")

    def test_should_return_string_type_from_repr(self, killed_result: TestResult):
        assert isinstance(repr(killed_result), str)

    def test_should_include_mutant_id_in_repr(self, killed_result: TestResult):
        assert "mutant=1" in repr(killed_result)

    def test_should_include_status_in_repr(self, killed_result: TestResult):
        assert "status='Killed'" in repr(killed_result)

    def test_should_include_failed_tests_in_repr(self, killed_result: TestResult):
        r = repr(killed_result)
        assert "failed_tests=" in r
        assert "test_add" in r
        assert "test_subtract" in r

    def test_should_include_execution_time_in_repr(self, killed_result: TestResult):
        assert "execution_time=0.42" in repr(killed_result)

    def test_should_show_empty_list_for_failed_tests_in_repr(
        self, survived_result: TestResult
    ):
        assert "failed_tests=[]" in repr(survived_result)

    def test_should_reflect_updated_status_in_repr(
        self, killed_result: TestResult
    ):
        killed_result.status = "Timeout"
        assert "status='Timeout'" in repr(killed_result)

    def test_should_reflect_updated_failed_tests_in_repr(
        self, killed_result: TestResult
    ):
        killed_result.failed_tests = ["test_only"]
        assert "test_only" in repr(killed_result)

    def test_should_reflect_updated_execution_time_in_repr(
        self, killed_result: TestResult
    ):
        killed_result.execution_time = 3.14
        assert "execution_time=3.14" in repr(killed_result)

    def test_should_not_include_test_result_class_name_due_to_bug_in_repr(
        self, killed_result: TestResult
    ):
        # [BUG] confirma que "TestResult(" está ausente no repr atual
        assert not repr(killed_result).startswith("TestResult(")
