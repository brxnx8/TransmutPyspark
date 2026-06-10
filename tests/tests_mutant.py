from __future__ import annotations

from pathlib import Path

import pytest

from src.model.mutant import Mutant


# ===========================================================================
# Fixtures compartilhadas
# ===========================================================================

@pytest.fixture()
def minimal_mutant() -> Mutant:
    """Mutant com apenas os campos obrigatórios preenchidos."""
    return Mutant(
        id=1,
        operator="AOR",
        original_path="/src/module.py",
        mutant_path="/workspace/mutants/module_1.py",
        modified_line="line 42: a + b → a - b",
    )


@pytest.fixture()
def full_mutant(tmp_path: Path) -> Mutant:
    """Mutant com todos os campos, incluindo test_files e test_functions."""
    t1 = tmp_path / "test_module.py"
    t2 = tmp_path / "test_integration.py"
    t1.write_text("")
    t2.write_text("")
    return Mutant(
        id=7,
        operator="ROR",
        original_path="/src/calc.py",
        mutant_path="/workspace/mutants/calc_7.py",
        modified_line="line 10: x > y → x >= y",
        test_files=[t1, t2],
        test_functions=["test_greater_than", "test_boundary"],
    )


# ===========================================================================
# Instanciação e campos obrigatórios
# ===========================================================================

class TestMutantInstantiation:

    def test_should_create_mutant_with_all_required_fields(
        self, minimal_mutant: Mutant
    ):
        assert minimal_mutant.id == 1
        assert minimal_mutant.operator == "AOR"
        assert minimal_mutant.original_path == "/src/module.py"
        assert minimal_mutant.mutant_path == "/workspace/mutants/module_1.py"
        assert minimal_mutant.modified_line == "line 42: a + b → a - b"

    def test_should_store_id_as_integer(self, minimal_mutant: Mutant):
        assert isinstance(minimal_mutant.id, int)

    def test_should_store_operator_as_string(self, minimal_mutant: Mutant):
        assert isinstance(minimal_mutant.operator, str)

    def test_should_store_original_path_as_string(self, minimal_mutant: Mutant):
        assert isinstance(minimal_mutant.original_path, str)

    def test_should_store_mutant_path_as_string(self, minimal_mutant: Mutant):
        assert isinstance(minimal_mutant.mutant_path, str)

    def test_should_store_modified_line_as_string(self, minimal_mutant: Mutant):
        assert isinstance(minimal_mutant.modified_line, str)

    def test_should_accept_zero_as_valid_id(self):
        m = Mutant(id=0, operator="LCR", original_path="a.py",
                   mutant_path="a_0.py", modified_line="line 1")
        assert m.id == 0

    def test_should_accept_negative_id(self):
        m = Mutant(id=-1, operator="LCR", original_path="a.py",
                   mutant_path="a_neg.py", modified_line="line 1")
        assert m.id == -1

    def test_should_accept_large_id(self):
        m = Mutant(id=999_999, operator="SVR", original_path="big.py",
                   mutant_path="big_999999.py", modified_line="line 99")
        assert m.id == 999_999

    def test_should_accept_empty_string_for_operator(self):
        m = Mutant(id=1, operator="", original_path="a.py",
                   mutant_path="a_1.py", modified_line="line 1")
        assert m.operator == ""

    def test_should_accept_empty_string_for_original_path(self):
        m = Mutant(id=1, operator="AOR", original_path="",
                   mutant_path="a_1.py", modified_line="line 1")
        assert m.original_path == ""

    def test_should_accept_empty_string_for_mutant_path(self):
        m = Mutant(id=1, operator="AOR", original_path="a.py",
                   mutant_path="", modified_line="line 1")
        assert m.mutant_path == ""

    def test_should_accept_empty_string_for_modified_line(self):
        m = Mutant(id=1, operator="AOR", original_path="a.py",
                   mutant_path="a_1.py", modified_line="")
        assert m.modified_line == ""


# ===========================================================================
# Campos opcionais com defaults
# ===========================================================================

class TestMutantDefaults:

    def test_should_initialize_test_files_as_empty_list_by_default(
        self, minimal_mutant: Mutant
    ):
        assert minimal_mutant.test_files == []

    def test_should_initialize_test_functions_as_empty_list_by_default(
        self, minimal_mutant: Mutant
    ):
        assert minimal_mutant.test_functions == []

    def test_should_not_share_test_files_list_between_instances(self):
        m1 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1")
        m2 = Mutant(id=2, operator="AOR", original_path="b.py",
                    mutant_path="b_2.py", modified_line="l2")
        m1.test_files.append(Path("test_x.py"))
        assert m2.test_files == [], (
            "test_files default não deve ser compartilhado entre instâncias"
        )

    def test_should_not_share_test_functions_list_between_instances(self):
        m1 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1")
        m2 = Mutant(id=2, operator="AOR", original_path="b.py",
                    mutant_path="b_2.py", modified_line="l2")
        m1.test_functions.append("test_something")
        assert m2.test_functions == [], (
            "test_functions default não deve ser compartilhado entre instâncias"
        )

    def test_should_accept_explicit_test_files_list(self, full_mutant: Mutant):
        assert len(full_mutant.test_files) == 2
        assert all(isinstance(f, Path) for f in full_mutant.test_files)

    def test_should_accept_explicit_test_functions_list(self, full_mutant: Mutant):
        assert full_mutant.test_functions == ["test_greater_than", "test_boundary"]

    def test_should_accept_single_test_file(self, tmp_path: Path):
        t = tmp_path / "test_one.py"
        t.write_text("")
        m = Mutant(id=1, operator="AOR", original_path="a.py",
                   mutant_path="a_1.py", modified_line="l1",
                   test_files=[t])
        assert m.test_files == [t]

    def test_should_accept_single_test_function(self):
        m = Mutant(id=1, operator="AOR", original_path="a.py",
                   mutant_path="a_1.py", modified_line="l1",
                   test_functions=["test_only"])
        assert m.test_functions == ["test_only"]

    def test_should_allow_mutation_of_test_files_after_creation(
        self, minimal_mutant: Mutant, tmp_path: Path
    ):
        new_test = tmp_path / "test_new.py"
        new_test.write_text("")
        minimal_mutant.test_files.append(new_test)
        assert new_test in minimal_mutant.test_files

    def test_should_allow_mutation_of_test_functions_after_creation(
        self, minimal_mutant: Mutant
    ):
        minimal_mutant.test_functions.append("test_added")
        assert "test_added" in minimal_mutant.test_functions


# ===========================================================================
# __repr__
# ===========================================================================

class TestMutantRepr:

    def test_should_start_with_mutant_class_name(self, minimal_mutant: Mutant):
        assert repr(minimal_mutant).startswith("Mutant(")

    def test_should_include_id_in_repr(self, minimal_mutant: Mutant):
        assert "id=1" in repr(minimal_mutant)

    def test_should_include_operator_in_repr(self, minimal_mutant: Mutant):
        assert "operator='AOR'" in repr(minimal_mutant)

    def test_should_include_original_path_in_repr(self, minimal_mutant: Mutant):
        assert "original='/src/module.py'" in repr(minimal_mutant)

    def test_should_include_mutant_path_in_repr(self, minimal_mutant: Mutant):
        assert "mutant='/workspace/mutants/module_1.py'" in repr(minimal_mutant)

    def test_should_include_modified_line_in_repr(self, minimal_mutant: Mutant):
        assert "line='line 42: a + b → a - b'" in repr(minimal_mutant)

    def test_should_show_empty_test_files_list_in_repr_when_default(
        self, minimal_mutant: Mutant
    ):
        assert "test_files=[]" in repr(minimal_mutant)

    def test_should_show_empty_test_functions_list_in_repr_when_default(
        self, minimal_mutant: Mutant
    ):
        assert "test_functions=[]" in repr(minimal_mutant)

    def test_should_show_only_file_names_in_test_files_repr(
        self, full_mutant: Mutant
    ):
        r = repr(full_mutant)
        assert "test_module.py" in r
        assert "test_integration.py" in r

    def test_should_not_show_full_path_in_test_files_repr(
        self, full_mutant: Mutant, tmp_path: Path
    ):
        r = repr(full_mutant)
        # O repr usa f.name, então o diretório pai não deve aparecer em test_files
        assert str(tmp_path) not in r.split("test_files=")[1].split("]")[0]

    def test_should_show_test_functions_names_in_repr(self, full_mutant: Mutant):
        r = repr(full_mutant)
        assert "test_greater_than" in r
        assert "test_boundary" in r

    def test_should_include_correct_id_in_repr_for_full_mutant(
        self, full_mutant: Mutant
    ):
        assert "id=7" in repr(full_mutant)

    def test_should_return_string_type_from_repr(self, minimal_mutant: Mutant):
        assert isinstance(repr(minimal_mutant), str)

    def test_should_reflect_updated_test_functions_in_repr(
        self, minimal_mutant: Mutant
    ):
        minimal_mutant.test_functions = ["test_x", "test_y"]
        r = repr(minimal_mutant)
        assert "test_x" in r
        assert "test_y" in r

    def test_should_reflect_updated_test_files_in_repr(
        self, minimal_mutant: Mutant, tmp_path: Path
    ):
        t = tmp_path / "test_dynamic.py"
        t.write_text("")
        minimal_mutant.test_files = [t]
        assert "test_dynamic.py" in repr(minimal_mutant)


# ===========================================================================
# Igualdade e identidade (dataclass)
# ===========================================================================

class TestMutantEquality:

    def test_should_be_equal_when_all_fields_are_identical(self):
        m1 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1")
        m2 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1")
        assert m1 == m2

    def test_should_not_be_equal_when_ids_differ(self):
        m1 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1")
        m2 = Mutant(id=2, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1")
        assert m1 != m2

    def test_should_not_be_equal_when_operators_differ(self):
        m1 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1")
        m2 = Mutant(id=1, operator="ROR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1")
        assert m1 != m2

    def test_should_not_be_equal_when_original_paths_differ(self):
        m1 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1")
        m2 = Mutant(id=1, operator="AOR", original_path="b.py",
                    mutant_path="a_1.py", modified_line="l1")
        assert m1 != m2

    def test_should_not_be_equal_when_mutant_paths_differ(self):
        m1 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1")
        m2 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_2.py", modified_line="l1")
        assert m1 != m2

    def test_should_not_be_equal_when_modified_lines_differ(self):
        m1 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1")
        m2 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l2")
        assert m1 != m2

    def test_should_not_be_equal_when_test_functions_differ(self):
        m1 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1",
                    test_functions=["test_a"])
        m2 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1",
                    test_functions=["test_b"])
        assert m1 != m2

    def test_should_not_be_same_object_when_created_separately(self):
        m1 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1")
        m2 = Mutant(id=1, operator="AOR", original_path="a.py",
                    mutant_path="a_1.py", modified_line="l1")
        assert m1 is not m2


# ===========================================================================
# Mutabilidade dos campos
# ===========================================================================

class TestMutantMutability:

    def test_should_allow_updating_id_after_creation(self, minimal_mutant: Mutant):
        minimal_mutant.id = 99
        assert minimal_mutant.id == 99

    def test_should_allow_updating_operator_after_creation(
        self, minimal_mutant: Mutant
    ):
        minimal_mutant.operator = "LCR"
        assert minimal_mutant.operator == "LCR"

    def test_should_allow_replacing_test_files_list_after_creation(
        self, minimal_mutant: Mutant, tmp_path: Path
    ):
        new_files = [tmp_path / "test_new.py"]
        (tmp_path / "test_new.py").write_text("")
        minimal_mutant.test_files = new_files
        assert minimal_mutant.test_files == new_files

    def test_should_allow_replacing_test_functions_list_after_creation(
        self, minimal_mutant: Mutant
    ):
        minimal_mutant.test_functions = ["test_replaced"]
        assert minimal_mutant.test_functions == ["test_replaced"]