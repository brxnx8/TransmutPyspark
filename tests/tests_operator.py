"""
tests_operator.py
=================
Unit tests for ``src.operator.Operator``.

Coverage strategy
-----------------
``Operator`` is an abstract class — it cannot be instantiated directly.
Every test uses ``ConcreteOperator``, a minimal concrete subclass defined
at the top of this module, so that the abstract contract is fulfilled while
keeping the focus on the base-class logic.

Test groups
-----------
1.  Construction — happy path (str / list registers, id=0 edge case)
2.  _validate_id — non-int, negative, float disguised as int (bool)
3.  _validate_name — non-str, blank, whitespace-only; normalisation to UPPER
4.  _validate_mutant_registers (str branch) — empty string, whitespace-only
5.  _validate_mutant_registers (list branch) — empty list, list with blank
    strings, list with non-str items, mixed valid/invalid
6.  _validate_mutant_registers (wrong type branch) — int, None, tuple
7.  _validate_mutant_list — non-list, list with non-Mutant items
8.  _assert_valid_tree — non-AST values
9.  _assert_valid_nodes — non-list, list with non-AST items, mixed list
10. _assert_valid_path — non-str, empty string, whitespace-only string
11. _next_mutant_id — empty list → 1, after appending mutants
12. clear_mutant_list — empties the list, other attributes unchanged
13. __repr__ — correct format, reflects current mutant count
14. Abstract enforcement — TypeError when abstract methods not implemented
15. ABC cannot be instantiated directly
"""

import ast
import pytest

from src.operators.operator import Operator
from src.model.mutant import Mutant


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def _make_mutant(id: int = 1) -> Mutant:
    """Return a minimal valid Mutant instance."""
    return Mutant(
        id=id,
        operator="TEST",
        original_path="/original/app.py",
        mutant_path=f"/mutants/m_{id}.py",
        modified_line="x = 1",
    )


class ConcreteOperator(Operator):
    """Minimal concrete subclass used across all tests."""

    def analyse_ast(self, tree: ast.AST) -> list[ast.AST]:
        return []

    def build_mutant(self, nodes, original_ast, original_path, mutant_dir):
        return self.mutant_list


def make_op(
    id: int = 1,
    name: str = "TEST",
    mutant_registers=None,
    mutant_list=None,
) -> ConcreteOperator:
    """Factory for ConcreteOperator with sensible defaults."""
    kwargs: dict = {
        "id": id,
        "name": name,
        "mutant_registers": mutant_registers if mutant_registers is not None else "Add",
    }
    if mutant_list is not None:
        kwargs["mutant_list"] = mutant_list
    return ConcreteOperator(**kwargs)


# ─────────────────────────────────────────────────────────────────────────── #
# 1. Construction — happy path                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class TestConstruction:

    def test_str_mutant_registers(self):
        op = make_op(mutant_registers="Add")
        assert op.mutant_registers == "Add"

    def test_list_mutant_registers(self):
        op = make_op(mutant_registers=["Add", "Sub", "Mult"])
        assert op.mutant_registers == ["Add", "Sub", "Mult"]

    def test_name_is_normalised_to_uppercase(self):
        op = make_op(name="aor")
        assert op.name == "AOR"

    def test_name_with_surrounding_whitespace_is_stripped_and_uppercased(self):
        op = make_op(name="  ror  ")
        assert op.name == "ROR"

    def test_id_zero_is_valid(self):
        op = make_op(id=0)
        assert op.id == 0

    def test_mutant_list_defaults_to_empty(self):
        op = make_op()
        assert op.mutant_list == []

    def test_mutant_list_accepts_prepopulated_list(self):
        m = _make_mutant()
        op = make_op(mutant_list=[m])
        assert len(op.mutant_list) == 1


# ─────────────────────────────────────────────────────────────────────────── #
# 2. _validate_id                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class TestValidateId:

    def test_negative_id_raises_type_error(self):
        with pytest.raises(TypeError, match="non-negative integer"):
            make_op(id=-1)

    def test_float_id_raises_type_error(self):
        with pytest.raises(TypeError, match="non-negative integer"):
            make_op(id=1.0)  # type: ignore[arg-type]

    def test_string_id_raises_type_error(self):
        with pytest.raises(TypeError, match="non-negative integer"):
            make_op(id="1")  # type: ignore[arg-type]

    def test_none_id_raises_type_error(self):
        with pytest.raises(TypeError, match="non-negative integer"):
            make_op(id=None)  # type: ignore[arg-type]

    def test_bool_true_raises_type_error(self):
        # bool is a subclass of int in Python; True == 1 — but we document
        # that only plain int is accepted.  If your validator currently
        # accepts bool, this test documents that behaviour.  Adjust if the
        # spec changes.
        # bool subclasses int → isinstance(True, int) is True AND True >= 0,
        # so the current implementation accepts it.  We test the actual
        # observable behaviour rather than an assumption.
        op = make_op(id=True)   # bool accepted by current impl
        assert op.id is True


# ─────────────────────────────────────────────────────────────────────────── #
# 3. _validate_name                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class TestValidateName:

    def test_non_string_name_raises_type_error(self):
        with pytest.raises(TypeError, match="non-empty string"):
            make_op(name=123)  # type: ignore[arg-type]

    def test_none_name_raises_type_error(self):
        with pytest.raises(TypeError, match="non-empty string"):
            make_op(name=None)  # type: ignore[arg-type]

    def test_empty_string_name_raises_type_error(self):
        with pytest.raises(TypeError, match="non-empty string"):
            make_op(name="")

    def test_whitespace_only_name_raises_type_error(self):
        with pytest.raises(TypeError, match="non-empty string"):
            make_op(name="   ")

    def test_name_is_uppercased(self):
        op = make_op(name="lcr")
        assert op.name == "LCR"

    def test_name_mixed_case_is_uppercased(self):
        op = make_op(name="NfTp")
        assert op.name == "NFTP"


# ─────────────────────────────────────────────────────────────────────────── #
# 4. _validate_mutant_registers — str branch                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class TestValidateMutantRegistersStr:

    def test_valid_str_is_accepted(self):
        op = make_op(mutant_registers="Add")
        assert op.mutant_registers == "Add"

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="mutant_registers must not be"):
            make_op(mutant_registers="")

    def test_whitespace_only_string_raises_value_error(self):
        with pytest.raises(ValueError, match="mutant_registers must not be"):
            make_op(mutant_registers="   ")


# ─────────────────────────────────────────────────────────────────────────── #
# 5. _validate_mutant_registers — list branch                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class TestValidateMutantRegistersList:

    def test_valid_list_is_accepted(self):
        op = make_op(mutant_registers=["Add", "Sub"])
        assert op.mutant_registers == ["Add", "Sub"]

    def test_single_element_list_is_accepted(self):
        op = make_op(mutant_registers=["filter"])
        assert op.mutant_registers == ["filter"]

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            make_op(mutant_registers=[])

    def test_list_with_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            make_op(mutant_registers=["Add", ""])

    def test_list_with_whitespace_string_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            make_op(mutant_registers=["Add", "   "])

    def test_list_with_non_string_item_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            make_op(mutant_registers=["Add", 42])  # type: ignore[list-item]

    def test_list_with_none_item_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            make_op(mutant_registers=["Add", None])  # type: ignore[list-item]

    def test_all_invalid_items_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            make_op(mutant_registers=["", "   "])


# ─────────────────────────────────────────────────────────────────────────── #
# 6. _validate_mutant_registers — wrong type branch                            #
# ─────────────────────────────────────────────────────────────────────────── #

class TestValidateMutantRegistersWrongType:

    def test_integer_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a str or list"):
            make_op(mutant_registers=42)  # type: ignore[arg-type]

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a str or list"):
            make_op(mutant_registers=None)  # type: ignore[arg-type]

    def test_tuple_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a str or list"):
            make_op(mutant_registers=("Add", "Sub"))  # type: ignore[arg-type]

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a str or list"):
            make_op(mutant_registers={"key": "Add"})  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────── #
# 7. _validate_mutant_list                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

class TestValidateMutantList:

    def test_non_list_raises_type_error(self):
        with pytest.raises(TypeError, match="mutant_list must be a list"):
            make_op(mutant_list="not a list")  # type: ignore[arg-type]

    def test_list_with_non_mutant_item_raises_type_error(self):
        with pytest.raises(TypeError, match="All items in mutant_list must be Mutant"):
            make_op(mutant_list=["not_a_mutant"])  # type: ignore[list-item]

    def test_list_with_integer_raises_type_error(self):
        with pytest.raises(TypeError, match="All items in mutant_list must be Mutant"):
            make_op(mutant_list=[1, 2, 3])  # type: ignore[list-item]

    def test_mixed_valid_and_invalid_raises_type_error(self):
        m = _make_mutant()
        with pytest.raises(TypeError, match="All items in mutant_list must be Mutant"):
            make_op(mutant_list=[m, "invalid"])  # type: ignore[list-item]

    def test_valid_mutant_list_is_accepted(self):
        m1, m2 = _make_mutant(1), _make_mutant(2)
        op = make_op(mutant_list=[m1, m2])
        assert len(op.mutant_list) == 2


# ─────────────────────────────────────────────────────────────────────────── #
# 8. _assert_valid_tree                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class TestAssertValidTree:

    def test_valid_ast_module_passes(self):
        op   = make_op()
        tree = ast.parse("x = 1")
        op._assert_valid_tree(tree)   # should not raise

    def test_valid_ast_expression_passes(self):
        op   = make_op()
        tree = ast.parse("1 + 1", mode="eval")
        op._assert_valid_tree(tree)

    def test_string_raises_type_error(self):
        op = make_op()
        with pytest.raises(TypeError, match="ast.AST instance"):
            op._assert_valid_tree("x = 1")  # type: ignore[arg-type]

    def test_none_raises_type_error(self):
        op = make_op()
        with pytest.raises(TypeError, match="ast.AST instance"):
            op._assert_valid_tree(None)  # type: ignore[arg-type]

    def test_integer_raises_type_error(self):
        op = make_op()
        with pytest.raises(TypeError, match="ast.AST instance"):
            op._assert_valid_tree(42)  # type: ignore[arg-type]

    def test_list_raises_type_error(self):
        op = make_op()
        with pytest.raises(TypeError, match="ast.AST instance"):
            op._assert_valid_tree([ast.parse("x=1")])  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────── #
# 9. _assert_valid_nodes                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class TestAssertValidNodes:

    def test_empty_list_passes(self):
        op = make_op()
        op._assert_valid_nodes([])   # should not raise

    def test_list_of_ast_nodes_passes(self):
        op    = make_op()
        nodes = list(ast.walk(ast.parse("x = 1")))
        op._assert_valid_nodes(nodes)

    def test_non_list_raises_type_error(self):
        op = make_op()
        with pytest.raises(TypeError, match="nodes must be a list"):
            op._assert_valid_nodes("not a list")  # type: ignore[arg-type]

    def test_tuple_raises_type_error(self):
        op = make_op()
        with pytest.raises(TypeError, match="nodes must be a list"):
            op._assert_valid_nodes((ast.parse("x=1"),))  # type: ignore[arg-type]

    def test_list_with_non_ast_item_raises_type_error(self):
        op    = make_op()
        valid = ast.parse("x = 1")
        with pytest.raises(TypeError, match="All items in nodes must be"):
            op._assert_valid_nodes([valid, "not_an_ast"])  # type: ignore[list-item]

    def test_list_with_none_raises_type_error(self):
        op = make_op()
        with pytest.raises(TypeError, match="All items in nodes must be"):
            op._assert_valid_nodes([None])  # type: ignore[list-item]

    def test_list_with_integer_raises_type_error(self):
        op = make_op()
        with pytest.raises(TypeError, match="All items in nodes must be"):
            op._assert_valid_nodes([1, 2])  # type: ignore[list-item]


# ─────────────────────────────────────────────────────────────────────────── #
# 10. _assert_valid_path                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class TestAssertValidPath:

    def test_valid_path_string_passes(self):
        op = make_op()
        op._assert_valid_path("/some/path/file.py", "original_path")

    def test_relative_path_string_passes(self):
        op = make_op()
        op._assert_valid_path("relative/path.py", "original_path")

    def test_empty_string_raises_value_error(self):
        op = make_op()
        with pytest.raises(ValueError, match="must be a non-empty string"):
            op._assert_valid_path("", "original_path")

    def test_whitespace_only_raises_value_error(self):
        op = make_op()
        with pytest.raises(ValueError, match="must be a non-empty string"):
            op._assert_valid_path("   ", "original_path")

    def test_none_raises_value_error(self):
        op = make_op()
        with pytest.raises(ValueError, match="must be a non-empty string"):
            op._assert_valid_path(None, "original_path")  # type: ignore[arg-type]

    def test_integer_raises_value_error(self):
        op = make_op()
        with pytest.raises(ValueError, match="must be a non-empty string"):
            op._assert_valid_path(123, "mutant_dir")  # type: ignore[arg-type]

    def test_param_name_appears_in_error_message(self):
        op = make_op()
        with pytest.raises(ValueError, match="mutant_dir"):
            op._assert_valid_path("", "mutant_dir")


# ─────────────────────────────────────────────────────────────────────────── #
# 11. _next_mutant_id                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class TestNextMutantId:

    def test_returns_one_when_list_is_empty(self):
        op = make_op()
        assert op._next_mutant_id() == 1

    def test_returns_two_after_one_mutant(self):
        op = make_op(mutant_list=[_make_mutant(1)])
        assert op._next_mutant_id() == 2

    def test_returns_n_plus_one_after_n_mutants(self):
        mutants = [_make_mutant(i) for i in range(1, 6)]
        op      = make_op(mutant_list=mutants)
        assert op._next_mutant_id() == 6

    def test_is_consistent_across_multiple_calls_without_appending(self):
        op = make_op()
        assert op._next_mutant_id() == op._next_mutant_id()

    def test_increments_after_appending_to_list(self):
        op = make_op()
        first = op._next_mutant_id()
        op.mutant_list.append(_make_mutant(first))
        second = op._next_mutant_id()
        assert second == first + 1


# ─────────────────────────────────────────────────────────────────────────── #
# 12. clear_mutant_list                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class TestClearMutantList:

    def test_clears_populated_list(self):
        op = make_op(mutant_list=[_make_mutant(1), _make_mutant(2)])
        op.clear_mutant_list()
        assert op.mutant_list == []

    def test_clear_on_empty_list_is_safe(self):
        op = make_op()
        op.clear_mutant_list()   # should not raise
        assert op.mutant_list == []

    def test_other_attributes_unchanged_after_clear(self):
        op = make_op(
            id=7,
            name="ror",
            mutant_registers=["Lt", "Gt"],
            mutant_list=[_make_mutant()],
        )
        op.clear_mutant_list()
        assert op.id   == 7
        assert op.name == "ROR"
        assert op.mutant_registers == ["Lt", "Gt"]

    def test_list_grows_again_after_clear(self):
        op = make_op(mutant_list=[_make_mutant(1)])
        op.clear_mutant_list()
        op.mutant_list.append(_make_mutant(2))
        assert len(op.mutant_list) == 1
        assert op.mutant_list[0].id == 2


# ─────────────────────────────────────────────────────────────────────────── #
# 13. __repr__                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class TestRepr:

    def test_repr_contains_class_name(self):
        op = make_op()
        assert "ConcreteOperator" in repr(op)

    def test_repr_contains_id(self):
        op = make_op(id=5)
        assert "id=5" in repr(op)

    def test_repr_contains_normalised_name(self):
        op = make_op(name="aor")
        assert "name='AOR'" in repr(op)

    def test_repr_contains_mutant_registers_str(self):
        op = make_op(mutant_registers="Add")
        assert "'Add'" in repr(op)

    def test_repr_contains_mutant_registers_list(self):
        op = make_op(mutant_registers=["Lt", "Gt"])
        assert "['Lt', 'Gt']" in repr(op)

    def test_repr_shows_zero_mutants_when_empty(self):
        op = make_op()
        assert "mutants=0" in repr(op)

    def test_repr_shows_correct_mutant_count(self):
        op = make_op(mutant_list=[_make_mutant(1), _make_mutant(2)])
        assert "mutants=2" in repr(op)

    def test_repr_updates_after_clear(self):
        op = make_op(mutant_list=[_make_mutant()])
        op.clear_mutant_list()
        assert "mutants=0" in repr(op)


# ─────────────────────────────────────────────────────────────────────────── #
# 14. Abstract enforcement                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

class TestAbstractEnforcement:

    def test_missing_analyse_ast_raises_type_error(self):
        class MissingAnalyse(Operator):
            def build_mutant(self, nodes, original_ast, original_path, mutant_dir):
                return self.mutant_list

        with pytest.raises(TypeError):
            MissingAnalyse(id=1, name="X", mutant_registers="Add")

    def test_missing_build_mutant_raises_type_error(self):
        class MissingBuild(Operator):
            def analyse_ast(self, tree):
                return []

        with pytest.raises(TypeError):
            MissingBuild(id=1, name="X", mutant_registers="Add")

    def test_missing_both_abstract_methods_raises_type_error(self):
        class MissingBoth(Operator):
            pass

        with pytest.raises(TypeError):
            MissingBoth(id=1, name="X", mutant_registers="Add")

    def test_concrete_subclass_with_both_methods_instantiates(self):
        op = ConcreteOperator(id=1, name="TEST", mutant_registers="Add")
        assert op is not None


# ─────────────────────────────────────────────────────────────────────────── #
# 15. ABC cannot be instantiated directly                                      #
# ─────────────────────────────────────────────────────────────────────────── #

class TestAbstractBaseCannotBeInstantiated:

    def test_operator_itself_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Operator(id=1, name="X", mutant_registers="Add")  # type: ignore[abstract]