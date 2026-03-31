"""
Unit tests for Operator (ABC)
==============================
Coverage targets
----------------
- Instantiation guards    : ABC cannot be instantiated directly; subclass without
                            full implementation raises TypeError
- __post_init__           : _validate_name, _validate_registers, _validate_code_ast
- set_code_ast()          : TypeError on non-AST, success + registers cleared,
                            replacing an existing codeAST
- clear_registers()       : empties registers, codeAST untouched
- _assert_code_ast_ready(): RuntimeError when None, no error when set
- _assert_node_in_registers(): ValueError when node absent, no error when present
- analyseAST()            : concrete impl called; RuntimeError guard works
- buildMutate()           : concrete impl called; TypeError / ValueError guards work
- __repr__                : before/after set_code_ast, with/without registers

Run with:
    pytest test_operator.py -v --cov=code.operator --cov-report=term-missing
"""

import ast

import pytest

from code.operator import Operator


# ═══════════════════════════════════════════════════════════════════════════ #
# Concrete test doubles                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

class ConcreteOperator(Operator):
    """
    Minimal concrete subclass used throughout the tests.
    Targets ast.Add nodes and replaces them with ast.Sub.
    """

    def analyseAST(self) -> list[ast.AST]:
        self._assert_code_ast_ready()
        self.registers = [
            node for node in ast.walk(self.codeAST)
            if isinstance(node, ast.Add)
        ]
        return self.registers

    def buildMutate(self, target_node: ast.AST) -> ast.AST:
        if not isinstance(target_node, ast.AST):
            raise TypeError(
                f"[ConcreteOperator] target_node must be ast.AST, "
                f"got: {type(target_node)}"
            )
        self._assert_node_in_registers(target_node)
        return ast.Sub()


class PartialOperator(Operator):
    """Subclass that only implements analyseAST — used to confirm ABC enforcement."""

    def analyseAST(self) -> list[ast.AST]:
        return []

    # buildMutate intentionally NOT implemented


# ═══════════════════════════════════════════════════════════════════════════ #
# Shared fixtures                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

@pytest.fixture
def simple_ast() -> ast.AST:
    """AST for 'x = 1 + 2' — contains exactly one ast.Add node."""
    tree = ast.parse("x = 1 + 2")
    ast.fix_missing_locations(tree)
    return tree


@pytest.fixture
def multi_add_ast() -> ast.AST:
    """AST for 'x = 1 + 2 + 3' — contains two ast.Add nodes."""
    tree = ast.parse("x = 1 + 2 + 3")
    ast.fix_missing_locations(tree)
    return tree


@pytest.fixture
def no_add_ast() -> ast.AST:
    """AST for 'x = 1' — contains zero ast.Add nodes."""
    tree = ast.parse("x = 1")
    ast.fix_missing_locations(tree)
    return tree


@pytest.fixture
def op() -> ConcreteOperator:
    """A fresh ConcreteOperator with no codeAST."""
    return ConcreteOperator(name="AOR")


@pytest.fixture
def op_with_ast(op, simple_ast) -> ConcreteOperator:
    """ConcreteOperator with codeAST already set."""
    op.set_code_ast(simple_ast)
    return op


@pytest.fixture
def op_analysed(op_with_ast) -> ConcreteOperator:
    """ConcreteOperator with codeAST set and analyseAST already called."""
    op_with_ast.analyseAST()
    return op_with_ast


# ═══════════════════════════════════════════════════════════════════════════ #
# ABC enforcement                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestABCEnforcement:

    def test_cannot_instantiate_operator_directly(self):
        with pytest.raises(TypeError):
            Operator(name="AOR")  # type: ignore[abstract]

    def test_cannot_instantiate_partial_subclass(self):
        """Subclass missing buildMutate must not be instantiable."""
        with pytest.raises(TypeError):
            PartialOperator(name="AOR")  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self):
        op = ConcreteOperator(name="AOR")
        assert op is not None


# ═══════════════════════════════════════════════════════════════════════════ #
# __post_init__ / construction validation                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestConstruction:

    # --- name ----------------------------------------------------------------

    def test_valid_name_accepted(self):
        op = ConcreteOperator(name="aor")
        assert op.name == "AOR"   # normalised to uppercase

    def test_name_stripped_and_uppercased(self):
        op = ConcreteOperator(name="  ror  ")
        assert op.name == "ROR"

    def test_name_not_string_raises_type_error(self):
        with pytest.raises(TypeError, match="name must be a non-empty string"):
            ConcreteOperator(name=123)

    def test_name_empty_string_raises_type_error(self):
        with pytest.raises(TypeError, match="name must be a non-empty string"):
            ConcreteOperator(name="")

    def test_name_whitespace_only_raises_type_error(self):
        with pytest.raises(TypeError, match="name must be a non-empty string"):
            ConcreteOperator(name="   ")

    def test_name_none_raises_type_error(self):
        with pytest.raises(TypeError, match="name must be a non-empty string"):
            ConcreteOperator(name=None)

    # --- registers -----------------------------------------------------------

    def test_default_registers_is_empty_list(self):
        op = ConcreteOperator(name="AOR")
        assert op.registers == []

    def test_custom_registers_with_ast_nodes_accepted(self):
        node = ast.Add()
        op = ConcreteOperator(name="AOR", registers=[node])
        assert op.registers == [node]

    def test_registers_not_a_list_raises_type_error(self):
        with pytest.raises(TypeError, match="registers must be a list"):
            ConcreteOperator(name="AOR", registers="not-a-list")

    def test_registers_tuple_raises_type_error(self):
        with pytest.raises(TypeError, match="registers must be a list"):
            ConcreteOperator(name="AOR", registers=(ast.Add(),))

    def test_registers_with_non_ast_item_raises_type_error(self):
        with pytest.raises(TypeError, match="All items in registers must be ast.AST"):
            ConcreteOperator(name="AOR", registers=[ast.Add(), "not-a-node"])

    def test_registers_with_integer_item_raises_type_error(self):
        with pytest.raises(TypeError, match="All items in registers must be ast.AST"):
            ConcreteOperator(name="AOR", registers=[42])

    # --- codeAST -------------------------------------------------------------

    def test_default_code_ast_is_none(self):
        op = ConcreteOperator(name="AOR")
        assert op.codeAST is None

    def test_valid_code_ast_accepted(self, simple_ast):
        op = ConcreteOperator(name="AOR", codeAST=simple_ast)
        assert op.codeAST is simple_ast

    def test_code_ast_non_ast_raises_type_error(self):
        with pytest.raises(TypeError, match="codeAST must be an ast.AST instance or None"):
            ConcreteOperator(name="AOR", codeAST="not-an-ast")

    def test_code_ast_integer_raises_type_error(self):
        with pytest.raises(TypeError, match="codeAST must be an ast.AST instance or None"):
            ConcreteOperator(name="AOR", codeAST=42)


# ═══════════════════════════════════════════════════════════════════════════ #
# set_code_ast()                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestSetCodeAst:

    def test_sets_code_ast(self, op, simple_ast):
        op.set_code_ast(simple_ast)
        assert op.codeAST is simple_ast

    def test_clears_registers_on_set(self, op, simple_ast, multi_add_ast):
        # Pre-populate registers manually
        op.registers = [ast.Add(), ast.Sub()]
        op.set_code_ast(simple_ast)
        assert op.registers == []

    def test_replaces_existing_code_ast(self, op, simple_ast, multi_add_ast):
        op.set_code_ast(simple_ast)
        op.set_code_ast(multi_add_ast)
        assert op.codeAST is multi_add_ast

    def test_non_ast_raises_type_error(self, op):
        with pytest.raises(TypeError, match="codeAST must be an ast.AST instance"):
            op.set_code_ast("not-an-ast")

    def test_none_raises_type_error(self, op):
        with pytest.raises(TypeError, match="codeAST must be an ast.AST instance"):
            op.set_code_ast(None)

    def test_integer_raises_type_error(self, op):
        with pytest.raises(TypeError, match="codeAST must be an ast.AST instance"):
            op.set_code_ast(99)


# ═══════════════════════════════════════════════════════════════════════════ #
# clear_registers()                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestClearRegisters:

    def test_empties_registers(self, op_analysed):
        assert len(op_analysed.registers) > 0
        op_analysed.clear_registers()
        assert op_analysed.registers == []

    def test_code_ast_untouched_after_clear(self, op_analysed, simple_ast):
        op_analysed.clear_registers()
        assert op_analysed.codeAST is simple_ast

    def test_clear_on_already_empty_registers_does_not_raise(self, op):
        op.clear_registers()   # must not raise
        assert op.registers == []


# ═══════════════════════════════════════════════════════════════════════════ #
# _assert_code_ast_ready()                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestAssertCodeAstReady:

    def test_raises_runtime_error_when_code_ast_is_none(self, op):
        with pytest.raises(RuntimeError, match="codeAST is not set"):
            op._assert_code_ast_ready()

    def test_does_not_raise_when_code_ast_is_set(self, op_with_ast):
        op_with_ast._assert_code_ast_ready()   # must not raise


# ═══════════════════════════════════════════════════════════════════════════ #
# _assert_node_in_registers()                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestAssertNodeInRegisters:

    def test_raises_value_error_when_node_not_in_registers(self, op):
        foreign_node = ast.Add()
        with pytest.raises(ValueError, match="not in registers"):
            op._assert_node_in_registers(foreign_node)

    def test_does_not_raise_when_node_is_in_registers(self, op_analysed):
        node = op_analysed.registers[0]
        op_analysed._assert_node_in_registers(node)   # must not raise

    def test_raises_after_registers_cleared(self, op_analysed):
        node = op_analysed.registers[0]
        op_analysed.clear_registers()
        with pytest.raises(ValueError, match="not in registers"):
            op_analysed._assert_node_in_registers(node)


# ═══════════════════════════════════════════════════════════════════════════ #
# analyseAST()  (tested via ConcreteOperator)                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestAnalyseAST:

    def test_raises_runtime_error_when_code_ast_not_set(self, op):
        with pytest.raises(RuntimeError, match="codeAST is not set"):
            op.analyseAST()

    def test_returns_list(self, op_with_ast):
        result = op_with_ast.analyseAST()
        assert isinstance(result, list)

    def test_returns_same_object_as_registers(self, op_with_ast):
        result = op_with_ast.analyseAST()
        assert result is op_with_ast.registers

    def test_finds_one_add_node(self, op_with_ast):
        op_with_ast.analyseAST()
        assert len(op_with_ast.registers) == 1

    def test_finds_two_add_nodes(self, op, multi_add_ast):
        op.set_code_ast(multi_add_ast)
        op.analyseAST()
        assert len(op.registers) == 2

    def test_finds_zero_add_nodes(self, op, no_add_ast):
        op.set_code_ast(no_add_ast)
        op.analyseAST()
        assert op.registers == []

    def test_registers_contain_ast_add_instances(self, op_with_ast):
        op_with_ast.analyseAST()
        for node in op_with_ast.registers:
            assert isinstance(node, ast.Add)

    def test_second_call_replaces_registers(self, op, simple_ast, multi_add_ast):
        """Calling analyseAST() twice must not accumulate — it must replace."""
        op.set_code_ast(simple_ast)
        op.analyseAST()
        assert len(op.registers) == 1

        op.set_code_ast(multi_add_ast)
        op.analyseAST()
        assert len(op.registers) == 2   # not 3


# ═══════════════════════════════════════════════════════════════════════════ #
# buildMutate()  (tested via ConcreteOperator)                               #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestBuildMutate:

    def test_raises_type_error_on_non_ast_node(self, op_analysed):
        with pytest.raises(TypeError, match="target_node must be ast.AST"):
            op_analysed.buildMutate("not-a-node")

    def test_raises_value_error_for_unregistered_node(self, op_analysed):
        foreign_node = ast.Add()   # a new, unregistered instance
        with pytest.raises(ValueError, match="not in registers"):
            op_analysed.buildMutate(foreign_node)

    def test_returns_ast_node(self, op_analysed):
        node = op_analysed.registers[0]
        result = op_analysed.buildMutate(node)
        assert isinstance(result, ast.AST)

    def test_returns_sub_node(self, op_analysed):
        node = op_analysed.registers[0]
        result = op_analysed.buildMutate(node)
        assert isinstance(result, ast.Sub)

    def test_each_call_returns_new_node(self, op, multi_add_ast):
        """buildMutate must return a fresh node on every call."""
        op.set_code_ast(multi_add_ast)
        op.analyseAST()
        node = op.registers[0]
        result_a = op.buildMutate(node)
        result_b = op.buildMutate(node)
        assert result_a is not result_b

    def test_raises_value_error_after_registers_cleared(self, op_analysed):
        node = op_analysed.registers[0]
        op_analysed.clear_registers()
        with pytest.raises(ValueError, match="not in registers"):
            op_analysed.buildMutate(node)


# ═══════════════════════════════════════════════════════════════════════════ #
# __repr__                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestRepr:

    def test_repr_contains_class_name(self, op):
        assert "ConcreteOperator" in repr(op)

    def test_repr_contains_operator_name(self, op):
        assert "'AOR'" in repr(op)

    def test_repr_shows_code_ast_not_set(self, op):
        assert "not set" in repr(op)

    def test_repr_shows_code_ast_set(self, op_with_ast):
        assert "set" in repr(op_with_ast)
        assert "not set" not in repr(op_with_ast)

    def test_repr_shows_zero_registers(self, op):
        assert "0 node(s)" in repr(op)

    def test_repr_shows_correct_register_count(self, op_analysed):
        assert "1 node(s)" in repr(op_analysed)

    def test_repr_updates_after_clear(self, op_analysed):
        op_analysed.clear_registers()
        assert "0 node(s)" in repr(op_analysed)


# ═══════════════════════════════════════════════════════════════════════════ #
# Integration                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestIntegration:

    def test_full_lifecycle_single_occurrence(self, simple_ast):
        op = ConcreteOperator(name="aor")
        assert op.name == "AOR"

        op.set_code_ast(simple_ast)
        nodes = op.analyseAST()

        assert len(nodes) == 1
        replacement = op.buildMutate(nodes[0])
        assert isinstance(replacement, ast.Sub)

    def test_full_lifecycle_multiple_occurrences(self, multi_add_ast):
        op = ConcreteOperator(name="AOR")
        op.set_code_ast(multi_add_ast)
        nodes = op.analyseAST()

        assert len(nodes) == 2
        for node in nodes:
            replacement = op.buildMutate(node)
            assert isinstance(replacement, ast.Sub)

    def test_reuse_operator_with_new_ast(self, simple_ast, multi_add_ast):
        """Operator reused on a different AST must reflect the new tree."""
        op = ConcreteOperator(name="AOR")

        op.set_code_ast(simple_ast)
        op.analyseAST()
        assert len(op.registers) == 1

        op.set_code_ast(multi_add_ast)
        op.analyseAST()
        assert len(op.registers) == 2

    def test_no_eligible_nodes_build_mutate_not_callable(self, no_add_ast):
        """When registers is empty, buildMutate must reject any node."""
        op = ConcreteOperator(name="AOR")
        op.set_code_ast(no_add_ast)
        op.analyseAST()
        assert op.registers == []

        with pytest.raises(ValueError, match="not in registers"):
            op.buildMutate(ast.Add())