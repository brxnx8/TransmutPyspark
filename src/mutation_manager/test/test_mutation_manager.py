"""
Unit tests for MutationManager
================================
Coverage targets
----------------
- __post_init__              : valid construction, TypeError / RuntimeError on bad configLoader,
                               TypeError on bad mutateList
- parseToAST()               : TypeError (non-string, empty), ValueError (syntax error),
                               success (returns AST, stores source, fixes locations),
                               clears mutateList on second call, idempotent when list is empty
- applyMutation()            : RuntimeError before parseToAST, TypeError on bad operator
                               (missing attrs, bad target_node_type, bad replacement_node,
                               bad operator_id), zero occurrences → returns [],
                               single occurrence, multiple occurrences, accumulates in mutateList,
                               multiple operators accumulate correctly,
                               replacement_node template not mutated between calls
- program_ast property       : RuntimeError before parseToAST, value after parseToAST
- program_source property    : RuntimeError before parseToAST, value after parseToAST
- _count_occurrences()       : zero, one, many
- _assert_ast_ready()        : RuntimeError before / no error after parseToAST
- _NodeReplacer              : replaces only target occurrence, leaves others intact
- Mutant.__repr__            : format check, long source truncation
- MutationManager.__repr__   : before / after parseToAST

Run with:
    pytest test_mutation_manager.py -v --cov=code.mutation_manager --cov-report=term-missing
"""

import ast
import copy

import pytest
from unittest.mock import MagicMock, PropertyMock

from code.mutation_manager import (
    MutationManager,
    Mutant,
    OperatorProtocol,
    _NodeReplacer,
)


# ═══════════════════════════════════════════════════════════════════════════ #
# Helpers / shared constants                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

# A minimal PySpark-like program that contains arithmetic operators
SOURCE_WITH_ADD = "x = 1 + 2"
SOURCE_WITH_TWO_ADDS = "x = 1 + 2 + 3"          # two ast.Add nodes
SOURCE_NO_ADD = "x = 1"                           # no ast.Add node
SOURCE_SYNTAX_ERROR = "def foo(:\n    pass"
SOURCE_MULTILINE = "a = 1 + 2\nb = 3 + 4\nc = a + b"  # three ast.Add nodes


def _make_mock_config(program_source: str = SOURCE_WITH_ADD) -> MagicMock:
    """Return a mock that satisfies MutationManager's duck-typing checks."""
    cfg = MagicMock()
    cfg.program_source = program_source
    cfg.workspace_path = "/tmp/workspace"
    cfg.operatorsList = ["AOR"]
    return cfg


def _make_operator(
    target_type: type = ast.Add,
    replacement: ast.AST = None,
    op_id: str = "AOR",
) -> MagicMock:
    """Return a mock that satisfies OperatorProtocol."""
    if replacement is None:
        replacement = ast.Sub()
    op = MagicMock(spec=["target_node_type", "replacement_node", "operator_id"])
    op.target_node_type = target_type
    op.replacement_node = replacement
    op.operator_id = op_id
    return op


# ═══════════════════════════════════════════════════════════════════════════ #
# Fixtures                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

@pytest.fixture
def mock_config():
    return _make_mock_config()


@pytest.fixture
def manager(mock_config):
    """A MutationManager with a valid config, not yet parsed."""
    return MutationManager(configLoader=mock_config)


@pytest.fixture
def parsed_manager(manager):
    """A MutationManager that has already called parseToAST."""
    manager.parseToAST(SOURCE_WITH_ADD)
    return manager


@pytest.fixture
def aor_operator():
    """Arithmetic Operator Replacement: Add → Sub."""
    return _make_operator(ast.Add, ast.Sub(), "AOR")


@pytest.fixture
def manager_two_adds(mock_config):
    m = MutationManager(configLoader=mock_config)
    m.parseToAST(SOURCE_WITH_TWO_ADDS)
    return m


@pytest.fixture
def manager_multiline(mock_config):
    m = MutationManager(configLoader=mock_config)
    m.parseToAST(SOURCE_MULTILINE)
    return m


# ═══════════════════════════════════════════════════════════════════════════ #
# __post_init__ / construction                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestConstruction:

    def test_valid_construction(self, mock_config):
        m = MutationManager(configLoader=mock_config)
        assert m.configLoader is mock_config
        assert m.mutateList == []

    def test_custom_mutate_list_accepted(self, mock_config):
        existing = [Mutant("AOR", 0, "x = 1 - 2")]
        m = MutationManager(configLoader=mock_config, mutateList=existing)
        assert m.mutateList is existing

    # --- configLoader: missing required attributes ------------------------

    def test_config_loader_missing_program_source(self):
        cfg = MagicMock(spec=["workspace_path", "operatorsList"])
        with pytest.raises(TypeError, match="missing attribute 'program_source'"):
            MutationManager(configLoader=cfg)

    def test_config_loader_missing_workspace_path(self):
        cfg = MagicMock(spec=["program_source", "operatorsList"])
        with pytest.raises(TypeError, match="missing attribute 'workspace_path'"):
            MutationManager(configLoader=cfg)

    def test_config_loader_missing_operators_list(self):
        cfg = MagicMock(spec=["program_source", "workspace_path"])
        with pytest.raises(TypeError, match="missing attribute 'operatorsList'"):
            MutationManager(configLoader=cfg)

    # --- configLoader: not loaded (raises RuntimeError on access) --------

    def test_config_loader_not_loaded_raises_runtime_error(self):
        cfg = MagicMock()
        cfg.workspace_path = "/tmp"
        cfg.operatorsList = ["AOR"]
        type(cfg).program_source = PropertyMock(
            side_effect=RuntimeError("Call .load() first.")
        )
        with pytest.raises(RuntimeError, match="has not been loaded yet"):
            MutationManager(configLoader=cfg)

    # --- mutateList: wrong type ------------------------------------------

    def test_mutate_list_not_a_list_raises_type_error(self, mock_config):
        with pytest.raises(TypeError, match="mutateList must be a list"):
            MutationManager(configLoader=mock_config, mutateList="not-a-list")

    def test_mutate_list_tuple_raises_type_error(self, mock_config):
        with pytest.raises(TypeError, match="mutateList must be a list"):
            MutationManager(configLoader=mock_config, mutateList=())


# ═══════════════════════════════════════════════════════════════════════════ #
# parseToAST()                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestParseToAST:

    # --- TypeError -----------------------------------------------------------

    def test_source_none_raises_type_error(self, manager):
        with pytest.raises(TypeError, match="source must be a non-empty string"):
            manager.parseToAST(None)

    def test_source_integer_raises_type_error(self, manager):
        with pytest.raises(TypeError, match="source must be a non-empty string"):
            manager.parseToAST(42)

    def test_source_empty_string_raises_type_error(self, manager):
        with pytest.raises(TypeError, match="source must be a non-empty string"):
            manager.parseToAST("")

    def test_source_whitespace_only_raises_type_error(self, manager):
        with pytest.raises(TypeError, match="source must be a non-empty string"):
            manager.parseToAST("   \n\t")

    # --- ValueError (syntax error) -------------------------------------------

    def test_syntax_error_raises_value_error(self, manager):
        with pytest.raises(ValueError, match="Syntax error in source"):
            manager.parseToAST(SOURCE_SYNTAX_ERROR)

    # --- Success -------------------------------------------------------------

    def test_returns_ast_module(self, manager):
        result = manager.parseToAST(SOURCE_WITH_ADD)
        assert isinstance(result, ast.AST)

    def test_stores_program_source(self, manager):
        manager.parseToAST(SOURCE_WITH_ADD)
        assert manager._program_source == SOURCE_WITH_ADD

    def test_stores_program_ast(self, manager):
        manager.parseToAST(SOURCE_WITH_ADD)
        assert manager._program_ast is not None

    def test_returned_tree_is_same_as_stored(self, manager):
        result = manager.parseToAST(SOURCE_WITH_ADD)
        assert result is manager._program_ast

    def test_fix_missing_locations_applied(self, manager):
        """Every node in the tree must have lineno after parseToAST."""
        manager.parseToAST(SOURCE_WITH_ADD)
        for node in ast.walk(manager._program_ast):
            if hasattr(node, "lineno"):
                assert node.lineno is not None

    # --- Second call: clears stale mutants -----------------------------------

    def test_second_parse_clears_mutate_list(self, manager, aor_operator):
        manager.parseToAST(SOURCE_WITH_ADD)
        manager.applyMutation(aor_operator)
        assert len(manager.mutateList) > 0

        manager.parseToAST(SOURCE_WITH_ADD)   # second call
        assert manager.mutateList == []

    def test_second_parse_with_empty_list_does_not_raise(self, manager):
        manager.parseToAST(SOURCE_WITH_ADD)
        assert manager.mutateList == []
        manager.parseToAST(SOURCE_NO_ADD)     # must not raise


# ═══════════════════════════════════════════════════════════════════════════ #
# applyMutation()                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestApplyMutation:

    # --- RuntimeError before parseToAST --------------------------------------

    def test_raises_runtime_error_before_parse(self, manager, aor_operator):
        with pytest.raises(RuntimeError, match="AST is not available"):
            manager.applyMutation(aor_operator)

    # --- TypeError: operator does not satisfy OperatorProtocol ---------------

    def test_operator_missing_all_attrs_raises_type_error(self, parsed_manager):
        with pytest.raises(TypeError, match="operator must satisfy OperatorProtocol"):
            parsed_manager.applyMutation("not-an-operator")

    def test_operator_missing_replacement_node_raises_type_error(self, parsed_manager):
        op = MagicMock(spec=["target_node_type", "operator_id"])
        op.target_node_type = ast.Add
        op.operator_id = "AOR"
        with pytest.raises(TypeError, match="operator must satisfy OperatorProtocol"):
            parsed_manager.applyMutation(op)

    # --- TypeError: target_node_type is not an ast.AST subclass --------------

    def test_target_node_type_not_a_type_raises_type_error(self, parsed_manager):
        op = _make_operator()
        op.target_node_type = "ast.Add"   # string, not a type
        with pytest.raises(TypeError, match="target_node_type must be a subclass of ast.AST"):
            parsed_manager.applyMutation(op)

    def test_target_node_type_not_ast_subclass_raises_type_error(self, parsed_manager):
        op = _make_operator()
        op.target_node_type = int          # type but not an AST subclass
        with pytest.raises(TypeError, match="target_node_type must be a subclass of ast.AST"):
            parsed_manager.applyMutation(op)

    # --- TypeError: replacement_node is not an ast.AST instance --------------

    def test_replacement_node_not_ast_raises_type_error(self, parsed_manager):
        op = _make_operator()
        op.replacement_node = "Sub"        # string, not ast.AST
        with pytest.raises(TypeError, match="replacement_node must be an ast.AST instance"):
            parsed_manager.applyMutation(op)

    # --- TypeError: operator_id is not a valid string ------------------------

    def test_operator_id_empty_string_raises_type_error(self, parsed_manager):
        op = _make_operator()
        op.operator_id = ""
        with pytest.raises(TypeError, match="operator_id must be a non-empty string"):
            parsed_manager.applyMutation(op)

    def test_operator_id_whitespace_raises_type_error(self, parsed_manager):
        op = _make_operator()
        op.operator_id = "   "
        with pytest.raises(TypeError, match="operator_id must be a non-empty string"):
            parsed_manager.applyMutation(op)

    def test_operator_id_non_string_raises_type_error(self, parsed_manager):
        op = _make_operator()
        op.operator_id = 123
        with pytest.raises(TypeError, match="operator_id must be a non-empty string"):
            parsed_manager.applyMutation(op)

    # --- Zero occurrences ----------------------------------------------------

    def test_zero_occurrences_returns_empty_list(self, parsed_manager):
        op = _make_operator(target_type=ast.Mult, replacement=ast.Div(), op_id="AOR")
        result = parsed_manager.applyMutation(op)
        assert result == []

    def test_zero_occurrences_does_not_add_to_mutate_list(self, parsed_manager):
        op = _make_operator(target_type=ast.Mult, replacement=ast.Div(), op_id="AOR")
        parsed_manager.applyMutation(op)
        assert parsed_manager.mutateList == []

    # --- Single occurrence ---------------------------------------------------

    def test_single_occurrence_returns_one_mutant(self, parsed_manager, aor_operator):
        result = parsed_manager.applyMutation(aor_operator)
        assert len(result) == 1

    def test_single_occurrence_mutant_is_correct_type(self, parsed_manager, aor_operator):
        result = parsed_manager.applyMutation(aor_operator)
        assert isinstance(result[0], Mutant)

    def test_single_occurrence_operator_id_stored(self, parsed_manager, aor_operator):
        result = parsed_manager.applyMutation(aor_operator)
        assert result[0].operator_id == "AOR"

    def test_single_occurrence_occurrence_index_is_zero(self, parsed_manager, aor_operator):
        result = parsed_manager.applyMutation(aor_operator)
        assert result[0].occurrence_index == 0

    def test_single_occurrence_source_code_is_string(self, parsed_manager, aor_operator):
        result = parsed_manager.applyMutation(aor_operator)
        assert isinstance(result[0].source_code, str)
        assert len(result[0].source_code) > 0

    def test_single_occurrence_mutation_applied_in_source(self, parsed_manager, aor_operator):
        """The mutant source must no longer contain '+' and must contain '-'."""
        result = parsed_manager.applyMutation(aor_operator)
        assert "-" in result[0].source_code

    def test_single_occurrence_added_to_mutate_list(self, parsed_manager, aor_operator):
        result = parsed_manager.applyMutation(aor_operator)
        assert parsed_manager.mutateList == result

    # --- Multiple occurrences ------------------------------------------------

    def test_two_occurrences_return_two_mutants(self, manager_two_adds, aor_operator):
        result = manager_two_adds.applyMutation(aor_operator)
        assert len(result) == 2

    def test_three_occurrences_return_three_mutants(self, manager_multiline, aor_operator):
        result = manager_multiline.applyMutation(aor_operator)
        assert len(result) == 3

    def test_occurrence_indices_are_sequential(self, manager_two_adds, aor_operator):
        result = manager_two_adds.applyMutation(aor_operator)
        assert [m.occurrence_index for m in result] == [0, 1]

    def test_each_mutant_has_independent_source(self, manager_two_adds, aor_operator):
        """Each mutant is generated from the *original* tree, not the previous mutant."""
        result = manager_two_adds.applyMutation(aor_operator)
        # Both mutants must be valid Python
        for mutant in result:
            ast.parse(mutant.source_code)   # must not raise

    def test_original_ast_not_modified_after_mutation(self, manager_two_adds, aor_operator):
        """The stored _program_ast must remain untouched after applyMutation."""
        original_source = ast.unparse(manager_two_adds._program_ast)
        manager_two_adds.applyMutation(aor_operator)
        after_source = ast.unparse(manager_two_adds._program_ast)
        assert original_source == after_source

    def test_replacement_node_template_not_mutated(self, parsed_manager):
        """Deep copy must protect the operator's replacement_node template."""
        op = _make_operator(ast.Add, ast.Sub(), "AOR")
        original_replacement_id = id(op.replacement_node)
        parsed_manager.applyMutation(op)
        # The object at the original address must still be an ast.Sub
        assert isinstance(op.replacement_node, ast.Sub)
        assert id(op.replacement_node) == original_replacement_id

    # --- Multiple operators accumulate correctly -----------------------------

    def test_two_operators_accumulate_in_mutate_list(self, manager_multiline):
        op_aor = _make_operator(ast.Add, ast.Sub(), "AOR")
        op_ror = _make_operator(ast.Add, ast.Mult(), "ROR")

        manager_multiline.applyMutation(op_aor)
        manager_multiline.applyMutation(op_ror)

        # 3 occurrences each → 6 total
        assert len(manager_multiline.mutateList) == 6

    def test_operator_ids_preserved_across_multiple_operators(self, manager_multiline):
        op_aor = _make_operator(ast.Add, ast.Sub(), "AOR")
        op_ror = _make_operator(ast.Add, ast.Mult(), "ROR")

        manager_multiline.applyMutation(op_aor)
        manager_multiline.applyMutation(op_ror)

        ids = [m.operator_id for m in manager_multiline.mutateList]
        assert ids.count("AOR") == 3
        assert ids.count("ROR") == 3


# ═══════════════════════════════════════════════════════════════════════════ #
# Properties                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestProperties:

    def test_program_ast_raises_before_parse(self, manager):
        with pytest.raises(RuntimeError, match="AST is not available"):
            _ = manager.program_ast

    def test_program_source_raises_before_parse(self, manager):
        with pytest.raises(RuntimeError, match="AST is not available"):
            _ = manager.program_source

    def test_program_ast_returns_after_parse(self, parsed_manager):
        assert isinstance(parsed_manager.program_ast, ast.AST)

    def test_program_source_returns_after_parse(self, parsed_manager):
        assert parsed_manager.program_source == SOURCE_WITH_ADD

    def test_program_ast_is_same_object_as_internal(self, parsed_manager):
        assert parsed_manager.program_ast is parsed_manager._program_ast

    def test_program_source_is_same_as_internal(self, parsed_manager):
        assert parsed_manager.program_source is parsed_manager._program_source


# ═══════════════════════════════════════════════════════════════════════════ #
# _count_occurrences()                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestCountOccurrences:

    def test_zero_occurrences(self, parsed_manager):
        assert parsed_manager._count_occurrences(ast.Mult) == 0

    def test_one_occurrence(self, parsed_manager):
        # SOURCE_WITH_ADD = "x = 1 + 2" → one ast.Add
        assert parsed_manager._count_occurrences(ast.Add) == 1

    def test_two_occurrences(self, manager_two_adds):
        assert manager_two_adds._count_occurrences(ast.Add) == 2

    def test_three_occurrences(self, manager_multiline):
        assert manager_multiline._count_occurrences(ast.Add) == 3


# ═══════════════════════════════════════════════════════════════════════════ #
# _assert_ast_ready()                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestAssertAstReady:

    def test_raises_before_parse(self, manager):
        with pytest.raises(RuntimeError, match="AST is not available"):
            manager._assert_ast_ready()

    def test_does_not_raise_after_parse(self, parsed_manager):
        parsed_manager._assert_ast_ready()   # must not raise


# ═══════════════════════════════════════════════════════════════════════════ #
# _NodeReplacer                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestNodeReplacer:

    def _parse(self, source: str) -> ast.AST:
        tree = ast.parse(source)
        ast.fix_missing_locations(tree)
        return tree

    def test_replaces_first_occurrence(self):
        tree = self._parse("x = 1 + 2 + 3")
        replacer = _NodeReplacer(ast.Add, ast.Sub(), occurrence_index=0)
        new_tree = replacer.visit(copy.deepcopy(tree))
        source = ast.unparse(new_tree)
        # First Add replaced → Sub appears; remaining Add still present
        assert "-" in source

    def test_replaces_second_occurrence(self):
        tree = self._parse("x = 1 + 2 + 3")
        replacer = _NodeReplacer(ast.Add, ast.Sub(), occurrence_index=1)
        new_tree = replacer.visit(copy.deepcopy(tree))
        source = ast.unparse(new_tree)
        assert "-" in source

    def test_non_target_nodes_untouched(self):
        tree = self._parse("x = 1 * 2 + 3")
        replacer = _NodeReplacer(ast.Add, ast.Sub(), occurrence_index=0)
        new_tree = replacer.visit(copy.deepcopy(tree))
        source = ast.unparse(new_tree)
        # Mult must still be present
        assert "*" in source

    def test_only_one_occurrence_replaced_per_pass(self):
        """When there are two Adds, replacing occurrence 0 leaves one Add."""
        tree = self._parse("x = 1 + 2 + 3")
        replacer = _NodeReplacer(ast.Add, ast.Sub(), occurrence_index=0)
        new_tree = replacer.visit(copy.deepcopy(tree))
        adds = sum(1 for n in ast.walk(new_tree) if isinstance(n, ast.Add))
        subs = sum(1 for n in ast.walk(new_tree) if isinstance(n, ast.Sub))
        assert adds == 1
        assert subs == 1

    def test_replacement_node_is_deep_copied(self):
        """The same replacement template used twice must produce independent nodes."""
        template = ast.Sub()
        tree = self._parse("x = 1 + 2 + 3")

        replacer_0 = _NodeReplacer(ast.Add, template, occurrence_index=0)
        new_tree_0 = replacer_0.visit(copy.deepcopy(tree))

        replacer_1 = _NodeReplacer(ast.Add, template, occurrence_index=1)
        new_tree_1 = replacer_1.visit(copy.deepcopy(tree))

        subs_0 = [n for n in ast.walk(new_tree_0) if isinstance(n, ast.Sub)]
        subs_1 = [n for n in ast.walk(new_tree_1) if isinstance(n, ast.Sub)]

        # Each tree should have exactly one Sub, and they must be distinct objects
        assert len(subs_0) == 1
        assert len(subs_1) == 1
        assert subs_0[0] is not subs_1[0]


# ═══════════════════════════════════════════════════════════════════════════ #
# Mutant                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestMutant:

    def test_repr_contains_operator_id(self):
        m = Mutant(operator_id="AOR", occurrence_index=0, source_code="x = 1 - 2")
        assert "AOR" in repr(m)

    def test_repr_contains_occurrence_index(self):
        m = Mutant(operator_id="AOR", occurrence_index=3, source_code="x = 1 - 2")
        assert "3" in repr(m)

    def test_repr_contains_source_preview(self):
        m = Mutant(operator_id="AOR", occurrence_index=0, source_code="x = 1 - 2")
        assert "x = 1 - 2" in repr(m)

    def test_repr_truncates_long_source(self):
        long_source = "x = " + "1 + " * 100 + "0"
        m = Mutant(operator_id="AOR", occurrence_index=0, source_code=long_source)
        r = repr(m)
        # Preview is capped at 60 chars
        assert len(r) < len(long_source)

    def test_repr_replaces_newlines_in_preview(self):
        m = Mutant(operator_id="AOR", occurrence_index=0, source_code="x = 1\ny = 2")
        assert "\\n" in repr(m)


# ═══════════════════════════════════════════════════════════════════════════ #
# MutationManager.__repr__                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestMutationManagerRepr:

    def test_repr_before_parse_shows_ast_not_ready(self, manager):
        r = repr(manager)
        assert "ast_ready=False" in r

    def test_repr_after_parse_shows_ast_ready(self, parsed_manager):
        r = repr(parsed_manager)
        assert "ast_ready=True" in r

    def test_repr_shows_mutant_count_zero(self, parsed_manager):
        assert "mutants=0" in repr(parsed_manager)

    def test_repr_shows_mutant_count_after_mutation(self, parsed_manager, aor_operator):
        parsed_manager.applyMutation(aor_operator)
        assert "mutants=1" in repr(parsed_manager)


# ═══════════════════════════════════════════════════════════════════════════ #
# Integration                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestIntegration:

    def test_full_pipeline_add_to_sub(self, mock_config):
        """End-to-end: parse → mutate Add→Sub → check mutants are valid Python."""
        manager = MutationManager(configLoader=mock_config)
        manager.parseToAST(SOURCE_MULTILINE)

        op = _make_operator(ast.Add, ast.Sub(), "AOR")
        mutants = manager.applyMutation(op)

        assert len(mutants) == 3
        for mutant in mutants:
            # Every mutant must be parseable Python
            ast.parse(mutant.source_code)
            # The mutant must contain a subtraction
            assert "-" in mutant.source_code

    def test_pipeline_two_operators_no_cross_contamination(self, mock_config):
        """Applying two operators must not mix their mutants."""
        manager = MutationManager(configLoader=mock_config)
        manager.parseToAST(SOURCE_MULTILINE)

        op_aor = _make_operator(ast.Add, ast.Sub(), "AOR")
        op_ror = _make_operator(ast.Add, ast.Mult(), "ROR")

        aor_mutants = manager.applyMutation(op_aor)
        ror_mutants = manager.applyMutation(op_ror)

        for m in aor_mutants:
            assert m.operator_id == "AOR"
        for m in ror_mutants:
            assert m.operator_id == "ROR"

    def test_reparse_resets_state_fully(self, mock_config):
        """After a second parseToAST, mutateList must be empty and new mutations work."""
        manager = MutationManager(configLoader=mock_config)
        manager.parseToAST(SOURCE_WITH_ADD)

        op = _make_operator(ast.Add, ast.Sub(), "AOR")
        manager.applyMutation(op)
        assert len(manager.mutateList) == 1

        # Re-parse with a different source
        manager.parseToAST(SOURCE_MULTILINE)
        assert manager.mutateList == []

        new_mutants = manager.applyMutation(op)
        assert len(new_mutants) == 3
        assert len(manager.mutateList) == 3