"""
Unit tests for MutationManager (refactored)
=============================================
Coverage targets
----------------
- __post_init__              : valid construction, TypeError on bad configLoader
                               (each missing attr), RuntimeError on unloaded
                               configLoader, TypeError on bad mutateList
- parseToAST()               : TypeError (non-string, empty, whitespace),
                               ValueError (syntax error), success (returns AST,
                               stores source + AST, fix_missing_locations),
                               clears mutateList on second call,
                               second call with empty list (no-op branch)
- applyMutation()            : RuntimeError before parseToAST,
                               TypeError (missing each required attr,
                               bad operator.name, bad operator.registers),
                               empty registers → returns [],
                               single node → 1 mutant,
                               multiple nodes → N mutants,
                               node-not-found branch (skip),
                               original AST never modified,
                               mutateList accumulation across operators,
                               buildMutate called once per registered node
- program_ast property       : RuntimeError before parse, correct value after
- program_source property    : RuntimeError before parse, correct value after
- _assert_ast_ready()        : RuntimeError / no error
- _NodeReplacer              : identity match replaces node,
                               non-matching node left intact,
                               deep-copies replacement template
- Mutant.__repr__            : fields present, newline escaping, truncation
- MutationManager.__repr__   : ast_ready flag, mutant count

Run with:
    pytest test_mutation_manager.py -v \
        --cov=code.mutation_manager \
        --cov-report=term-missing
"""

import ast
import copy

import pytest
from unittest.mock import MagicMock, PropertyMock

from code.mutation_manager import MutationManager, Mutant, _NodeReplacer


# ═══════════════════════════════════════════════════════════════════════════ #
# Source code constants                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

SRC_ONE_ADD       = "x = 1 + 2"
SRC_TWO_ADDS      = "x = 1 + 2 + 3"
SRC_THREE_ADDS    = "a = 1 + 2\nb = 3 + 4\nc = a + b"
SRC_NO_ADD        = "x = 1"
SRC_SYNTAX_ERROR  = "def foo(:\n    pass"


# ═══════════════════════════════════════════════════════════════════════════ #
# Helpers                                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

def _make_config(source: str = SRC_ONE_ADD) -> MagicMock:
    """Return a mock ConfigLoader that satisfies duck-type checks."""
    cfg = MagicMock()
    cfg.program_source = source
    cfg.workspace_path = "/tmp/ws"
    cfg.operatorsList  = ["AOR"]
    return cfg


def _make_operator(name: str = "AOR",
                   registers: list | None = None,
                   replacement: ast.AST | None = None) -> MagicMock:
    """
    Return a mock Operator with the full interface MutationManager expects.
    ``buildMutate`` always returns a deep-copy of ``replacement`` (default Sub).
    """
    if replacement is None:
        replacement = ast.Sub()

    op = MagicMock()
    op.name        = name
    op.registers   = registers if registers is not None else []
    op.analyseAST  = MagicMock(return_value=op.registers)
    op.buildMutate = MagicMock(side_effect=lambda _: copy.deepcopy(replacement))
    return op


def _add_nodes_from(manager: MutationManager) -> list[ast.AST]:
    """Return the actual ast.Add node instances from the manager's AST."""
    return [n for n in ast.walk(manager._program_ast) if isinstance(n, ast.Add)]


# ═══════════════════════════════════════════════════════════════════════════ #
# Fixtures                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

@pytest.fixture
def cfg():
    return _make_config()


@pytest.fixture
def manager(cfg):
    return MutationManager(configLoader=cfg)


@pytest.fixture
def parsed_manager(manager):
    manager.parseToAST(SRC_ONE_ADD)
    return manager


@pytest.fixture
def manager_two_adds(cfg):
    m = MutationManager(configLoader=cfg)
    m.parseToAST(SRC_TWO_ADDS)
    return m


@pytest.fixture
def manager_three_adds(cfg):
    m = MutationManager(configLoader=cfg)
    m.parseToAST(SRC_THREE_ADDS)
    return m


# ═══════════════════════════════════════════════════════════════════════════ #
# Construction / __post_init__                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestConstruction:

    def test_valid_construction(self, cfg):
        m = MutationManager(configLoader=cfg)
        assert m.configLoader is cfg
        assert m.mutateList == []

    def test_custom_mutate_list_accepted(self, cfg):
        existing = [Mutant("AOR", 0, "x = 1 - 2")]
        m = MutationManager(configLoader=cfg, mutateList=existing)
        assert m.mutateList is existing

    # --- configLoader: missing attributes ------------------------------------

    def test_missing_program_source_raises_type_error(self):
        cfg = MagicMock(spec=["workspace_path", "operatorsList"])
        with pytest.raises(TypeError, match="missing attribute 'program_source'"):
            MutationManager(configLoader=cfg)

    def test_missing_workspace_path_raises_type_error(self):
        cfg = MagicMock(spec=["program_source", "operatorsList"])
        with pytest.raises(TypeError, match="missing attribute 'workspace_path'"):
            MutationManager(configLoader=cfg)

    def test_missing_operators_list_raises_type_error(self):
        cfg = MagicMock(spec=["program_source", "workspace_path"])
        with pytest.raises(TypeError, match="missing attribute 'operatorsList'"):
            MutationManager(configLoader=cfg)

    # --- configLoader: not loaded --------------------------------------------

    def test_unloaded_config_loader_raises_runtime_error(self):
        cfg = MagicMock()
        cfg.workspace_path = "/tmp"
        cfg.operatorsList  = ["AOR"]
        type(cfg).program_source = PropertyMock(
            side_effect=RuntimeError("Call .load() first.")
        )
        with pytest.raises(RuntimeError, match="has not been loaded yet"):
            MutationManager(configLoader=cfg)

    # --- mutateList: wrong type ----------------------------------------------

    def test_mutate_list_string_raises_type_error(self, cfg):
        with pytest.raises(TypeError, match="mutateList must be a list"):
            MutationManager(configLoader=cfg, mutateList="bad")

    def test_mutate_list_tuple_raises_type_error(self, cfg):
        with pytest.raises(TypeError, match="mutateList must be a list"):
            MutationManager(configLoader=cfg, mutateList=())

    def test_mutate_list_none_raises_type_error(self, cfg):
        with pytest.raises(TypeError, match="mutateList must be a list"):
            MutationManager(configLoader=cfg, mutateList=None)


# ═══════════════════════════════════════════════════════════════════════════ #
# parseToAST()                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestParseToAST:

    # --- TypeError -----------------------------------------------------------

    def test_none_raises_type_error(self, manager):
        with pytest.raises(TypeError, match="source must be a non-empty string"):
            manager.parseToAST(None)

    def test_integer_raises_type_error(self, manager):
        with pytest.raises(TypeError, match="source must be a non-empty string"):
            manager.parseToAST(42)

    def test_empty_string_raises_type_error(self, manager):
        with pytest.raises(TypeError, match="source must be a non-empty string"):
            manager.parseToAST("")

    def test_whitespace_only_raises_type_error(self, manager):
        with pytest.raises(TypeError, match="source must be a non-empty string"):
            manager.parseToAST("   \n\t")

    # --- ValueError ----------------------------------------------------------

    def test_syntax_error_raises_value_error(self, manager):
        with pytest.raises(ValueError, match="Syntax error in source"):
            manager.parseToAST(SRC_SYNTAX_ERROR)

    # --- Success -------------------------------------------------------------

    def test_returns_ast_module(self, manager):
        result = manager.parseToAST(SRC_ONE_ADD)
        assert isinstance(result, ast.AST)

    def test_stores_program_source(self, manager):
        manager.parseToAST(SRC_ONE_ADD)
        assert manager._program_source == SRC_ONE_ADD

    def test_stores_program_ast(self, manager):
        manager.parseToAST(SRC_ONE_ADD)
        assert manager._program_ast is not None

    def test_returned_tree_is_stored_tree(self, manager):
        result = manager.parseToAST(SRC_ONE_ADD)
        assert result is manager._program_ast

    def test_fix_missing_locations_applied(self, manager):
        manager.parseToAST(SRC_ONE_ADD)
        for node in ast.walk(manager._program_ast):
            if hasattr(node, "lineno"):
                assert node.lineno is not None

    # --- Second call behaviour -----------------------------------------------

    def test_second_call_clears_mutate_list(self, manager):
        manager.parseToAST(SRC_ONE_ADD)
        adds = _add_nodes_from(manager)
        op   = _make_operator(registers=adds)
        manager.applyMutation(op)
        assert len(manager.mutateList) > 0

        manager.parseToAST(SRC_ONE_ADD)
        assert manager.mutateList == []

    def test_second_call_with_empty_list_does_not_raise(self, manager):
        """The 'if self.mutateList' branch must be skipped gracefully."""
        manager.parseToAST(SRC_ONE_ADD)
        assert manager.mutateList == []
        manager.parseToAST(SRC_NO_ADD)   # must not raise


# ═══════════════════════════════════════════════════════════════════════════ #
# applyMutation()                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestApplyMutation:

    # --- RuntimeError before parseToAST --------------------------------------

    def test_raises_before_parse(self, manager):
        op = _make_operator()
        with pytest.raises(RuntimeError, match="AST is not available"):
            manager.applyMutation(op)

    # --- TypeError: missing operator attributes ------------------------------

    def test_missing_name_raises_type_error(self, parsed_manager):
        op = MagicMock(spec=["registers", "analyseAST", "buildMutate"])
        op.registers = []
        with pytest.raises(TypeError, match="missing attribute / method 'name'"):
            parsed_manager.applyMutation(op)

    def test_missing_registers_raises_type_error(self, parsed_manager):
        op = MagicMock(spec=["name", "analyseAST", "buildMutate"])
        op.name = "AOR"
        with pytest.raises(TypeError, match="missing attribute / method 'registers'"):
            parsed_manager.applyMutation(op)

    def test_missing_analyse_ast_raises_type_error(self, parsed_manager):
        op = MagicMock(spec=["name", "registers", "buildMutate"])
        op.name      = "AOR"
        op.registers = []
        with pytest.raises(TypeError, match="missing attribute / method 'analyseAST'"):
            parsed_manager.applyMutation(op)

    def test_missing_build_mutate_raises_type_error(self, parsed_manager):
        op = MagicMock(spec=["name", "registers", "analyseAST"])
        op.name      = "AOR"
        op.registers = []
        with pytest.raises(TypeError, match="missing attribute / method 'buildMutate'"):
            parsed_manager.applyMutation(op)

    # --- TypeError: bad operator.name ----------------------------------------

    def test_empty_operator_name_raises_type_error(self, parsed_manager):
        op = _make_operator(name="")
        with pytest.raises(TypeError, match="operator.name must be a non-empty string"):
            parsed_manager.applyMutation(op)

    def test_whitespace_operator_name_raises_type_error(self, parsed_manager):
        op = _make_operator(name="  ")
        with pytest.raises(TypeError, match="operator.name must be a non-empty string"):
            parsed_manager.applyMutation(op)

    def test_non_string_operator_name_raises_type_error(self, parsed_manager):
        op = _make_operator()
        op.name = 123
        with pytest.raises(TypeError, match="operator.name must be a non-empty string"):
            parsed_manager.applyMutation(op)

    # --- TypeError: bad operator.registers -----------------------------------

    def test_registers_not_a_list_raises_type_error(self, parsed_manager):
        op = _make_operator()
        op.registers = "not-a-list"
        with pytest.raises(TypeError, match="operator.registers must be a list"):
            parsed_manager.applyMutation(op)

    # --- Empty registers → returns [] ----------------------------------------

    def test_empty_registers_returns_empty_list(self, parsed_manager):
        op = _make_operator(registers=[])
        result = parsed_manager.applyMutation(op)
        assert result == []

    def test_empty_registers_does_not_add_to_mutate_list(self, parsed_manager):
        op = _make_operator(registers=[])
        parsed_manager.applyMutation(op)
        assert parsed_manager.mutateList == []

    # --- Single node → 1 mutant ----------------------------------------------

    def test_single_node_returns_one_mutant(self, parsed_manager):
        adds = _add_nodes_from(parsed_manager)
        op   = _make_operator(registers=adds)
        result = parsed_manager.applyMutation(op)
        assert len(result) == 1

    def test_single_node_mutant_type(self, parsed_manager):
        adds = _add_nodes_from(parsed_manager)
        op   = _make_operator(registers=adds)
        result = parsed_manager.applyMutation(op)
        assert isinstance(result[0], Mutant)

    def test_single_node_operator_name_stored(self, parsed_manager):
        adds = _add_nodes_from(parsed_manager)
        op   = _make_operator(name="AOR", registers=adds)
        result = parsed_manager.applyMutation(op)
        assert result[0].operator_name == "AOR"

    def test_single_node_occurrence_index_is_zero(self, parsed_manager):
        adds = _add_nodes_from(parsed_manager)
        op   = _make_operator(registers=adds)
        result = parsed_manager.applyMutation(op)
        assert result[0].occurrence_index == 0

    def test_single_node_source_code_is_valid_python(self, parsed_manager):
        adds = _add_nodes_from(parsed_manager)
        op   = _make_operator(registers=adds)
        result = parsed_manager.applyMutation(op)
        ast.parse(result[0].source_code)

    def test_single_node_mutation_reflected_in_source(self, parsed_manager):
        adds = _add_nodes_from(parsed_manager)
        op   = _make_operator(registers=adds, replacement=ast.Sub())
        result = parsed_manager.applyMutation(op)
        assert "-" in result[0].source_code

    def test_single_node_added_to_mutate_list(self, parsed_manager):
        adds = _add_nodes_from(parsed_manager)
        op   = _make_operator(registers=adds)
        result = parsed_manager.applyMutation(op)
        assert parsed_manager.mutateList == result

    def test_build_mutate_called_once_per_node(self, parsed_manager):
        adds = _add_nodes_from(parsed_manager)
        op   = _make_operator(registers=adds)
        parsed_manager.applyMutation(op)
        assert op.buildMutate.call_count == len(adds)

    # --- Multiple nodes → N mutants ------------------------------------------

    def test_two_nodes_return_two_mutants(self, manager_two_adds):
        adds = _add_nodes_from(manager_two_adds)
        op   = _make_operator(registers=adds)
        result = manager_two_adds.applyMutation(op)
        assert len(result) == 2

    def test_three_nodes_return_three_mutants(self, manager_three_adds):
        adds = _add_nodes_from(manager_three_adds)
        op   = _make_operator(registers=adds)
        result = manager_three_adds.applyMutation(op)
        assert len(result) == 3

    def test_occurrence_indices_are_sequential(self, manager_two_adds):
        adds = _add_nodes_from(manager_two_adds)
        op   = _make_operator(registers=adds)
        result = manager_two_adds.applyMutation(op)
        assert [m.occurrence_index for m in result] == [0, 1]

    def test_each_mutant_source_is_valid_python(self, manager_three_adds):
        adds = _add_nodes_from(manager_three_adds)
        op   = _make_operator(registers=adds)
        for mutant in manager_three_adds.applyMutation(op):
            ast.parse(mutant.source_code)

    def test_each_mutant_has_exactly_one_substitution(self, manager_two_adds):
        """Only the targeted occurrence is replaced; remaining Adds stay."""
        adds = _add_nodes_from(manager_two_adds)
        op   = _make_operator(registers=adds, replacement=ast.Sub())
        mutants = manager_two_adds.applyMutation(op)

        for mutant in mutants:
            tree = ast.parse(mutant.source_code)
            subs           = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Sub))
            remaining_adds = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Add))
            assert subs == 1
            assert remaining_adds == len(adds) - 1

    # --- Original AST unmodified ---------------------------------------------

    def test_original_ast_not_modified(self, manager_two_adds):
        original_src = ast.unparse(manager_two_adds._program_ast)
        adds = _add_nodes_from(manager_two_adds)
        op   = _make_operator(registers=adds)
        manager_two_adds.applyMutation(op)
        assert ast.unparse(manager_two_adds._program_ast) == original_src

    # --- Node-not-found branch (skip) ----------------------------------------

    def test_foreign_node_is_skipped(self, parsed_manager):
        """
        A node that is not part of the manager's AST cannot be found in
        ast.walk → applyMutation must skip it without raising.
        """
        foreign_node = ast.Add()
        op = _make_operator(registers=[foreign_node])
        result = parsed_manager.applyMutation(op)
        assert result == []
        assert parsed_manager.mutateList == []

    # --- Accumulation across multiple operators ------------------------------

    def test_two_operators_accumulate_in_mutate_list(self, manager_three_adds):
        adds   = _add_nodes_from(manager_three_adds)
        op_aor = _make_operator(name="AOR", registers=adds, replacement=ast.Sub())
        op_ror = _make_operator(name="ROR", registers=adds, replacement=ast.Mult())

        manager_three_adds.applyMutation(op_aor)
        manager_three_adds.applyMutation(op_ror)

        assert len(manager_three_adds.mutateList) == 6

    def test_operator_names_preserved_in_mutate_list(self, manager_three_adds):
        adds   = _add_nodes_from(manager_three_adds)
        op_aor = _make_operator(name="AOR", registers=adds)
        op_ror = _make_operator(name="ROR", registers=adds)

        manager_three_adds.applyMutation(op_aor)
        manager_three_adds.applyMutation(op_ror)

        names = [m.operator_name for m in manager_three_adds.mutateList]
        assert names.count("AOR") == 3
        assert names.count("ROR") == 3

    def test_reparse_resets_mutate_list_between_operators(self, cfg):
        m = MutationManager(configLoader=cfg)
        m.parseToAST(SRC_ONE_ADD)

        op = _make_operator(registers=_add_nodes_from(m))
        m.applyMutation(op)
        assert len(m.mutateList) == 1

        m.parseToAST(SRC_THREE_ADDS)
        assert m.mutateList == []

        op2 = _make_operator(registers=_add_nodes_from(m))
        m.applyMutation(op2)
        assert len(m.mutateList) == 3


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

    def test_program_ast_after_parse(self, parsed_manager):
        assert isinstance(parsed_manager.program_ast, ast.AST)

    def test_program_source_after_parse(self, parsed_manager):
        assert parsed_manager.program_source == SRC_ONE_ADD

    def test_program_ast_is_internal_object(self, parsed_manager):
        assert parsed_manager.program_ast is parsed_manager._program_ast

    def test_program_source_is_internal_object(self, parsed_manager):
        assert parsed_manager.program_source is parsed_manager._program_source


# ═══════════════════════════════════════════════════════════════════════════ #
# _assert_ast_ready()                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestAssertAstReady:

    def test_raises_before_parse(self, manager):
        with pytest.raises(RuntimeError, match="AST is not available"):
            manager._assert_ast_ready()

    def test_does_not_raise_after_parse(self, parsed_manager):
        parsed_manager._assert_ast_ready()


# ═══════════════════════════════════════════════════════════════════════════ #
# _NodeReplacer                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestNodeReplacer:

    def _tree(self, src: str) -> ast.AST:
        t = ast.parse(src)
        ast.fix_missing_locations(t)
        return t

    def test_replaces_target_node_by_identity(self):
        tree = self._tree("x = 1 + 2")
        add  = next(n for n in ast.walk(tree) if isinstance(n, ast.Add))
        new_tree = _NodeReplacer(add, ast.Sub()).visit(copy.deepcopy(tree))
        assert sum(1 for n in ast.walk(new_tree) if isinstance(n, ast.Sub)) == 1

    def test_non_target_nodes_untouched(self):
        tree = self._tree("x = 1 * 2 + 3")
        add  = next(n for n in ast.walk(tree) if isinstance(n, ast.Add))
        new_tree = _NodeReplacer(add, ast.Sub()).visit(copy.deepcopy(tree))
        assert sum(1 for n in ast.walk(new_tree) if isinstance(n, ast.Mult)) == 1

    def test_only_matching_identity_replaced(self):
        tree = self._tree("x = 1 + 2 + 3")
        adds = [n for n in ast.walk(tree) if isinstance(n, ast.Add)]
        new_tree = _NodeReplacer(adds[0], ast.Sub()).visit(copy.deepcopy(tree))
        assert sum(1 for n in ast.walk(new_tree) if isinstance(n, ast.Add)) == 1
        assert sum(1 for n in ast.walk(new_tree) if isinstance(n, ast.Sub)) == 1

    def test_replacement_node_is_deep_copied(self):
        tree     = self._tree("x = 1 + 2")
        add      = next(n for n in ast.walk(tree) if isinstance(n, ast.Add))
        template = ast.Sub()
        replacer = _NodeReplacer(add, template)
        r1 = replacer.generic_visit(add)
        r2 = replacer.generic_visit(add)
        assert isinstance(r1, ast.Sub)
        assert r1 is not template
        assert r2 is not template

    def test_unmatched_node_delegates_to_super(self):
        tree  = self._tree("x = 1")
        const = next(n for n in ast.walk(tree) if isinstance(n, ast.Constant))
        replacer = _NodeReplacer(ast.Add(), ast.Sub())   # different target
        result = replacer.generic_visit(const)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════ #
# Mutant                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestMutant:

    def test_repr_contains_operator_name(self):
        assert "AOR" in repr(Mutant("AOR", 0, "x = 1 - 2"))

    def test_repr_contains_occurrence_index(self):
        assert "7" in repr(Mutant("AOR", 7, "x = 1 - 2"))

    def test_repr_contains_source_preview(self):
        assert "x = 1 - 2" in repr(Mutant("AOR", 0, "x = 1 - 2"))

    def test_repr_truncates_long_source(self):
        long_src = "x = " + "+ ".join(["1"] * 50)
        assert len(repr(Mutant("AOR", 0, long_src))) < len(long_src) + 60

    def test_repr_escapes_newlines(self):
        assert "\\n" in repr(Mutant("AOR", 0, "x = 1\ny = 2"))


# ═══════════════════════════════════════════════════════════════════════════ #
# MutationManager.__repr__                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestMutationManagerRepr:

    def test_repr_before_parse(self, manager):
        assert "ast_ready=False" in repr(manager)

    def test_repr_after_parse(self, parsed_manager):
        assert "ast_ready=True" in repr(parsed_manager)

    def test_repr_zero_mutants(self, parsed_manager):
        assert "mutants=0" in repr(parsed_manager)

    def test_repr_mutant_count_after_mutation(self, parsed_manager):
        op = _make_operator(registers=_add_nodes_from(parsed_manager))
        parsed_manager.applyMutation(op)
        assert "mutants=1" in repr(parsed_manager)


# ═══════════════════════════════════════════════════════════════════════════ #
# Integration                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestIntegration:

    def test_full_pipeline_single_occurrence(self, cfg):
        m = MutationManager(configLoader=cfg)
        m.parseToAST(SRC_ONE_ADD)
        adds    = _add_nodes_from(m)
        mutants = m.applyMutation(_make_operator(registers=adds, replacement=ast.Sub()))

        assert len(mutants) == 1
        ast.parse(mutants[0].source_code)
        assert "-" in mutants[0].source_code

    def test_full_pipeline_multiple_occurrences(self, cfg):
        m = MutationManager(configLoader=cfg)
        m.parseToAST(SRC_THREE_ADDS)
        adds    = _add_nodes_from(m)
        mutants = m.applyMutation(_make_operator(registers=adds, replacement=ast.Sub()))

        assert len(mutants) == 3
        for mu in mutants:
            ast.parse(mu.source_code)

    def test_two_operators_no_cross_contamination(self, cfg):
        m = MutationManager(configLoader=cfg)
        m.parseToAST(SRC_THREE_ADDS)
        adds = _add_nodes_from(m)

        aor = m.applyMutation(_make_operator(name="AOR", registers=adds, replacement=ast.Sub()))
        ror = m.applyMutation(_make_operator(name="ROR", registers=adds, replacement=ast.Mult()))

        assert all(mu.operator_name == "AOR" for mu in aor)
        assert all(mu.operator_name == "ROR" for mu in ror)

    def test_reparse_fully_resets_state(self, cfg):
        m = MutationManager(configLoader=cfg)
        m.parseToAST(SRC_ONE_ADD)
        m.applyMutation(_make_operator(registers=_add_nodes_from(m)))
        assert len(m.mutateList) == 1

        m.parseToAST(SRC_THREE_ADDS)
        assert m.mutateList == []
        m.applyMutation(_make_operator(registers=_add_nodes_from(m)))
        assert len(m.mutateList) == 3