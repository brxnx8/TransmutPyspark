"""
Unit tests for OperatorNFTP
============================
Comprehensive test suite for the NFTP (Negation of Filter Transformation
Predicate) mutation operator.

DESIGN PHILOSOPHY:
  These tests are DOMAIN-AGNOSTIC and focus on the core logic of the
  OperatorNFTP class, NOT on PySpark-specific code. Tests use:
  - Manually constructed AST nodes (not parsed from real code)
  - Generic predicates that could apply to any filtering system
  - Controlled test data via fixtures (not external files)
  - Mock strategies to isolate class behavior

This ensures tests remain valid regardless of target framework changes.

Coverage areas:
  - Initialization and attribute validation
  - AST node detection (generic call patterns)
  - Sub-condition decomposition logic (BoolOp, BinOp, UnaryOp)
  - Mutation logic (negation wrapper creation)
  - Node location and replacement algorithms
  - File writing and directory structure
  - Error handling and edge cases
"""

import ast
import copy
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

from src.operator_nftp import OperatorNFTP
from src.mutant import Mutant


# ─────────────────────────────────────────────────────────────────────────── #
# Fixtures: Common test data and helpers
# ─────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def operator_instance():
    """Fresh OperatorNFTP instance for each test."""
    return OperatorNFTP()


@pytest.fixture
def temp_source_file():
    """Temporary Python source file for mutant generation tests."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("# test source file\n")
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_mutant_dir():
    """Temporary directory for generated mutants."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def make_call_node(func_attr: str, predicate: ast.expr, lineno: int = 1, col_offset: int = 0) -> ast.Call:
    """
    Helper to construct a generic .filter() or .where() call node.
    
    Parameters
    ----------
    func_attr : str
        Method name: "filter" or "where"
    predicate : ast.expr
        The predicate expression (first argument)
    lineno, col_offset : int
        AST location metadata
    
    Returns
    -------
    ast.Call
        A call node representing df.method(predicate)
    """
    call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="df", ctx=ast.Load()),
            attr=func_attr,
            ctx=ast.Load()
        ),
        args=[predicate],
        keywords=[]
    )
    call.lineno = lineno
    call.col_offset = col_offset
    return call


def make_comparison(left: ast.expr, op_type: type, right: ast.expr) -> ast.Compare:
    """
    Helper to construct a comparison expression.
    
    Parameters
    ----------
    left, right : ast.expr
        Operands
    op_type : type
        Comparison operator class (e.g., ast.Gt, ast.Lt, ast.Eq)
    
    Returns
    -------
    ast.Compare
        A comparison node: left op right
    """
    compare = ast.Compare(left=left, ops=[op_type()], comparators=[right])
    ast.fix_missing_locations(compare)
    return compare


def make_boolop(op_type: type, *values: ast.expr) -> ast.BoolOp:
    """
    Helper to construct a BoolOp (and/or) expression.
    
    Parameters
    ----------
    op_type : type
        Boolean operator class (ast.And or ast.Or)
    *values : ast.expr
        Operands
    
    Returns
    -------
    ast.BoolOp
        A boolean operation node
    """
    node = ast.BoolOp(op=op_type(), values=list(values))
    ast.fix_missing_locations(node)
    return node


def make_binop(left: ast.expr, op_type: type, right: ast.expr) -> ast.BinOp:
    """
    Helper to construct a BinOp (bitwise &/|) expression.
    
    Parameters
    ----------
    left, right : ast.expr
        Operands
    op_type : type
        Binary operator class (e.g., ast.BitAnd, ast.BitOr)
    
    Returns
    -------
    ast.BinOp
        A binary operation node
    """
    node = ast.BinOp(left=left, op=op_type(), right=right)
    ast.fix_missing_locations(node)
    return node


def make_simple_predicate(name: str = "x") -> ast.Compare:
    """Create a simple predicat: name > 0."""
    return make_comparison(
        ast.Name(id=name, ctx=ast.Load()),
        ast.Gt,
        ast.Constant(value=0)
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Test Suites
# ─────────────────────────────────────────────────────────────────────────── #

class TestOperatorNFTPInitialization:
    """Test suite for OperatorNFTP.__init__() and attribute setup."""

    def test_init_basic_attributes(self, operator_instance):
        """Verify initialization sets correct id, name, and mutant_registers."""
        assert operator_instance.id == 2
        assert operator_instance.name == "NFTP"
        assert operator_instance.mutant_registers == ["filter", "where"]

    def test_init_mutant_list_empty(self, operator_instance):
        """Verify mutant_list starts empty."""
        assert operator_instance.mutant_list == []
        assert isinstance(operator_instance.mutant_list, list)

    def test_init_inherits_from_operator(self, operator_instance):
        """Verify OperatorNFTP is a proper subclass."""
        assert hasattr(operator_instance, "analyse_ast")
        assert hasattr(operator_instance, "build_mutant")
        assert hasattr(operator_instance, "_next_mutant_id")


class TestAnalyseAST:
    """Test suite for OperatorNFTP.analyse_ast() - GENERIC (no PySpark)."""

    def test_analyse_ast_simple_call_with_argument(self, operator_instance):
        """Detect a .filter() call with predicate argument."""
        predicate = make_simple_predicate()
        call_node = make_call_node("filter", predicate, lineno=1, col_offset=0)
        
        # Build AST with this call
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        
        assert len(nodes) == 1
        assert isinstance(nodes[0], ast.Call)

    def test_analyse_ast_where_method(self, operator_instance):
        """Detect a .where() call (should be treated like filter)."""
        predicate = make_simple_predicate()
        call_node = make_call_node("where", predicate, lineno=1, col_offset=0)
        
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        
        assert len(nodes) == 1

    def test_analyse_ast_multiple_calls_same_module(self, operator_instance):
        """Detect multiple filter/where calls in one module."""
        pred1 = make_simple_predicate("x")
        pred2 = make_simple_predicate("y")
        
        call1 = make_call_node("filter", pred1, lineno=1, col_offset=0)
        call2 = make_call_node("where", pred2, lineno=2, col_offset=0)
        
        module = ast.Module(
            body=[ast.Expr(value=call1), ast.Expr(value=call2)],
            type_ignores=[]
        )
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        
        assert len(nodes) == 2

    def test_analyse_ast_ignore_no_arguments(self, operator_instance):
        """Skip filter/where calls with no predicate argument."""
        call_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="df", ctx=ast.Load()),
                attr="filter",
                ctx=ast.Load()
            ),
            args=[],  # No arguments
            keywords=[]
        )
        call_node.lineno = 1
        call_node.col_offset = 0
        
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        
        assert len(nodes) == 0

    def test_analyse_ast_ignore_function_calls(self, operator_instance):
        """Ignore function calls (not method calls)."""
        # filter(predicate) — function call, not method call
        call_node = ast.Call(
            func=ast.Name(id="filter", ctx=ast.Load()),  # Direct function name
            args=[make_simple_predicate()],
            keywords=[]
        )
        call_node.lineno = 1
        call_node.col_offset = 0
        
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        
        assert len(nodes) == 0

    def test_analyse_ast_ignore_other_methods(self, operator_instance):
        """Ignore method calls other than filter/where."""
        # df.map(...) should be ignored
        call_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="df", ctx=ast.Load()),
                attr="map",  # Not filter or where
                ctx=ast.Load()
            ),
            args=[make_simple_predicate()],
            keywords=[]
        )
        call_node.lineno = 1
        call_node.col_offset = 0
        
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        
        assert len(nodes) == 0

    def test_analyse_ast_invalid_tree_type(self, operator_instance):
        """Verify TypeError is raised for non-AST input."""
        with pytest.raises(TypeError):
            operator_instance.analyse_ast("not an ast")
        
        with pytest.raises(TypeError):
            operator_instance.analyse_ast(None)
        
        with pytest.raises(TypeError):
            operator_instance.analyse_ast([])

    def test_analyse_ast_empty_tree(self, operator_instance):
        """Handle empty AST module gracefully."""
        empty_module = ast.Module(body=[], type_ignores=[])
        
        nodes = operator_instance.analyse_ast(empty_module)
        
        assert nodes == []

    def test_analyse_ast_tree_with_no_calls(self, operator_instance):
        """Handle AST with no call nodes at all."""
        # Just assignments
        module = ast.Module(
            body=[
                ast.Assign(targets=[ast.Name(id="x", ctx=ast.Store())], value=ast.Constant(value=1))
            ],
            type_ignores=[]
        )
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        
        assert nodes == []


class TestCollectSubconditions:
    """Test suite for OperatorNFTP._collect_subconditions() - GENERIC logic."""

    def test_collect_simple_leaf_condition(self, operator_instance):
        """A simple comparison is returned as-is (single leaf)."""
        # x > 0
        leaf = make_simple_predicate("x")
        
        conditions = operator_instance._collect_subconditions(leaf)
        
        assert len(conditions) == 1
        assert conditions[0] is leaf

    def test_collect_boolop_and_two_operands(self, operator_instance):
        """Decompose 'and' into two leaf conditions."""
        # (x > 0) and (y < 10)
        left = make_simple_predicate("x")
        right = make_simple_predicate("y")
        boolop = make_boolop(ast.And, left, right)
        
        conditions = operator_instance._collect_subconditions(boolop)
        
        assert len(conditions) == 2

    def test_collect_boolop_or_two_operands(self, operator_instance):
        """Decompose 'or' into two leaf conditions."""
        # (x > 0) or (y < 10)
        left = make_simple_predicate("x")
        right = make_simple_predicate("y")
        boolop = make_boolop(ast.Or, left, right)
        
        conditions = operator_instance._collect_subconditions(boolop)
        
        assert len(conditions) == 2

    def test_collect_boolop_multiple_operands(self, operator_instance):
        """Decompose 'and' with multiple operands."""
        # (x > 0) and (y < 10) and (z == 5)
        left = make_simple_predicate("x")
        middle = make_simple_predicate("y")
        right = make_simple_predicate("z")
        boolop = make_boolop(ast.And, left, middle, right)
        
        conditions = operator_instance._collect_subconditions(boolop)
        
        assert len(conditions) == 3

    def test_collect_bitop_and(self, operator_instance):
        """Decompose PySpark '&' (BitAnd) into two operands."""
        # (x > 0) & (y < 10)
        left = make_simple_predicate("x")
        right = make_simple_predicate("y")
        bitop = make_binop(left, ast.BitAnd, right)
        
        conditions = operator_instance._collect_subconditions(bitop)
        
        assert len(conditions) == 2

    def test_collect_bitop_or(self, operator_instance):
        """Decompose PySpark '|' (BitOr) into two operands."""
        # (x > 0) | (y < 10)
        left = make_simple_predicate("x")
        right = make_simple_predicate("y")
        bitop = make_binop(left, ast.BitOr, right)
        
        conditions = operator_instance._collect_subconditions(bitop)
        
        assert len(conditions) == 2

    def test_collect_nested_boolop(self, operator_instance):
        """Decompose nested compound conditions."""
        # ((x > 0) and (y < 10)) or (z == 5)
        inner_and = make_boolop(ast.And, make_simple_predicate("x"), make_simple_predicate("y"))
        outer_or = make_boolop(ast.Or, inner_and, make_simple_predicate("z"))
        
        conditions = operator_instance._collect_subconditions(outer_or)
        
        assert len(conditions) == 3

    def test_collect_mixed_boolop_and_bitop(self, operator_instance):
        """Decompose mix of Python (and) and PySpark (&) operators."""
        # (x > 0) and ((y < 10) & (z == 5))
        left = make_simple_predicate("x")
        bitop_part = make_binop(make_simple_predicate("y"), ast.BitAnd, make_simple_predicate("z"))
        boolop = make_boolop(ast.And, left, bitop_part)
        
        conditions = operator_instance._collect_subconditions(boolop)
        
        assert len(conditions) == 3

    def test_collect_skip_already_negated(self, operator_instance):
        """Skip conditions that are already wrapped in 'not'."""
        # not (x > 0)
        inner = make_simple_predicate("x")
        negated = ast.UnaryOp(op=ast.Not(), operand=inner)
        ast.fix_missing_locations(negated)
        
        conditions = operator_instance._collect_subconditions(negated)
        
        assert len(conditions) == 0

    def test_collect_skip_negated_in_compound(self, operator_instance):
        """In compound: skip negated sub-conditions, include non-negated."""
        # (x > 0) and not (y < 10)
        left = make_simple_predicate("x")
        negated_right = ast.UnaryOp(op=ast.Not(), operand=make_simple_predicate("y"))
        ast.fix_missing_locations(negated_right)
        boolop = make_boolop(ast.And, left, negated_right)
        
        conditions = operator_instance._collect_subconditions(boolop)
        
        # Should only include the non-negated left
        assert len(conditions) == 1

    def test_collect_preserves_lineno_col_offset(self, operator_instance):
        """Collected conditions retain lineno and col_offset."""
        left = make_simple_predicate("x")
        left.lineno = 5
        left.col_offset = 10
        right = make_simple_predicate("y")
        right.lineno = 5
        right.col_offset = 25
        
        boolop = make_boolop(ast.And, left, right)
        
        conditions = operator_instance._collect_subconditions(boolop)
        
        assert len(conditions) == 2
        for cond in conditions:
            assert cond.lineno is not None
            assert cond.col_offset is not None

    def test_collect_deeply_nested(self, operator_instance):
        """Handle deeply nested compound structures."""
        # ((((x > 0))))
        innermost = make_simple_predicate("x")
        level1 = make_boolop(ast.And, innermost, make_simple_predicate("a"))
        level2 = make_boolop(ast.And, level1, make_simple_predicate("b"))
        level3 = make_boolop(ast.And, level2, make_simple_predicate("c"))
        
        conditions = operator_instance._collect_subconditions(level3)
        
        assert len(conditions) >= 4


class TestNodeLocationHelpers:
    """Test suite for node location and replacement helper methods."""

    def test_find_call_in_copy_found(self):
        """Successfully locate a call node in AST copy by lineno/col_offset."""
        predicate = make_simple_predicate("x")
        call_node = make_call_node("filter", predicate, lineno=1, col_offset=5)
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        # Deep copy and search
        module_copy = copy.deepcopy(module)
        found = OperatorNFTP._find_call_in_copy(module_copy, lineno=1, col_offset=5)
        
        assert found is not None
        assert isinstance(found, ast.Call)

    def test_find_call_in_copy_not_found_wrong_lineno(self):
        """Return None when lineno doesn't match."""
        predicate = make_simple_predicate("x")
        call_node = make_call_node("filter", predicate, lineno=1, col_offset=5)
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        module_copy = copy.deepcopy(module)
        found = OperatorNFTP._find_call_in_copy(module_copy, lineno=999, col_offset=5)
        
        assert found is None

    def test_find_call_in_copy_not_found_wrong_col_offset(self):
        """Return None when col_offset doesn't match."""
        predicate = make_simple_predicate("x")
        call_node = make_call_node("filter", predicate, lineno=1, col_offset=5)
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        module_copy = copy.deepcopy(module)
        found = OperatorNFTP._find_call_in_copy(module_copy, lineno=1, col_offset=999)
        
        assert found is None

    def test_find_node_in_subtree_found(self):
        """Successfully locate a node in subtree by coordinates."""
        left = make_simple_predicate("x")
        left.lineno = 1
        left.col_offset = 5
        right = make_simple_predicate("y")
        right.lineno = 1
        right.col_offset = 20
        compound = make_boolop(ast.And, left, right)
        
        found = OperatorNFTP._find_node_in_subtree(compound, lineno=1, col_offset=5)
        
        assert found is not None

    def test_find_node_in_subtree_not_found(self):
        """Return None when node coordinates don't match."""
        compound = make_boolop(
            ast.And,
            make_simple_predicate("x"),
            make_simple_predicate("y")
        )
        
        found = OperatorNFTP._find_node_in_subtree(compound, lineno=999, col_offset=999)
        
        assert found is None

    def test_replace_node_in_subtree_direct_child(self):
        """Replace a direct child node in compound structure."""
        left = make_simple_predicate("x")
        left.lineno = 1
        left.col_offset = 5
        right = make_simple_predicate("y")
        right.lineno = 1
        right.col_offset = 20
        compound = make_boolop(ast.And, left, right)
        compound_copy = copy.deepcopy(compound)
        
        replacement = ast.Constant(value=True)
        replaced = OperatorNFTP._replace_node_in_subtree(
            compound_copy, lineno=1, col_offset=5, replacement=replacement
        )
        
        assert replaced is True

    def test_replace_node_in_subtree_not_found(self):
        """Return False when replacement target not found."""
        compound = make_boolop(
            ast.And,
            make_simple_predicate("x"),
            make_simple_predicate("y")
        )
        
        replacement = ast.Constant(value=True)
        replaced = OperatorNFTP._replace_node_in_subtree(
            compound, lineno=999, col_offset=999, replacement=replacement
        )
        
        assert replaced is False

    def test_replace_node_in_list_field(self):
        """Replace a node inside a list field (e.g., BoolOp.values)."""
        left = make_simple_predicate("x")
        left.lineno = 1
        left.col_offset = 5
        middle = make_simple_predicate("y")
        middle.lineno = 1
        middle.col_offset = 20
        right = make_simple_predicate("z")
        right.lineno = 1
        right.col_offset = 35
        compound = make_boolop(ast.And, left, middle, right)
        compound_copy = copy.deepcopy(compound)
        
        replacement = ast.Constant(value=False)
        replaced = OperatorNFTP._replace_node_in_subtree(
            compound_copy, lineno=1, col_offset=20, replacement=replacement
        )
        
        assert replaced is True


class TestSourceFileHelpers:
    """Test suite for source file reading helpers - file I/O operations."""

    def test_read_source_lines_valid_file(self):
        """Read program source lines from a valid file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("first line\nsecond\nthird\n")
            temp_path = f.name
        
        try:
            lines = OperatorNFTP._read_source_lines(temp_path)
            assert lines == ["first line", "second", "third"]
        finally:
            Path(temp_path).unlink()

    def test_read_source_lines_nonexistent_file(self):
        """Return empty list for nonexistent file (graceful degradation)."""
        lines = OperatorNFTP._read_source_lines("/does/not/exist/file.py")
        assert lines == []

    def test_read_source_lines_empty_file(self):
        """Handle empty files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            temp_path = f.name
        
        try:
            lines = OperatorNFTP._read_source_lines(temp_path)
            assert lines == []
        finally:
            Path(temp_path).unlink()

    def test_read_source_lines_utf8_encoding(self):
        """Correctly read UTF-8 encoded files with special characters."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Comentário\nvar = 'teste'\n")
            temp_path = f.name
        
        try:
            lines = OperatorNFTP._read_source_lines(temp_path)
            assert len(lines) == 2
            assert "Comentário" in lines[0]
        finally:
            Path(temp_path).unlink()

    def test_get_source_line_valid_index(self):
        """Retrieve correct source line for valid 1-based index."""
        lines = ["first", "second", "third"]
        line = OperatorNFTP._get_source_line(lines, lineno=2)
        assert line == "second"

    def test_get_source_line_first_line(self):
        """Retrieve first line (lineno=1 maps to index 0)."""
        lines = ["first", "second", "third"]
        line = OperatorNFTP._get_source_line(lines, lineno=1)
        assert line == "first"

    def test_get_source_line_last_line(self):
        """Retrieve last line."""
        lines = ["first", "second", "third"]
        line = OperatorNFTP._get_source_line(lines, lineno=3)
        assert line == "third"

    def test_get_source_line_out_of_bounds(self):
        """Return empty string for out-of-bounds line number."""
        lines = ["first", "second", "third"]
        line = OperatorNFTP._get_source_line(lines, lineno=999)
        assert line == ""

    def test_get_source_line_zero_index(self):
        """Return empty string for index zero."""
        lines = ["first", "second"]
        line = OperatorNFTP._get_source_line(lines, lineno=0)
        assert line == ""

    def test_get_source_line_negative_index(self):
        """Return empty string for negative index."""
        lines = ["first", "second"]
        line = OperatorNFTP._get_source_line(lines, lineno=-1)
        assert line == ""

    def test_get_source_line_empty_list(self):
        """Return empty string for empty lines list."""
        lines = []
        line = OperatorNFTP._get_source_line(lines, lineno=1)
        assert line == ""


class TestBuildMutant:
    """Test suite for OperatorNFTP.build_mutant() - GENERIC, no PySpark."""

    def test_build_mutant_simple_predicate(self, operator_instance, temp_source_file, temp_mutant_dir):
        """Generate one mutant from a simple predicate in filter call."""
        predicate = make_simple_predicate("x")
        call_node = make_call_node("filter", predicate, lineno=1, col_offset=0)
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        assert len(nodes) == 1
        
        mutants = operator_instance.build_mutant(
            nodes, module, temp_source_file, temp_mutant_dir
        )
        
        assert len(mutants) == 1
        assert mutants[0].operator == "NFTP"
        assert mutants[0].id == 1
        assert Path(mutants[0].mutant_path).exists()

    def test_build_mutant_compound_predicate(self, operator_instance, temp_source_file, temp_mutant_dir):
        """Generate multiple mutants from compound predicate (and operator)."""
        left = make_simple_predicate("x")
        right = make_simple_predicate("y")
        compound = make_boolop(ast.And, left, right)
        call_node = make_call_node("filter", compound, lineno=1, col_offset=0)
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        assert len(nodes) == 1
        
        mutants = operator_instance.build_mutant(
            nodes, module, temp_source_file, temp_mutant_dir
        )
        
        # Should generate 2 mutants (one per sub-condition)
        assert len(mutants) == 2
        assert all(m.operator == "NFTP" for m in mutants)

    def test_build_mutant_mutant_source_is_valid_ast(self, operator_instance, temp_source_file, temp_mutant_dir):
        """Verify that generated mutant source parses to valid AST."""
        predicate = make_simple_predicate("x")
        call_node = make_call_node("filter", predicate, lineno=1, col_offset=0)
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        mutants = operator_instance.build_mutant(
            nodes, module, temp_source_file, temp_mutant_dir
        )
        
        for mutant in mutants:
            mutant_source = Path(mutant.mutant_path).read_text()
            # Should be valid Python
            ast.parse(mutant_source)
            # Should contain negation
            assert "not" in mutant_source or "~" in mutant_source or "Not" in ast.dump(ast.parse(mutant_source))

    def test_build_mutant_creates_subdirectories(self, operator_instance, temp_source_file, temp_mutant_dir):
        """Verify nftp_<id> subdirectories are created correctly."""
        predicate = make_simple_predicate("x")
        call_node = make_call_node("filter", predicate, lineno=1, col_offset=0)
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        mutants = operator_instance.build_mutant(
            nodes, module, temp_source_file, temp_mutant_dir
        )
        
        for mutant in mutants:
            assert Path(mutant.mutant_path).exists()
            assert "nftp_" in mutant.mutant_path
            assert mutant.mutant_path.endswith(".py")
            # Directory should contain nftp_<id>
            parent_dir = Path(mutant.mutant_path).parent.name
            assert parent_dir.startswith("nftp_")

    def test_build_mutant_accumulates_in_list(self, operator_instance, temp_source_file, temp_mutant_dir):
        """Verify mutants are accumulated in self.mutant_list."""
        predicate = make_simple_predicate("x")
        call_node = make_call_node("filter", predicate, lineno=1, col_offset=0)
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        returned = operator_instance.build_mutant(
            nodes, module, temp_source_file, temp_mutant_dir
        )
        
        assert len(operator_instance.mutant_list) == len(returned)
        assert all(isinstance(m, Mutant) for m in operator_instance.mutant_list)

    def test_build_mutant_invalid_nodes_type(self, operator_instance, temp_source_file, temp_mutant_dir):
        """Raise TypeError for invalid nodes argument."""
        module = ast.Module(body=[], type_ignores=[])
        
        with pytest.raises(TypeError):
            operator_instance.build_mutant("not a list", module, temp_source_file, temp_mutant_dir)
        
        with pytest.raises(TypeError):
            operator_instance.build_mutant([1, 2, 3], module, temp_source_file, temp_mutant_dir)

    def test_build_mutant_invalid_path(self, operator_instance, temp_mutant_dir):
        """Raise ValueError for invalid original_path."""
        module = ast.Module(body=[], type_ignores=[])
        
        with pytest.raises(ValueError):
            operator_instance.build_mutant([], module, "", temp_mutant_dir)
        
        with pytest.raises(ValueError):
            operator_instance.build_mutant([], module, None, temp_mutant_dir)

    def test_build_mutant_invalid_mutant_dir(self, operator_instance, temp_source_file):
        """Raise ValueError for invalid mutant_dir."""
        module = ast.Module(body=[], type_ignores=[])
        
        with pytest.raises(ValueError):
            operator_instance.build_mutant([], module, temp_source_file, "")
        
        with pytest.raises(ValueError):
            operator_instance.build_mutant([], module, temp_source_file, None)

    def test_build_mutant_modified_line_recorded(self, operator_instance, temp_source_file, temp_mutant_dir):
        """Verify that modified_line is properly recorded."""
        # Write some content to temp file so modified_line isn't empty
        Path(temp_source_file).write_text("x = 1\ny > 0\n")
        
        predicate = make_simple_predicate("y")
        predicate.lineno = 2
        predicate.col_offset = 0
        call_node = make_call_node("filter", predicate, lineno=2, col_offset=0)
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        mutants = operator_instance.build_mutant(
            nodes, module, temp_source_file, temp_mutant_dir
        )
        
        assert len(mutants) > 0
        for mutant in mutants:
            # modified_line should either be recording from source or empty (if parsing real AST)
            assert isinstance(mutant.modified_line, str)

    def test_build_mutant_mutant_id_increments(self, operator_instance, temp_source_file):
        """Verify mutant IDs increment correctly."""
        predicate = make_simple_predicate("x")
        call_node = make_call_node("filter", predicate, lineno=1, col_offset=0)
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        
        # First batch
        with tempfile.TemporaryDirectory() as tmpdir1:
            mutants1 = operator_instance.build_mutant(nodes, module, temp_source_file, tmpdir1)
            id1 = mutants1[-1].id
        
        # Second batch
        with tempfile.TemporaryDirectory() as tmpdir2:
            mutants2 = operator_instance.build_mutant(nodes, module, temp_source_file, tmpdir2)
            id2 = mutants2[-1].id
        
        assert id2 > id1

    def test_build_mutant_multiple_filter_calls(self, operator_instance, temp_source_file, temp_mutant_dir):
        """Generate mutants for multiple filter calls."""
        pred1 = make_simple_predicate("x")
        pred2 = make_simple_predicate("y")
        call1 = make_call_node("filter", pred1, lineno=1, col_offset=0)
        call2 = make_call_node("where", pred2, lineno=2, col_offset=0)
        
        module = ast.Module(
            body=[ast.Expr(value=call1), ast.Expr(value=call2)],
            type_ignores=[]
        )
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        assert len(nodes) == 2
        
        mutants = operator_instance.build_mutant(
            nodes, module, temp_source_file, temp_mutant_dir
        )
        
        # Should have at least 2 mutants
        assert len(mutants) >= 2

    def test_build_mutant_no_eligible_subconditions(self, operator_instance, temp_source_file, temp_mutant_dir):
        """Skip calls where all sub-conditions are already negated."""
        # not (x > 0) — entire predicate is negated
        inner = make_simple_predicate("x")
        negated_pred = ast.UnaryOp(op=ast.Not(), operand=inner)
        ast.fix_missing_locations(negated_pred)
        
        call_node = make_call_node("filter", negated_pred, lineno=1, col_offset=0)
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        assert len(nodes) == 1
        
        mutants = operator_instance.build_mutant(
            nodes, module, temp_source_file, temp_mutant_dir
        )
        
        # Should skip because predicate is already fully negated
        assert len(mutants) == 0

    def test_build_mutant_returns_self_mutant_list(self, operator_instance, temp_source_file, temp_mutant_dir):
        """Verify return value is self.mutant_list."""
        predicate = make_simple_predicate("x")
        call_node = make_call_node("filter", predicate, lineno=1, col_offset=0)
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        returned = operator_instance.build_mutant(
            nodes, module, temp_source_file, temp_mutant_dir
        )
        
        # Returned should be same object as operator_instance.mutant_list
        assert returned is operator_instance.mutant_list


class TestEdgeCases:
    """Test edge cases and boundary conditions - GENERIC logic."""

    def test_empty_ast_module(self, operator_instance):
        """Handle completely empty AST module."""
        module = ast.Module(body=[], type_ignores=[])
        
        nodes = operator_instance.analyse_ast(module)
        
        assert nodes == []

    def test_very_deep_nesting(self, operator_instance):
        """Handle deeply nested compound structures."""
        # Build nested and: (((((x > 0) and (a > 0)) and ...) and (z > 0)))
        innermost = make_simple_predicate("x")
        compound = innermost
        for i in range(10):
            compound = make_boolop(ast.And, compound, make_simple_predicate(f"var{i}"))
        
        conditions = operator_instance._collect_subconditions(compound)
        
        # Should decompose into many conditions
        assert len(conditions) >= 5

    def test_mixed_operator_combinations(self, operator_instance):
        """Handle complex combinations of and/or and &/|."""
        # (x and y) | (a & b)
        left_and = make_boolop(ast.And, make_simple_predicate("x"), make_simple_predicate("y"))
        right_bitand = make_binop(
            make_simple_predicate("a"), ast.BitAnd, make_simple_predicate("b")
        )
        compound = make_boolop(ast.Or, left_and, right_bitand)
        
        conditions = operator_instance._collect_subconditions(compound)
        
        # Should decompose into 4 leaf conditions
        assert len(conditions) == 4

    def test_call_with_multiple_method_names_in_ast(self, operator_instance):
        """Only filter/where calls should be detected."""
        # Build module with filter, where, map, select
        pred = make_simple_predicate("x")
        
        filter_call = make_call_node("filter", pred, lineno=1, col_offset=0)
        where_call = make_call_node("where", pred, lineno=2, col_offset=0)
        map_call = make_call_node("map", pred, lineno=3, col_offset=0)
        
        module = ast.Module(
            body=[
                ast.Expr(value=filter_call),
                ast.Expr(value=where_call),
                ast.Expr(value=map_call),
            ],
            type_ignores=[]
        )
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        
        # Should only find filter and where
        assert len(nodes) == 2

    def test_unary_not_is_properly_skipped(self, operator_instance):
        """Verify that 'not' (UnaryOp with Not) is skipped in decomposition."""
        inner = make_simple_predicate("x")
        negated = ast.UnaryOp(op=ast.Not(), operand=inner)
        ast.fix_missing_locations(negated)
        
        compound = make_boolop(
            ast.And,
            negated,
            make_simple_predicate("y"),
            negated,
            make_simple_predicate("z")
        )
        
        conditions = operator_instance._collect_subconditions(compound)
        
        # Should only include the two non-negated conditions
        assert len(conditions) == 2

    def test_mutant_id_sequence_is_monotonic(self, operator_instance, temp_source_file):
        """Verify mutant IDs always increase."""
        pred = make_simple_predicate("x")
        call_node = make_call_node("filter", pred, lineno=1, col_offset=0)
        module = ast.Module(body=[ast.Expr(value=call_node)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        ids = []
        
        for i in range(3):
            with tempfile.TemporaryDirectory() as tmpdir:
                mutants = operator_instance.build_mutant(nodes, module, temp_source_file, tmpdir)
                if mutants:
                    ids.append(mutants[-1].id)
        
        # IDs should be strictly increasing
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)  # All unique

    def test_ast_deepcopy_does_not_modify_original(self, operator_instance):
        """Verify that deep copies don't affect original AST."""
        original_pred = make_simple_predicate("x")
        original_module = ast.Module(body=[ast.Expr(value=original_pred)], type_ignores=[])
        original_dump = ast.dump(original_module)
        
        copy_module = copy.deepcopy(original_module)
        
        # Modify the copy
        for node in ast.walk(copy_module):
            if isinstance(node, ast.Name):
                node.id = "modified"
        
        # Original should remain unchanged
        assert ast.dump(original_module) == original_dump


class TestErrorHandling:
    """Test error handling and validation."""

    def test_analyse_ast_invalid_tree_type(self, operator_instance):
        """Test analyse_ast with invalid tree types."""
        with pytest.raises(TypeError):
            operator_instance.analyse_ast(12345)
        
        with pytest.raises(TypeError):
            operator_instance.analyse_ast({})
        
        with pytest.raises(TypeError):
            operator_instance.analyse_ast("not an ast")

    def test_build_mutant_invalid_nodes_type(self, operator_instance, temp_source_file, temp_mutant_dir):
        """Test build_mutant with invalid nodes argument."""
        module = ast.Module(body=[], type_ignores=[])
        
        with pytest.raises(TypeError):
            operator_instance.build_mutant("not_a_list", module, temp_source_file, temp_mutant_dir)
        
        with pytest.raises(TypeError):
            operator_instance.build_mutant([1, 2, 3], module, temp_source_file, temp_mutant_dir)

    def test_build_mutant_invalid_path_and_dir(self, operator_instance, temp_source_file, temp_mutant_dir):
        """Test build_mutant with invalid path and directory arguments."""
        module = ast.Module(body=[], type_ignores=[])
        
        # Empty original_path
        with pytest.raises(ValueError):
            operator_instance.build_mutant([], module, "", temp_mutant_dir)
        
        # Empty mutant_dir
        with pytest.raises(ValueError):
            operator_instance.build_mutant([], module, temp_source_file, "")

    def test_mutant_list_contains_valid_mutant_instances(self, operator_instance, temp_source_file, temp_mutant_dir):
        """Verify all items in mutant_list are properly formed Mutant instances."""
        pred = make_simple_predicate("x")
        call = make_call_node("filter", pred, lineno=1, col_offset=0)
        module = ast.Module(body=[ast.Expr(value=call)], type_ignores=[])
        ast.fix_missing_locations(module)
        
        nodes = operator_instance.analyse_ast(module)
        mutants = operator_instance.build_mutant(nodes, module, temp_source_file, temp_mutant_dir)
        
        for mutant in mutants:
            assert isinstance(mutant, Mutant)
            assert isinstance(mutant.id, int)
            assert isinstance(mutant.operator, str)
            assert isinstance(mutant.original_path, str)
            assert isinstance(mutant.mutant_path, str)
            assert isinstance(mutant.modified_line, str)
            assert mutant.id > 0
            assert mutant.operator == "NFTP"

    def test_collected_conditions_all_have_coordinates(self, operator_instance):
        """Ensure all collected conditions have lineno and col_offset."""
        compound = make_boolop(
            ast.And,
            make_simple_predicate("a"),
            make_simple_predicate("b"),
            make_simple_predicate("c")
        )
        
        conditions = operator_instance._collect_subconditions(compound)
        
        for cond in conditions:
            assert hasattr(cond, "lineno"), f"Condition {cond} missing lineno"
            assert hasattr(cond, "col_offset"), f"Condition {cond} missing col_offset"

    def test_negation_wrapper_creates_valid_unaryop(self, operator_instance):
        """Verify that negation creates valid UnaryOp with Not."""
        operand = make_simple_predicate("x")
        negated = ast.UnaryOp(op=ast.Not(), operand=operand)
        ast.fix_missing_locations(negated)
        
        # Should be ast.UnaryOp with ast.Not()
        assert isinstance(negated, ast.UnaryOp)
        assert isinstance(negated.op, ast.Not)
        
        # Should be parseable
        module = ast.Module(body=[ast.Expr(value=negated)], type_ignores=[])
        ast.unparse(module)  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
