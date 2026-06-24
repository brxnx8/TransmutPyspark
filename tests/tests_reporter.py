import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Ajuste o import conforme a estrutura real do projeto
from src.reporter.reporter import (
    Reporter,
    _normalise_source,
    _normalised_lines,
    _compute_diff,
)
from src.model.test_result import TestResult


# --- Fixtures e Mock Factories ---

def make_result(mutant_id: int, status: str = "killed", failed_tests: list = None, exec_time: float = 0.5):
    """Cria um TestResult mockado que passa pela verificação isinstance(obj, TestResult)."""
    m = MagicMock(spec=TestResult)
    m.mutant = mutant_id
    m.status = status
    m.failed_tests = failed_tests or []
    m.execution_time = exec_time
    return m

def make_mutant(m_id: int, operator: str = "ATR", orig_path: str = "orig.py", mut_path: str = "mut.py", mod_line: str = "line 1"):
    """Cria um objeto simulando um mutante com os atributos essenciais."""
    m = MagicMock()
    m.id = m_id
    m.operator = operator
    m.original_path = orig_path
    m.mutant_path = mut_path
    m.modified_line = mod_line
    return m

@pytest.fixture
def reporter_data():
    r1 = make_result(1, "killed", ["test_abc"])
    r2 = make_result(2, "survived")
    r3 = make_result(3, "timeout")
    r4 = make_result(4, "error")

    m1 = make_mutant(1, orig_path="source_a.py")
    m2 = make_mutant(2, operator="MTR", orig_path="source_b.py")
    m3 = make_mutant(3, orig_path="source_a.py")
    m4 = make_mutant(4, orig_path="source_b.py")

    return {
        "results": [r1, r2, r3, r4],
        "mutants": [m1, m2, m3, m4],
        "code_original": "a = 1\n",
        "output_dir": Path("/tmp/report")
    }

@pytest.fixture
def reporter(reporter_data):
    return Reporter(
        result_list=reporter_data["results"],
        code_original=reporter_data["code_original"],
        mutant_list=reporter_data["mutants"],
        output_dir=reporter_data["output_dir"]
    )


# --- Testes de Helpers ---

def test_should_normalize_source_when_valid_python_code():
    code = "def f():\n  x=1\n  return x"
    norm = _normalise_source(code)
    # ast.unparse padroniza a formatação adicionando espaços ao redor do '='
    assert "def f():" in norm
    assert "x = 1" in norm

def test_should_return_original_source_when_syntax_error_occurs():
    code = "def invalid() -> :\n missing"
    assert _normalise_source(code) == code

def test_should_return_normalized_lines_with_newlines():
    code = "a=1\nb=2"
    lines = _normalised_lines(code)
    assert lines == ["a = 1\n", "b = 2\n"]

def test_should_compute_unified_diff_between_sources():
    orig = "a = 1"
    mut = "a = 2"
    diff = _compute_diff(orig, mut, 1)
    
    assert "--- original.py" in diff
    assert "+++ mutant_1.py" in diff
    assert "-a = 1" in diff
    assert "+a = 2" in diff


# --- Testes de Validação (Guards) ---

def test_should_raise_type_error_when_result_list_is_not_list():
    with pytest.raises(TypeError, match="must be a list"):
        Reporter("not a list", "code", [], Path("."))

def test_should_raise_type_error_when_result_list_contains_invalid_items():
    with pytest.raises(TypeError, match="must be TestResult instances"):
        Reporter(["invalid object"], "code", [], Path("."))

def test_should_raise_value_error_when_code_original_is_invalid():
    with pytest.raises(ValueError, match="must be a non-empty string"):
        Reporter([], "", [], Path("."))

def test_should_raise_type_error_when_mutant_list_is_not_list():
    with pytest.raises(TypeError, match="must be a list"):
        Reporter([], "code", "not a list", Path("."))

def test_should_raise_type_error_when_output_dir_is_not_path():
    with pytest.raises(TypeError, match="must be a Path instance"):
        Reporter([], "code", [], "/tmp/string/path")


# --- Testes da API Principal ---

def test_should_calculate_mutation_score_with_mixed_results(reporter):
    reporter.calculate()
    calc = reporter.result_calculate

    assert calc["total"] == 4
    assert calc["killed"] == 1
    assert calc["survived"] == 1
    assert calc["timeout"] == 1
    assert calc["error"] == 1
    assert calc["mutation_score"] == 0.25
    assert "ATR" in calc["by_operator"]
    assert "MTR" in calc["by_operator"]

def test_should_calculate_zero_score_when_no_results_provided():
    r = Reporter([], "code", [], Path("."))
    r.calculate()
    assert r.result_calculate["mutation_score"] == 0.0


def test_should_raise_runtime_error_when_make_diff_called_before_calculate(reporter):
    with pytest.raises(RuntimeError, match="Call calculate() before"):
        reporter.make_diff()

def custom_read_text_success(self, encoding="utf-8"):
    if "mut.py" in str(self):
        return "a = 2"
    return "a = 1"

@patch.object(Path, "read_text", autospec=True, side_effect=custom_read_text_success)
def test_should_generate_diffs_when_mutants_and_files_exist(mock_read_text, reporter):
    reporter.calculate()
    reporter.make_diff()

    diffs = reporter.result_calculate["diff_original_code"]
    assert len(diffs) == 4
    assert "-a = 1" in diffs[0]["diff"]
    assert "+a = 2" in diffs[0]["diff"]
    assert diffs[0]["mutant_source"] == "a = 2"

def custom_read_text_missing_mutant(self, encoding="utf-8"):
    raise FileNotFoundError("Mutant not found")

@patch.object(Path, "read_text", autospec=True, side_effect=custom_read_text_missing_mutant)
def test_should_skip_diff_when_mutant_file_not_found(mock_read_text, reporter):
    reporter.calculate()
    reporter.make_diff()
    diffs = reporter.result_calculate["diff_original_code"]
    # Todos os mutantes dispararão exceção, logo nenhum diff será montado.
    assert len(diffs) == 0

def custom_read_text_missing_original(self, encoding="utf-8"):
    if "mut.py" in str(self):
        return "a = 2"
    raise FileNotFoundError("Original not found")

@patch.object(Path, "read_text", autospec=True, side_effect=custom_read_text_missing_original)
def test_should_fallback_to_code_original_when_original_file_not_found(mock_read_text, reporter):
    reporter.calculate()
    reporter.make_diff()
    diffs = reporter.result_calculate["diff_original_code"]
    
    assert len(diffs) == 4
    # Como não encontrou o arquivo original, fará o diff usando o self.code_original ("a = 1\n")
    assert "-a = 1" in diffs[0]["diff"]
    assert "+a = 2" in diffs[0]["diff"]


def test_should_raise_runtime_error_when_show_results_called_before_calculate(reporter):
    with pytest.raises(RuntimeError):
        reporter.show_results()

@patch.object(Path, "write_text", autospec=True)
def test_should_build_and_write_html_report_to_output_dir(mock_write_text, reporter):
    reporter.calculate()
    reporter.show_results()
    
    mock_write_text.assert_called_once()
    args, kwargs = mock_write_text.call_args
    # args[0] será o path construído
    assert "report.html" in str(args[0])
    # args[1] será o HTML
    assert "<html" in args[1]


# --- Testes do Construtor HTML e Formatação ---

def test_should_generate_html_with_strong_score(reporter_data):
    r1 = make_result(1, "killed")
    r2 = make_result(2, "killed")
    rep = Reporter([r1, r2], "code", reporter_data["mutants"][:2], Path("."))
    rep.calculate()
    html = rep._build_html()
    assert "STRONG" in html

def test_should_generate_html_with_moderate_score(reporter_data):
    r1 = make_result(1, "killed")
    r2 = make_result(2, "survived")
    rep = Reporter([r1, r2], "code", reporter_data["mutants"][:2], Path("."))
    rep.calculate()
    html = rep._build_html()
    assert "MODERATE" in html

def test_should_generate_html_with_weak_score(reporter_data):
    r1 = make_result(1, "survived")
    r2 = make_result(2, "survived")
    rep = Reporter([r1, r2], "code", reporter_data["mutants"][:2], Path("."))
    rep.calculate()
    html = rep._build_html()
    assert "WEAK" in html

def test_should_handle_mutant_with_no_result():
    m = make_mutant(99, operator="UNKNOWN")
    rep = Reporter([], "code", [m], Path("."))
    rep.calculate()
    html = rep._build_html()
    
    # Se o resultado não estiver indexado, o código entra nas fallbacks
    assert "UNKNOWN" in html
    assert "status-unknown" in html

def test_should_handle_missing_diff_and_source_in_html(reporter):
    reporter.calculate()
    # Chama _build_html DIRETAMENTE sem chamar make_diff() antes
    # Isso simulará mutantes que não possuíram processamento de diff com sucesso
    html = reporter._build_html()
    
    assert "No diff detected" in html
    assert "Source unavailable" in html
    assert "test_abc" in html  # Garante que um failed test presente na fixture será listado

def test_should_return_no_results_message_when_no_mutants_exist():
    rep = Reporter([], "code", [], Path("."))
    rep.calculate()
    html = rep._build_html()
    assert "No results available." in html

def test_should_escape_html_characters():
    text = 'a & b < c > d "e"'
    escaped = Reporter._escape(text)
    assert escaped == 'a &amp; b &lt; c &gt; d &quot;e&quot;'

def test_should_colorise_diff_lines():
    diff = "+++ file_a\n--- file_b\n@@ -1 +1 @@\n+add line\n-del line\n context line"
    html = Reporter._colorise_diff(diff)
    
    assert "<span class='diff-hdr'>+++ file_a</span>" in html
    assert "<span class='diff-hdr'>--- file_b</span>" in html
    assert "<span class='diff-hdr'>@@ -1 +1 @@</span>" in html
    assert "<span class='diff-add'>+add line</span>" in html
    assert "<span class='diff-del'>-del line</span>" in html
    assert " context line" in html


# --- Outros Métodos ---

def test_should_return_string_representation_of_reporter(reporter):
    rep = repr(reporter)
    assert "Reporter(results=4, mutants=4," in rep
    assert "not calculated" in rep

    reporter.calculate()
    rep_calc = repr(reporter)
    assert "score=0.25" in rep_calc